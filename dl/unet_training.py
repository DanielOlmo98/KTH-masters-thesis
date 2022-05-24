import gc
import os

import torch
import json
from tqdm import tqdm
import dl.metrics
import utils
from unet_model import Unet
from old_wavelet_unet_model import OldWaveletUnet
from wavelet_unet_model import WaveletUnet
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from dl.dataloader import CamusDatasetPNG, KFoldLoaders, get_transforms, get_full_dataset_loader
from colorama import Fore, Style
import torch.optim as optim
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from pynvml.smi import nvidia_smi
from pytorch_wavelets import DWTForward, DWTInverse


def train_loop(unet, train_loader, val_loader, savename, epochs, optimizer, loss_func, **kwargs):
    bar = Fore.WHITE + '{l_bar}{bar}' + Fore.WHITE + '| {n_fmt}/{total_fmt} [{elapsed}{postfix}]'
    bar_val = Fore.GREEN + '{l_bar}{bar}' + Fore.GREEN + '| {n_fmt}/{total_fmt} [{elapsed}  {postfix}]'
    val_min_loss = 9
    # gc.collect()
    # torch.cuda.empty_cache()
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):

        train_loop = tqdm(train_loader, colour='white', desc=f'Epoch {epoch + 1:03}/{epochs:03},   training',
                          bar_format=bar, leave=True, position=0)
        train_loss_epoch = train(optimizer, loss_func, train_loop, unet)
        del train_loop
        if val_loader is not None:
            val_loop = tqdm(val_loader, colour='green', desc=f'               validating', bar_format=bar_val,
                            leave=True,
                            position=0)
            val_loss_epoch = val(loss_func, val_loop, unet)
            del val_loop

            val_loss_list.append(val_loss_epoch)
        else:
            val_loss_epoch = train_loss_epoch

        if val_loss_epoch < val_min_loss:
            torch.save(unet.state_dict(), f'{savename}.pt')
            print("\nModel Saved")
            val_min_loss = val_loss_epoch

        gc.collect()
        torch.cuda.empty_cache()

        train_loss_list.append(train_loss_epoch)

        # Wait for augmentation queues to finish if there are any
        if train_loader.dataset.n_aug_threads > 0:
            train_loader.dataset.join_queues()

    utils.plot_losses(train_loss_list, val_loss_list, filename=f'{savename}_loss.png')


def train(optimizer, loss_func, train_loop, unet):
    running_loss = 0.
    torch.enable_grad()
    unet.train()
    for _, data in enumerate(train_loop):
        img, gt, _ = data
        # utils.plot_image_g(img[0])
        # input_names = ['img']
        # output_names = ['seg']
        # torch.onnx.export(unet, img, 'unet.onnx', input_names=input_names, output_names=output_names)
        optimizer.zero_grad()
        del data
        # with torch.cuda.amp.autocast():
        output = unet(img)
        loss = loss_func(output, gt)

        loss.backward()
        optimizer.step()

        train_loop.set_postfix(train_loss=loss.item())
        running_loss += loss.item()
    return running_loss / train_loop.n


def val(loss_func, val_loop, unet):
    unet.eval()
    torch.no_grad()
    val_running_loss = 0.
    for _, data in enumerate(val_loop):
        val_img, val_gt, _ = data
        del data
        with torch.cuda.amp.autocast():
            val_output = unet(val_img)
            val_loss = loss_func(val_output, val_gt)

        val_loop.set_postfix(val_loss=val_loss.item())
        val_running_loss += val_loss.item()
    return val_running_loss / val_loop.n


def kfold_train_unet(unet, foldername, train_settings, dataloader_settings, **kwargs):
    """
    Train unet and calculate metrics in k-folds. kwargs are ignored.
    """

    if dataloader_settings['augments'] is True:
        dataloader_settings['augments'] = get_transforms(**aug_settings)
    else:
        dataloader_settings['augments'] = None

    kf_loader = KFoldLoaders(**dataloader_settings)

    fold_count = 0

    for i in range(dataloader_settings['split']):
        if os.path.exists(f'{foldername}fold_{i}_loss.png'):
            print(f'Skipping fold {i}')
            fold_count = i + 1
            kf_loader.skip_fold()

    for fold, (train_loader, val_loader) in enumerate(kf_loader):
        fold += fold_count
        path = f'{foldername}fold_{fold}'
        print(f'Fold #{fold}')

        train_loop(unet, train_loader, val_loader, path, **train_settings)
        del train_loader.dataset.aug_imgs, train_loader.dataset.aug_segs
        del val_loader.dataset.aug_imgs, val_loader.dataset.aug_segs
        del train_loader, val_loader
        unet.zero_grad()
        unet.reset_params()
        gc.collect()
        torch.cuda.empty_cache()


def full_train_unet(unet, foldername, train_settings, dataloader_settings, **kwargs):
    if dataloader_settings['augments'] is True:
        dataloader_settings['augments'] = get_transforms(**aug_settings)
    else:
        dataloader_settings['augments'] = None

    train_loader = get_full_dataset_loader(**dataloader_settings)
    train_loop(unet, train_loader, val_loader=None, savename=f'{foldername}full_dataset', **train_settings)


def kfold_train(dataset='camus_png'):
    unet_settings = {
        'levels': 5,
        'top_feature_ch': 16,
        'output_ch': 4,
        'wavelet': False


    }

    wavelet_unet = unet_settings['wavelet']
    if wavelet_unet:
        if wavelet_unet == 'same':
            unet = OldWaveletUnet(**unet_settings).cuda()
        elif wavelet_unet == 'decrease':
            unet = WaveletUnet(**unet_settings).cuda()
    else:
        unet = Unet(**unet_settings).cuda()

    pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    unet_settings['trainable_params'] = pytorch_total_params
    train_settings = {
        "epochs": 100,
        "do_val": True,
        "loss_func": dl.metrics.FscoreLoss(class_weights=torch.tensor([0.01, 1, 1, 1], device='cuda:0'),
                                           f1_weight=0.7),
        "optimizer": optim.Adam(unet.parameters(), lr=5e-5, weight_decay=1e-4),
    }

    aug_settings = {
        "h_flip_p": 0.5,
        "elastic_alpha": 110,
        "elastic_sigma": 15,
        "elastic_affine": 7,
        "bright_lim": 0.4,
        "contrast_lim": 0.2,

    }

    # import albumentations as A
    # import albumentations.pytorch
    #
    # transformer = [A.GaussNoise(p=1., var_limit=(10, 50)), A.pytorch.ToTensorV2()]
    # transformer = A.Compose(transformer)

    dataloader_settings = {
        "batch_size": 8,
        "split": 8,
        "dataset": CamusDatasetPNG(dataset=dataset),
        "augments": False,
        "n_train_aug_threads": 2,
    }

    settings = {'unet_settings': unet_settings,
                'train_settings': train_settings,
                'aug_settings': aug_settings,
                'dataloader_settings': dataloader_settings,
                }
    # {dataloader_settings['augments']}
    waveletstr = f'wavelet_{wavelet_unet}' if wavelet_unet else ''
    foldername = f"train_results/{dataset}/{waveletstr}unet_{unet_settings['levels']}levels" \
                 f"_augment_{dataloader_settings['augments']}" \
                 f"_{unet_settings['top_feature_ch']}top/"
    print(f'Trainable parameters: {pytorch_total_params}')
    # print(f'Feature maps: {unet.channels}')
    os.makedirs(foldername, exist_ok=True)
    with open(f'{foldername}settings.json', 'w') as file:
        json.dump(settings, file, indent=2, default=utils.call_json_serializer)

    kfold_train_unet(unet, foldername, **settings)


if __name__ == '__main__':
    kfold_train(dataset="camus_combined_50-0.1_w0.7_eps0.001")

    ''' TODO
        - tv vs combined
        - check performance on high/low quality images for networks
        - calc psnr of noise aug
    '''
