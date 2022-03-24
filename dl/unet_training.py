import gc
import os

import torch
import json
from tqdm import tqdm
import dl.metrics
import utils
from unet_model import Unet
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from dl.dataloader import CamusDatasetPNG, KFoldLoaders, get_transforms, get_full_dataset_loader
from colorama import Fore, Style
import torch.optim as optim
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from pynvml.smi import nvidia_smi


def train_loop(unet, train_loader, val_loader, savename, val_metrics, epochs, optimizer, loss_func, **kwargs):
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
    if val_loader is not None:
        evaluate_unet(unet, val_loader, val_metrics)


def train(optimizer, loss_func, train_loop, unet):
    running_loss = 0.
    torch.enable_grad()
    unet.train()
    for _, data in enumerate(train_loop):
        img, gt, _ = data
        # input_names = ['img']
        # output_names = ['seg']
        # torch.onnx.export(unet, img, 'unet.onnx', input_names=input_names, output_names=output_names)
        optimizer.zero_grad()
        del data
        with torch.cuda.amp.autocast():
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


def evaluate_unet(unet, val_loader, val_metrics):
    unet.eval()
    val_loader = iter(val_loader)
    next_batch = next(val_loader)
    n_classes = next_batch[1].size()[1]
    metric_lists_ES = [[] for _ in range(n_classes)]  # one list per class
    metric_lists_ED = [[] for _ in range(n_classes)]  # one list per class
    try:
        while True:
            with torch.no_grad():
                img, seg, ED_or_ES = next_batch
                prediction = torch.softmax(unet(img), dim=1)

            for i in range(img.shape[0]):
                for n in range(n_classes):
                    # append 1x3 tensor per class containing precision recall and f1 for the class
                    if ED_or_ES[i] == 1:
                        metric_lists_ED[n].append(dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :]))
                    elif ED_or_ES[i] == 2:
                        metric_lists_ES[n].append(dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :]))

            next_batch = next(val_loader)

    except StopIteration:
        metric_lists_ED = torch.FloatTensor(metric_lists_ED)
        metric_lists_ES = torch.FloatTensor(metric_lists_ES)
        for metric_list, val_m_dict_key in zip([metric_lists_ED, metric_lists_ES], ['ED', 'ES']):
            p_list = []
            r_list = []
            f1_list = []
            for n in range(n_classes):
                p_list.append(torch.mean(metric_list[n, :, 0]))
                r_list.append(torch.mean(metric_list[n, :, 1]))
                f1_list.append(torch.mean(metric_list[n, :, 2]))
            val_metrics[val_m_dict_key]['p'].append(p_list)
            val_metrics[val_m_dict_key]['r'].append(r_list)
            val_metrics[val_m_dict_key]['f1'].append(f1_list)
        pass


def save_metrics(savename, val_metrics_ES_and_ED):
    m_frames = []
    for val_metrics in val_metrics_ES_and_ED.values():
        fold_arrays = []
        p_tensor = torch.tensor(val_metrics['p'])
        r_tensor = torch.tensor(val_metrics['r'])
        f1_tensor = torch.tensor(val_metrics['f1'])
        folds, n_classes = p_tensor.shape
        for fold in range(folds):
            fold_arrays.append(np.stack((p_tensor[fold], r_tensor[fold], f1_tensor[fold])))

        row_idxs = pd.MultiIndex.from_product(
            [range(folds), list(val_metrics.keys())],
            names=['fold', 'metric']
        )
        col_sub_idxs = pd.MultiIndex.from_product([range(n_classes)], names=['class'])
        metrics_frame = pd.DataFrame(np.vstack(fold_arrays), index=row_idxs, columns=col_sub_idxs)
        avgs = calc_metric_avgs(metrics_frame, list(val_metrics.keys()))
        m_frames.append(pd.concat([metrics_frame, avgs]))

    metrics_frame_full = pd.concat(m_frames, keys=list(val_metrics_ES_and_ED.keys()), axis=1)
    metrics_frame_full.to_csv(f'{savename}val_metrics.csv')
    return metrics_frame_full


def calc_metric_avgs(metrics_frame, metrics_name):
    avgs = []
    f1_std = []
    for metric in metrics_name:
        avgs.append(metrics_frame.xs(metric, level=1).mean())
        if metric == 'f1':
            f1_std.append(metrics_frame.xs(metric, level=1).std())

    metrics_name.append('f1_std_dev')
    row_idxs = pd.MultiIndex.from_product(
        [['avg'], metrics_name],
        names=['fold', 'metric']
    )
    return pd.DataFrame([*avgs, *f1_std], index=row_idxs)


def kfold_train_unet(unet, foldername, train_settings, dataloader_settings, **kwargs):
    """
    Train unet and calculate metrics in k-folds. kwargs are ignored.
    """
    val_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}

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

        train_loop(unet, train_loader, val_loader, path, val_metrics, **train_settings)
        del train_loader.dataset.aug_imgs, train_loader.dataset.aug_segs
        del val_loader.dataset.aug_imgs, val_loader.dataset.aug_segs
        del train_loader, val_loader
        unet.zero_grad()
        unet.reset_params()
        gc.collect()
        torch.cuda.empty_cache()

    metrics_frame = save_metrics(foldername, val_metrics)
    print('ED')
    print(metrics_frame.xs('avg').xs('ED', axis=1))
    print('\nES')
    print(metrics_frame.xs('avg').xs('ES', axis=1))


def full_train_unet(unet, foldername, train_settings, dataloader_settings, **kwargs):
    val_metrics = {'ED': {'p': [], 'r': [], 'f1': []}, 'ES': {'p': [], 'r': [], 'f1': []}}

    if dataloader_settings['augments'] is True:
        dataloader_settings['augments'] = get_transforms(**aug_settings)
    else:
        dataloader_settings['augments'] = None

    train_loader = get_full_dataset_loader(**dataloader_settings)


if __name__ == '__main__':
    # unet = load_unet(filename, channels=n_ch, levels=levels)

    unet_settings = {
        'levels': 5,
        'top_feature_ch': 32,
        'output_ch': 4
    }
    unet = Unet(**unet_settings).cuda()

    train_settings = {
        "epochs": 100,
        "do_val": True,
        "loss_func": dl.metrics.FscoreLoss(class_weights=torch.tensor([0.01, 1, 1, 1], device='cuda:0'),
                                           f1_weight=0.7),
        "optimizer": optim.Adam(unet.parameters(), lr=1e-5, weight_decay=1e-4),
    }

    aug_settings = {
        "h_flip_p": 0.5,
        "elastic_alpha": 110,
        "elastic_sigma": 15,
        "elastic_affine": 7,
        "bright_lim": 0.4,
        "contrast_lim": 0.2,

    }

    dataset = "camus_png"
    dataloader_settings = {
        "batch_size": 8,
        "split": 8,
        "dataset": CamusDatasetPNG(dataset=dataset),
        "augments": True,
        "n_train_aug_threads": 2,
    }

    settings = {'unet_settings': unet_settings,
                'train_settings': train_settings,
                'aug_settings': aug_settings,
                'dataloader_settings': dataloader_settings,
                }

    foldername = f"train_results/{dataset}/unet_{unet_settings['levels']}" \
                 f"levels_augment_{dataloader_settings['augments']}" \
                 f"_{unet_settings['top_feature_ch']}top/"
    pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f'Trainable parameters: {pytorch_total_params}')
    print(f'Feature maps: {unet.channels}')
    os.makedirs(foldername, exist_ok=True)
    with open(f'{foldername}settings.json', 'w') as file:
        json.dump(settings, file, indent=2, default=utils.call_json_serializer)

    kfold_train_unet(unet, foldername, **settings)

    ''' TODO
        - change aug params
        - put us sim img gt though coord transform
        - plot CNN
        - run 4 level 64 top and 5 level 64 top
        - denoise and then noise
    '''
