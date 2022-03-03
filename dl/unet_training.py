import gc
import os

import torch
from tqdm import tqdm

import dl.metrics
import utils
from unet_model import Unet
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from dl.dataloader import CamusDatasetPNG, KFoldLoaders, get_loaders
from colorama import Fore, Style
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from pynvml.smi import nvidia_smi


def train_loop(unet, train_loader, val_loader, savename, val_metrics, epochs, optimizer, loss_func, do_val=True):
    tb_writer = SummaryWriter()
    bar = Fore.WHITE + '{l_bar}{bar}' + Fore.WHITE + '| {n_fmt}/{total_fmt} [{elapsed}{postfix}]'
    bar_val = Fore.GREEN + '{l_bar}{bar}' + Fore.GREEN + '| {n_fmt}/{total_fmt} [{elapsed}  {postfix}]'
    val_min_loss = 9
    # gc.collect()
    # torch.cuda.empty_cache()

    for epoch in range(epochs):

        train_loop = tqdm(train_loader, colour='white', desc=f'Epoch {epoch + 1:03}/{epochs:03},   training',
                          bar_format=bar, leave=True, position=0)
        train_loss_epoch = train(optimizer, loss_func, train_loop, unet)
        del train_loop

        if do_val:
            val_loop = tqdm(val_loader, colour='green', desc=f'               validating', bar_format=bar_val,
                            leave=True,
                            position=0)
            val_loss_epoch = val(loss_func, val_loop, unet)
            del val_loop

            if val_loss_epoch < val_min_loss:
                torch.save(unet.state_dict(), savename)
                print("\nModel Saved")
                val_min_loss = val_loss_epoch

            tb_writer.add_scalars("loss per epoch", {
                'train loss': train_loss_epoch,
                'val loss': val_loss_epoch,
            }, epoch)
        else:
            tb_writer.add_scalar('Loss/train', train_loss_epoch, epoch)
            if epoch % 25 == 0:
                print("\nModel Saved")
                torch.save(unet.state_dict(), savename)

        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # print(f"\nReserved:  {r * 1e-9:.2f} GB")
        # print(f"Allocated: {a * 1e-9:.2f} GB")

    tb_writer.close()
    evaluate_unet(unet, val_loader, val_metrics)


def train(optimizer, loss_func, train_loop, unet):
    running_loss = 0.
    torch.enable_grad()
    unet.train()
    for _, data in enumerate(train_loop):
        img, gt = data
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
        val_img, val_gt = data
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
    metric_lists = [[] for _ in range(n_classes)]  # one list per class
    try:
        while True:
            with torch.no_grad():
                img, seg = next_batch
                prediction = torch.softmax(unet(img), dim=1)

            for n in range(n_classes):
                # append 1x3 tensor per class containing precision recall and f1 for the class
                metric_lists[n].append(dl.metrics.get_f1_metrics(prediction[:, n, :, :], seg[:, n, :, :]))

            next_batch = next(val_loader)

    except StopIteration:
        metric_lists = torch.FloatTensor(metric_lists)
        p_list = []
        r_list = []
        f1_list = []
        for n in range(n_classes):
            p_list.append(torch.mean(metric_lists[n, :, 0]))
            r_list.append(torch.mean(metric_lists[n, :, 1]))
            f1_list.append(torch.mean(metric_lists[n, :, 2]))
        val_metrics['p'].append(p_list)
        val_metrics['r'].append(r_list)
        val_metrics['f1'].append(f1_list)
        pass


def save_metrics(savename, val_metrics):
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
    col_idxs = pd.MultiIndex.from_product([range(n_classes)], names=['class'])

    metrics_frame = pd.DataFrame(np.vstack(fold_arrays), index=row_idxs, columns=col_idxs)
    metrics_frame.to_csv(f'{savename}metrics.csv')
    return metrics_frame


def calc_metric_avgs(metrics_df, metrics_name):
    avgs = []
    for metric in metrics_name:
        avgs.append(metrics_df.xs(metric, level=1).mean())

    return pd.DataFrame(avgs, index=metrics_name)


def load_unet(filename, channels=2, levels=4):
    saved_unet = Unet(output_ch=channels, levels=levels)
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet.cuda()


def check_predictions(unet, val_loader, loss):
    unet.eval()
    with torch.no_grad():
        for i in range(2):
            img, seg = next(iter(val_loader))
            seg = seg[i]
            img = img[i:i + 1]
            # img = img.unsqueeze(dim=0)
            prediction = unet(img)

            loss_score = loss(prediction, seg).item()
            print(f"Loss: {loss_score:.3f}")
            prediction = torch.softmax(prediction, dim=1)
            prediction = (prediction > 0.5).float().squeeze(dim=0)
            dl.metrics.print_metrics(prediction, seg)
            prediction = prediction.cpu().detach().numpy()
            img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
            seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
            # utils.plot_onehot_seg(img, seg, title='Ground Truth')
            # utils.plot_onehot_seg(img, prediction, title='Prediction')
            utils.plot_onehot_seg(img, prediction, outline=seg)
            '''
            green: overlap
            orange: missed
            red: segmented background
            '''
            # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


def train_unet(unet, filename, train_settings, dataloader_settings):
    val_metrics = {'p': [], 'r': [], 'f1': [], }

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,use_cuda=True) as prof:
    #     train_unet(unet, **train_settings)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    os.makedirs(filename, exist_ok=True)
    kf_loader = KFoldLoaders(**dataloader_settings)
    for fold, (train_loader, val_loader) in enumerate(kf_loader):
        print(f'Fold #{fold}')
        train_loop(unet, train_loader, val_loader, f'{filename}fold_{fold}.pt', val_metrics, **train_settings)
        unet.zero_grad()
        unet.reset_params()

    metrics_frame = save_metrics(filename, val_metrics)
    avg_df = calc_metric_avgs(metrics_frame, list(val_metrics.keys()))
    print(avg_df)
    # check_predictions(load_unet(filename, channels=n_ch, levels=levels), val_loader, loss_func)


if __name__ == '__main__':
    class_weights = torch.tensor([0.1, 1, 1, 1.5], device='cuda:0')
    loss_func = dl.metrics.FscoreLoss(class_weights=class_weights, f1_weight=0.6)
    filename = "checkpoints/unet1-1a/"
    unet = Unet(output_ch=class_weights.size()[0], levels=4).cuda()
    dataset = CamusDatasetPNG(augment=True)
    # unet = load_unet(filename, channels=n_ch, levels=levels)
    pytorch_total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f'Trainable parameters: {pytorch_total_params}')
    print(f'Feature maps: {unet.channels}')

    train_settings = {
        "epochs": 30,
        "loss_func": loss_func,
        "optimizer": optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-3),
        "do_val": True
    }

    dataloader_settings = {
        "batch_size": 10,
        "split": 10,
        "dataset": dataset,
    }

    train_unet(unet, filename, train_settings, dataloader_settings)

    ''' TODO
        - change aug params
        - loss graph per fold
        - store scores for each patient
        - test augmentation
    '''
