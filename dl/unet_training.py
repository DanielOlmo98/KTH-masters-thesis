import torch
from tqdm import tqdm

import dl.metrics
import utils
from unet_model import Unet
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from dl.dataloader import CamusDataset
from colorama import Fore, Style
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_unet(unet, epochs, optimizer, loss_func, batch_size, dataset, savename):
    tb_writer = SummaryWriter()
    train_loader, val_loader = get_loaders(batch_size, dataset)
    bar = Fore.WHITE + '{l_bar}{bar}' + Fore.WHITE + '| {n_fmt}/{total_fmt} [{elapsed}{postfix}]'
    bar_val = Fore.GREEN + '{l_bar}{bar}' + Fore.GREEN + '| {n_fmt}/{total_fmt} [{elapsed}  {postfix}]'
    val_min_loss = 9
    for epoch in range(epochs):
        running_loss = 0.
        val_running_loss = 0.
        train_loop = tqdm(train_loader, colour='white', desc=f'Epoch {epoch + 1:03}/{epochs:03},   training',
                          bar_format=bar)

        torch.enable_grad()
        unet.train()
        for i, data in enumerate(train_loop):
            img, gt = data
            with torch.cuda.amp.autocast():
                output = unet(img)
                loss = loss_func(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(train_loss=loss.item())
            running_loss += loss.item()

        val_loop = tqdm(val_loader, colour='green', desc=f'               validating', bar_format=bar_val)

        unet.eval()
        torch.no_grad()
        for i, data in enumerate(val_loop):
            val_img, val_gt = data
            with torch.cuda.amp.autocast():
                val_output = unet(val_img)
                val_loss = loss_func(val_output, val_gt)

            val_loop.set_postfix(val_loss=val_loss.item())
            val_running_loss += val_loss.item()

        train_loss_epoch = running_loss / train_loop.n
        val_loss_epoch = val_running_loss / val_loop.n

        if val_loss_epoch < val_min_loss:
            torch.save(unet.state_dict(), savename)
            print("\nSaving Model")
            val_min_loss = val_loss_epoch

        tb_writer.add_scalars("loss per epoch", {
            'train loss': train_loss_epoch,
            'val loss': val_loss_epoch,
        }, epoch)

    tb_writer.close()


def get_loaders(batch_size, dataset):
    test_size = len(dataset) // 5
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size],
                                         generator=torch.manual_seed(1))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader


def load_unet(filename):
    saved_unet = Unet()
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet


def check_predictions(network, dataset, loss):
    unet = network.to('cuda')
    with torch.no_grad():
        img, seg = dataset[19]
        img = img.unsqueeze(dim=0)
        prediction = unet(img)

    loss_score = loss(prediction, seg.unsqueeze(dim=0)).item()
    print(f"Loss: {loss_score:.3f}")
    prediction = torch.softmax(prediction, dim=1)
    prediction = (prediction > 0.5).float().squeeze(dim=0)
    dl.metrics.print_metrics(prediction, seg)
    prediction = prediction.cpu().detach().numpy()
    img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
    seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
    utils.plot_image_g(img, overlay_img=seg[1], title='Ground truth')
    # utils.plot_image_g(seg[0])
    # utils.plot_image_g(seg[1])
    '''
    green: overlap
    orange: missed
    red: segmented background
    '''
    utils.plot_image_g(img, overlay_img=prediction[1] + 2 * seg[1], title='Prediction')
    # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


if __name__ == '__main__':
    unet = Unet().cuda()
    # loss_func = dl.metrics.DiceLoss(num_classes=2, weights=torch.tensor([0.3, 3], device='cuda:0'), f1_weight=0.3)
    class_weights = torch.tensor([0.3, 3], device='cuda:0')
    loss_func = dl.metrics.FscoreLoss(num_classes=2, class_weights=class_weights, f1_weight=0.7)

    filename = "unet_weighted_t5.pt"
    train_settings = {
        "batch_size": 24,
        "epochs": 100,
        "loss_func": loss_func,
        # 'loss_func': nn.CrossEntropyLoss(),
        # "optimizer": optim.SGD(unet.parameters(), lr=1e-4, momentum=0),
        "optimizer": optim.Adam(unet.parameters(), lr=1e-4, weight_decay=5e-5),
        "dataset": CamusDataset(binary=True),
        "savename": filename
    }

    # unet = load_unet("unet_camus_bce.pt").cuda()

    '''
    regularization?
    '''

    train_unet(unet, **train_settings)
    check_predictions(load_unet(filename), CamusDataset(set="training/", binary=True), loss_func)
