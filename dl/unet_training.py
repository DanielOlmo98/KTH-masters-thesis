import torch
from tqdm import tqdm

import dl.metrics
import utils
from unet_model import Unet
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from dl.dataloader import CamusDataset
from colorama import Fore
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_unet(unet, epochs, optimizer, loss_func, batch_size, dataset):
    tb_writer = SummaryWriter()
    train_loader, val_loader = get_loaders(batch_size, dataset)
    for epoch in range(epochs):
        running_loss = 0.
        val_running_loss = 0.
        bar = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]'
        train_loop = tqdm(train_loader, colour='white', desc=f'Epoch {epoch + 1:03}/{epochs:03},   training',
                          bar_format=bar)

        torch.enable_grad()
        unet.train()
        for i, data in enumerate(train_loop):
            img, gt = data
            gt = 1 - gt[:, 0:2, :, :]
            with torch.cuda.amp.autocast():
                output = unet(img)
                loss = loss_func(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(train_loss=loss.item())
            running_loss += loss.item()

        val_loop = tqdm(val_loader, colour='green', desc=f'               validating', bar_format=bar)

        unet.eval()
        torch.no_grad()
        for i, data in enumerate(val_loop):
            val_img, val_gt = data
            val_gt = 1 - val_gt[:, 0:2, :, :]
            with torch.cuda.amp.autocast():
                val_output = unet(val_img)
                val_loss = loss_func(val_output, val_gt)

            val_loop.set_postfix(val_loss=val_loss.item())
            val_running_loss += val_loss.item()

        last_loss = running_loss / train_loop.n
        val_last_loss = val_running_loss / val_loop.n
        tb_writer.add_scalars("loss per epoch", {
            'train loss': last_loss,
            'val loss': val_last_loss,
        }, epoch)

    tb_writer.close()


def get_loaders(batch_size, dataset):
    test_size = len(dataset) // 5
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size],
                                         generator=torch.manual_seed(1))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def load_unet(filename):
    saved_unet = Unet()
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet


def check_predictions(network, dataset):
    unet = network.to('cuda')
    with torch.no_grad():
        img, seg = dataset[15]
        seg = seg[0:2]
        img = img.unsqueeze(dim=0)
        prediction = unet(img)

    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float().squeeze(dim=0)
    dl.metrics.print_metrics(prediction[0:2], seg)
    prediction = prediction.cpu().detach().numpy()
    img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
    seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
    utils.plot_image_g(img, overlay_img=seg[0], title='img')
    utils.plot_image_g(prediction[0], title='Prediction class 1')
    utils.plot_image_g(img, overlay_img=prediction[1], title='Prediction')
    # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


if __name__ == '__main__':
    unet = Unet().cuda()

    train_settings = {
        "batch_size": 16,
        "epochs": 100,
        "loss_func": dl.metrics.DiceLoss(num_classes=2, weights=torch.tensor([0.25, 4], device='cuda:0')),
        # 'loss_func': nn.CrossEntropyLoss(),
        # "optimizer": optim.SGD(unet.parameters(), lr=1e-4, momentum=0),
        "optimizer": optim.AdamW(unet.parameters(), lr=1e-4),
        "dataset": CamusDataset()
    }

    # unet = load_unet("unet_camus_bce.pt").cuda()

    '''
    regularization?
    '''

    train_unet(unet, **train_settings)
    torch.save(unet.state_dict(), "unet_weighted_dice_adam.pt")

    check_predictions(unet, CamusDataset(set="training/"))
