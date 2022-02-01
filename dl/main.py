import torch
from tqdm import tqdm

import dl.metrics
import utils
from unet_model import Unet
from dataloader import CellSegDataset
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np


def training(unet, epochs, tb_writer, optimizer, loss_func, batch_size, device):
    train_loader, val_loader = get_loaders(batch_size)
    for epoch in range(epochs):
        running_loss = 0.
        val_running_loss = 0.
        train_loop = tqdm(train_loader, colour='white', desc=f'Epoch {epoch:03},   training')

        for i, data in enumerate(train_loop):
            img, gt = data
            unet.train()
            with torch.cuda.amp.autocast():
                output = unet(img)
                loss = loss_func(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(train_loss=loss.item())
            running_loss += loss.item()

        val_loop = tqdm(val_loader, colour='green', desc=f'           validating')

        for i, data in enumerate(val_loop):
            val_img, val_gt = data
            unet.eval()
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


def get_loaders(batch_size):
    dataset = CellSegDataset()
    test_size = len(dataset) // 5
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size],
                                         generator=torch.manual_seed(1))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train_unet(device, epochs, batch_size):
    unet = Unet().cuda()
    # unet.to(device)

    # loss = nn.BCEWithLogitsLoss()
    loss = dl.metrics.DiceLoss()

    optimizer = optim.SGD(unet.parameters(), lr=1e-2, momentum=0)

    tb_writer = SummaryWriter()

    training(unet, epochs, tb_writer, optimizer, loss, batch_size, device)

    torch.save(unet.state_dict(), 'unet3.pt')

    tb_writer.close()


def load_unet(filename):
    saved_unet = Unet()
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet


def vis_prediction():
    unet = load_unet('unet3.pt').to('cuda')
    with torch.no_grad():
        img, seg = CellSegDataset(img_dir="02")[1]
        img = img.unsqueeze(dim=0)
        prediction = unet(img)

    prediction = torch.sigmoid(prediction)
    prediction = (prediction > 0.5).float().squeeze(dim=0)
    scores = dl.metrics.dice_calc_multiclass(prediction, seg)
    print(f"Dice:\n   Background: {scores[0]:.3f}\n   Target: {scores[1]:.3f}")
    prediction = prediction.cpu().detach().numpy()
    img = img.cpu().detach().squeeze(dim=0).squeeze(dim=0).numpy()
    seg = seg.cpu().detach().squeeze(dim=0).numpy().astype('float32')
    utils.plot_image_g(img, title='img')
    utils.plot_image_g(prediction[1], title='Prediction')
    utils.plot_image_g(np.abs(seg[1] - prediction[1]), title='Difference')


if __name__ == '__main__':
    batch_size = 16
    epochs = 200
    device = torch.device('cuda')
    # train_unet(device, epochs, batch_size)
    vis_prediction()

    print()
