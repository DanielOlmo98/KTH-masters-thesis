import torch
from tqdm import tqdm

import utils
from unet_model import Unet
from dataloader import CellSegDataset
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_loop(unet, epochs, tb_writer, optimizer, loss_func, batch_size, device):
    train_loader, test_loader = get_loaders(batch_size)
    for epoch in range(epochs):
        running_loss = 0.
        val_running_loss = 0.
        loop = tqdm(train_loader)

        for i, data in enumerate(loop):
            img, gt = data
            img = img.to(device)
            gt = gt.to(device)

            with torch.cuda.amp.autocast():
                output = unet(img)
                loss = loss_func(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

            running_loss += loss.item()

        for i, data in enumerate(test_loader):
            val_img, val_gt = data
            val_img = val_img.to(device)
            with torch.cuda.amp.autocast():
                val_gt = val_gt.to(device)
                val_output = unet(val_img)
            val_loss = loss_func(val_output, val_gt)

            val_running_loss += val_loss

        last_loss = running_loss / batch_size
        val_last_loss = val_running_loss / batch_size
        print(f"\nEpoch {epoch}, Losses:  Val: {val_last_loss:.2f} | Test {last_loss:.2f}")
        tb_writer.add_scalars("loss per epoch", {
            'train loss': last_loss,
            'val loss': val_last_loss,
        }, epoch)
        # tb_writer.add_scalar("Epoch train loss", last_loss, epoch)
        # tb_writer.add_scalar("Epoch val loss", val_last_loss, epoch)


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
    unet = Unet()
    unet.to(device)

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unet.parameters(), lr=1e-3, momentum=0)

    tb_writer = SummaryWriter()

    train_loop(unet, epochs, tb_writer, optimizer, loss, batch_size, device)

    torch.save(unet.state_dict(), 'unet2.pt')

    tb_writer.close()


def load_unet(filename):
    saved_unet = Unet()
    saved_unet.load_state_dict(torch.load(filename))
    return saved_unet


def vis_prediction():
    unet = load_unet('unet2.pt')
    with torch.no_grad():
        img, seg = CellSegDataset()[1]
        img = torch.unsqueeze(img, 0).type(torch.FloatTensor)
        prediction = unet(img)

    plot_output(img, prediction)


def plot_output(img, prediction):
    img = np.squeeze(img.cpu().detach().numpy())
    prediction = torch.sigmoid(prediction)
    prediction = np.squeeze(prediction.cpu().detach().numpy())
    prediction = utils.symmetric_threshold(prediction, threshold=0.5)

    utils.plot_image_g(img)
    utils.plot_image_g(prediction[1])


if __name__ == '__main__':
    batch_size = 16
    epochs = 100
    device = torch.device('cuda:0')
    vis_prediction()
    # train_unet(device, epochs, batch_size)

    print()
