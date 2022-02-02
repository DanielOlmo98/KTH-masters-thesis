import torch
import dl.metrics
import utils
from unet_model import Unet
from cell_dataloader import CellSegDataset
from torch.utils.tensorboard import SummaryWriter
from unet_training import train_unet, load_unet, check_predictions
import torch.optim as optim
import numpy as np

if __name__ == '__main__':
    unet = Unet().cuda()

    train_settings = {
        "batch_size": 16,
        "epochs": 10,
        "loss_func": dl.metrics.DiceLoss(),
        "optimizer": optim.SGD(unet.parameters(), lr=1e-2, momentum=0),
        "dataset": CellSegDataset()
    }

    check_predictions(load_unet('unet4.pt'), CellSegDataset(img_dir="02"))
    check_predictions(load_unet('unet3.pt'), CellSegDataset(img_dir="02"))
    # train_unet(unet, **train_settings)
    # torch.save(unet.state_dict(), 'unet4.pt')
