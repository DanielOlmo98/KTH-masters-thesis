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
    evaluate_unet(unet, val_loader, savename)


def evaluate_unet(unet, val_loader, savename):
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
                metric_lists[n].append(dl.metrics.get_f1_metrics(prediction.unsqueeze(dim=0), seg.unsqueeze(dim=0)))

            next_batch = next(val_loader)

    except StopIteration:
        metric_lists = torch.FloatTensor(metric_lists)
        with open(savename + "eval_metrics.txt", "w") as file:
            for n in range(n_classes):
                p = torch.mean(metric_lists[n, :, 0])
                r = torch.mean(metric_lists[n, :, 1])
                f1 = torch.mean(metric_lists[n, :, 2])
                file.write(f"Class {n + 1}:\n  Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}\n")
        pass


def get_loaders(batch_size, dataset, split=5):
    test_size = len(dataset) // split
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size],
                                         generator=torch.manual_seed(1))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, test_loader


def load_unet(filename, channels=2):
    saved_unet = Unet(output_ch=channels).cuda()
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
    utils.plot_image_g(prediction[1] + 2 * prediction[2] + 3 * prediction[3])
    '''
    green: overlap
    orange: missed
    red: segmented background
    '''
    utils.plot_image_g(img, overlay_img=prediction[1] + 2 * seg[1], title='Prediction')
    # utils.plot_image_g(np.abs(seg - prediction[0]), title='Difference')


if __name__ == '__main__':
    # loss_func = dl.metrics.DiceLoss(num_classes=2, weights=torch.tensor([0.3, 3], device='cuda:0'), f1_weight=0.3)
    class_weights = torch.tensor([0.3, 1, 1, 1], device='cuda:0')
    loss_func = dl.metrics.FscoreLoss(class_weights=class_weights, f1_weight=0.6)

    filename = "unet_multiclass.pt"
    unet = Unet(output_ch=4).cuda()
    # unet = load_unet("unet_weighted_t5.pt")

    train_settings = {
        "batch_size": 24,
        "epochs": 100,
        "loss_func": loss_func,
        # 'loss_func': nn.CrossEntropyLoss(),
        # "optimizer": optim.SGD(unet.parameters(), lr=1e-4, momentum=0),
        "optimizer": optim.Adam(unet.parameters(), lr=5e-5, weight_decay=5e-5),
        "dataset": CamusDataset(binary=False),
        "savename": filename
    }

    # unet = load_unet("unet_camus_bce.pt").cuda()

    '''
    regularization?
    '''

    train_unet(unet, **train_settings)
    check_predictions(load_unet(filename, channels=4), CamusDataset(set="training/", binary=False), loss_func)
