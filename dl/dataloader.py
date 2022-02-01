import os
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from torchvision.transforms import Resize, Compose, ToTensor
from torch.nn.functional import one_hot

import utils


@dataclass
class CellSegDataset(Dataset):
    def __init__(self, transform=None, img_size=256, img_dir="01"):
        self.path = "D:/KTH/Thesis/dataset/Fluo-N2DH-GOWT1/"
        self.img_dir_path = self.path + img_dir
        self.seg_dir_path = self.path + img_dir + "_ST/SEG"

        self.img_list = os.listdir(self.img_dir_path)
        self.seg_list = os.listdir(self.seg_dir_path)
        self.transform = transform
        self.img_size = img_size
        self.resize = Resize(img_size)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir_path, self.img_list[idx])
        seg_path = os.path.join(self.seg_dir_path, self.seg_list[idx])

        img = imread(img_path)
        img = utils.normalize_0_1(img.astype(dtype=np.float32))
        img = torch.from_numpy(img).to('cuda')
        seg = utils.binarize(imread(seg_path).astype(dtype=np.int64))
        seg = torch.from_numpy(seg).to('cuda')

        img = self.resize(img.unsqueeze(0))
        seg = self.resize(seg.unsqueeze(0)).squeeze(0)

        seg = one_hot(seg, num_classes=2).permute(2, 0, 1)
        seg = seg.type(torch.float16)

        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)

        return img, seg


def transforms():
    return Compose([Resize(256), ToTensor()])


if __name__ == '__main__':
    train_data = CellSegDataset()
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    img, seg = next(iter(train_loader))
    img = img[0]
    seg = seg[0]

    from utils import plot_image_g

    plot_image_g(img)
    plot_image_g(seg)
