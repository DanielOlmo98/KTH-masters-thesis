import os
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread

import utils


@dataclass
class CellSegDataset(Dataset):
    def __init__(self, transform=None):
        self.path = "D:/KTH/Thesis/dataset/Fluo-N2DH-GOWT1/"
        self.img_dir_path = self.path + "01"
        self.seg_dir_path = self.path + "01_st/SEG"

        self.img_list = os.listdir(self.img_dir_path)
        self.seg_list = os.listdir(self.seg_dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir_path, self.img_list[idx])
        seg_path = os.path.join(self.seg_dir_path, self.seg_list[idx])
        img = torch.from_numpy(imread(img_path))
        seg = torch.from_numpy(utils.binarize(imread(seg_path).astype(dtype=np.uint8)))
        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)

        return img, seg


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
