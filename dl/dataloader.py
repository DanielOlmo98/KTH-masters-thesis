import os

import albumentations.pytorch
import numpy as np
import torch
import utils
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from torchvision.transforms import Resize, Compose
from torch.nn.functional import one_hot


class CamusDataset(Dataset):
    def __init__(self, augment=False, img_size=256, set="training/", binary=False):
        self.data_path = utils.get_project_root() + "/dataset/" + set
        self.img_paths, self.seg_paths = self._get_image_paths()
        self.transform_list = [A.Resize(img_size, img_size)]
        if augment:
            self.transform_list.extend(get_transforms())
        self.transform_list.extend([A.pytorch.ToTensorV2()])
        self.transform = A.Compose(self.transform_list)
        self.binary = binary

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        seg = imread(self.seg_paths[idx])
        img = img.astype(dtype=np.float32)

        augmented = self.transform(image=img[0], mask=seg[0])

        # seg = torch.from_numpy(seg.squeeze().astype(dtype=np.int64)).to('cuda')
        img = augmented['image']
        seg = one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1)
        del augmented
        # seg = seg.type(torch.float32)
        # img = self.transform(img)
        # seg = self.transform(seg)
        # seg = (seg > 0.5).float()

        if self.binary:
            seg = seg[0:2]
            seg[1] = 1 - seg[0]

        return img.to('cuda').div(255.), seg.to('cuda')

    def _get_image_paths(self):
        gt_paths = []
        img_paths = []
        for patient_folder in os.listdir(self.data_path):
            image_paths = os.listdir(self.data_path + patient_folder)
            gt_path = [self.data_path + patient_folder + "/" + file for file in image_paths if file[-7:] == "_gt.mhd"]
            img_path = [file[:-7] + ".mhd" for file in gt_path]
            gt_paths.extend(gt_path)
            img_paths.extend(img_path)

        return img_paths, gt_paths


def get_transforms():
    return [
        # A.Normalize(max_pixel_value=1.0),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=1, alpha=100, sigma=15, alpha_affine=6, border_mode=1),
        A.RandomBrightnessContrast(p=0.7, brightness_by_max=False),

    ]


if __name__ == '__main__':
    loader = DataLoader(CamusDataset(augment=True), batch_size=6)
    iter_data = iter(loader)
    for _ in range(6):
        img, seg = next(iter_data)
        utils.plot_onehot_seg(img.cpu().numpy()[0, 0, :, :], seg=seg.cpu().numpy()[0])
