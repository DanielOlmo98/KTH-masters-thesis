import os

import albumentations.pytorch
import numpy as np
import torch
import utils
import h5py
import cv2
import kornia
import albumentations as A
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imsave
from torchvision.transforms import Resize, Compose
from torch.nn.functional import one_hot


class CamusDatasetPNG(Dataset):
    def __init__(self):
        self.imgs, self.segs = self.load_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        seg = self.segs[idx]
        transform = nn.Sequential(kornia.augmentation.RandomElasticTransform(same_on_batch=True))
        tf = transform(img)
        tf2 = transform(seg)
        utils.plot_image_g(kornia.utils.image.tensor_to_image(tf[0]))
        utils.plot_image_g(kornia.utils.image.tensor_to_image(tf2[0]))
        print()

    def load_dataset(self):
        data_path = utils.get_project_root() + "/dataset/camus_png/"
        img_paths, seg_paths = get_image_paths(data_path, extension='.png')
        img_list = []
        seg_list = []
        gpu = torch.device('cuda:0')
        for img_p, seg_p in zip(img_paths, seg_paths):
            img = cv2.imread(img_p, 0)
            img_list.append(kornia.utils.image_to_tensor(img).to(gpu).type(torch.float32).div(255.))
            seg = cv2.imread(seg_p, 0)
            seg = kornia.utils.image_to_tensor(seg)
            seg_list.append(kornia.utils.one_hot(seg.type(torch.int64), 4).type(torch.int64).to(gpu))

        return img_list, seg_list


class DataAugmentation(nn.Module):
    def __init__(self):
        super(DataAugmentation, self).__init__()
        kornia.geometry.elastic_transform2d()


class CamusDataset(Dataset):
    def __init__(self, augment=False, img_size=256, set="training/", binary=False):
        self.data_path = utils.get_project_root() + "/dataset/" + set
        self.img_paths, self.seg_paths = get_image_paths(self.data_path)
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


def get_image_paths(data_path, extension=".mhd"):
    gt_paths = []
    img_paths = []
    for patient_folder in os.listdir(data_path):
        image_paths = os.listdir(data_path + patient_folder)
        gt_path = [data_path + patient_folder + "/" + file for file in image_paths if file[-7:] == "_gt" + extension]
        img_path = [file[:-7] + extension for file in gt_path]
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


def dataset_convert():
    resize = A.Resize(256, 256)
    dataset_path = utils.get_project_root() + '/dataset/' + 'training/'
    converted_path = utils.get_project_root() + '/dataset/' + 'camus_png/'

    img_paths, seg_paths = get_image_paths(dataset_path)

    folders = os.listdir(dataset_path)
    for folder in folders:
        os.makedirs(converted_path + folder)

    for img_path, seg_path in zip(img_paths, seg_paths):
        data = resize(image=imread(img_path)[0], mask=imread(seg_path)[0])
        imsave(converted_path + img_path[31:-3] + 'png', data['image'])
        imsave(converted_path + seg_path[31:-3] + 'png', data['mask'])


if __name__ == '__main__':
    loader = DataLoader(CamusDatasetPNG(), batch_size=6)
    iter_data = iter(loader)
    for _ in range(6):
        img, seg = next(iter_data)
        utils.plot_onehot_seg(img.cpu().numpy()[0, 0, :, :], seg=seg.cpu().numpy()[0])
