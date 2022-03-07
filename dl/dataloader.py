import os
import asyncio
import threading, queue
import albumentations as A
import albumentations.pytorch
import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import gc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from skimage.io import imread, imsave
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import utils


class CamusDatasetPNG(Dataset):
    """
    Loads the dataset into CPU memory, then creates an augmented copy in GPU memory.
    When a sample is requested it is returned from the GPU and put in an asynchronous queue that augments the sample
    from CPU and refills the GPU dataset.
    """

    def __init__(self):
        self.imgs, self.segs = self.load_np()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.segs[idx]

    def load_np(self):
        data_path = utils.get_project_root() + "/dataset/camus_png/"
        img_paths, seg_paths = get_image_paths(data_path, extension='.png')
        img_list = []
        seg_list = []

        for img_p, seg_p in zip(img_paths, seg_paths):
            img = cv2.imread(img_p, 0)
            img_list.append(img)
            seg = cv2.imread(seg_p, 0)
            seg_list.append(seg)

        return img_list, seg_list

    def __str__(self):
        return f'{type(self)}\n    n_images: {len(self)}'


class MySubset(Dataset):
    def __init__(self, dataset, indices, augment=True):
        self.dataset = dataset
        self.indices = indices
        self.aug_imgs = []
        self.aug_segs = []
        self.augment = augment
        if self.augment:
            self.transformer = A.Compose(get_transforms())
        else:
            self.transformer = A.Compose([A.pytorch.ToTensorV2()])

        self.augment_dataset()
        if self.augment:
            self.q = queue.Queue()
            self.thread = threading.Thread(target=self.augment_idx, daemon=True).start()

    def __getitem__(self, idx):
        # if isinstance(idx, list):
        #     return self._get_item([[self.indices[i] for i in idx]])
        if self.augment:
            self.q.put(idx)
        return self.aug_imgs[idx], self.aug_segs[idx]

    def __len__(self):
        return len(self.indices)

    def augment_dataset(self):
        for img, seg in [self.dataset[i] for i in self.indices]:
            augmented = self.transformer(image=img, mask=seg)
            self.aug_imgs.append(augmented['image'].type(torch.float32).div(255.).to('cuda'))
            self.aug_segs.append(
                one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1).to('cuda'))
            gc.collect()
            torch.cuda.empty_cache()
        return

    def augment_idx(self):
        while True:
            idx = self.q.get()
            img, seg = self.dataset[self.indices[idx]]
            augmented = self.transformer(image=img, mask=seg)
            self.aug_imgs[idx] = (augmented['image'].type(torch.float32).div(255.).to('cuda'))
            self.aug_segs[idx] = (
                one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1).to('cuda'))
            self.q.task_done()


class KFoldLoaders:
    """
    Uses sklearn KFold to create and iterator that returns loaders
    """

    def __init__(self, batch_size, split, dataset, augment=False):
        self.dataset = dataset
        self.kf = KFold(n_splits=split).split(self.dataset)
        self.batch_size = batch_size
        self.augment = augment

    def __iter__(self):
        return self

    def __next__(self):
        train_indices, val_indices = next(self.kf)

        train_data = MySubset(self.dataset, indices=train_indices, augment=self.augment)
        val_data = MySubset(self.dataset, indices=val_indices, augment=False)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
        del train_data, val_data
        gc.collect()
        torch.cuda.empty_cache()
        return train_loader, val_loader


def get_loaders(batch_size, dataset, train_indices, val_indices):
    train_data = Subset(dataset, indices=train_indices)
    val_data = Subset(dataset, indices=val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=0)
    return train_loader, val_loader


def get_transforms():
    return [
        # A.Normalize(max_pixel_value=1.0),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=1, alpha=110, sigma=15, alpha_affine=7, border_mode=0),
        A.RandomBrightnessContrast(p=1., brightness_by_max=False, brightness_limit=0.4, contrast_limit=0.2),
        A.pytorch.ToTensorV2(),

    ]


class DataAugmentation(nn.Module):
    def __init__(self):
        super(DataAugmentation, self).__init__()
        self.transforms = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomElasticTransform(kernel_size=(85, 85), alpha=(2., 2.), sigma=(24., 24.), p=1.),
            kornia.augmentation.RandomHorizontalFlip(),
            keepdim=True
            # kornia.augmentation.ColorJitter(0.1, 0.1, 0., 0., p=1.),
        )

        # self.t1.device = torch.device('cuda:0')
        # self.t1.to('cuda:0')

    def forward(self, img: torch.Tensor, seg: torch.Tensor):
        # noise = self.t1.generate_parameters([1, *img.shape])
        img_o = self.transforms(img)
        seg_o = self.transforms(seg, self.transforms._params).type(torch.int64)
        return img_o, seg_o


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


def timetest():
    import timeit

    imports = """from dl.dataloader import DataAugmentation, CamusDatasetPNG
import torch
from torch.utils.data import DataLoader"""

    testcode = """loader = DataLoader(CamusDatasetPNG(kornia=False), batch_size=4)    
iter_data = iter(loader)
# aug = DataAugmentation()
for _ in range(2):
    img, seg = next(iter_data)
    #torch.cuda.empty_cache()
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
print(f"Reserved:  {r*1e-9:.2f} GB")
print(f"Allocated: {a*1e-9:.2f} GB")
            """
    print(timeit.timeit(stmt=testcode, setup=imports, number=10))


if __name__ == '__main__':
    timetest()
    # loader = DataLoader(CamusDatasetPNG(), batch_size=4)
    # iter_data = iter(loader)
    # for _ in range(2):
    #     img, seg = next(iter_data)
    #     for i in range(4):
    #         utils.plot_onehot_seg(img.cpu().numpy()[i].squeeze(), seg=seg.cpu().numpy()[i], title="aaaaaa")
