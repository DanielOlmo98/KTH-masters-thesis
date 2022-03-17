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
from itertools import cycle
import utils


class CamusDatasetPNG(Dataset):
    """
    Loads the dataset into CPU memory. It also keeps track if an images is end systole (ES) or end diastole (ED).
    """

    def __init__(self, dataset="camus_png"):
        self.path = f'/dataset/{dataset}/'
        self.imgs, self.segs, self.ED_or_ES = self._load_np(self.path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.segs[idx], self.ED_or_ES[idx]

    def _load_np(self, path):
        data_path = utils.get_project_root() + path
        img_paths, seg_paths = get_image_paths(data_path, extension='.png')
        img_list = []
        seg_list = []
        ED_or_ES = []

        for img_p, seg_p in zip(img_paths, seg_paths):
            img = cv2.imread(img_p, 0)
            img_list.append(img)
            seg = cv2.imread(seg_p, 0)
            seg_list.append(seg)
            if 'ED' in img_p[-7:]:
                ED_or_ES.append(1)
            elif 'ES' in img_p[-7:]:
                ED_or_ES.append(2)
            else:
                ED_or_ES.append(0)

        return img_list, seg_list, torch.tensor(ED_or_ES)

    def __str__(self):
        return f'{type(self)}\n    n_images: {len(self)}'

    def to_json(self):
        return {'n_images': len(self),
                'path': self.path}


class MySubset(Dataset):
    """
    Creates a subset of a dataset in GPU memory, this subset handles augmentation so that different augmentation can
    be used for different subsets (for example augmentation in train but not in val). There is an optional threaded
    queue that queues elements for re-augmentation when they are returned, this 'refills' the augmented elemts so a
    new augmented version of the element is returned each time. The class keeps a reference to the parent dataset,
    the indices of the parent dataset that are in the subclass.

    The parent dataset is expected to be in numpy array format, images in grayscale with intensities 0 to 255,
    and the segmentations label encoded. The subset will convert the data to pytorch tensors, one-hot encode the
    labels and change the intensity range of the image form 0 to 1.

    A subset of the parent dataset indicated by the indices will be transformed. If there are any augmentation
    threads enabled, when an element is requested and returned its index will be put in a queue so that the element
    is augmented again. The elements in the queues will be augmented in parallel in another thread.

    To wait for the queues to finish augmenting use the 'join()'.
    """

    # todo update this documentation
    def __init__(self, dataset, indices, transformer=None, n_aug_threads=0):
        """
        :param dataset: Dataset that returns images and segmentations as numpy arrays. Image intensity must be in the
                        range 0-255. Segmentations must be label encoded.
        :param indices: List of indices of the parent dataset that will form the subset
        :param transformer: List of transforms to apply both in the full subset augmentation and the queue if
                            enabled. If None the augmentation queue is disabled.
        :param n_aug_threads: Number of threads to use for augmentation.
        """
        self.dataset = dataset
        self.indices = indices
        self.aug_imgs = []
        self.aug_segs = []
        self.n_aug_threads = n_aug_threads

        if transformer is None:
            transformer = [A.pytorch.ToTensorV2()]
            self.transformer = A.Compose(transformer)
            self._augment_dataset()
            self.n_aug_threads = 0
        else:
            self.transformer = A.Compose(transformer)
            if self.n_aug_threads < 1:
                raise ValueError('Must have at least 1 thread for augmentation')
            self.q_list = []
            self.thread_list = []
            split_indices = np.array_split(range(len(self.indices)), self.n_aug_threads)
            for i in range(self.n_aug_threads):
                q = queue.Queue()
                self.q_list.append(q)
                self.thread_list.append(
                    threading.Thread(target=self._augment_idx, daemon=True, args=([q])).start())
                self.q_cycler = cycle(self.q_list)
                self.aug_imgs = [None] * len(self.indices)
                self.aug_segs = [None] * len(self.indices)
                for idx in split_indices[i]:
                    q.put(idx)

            self.join_queues()

    def __getitem__(self, idx):
        # if isinstance(idx, list):
        #     return self._get_item([[self.indices[i] for i in idx]])
        if self.n_aug_threads > 0:
            q = next(self.q_cycler)
            q.put(idx)
        return self.aug_imgs[idx], self.aug_segs[idx], self.dataset.ED_or_ES[idx]

    def __len__(self):
        return len(self.indices)

    def _augment_dataset(self):
        """
        Augments the whole subset.
        """
        for img, seg, _ in [self.dataset[i] for i in self.indices]:
            augmented = self.transformer(image=img, mask=seg)
            self.aug_imgs.append(augmented['image'].type(torch.float32).div(255.).to('cuda'))
            self.aug_segs.append(
                one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1).to('cuda'))
            gc.collect()
            torch.cuda.empty_cache()

    def _augment_idx(self, q):
        """
        Augments the indices in the queue.
        """
        while True:
            idx = q.get()
            img, seg, _ = self.dataset[self.indices[idx]]
            augmented = self.transformer(image=img, mask=seg)
            self.aug_imgs[idx] = (augmented['image'].type(torch.float32).div(255.).to('cuda'))
            self.aug_segs[idx] = (
                one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1).to('cuda'))
            q.task_done()
            # if q.qsize() % 100 == 0:
            #     print(f'Augmenting {q.qsize()} images...')
            gc.collect()
            torch.cuda.empty_cache()

    def join_queues(self):
        for q in self.q_list:
            q.join()

        gc.collect()
        torch.cuda.empty_cache()


class KFoldLoaders:
    """
    Uses sklearns KFold to create and iterator that returns training and validation loaders for each fold.
    """

    def __init__(self, batch_size, split, dataset, augments=None, n_train_aug_threads=1):
        """
        Creates the KFold loaders iterator.
        :param batch_size: Batch size of the train loader.
        :param split: How many folds to split the dataset in.
        :param dataset: Dataset to split.
        """
        self.dataset = dataset
        self.kf = KFold(n_splits=split).split(self.dataset)
        self.batch_size = batch_size
        self.augments = augments
        self.n_train_aug_threads = n_train_aug_threads

    def __iter__(self):
        return self

    def __next__(self):
        train_indices, val_indices = next(self.kf)

        train_data = MySubset(self.dataset, indices=train_indices, transformer=self.augments,
                              n_aug_threads=self.n_train_aug_threads)
        val_data = MySubset(self.dataset, indices=val_indices, n_aug_threads=0)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
        del train_data, val_data
        gc.collect()
        torch.cuda.empty_cache()
        return train_loader, val_loader


class KFoldValLoaders:
    """
    Indexable class that has validation dataloaders, used with kfold checkpoints to calculate validation metrics for
    each fold.
    """

    def __init__(self, dataset, split):
        self.dataset = dataset
        self.kf = list(KFold(n_splits=split).split(self.dataset))

    def __len__(self):
        return len(self.kf)

    def __getitem__(self, item: int):
        val_data = MySubset(self.dataset, indices=self.kf[item][1], transformer=None)
        return DataLoader(val_data, batch_size=1, shuffle=True)


def get_loaders(batch_size, dataset, train_indices, val_indices):
    """
    Splits the dataset into train and validation subsets given indices, then creates loaders for each of them.
    Validation loader batch size is set to 2 for memory reasons.
    Unused as the class KFoldLoaders implements k-folds and loader creation in one.
    :param batch_size: Batch size of the training loader.
    :param dataset: Dataset to create loaders for.
    :param train_indices: Indices of the dataset that will be split to the training loader.
    :param val_indices: Indices of the dataset that will be split to the validation loader.
    :return: Training and validation loaders.
    """
    train_data = Subset(dataset, indices=train_indices)
    val_data = Subset(dataset, indices=val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=0)
    return train_loader, val_loader


def get_transforms(h_flip_p, elastic_alpha, elastic_sigma, elastic_affine, bright_lim, contrast_lim):
    """
    Get array of transforms used for data augmentation, including a numpy to tensor conversion.
    :return: Array of transforms.
    """
    return [
        # A.Normalize(max_pixel_value=1.0),
        A.HorizontalFlip(p=h_flip_p),
        A.ElasticTransform(p=1, alpha=elastic_alpha, sigma=elastic_sigma, alpha_affine=elastic_affine, border_mode=0),
        A.RandomBrightnessContrast(p=1., brightness_by_max=False, brightness_limit=bright_lim,
                                   contrast_limit=contrast_lim),
        A.pytorch.ToTensorV2(),

    ]


class DataAugmentation(nn.Module):
    """
    Kornia augmentation class, unused as elastic transforms don't seem to work on CUDA tensors so albumentations is a
    better alternative.
    """

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
    """
    Dataset class that uses the original .mhd files from the dataset.
    It will resize the files from disk and augment them as requested.
    """

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
        img = augmented['image']
        seg = one_hot(augmented['mask'].type(torch.int64), num_classes=4).permute(2, 0, 1)
        del augmented

        if self.binary:
            seg = seg[0:2]
            seg[1] = 1 - seg[0]

        return img.to('cuda').div(255.), seg.to('cuda')


def get_image_paths(data_path, extension=".mhd"):
    """
    Returns lists of the paths of the images and ground truth of a file type.
    Ground truth file name must end with _gt.extension
    :param data_path: Data path containing patient folders with the data.
    :param extension: Extension of the image and ground truth files.
    :return: A list of image path and ground truth paths.
    """
    gt_paths = []
    img_paths = []
    for patient_folder in os.listdir(data_path):
        image_paths = os.listdir(data_path + patient_folder)
        gt_path = [data_path + patient_folder + "/" + file for file in image_paths if file[-7:] == "_gt" + extension]
        img_path = [file[:-7] + extension for file in gt_path]
        gt_paths.extend(gt_path)
        img_paths.extend(img_path)

    return img_paths, gt_paths


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
