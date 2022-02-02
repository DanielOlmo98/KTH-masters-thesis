import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from torchvision.transforms import Resize, Compose
from torch.nn.functional import one_hot
import utils


class CamusDataset(Dataset):
    def __init__(self, transforms=None, img_size=256, set="training/"):
        self.data_path = utils.get_project_root() + "/dataset/" + set
        self.img_paths, self.seg_paths = self._get_image_paths()
        self.transform_list = [Resize((img_size, img_size))]
        if transforms is not None:
            self.transform_list.extend(transforms)
        self.transform = Compose(self.transform_list)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        # img = resize(img, (1, self.img_size, self.img_size))
        img = utils.normalize_0_1(img.astype(dtype=np.float32))
        img = torch.from_numpy(img).to('cuda')

        seg = imread(self.seg_paths[idx])
        seg = torch.from_numpy(seg.squeeze().astype(dtype=np.int64)).to('cuda')
        seg = one_hot(seg, num_classes=4).permute(2, 0, 1)
        seg = seg.type(torch.float32)
        # seg = downscale_local_mean(seg, (1, 5, 4))
        # seg = resize(seg.squeeze(), (self.img_size, self.img_size))
        # seg = utils.binarize(seg.astype(dtype=np.int64))

        img = self.transform(img)
        seg = self.transform(seg)
        seg = (seg > 0.5).float()

        return img, seg

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


if __name__ == '__main__':
    from torchvision import transforms

    t = transforms.RandomCrop((100, 100))
    t2 = transforms.RandomCrop((70, 100))
    loader = DataLoader(CamusDataset(transforms=[t, t2]), batch_size=1)
    img, seg = next(iter(loader))
    utils.plot_image_g(img[0][0].cpu().numpy(), overlay_img=seg[0][1].cpu().numpy())
