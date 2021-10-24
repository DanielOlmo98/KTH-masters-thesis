import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color


def binarize(img):
    img[img > 0] = 1
    return img


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def normalize_0_255(img):
    return (normalize(img) * 255).astype(np.uint8)


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def plot_image_g(img):
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    print(get_project_root())
