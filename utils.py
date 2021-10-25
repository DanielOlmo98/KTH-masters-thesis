import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color


def binarize(img):
    img[img > 0] = 1
    return img


def normalize_0_1(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def normalize_neg1_to_1(img):
    return 2 * (img - np.min(img)) / (np.max(img) - np.min(img)) - 1


def normalize_0_255(img):
    return (normalize_0_1(img) * 255).astype(np.uint8)


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def plot_image_g(img, title=None, ax=None):
    if ax is None:
        plt.imshow(img, cmap='gray')
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        ax.imshow(img, cmap='gray')
        if title is not None:
            ax.title(title)
        return ax


if __name__ == '__main__':
    print(get_project_root())
