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


def symmetric_threshold(img, threshold, zero_point=0, invert=True):
    d_n_bool = np.logical_or(img < zero_point - threshold, img > zero_point + threshold)
    if invert:
        img[~d_n_bool] = zero_point
    else:
        img[d_n_bool] = zero_point
    return img


def plot_image_g(img, title=None, ax=None, overlay_img=None, alpha_overlay=0.5):
    if ax is None:
        plt.imshow(img, cmap='gray')
        if overlay_img is not None:
            plt.imshow(overlay_img, cmap='seismic', alpha=alpha_overlay)
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        ax.imshow(img, cmap='gray')
        if title is not None:
            ax.set_title(title)
        return ax


def heightmap(array, ax=None, title=None, elev=None, azim=None):
    height, width = array.shape
    x = np.arange(0, height, 1)
    y = np.arange(0, width, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.transpose(array.data)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='hot')
        ax.view_init(elev=elev, azim=azim)

        if title is not None:
            ax.set_title(title)

        plt.show()
        return
    else:
        ax.plot_surface(X, Y, Z, cmap='hot')
        ax.view_init(elev=elev, azim=azim)

        if title is not None:
            ax.set_title(title)
        return ax


if __name__ == '__main__':
    print(get_project_root())
