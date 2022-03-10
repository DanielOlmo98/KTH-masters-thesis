import os
import SimpleITK as sitk
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion
from matplotlib.colors import ListedColormap
from skimage.io import imread
from skimage import color, img_as_float32

import cv2


def binarize(img, threshold=0.5):
    img[img >= threshold] = 1
    img[img < threshold] = 0
    return img


def normalize_0_1(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)


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


def load_images(path=(get_project_root() + '/image/')):
    file_list = os.listdir(path)
    images = []
    for file in file_list:
        common_file_path = path + file
        if not os.path.isfile(common_file_path):
            continue

        image = imread(common_file_path)
        if image.ndim == 3:
            image = color.rgb2gray(image)
        images.append(img_as_float32(image))
    return images


def plot_image_g(img, overlay_img=None, title=None, ax=None, cmap_overlay=None, alpha_overlay=0.2):
    if cmap_overlay is None:
        # cmap_overlay = ListedColormap([(0, 0, 0, 0), "red", "orange", "lime"])
        colors = ['none', 'gold', 'lime', 'blue', 'red']
        cmap_overlay = ListedColormap(['none', *colors])

    if ax is None:
        plt.figure(figsize=np.divide(img.shape[::-1], 100))
        plt.imshow(img, cmap='gray')
        if overlay_img is not None:
            plt.imshow(overlay_img, cmap=cmap_overlay, alpha=alpha_overlay)
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        ax.imshow(img, cmap='gray')
        if title is not None:
            ax.set_title(title)
        return ax


def plot_onehot_seg(img, seg, outline=None, alpha_overlay=0.2, title=None):
    colors = ['none', 'gold', 'lime', 'blue', 'red']
    plt.imshow(img, cmap='gray')
    for n in range(seg.shape[0]):
        plt.imshow(seg[n], alpha=alpha_overlay, cmap=ListedColormap(['none', colors[n]]))
        if outline is not None:
            plt.imshow(get_outline(outline[n]), alpha=0.7, cmap=ListedColormap(['none', colors[n]]))

    if title is not None:
        plt.title(title)
    plt.show()
    return


def get_outline(seg):
    eroded_seg = binary_erosion(seg)
    return seg - eroded_seg


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


def load_patient(patient='0001', CH=2, ED_or_ES='ED', png=False, rotate=False):
    from medpy.io.load import load
    if png:
        dataset_path = 'camus_png'
        filetype = '.png'
    else:
        dataset_path = 'training'
        filetype = '.mhd'

    patient_path = get_project_root() + \
                   f'/dataset/{dataset_path}/patient{patient}/patient{patient}_{CH}CH_{ED_or_ES}'
    img_path = patient_path + filetype
    seg_path = patient_path + '_gt' + filetype

    if png:
        img = cv2.imread(img_path, 0)
        seg = cv2.imread(seg_path, 0)
        return img, seg
    else:
        img, img_header = load(img_path)
        seg, seg_header = load(seg_path)
        if rotate:
            img = np.rot90(img, axes=(1, 0))
            seg = np.rot90(seg, axes=(1, 0))
        return img, img_header, seg, seg_header


def slice_view_3d(volume):
    # not working
    class IndexTracker:
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices // 2

            self.im = ax.imshow(self.X[:, :, self.ind])
            self.update()

        def on_scroll(self, event):
            print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[:, :, self.ind])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    fig, ax = plt.subplots(1, 1)
    fig.canvas.mpl_connect('scroll_event', IndexTracker(ax, volume).on_scroll)
    plt.show()

    return


def plot_losses(train_losses, val_losses, show=True, filename=None, title='Losses'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim([0, 3])
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


if __name__ == '__main__':
    from torch.nn.functional import one_hot
    import torch

    img, seg = load_patient(png=True, patient='0003', ED_or_ES='ED', CH=4)
    seg = one_hot(torch.tensor(seg.squeeze()).type(torch.int64), num_classes=4).permute(2, 0, 1)
    # img = np.rot90(img.squeeze(), axes=(1, 0))
    # seg = np.rot90(seg, axes=(2, 1))
    plot_onehot_seg(img, seg=seg, alpha_overlay=0.2)

    print(np.unique(seg))
