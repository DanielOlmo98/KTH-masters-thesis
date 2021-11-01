import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color, img_as_float32
from noise_filtering.denoise import sitk_noisefilter
from noise_filtering.wavelet_denoise import wavelet_exp
from noise_filtering.dct import dct_exp
from utils import *


# TODO: total variation optimization problem denoise

def srad_test(image, steps, step_size):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_srad

    image_n = image
    ci = []
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_srad(array=image_n.clip(1e-10), iter=n, step=step_size)

    ci = np.asarray(ci.base)
    title = 'Final cy_SRAD ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    fig = plt.figure(figsize=(6, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    plot_image_g(image_n, ax=ax1, title=title)
    heightmap(ci, ax=ax2, azim=10, elev=40)
    plt.show()


def csrad_test(image, steps, step_size):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad

    image_n = image
    ci = np.zeros_like(image)
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_csrad(array=image_n.clip(1e-10), ci_1=ci, iter=n, step=step_size)
        ci = np.asarray(ci.base)

    title = 'Final cy_CSRAD ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    fig = plt.figure(figsize=(6, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    plot_image_g(image_n, ax=ax1, title=title)
    heightmap(ci, ax=ax2, azim=10, elev=40)
    plt.show()


if __name__ == '__main__':
    images = load_images()
    image = images[0]
    # image = image[130:300, 200:450]
    epsilon = 10e-5
    image += epsilon
    steps = 75
    step_size = 0.05
    csrad_test(image, steps=steps, step_size=step_size)
    srad_test(image, steps=steps, step_size=step_size)
