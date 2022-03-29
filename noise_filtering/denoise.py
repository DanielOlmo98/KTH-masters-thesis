import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import utils
import noise_filtering
from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad, cy_srad
from noise_filtering.combined_method import combined_method
from noise_filtering.HMF import hybrid_median_filtering
from skimage.restoration import denoise_tv_bregman


def wavelet_denoise(img, mode, sigma):
    if mode not in ['visu', 'bayes']:
        raise ValueError("Mode must be either 'visu' or 'bayes'")
    img = utils.normalize_0_1(img).astype(np.float32)

    return noise_filtering.wavelet_denoise.wavelet_denoise(img, mode, sigma)


def tv_denoise(img, weight):
    img = utils.normalize_0_1(img).astype(np.float32)
    return denoise_tv_bregman(img, weight)


def combined_method_denoise(img, steps, step_size, weight):
    img = utils.normalize_0_1(img).astype(np.float32)
    return combined_method(img, steps, step_size, weight)


def tv_csrad_denoise(img, steps, step_size, weight):
    img = utils.normalize_0_1(img).astype(np.float32)
    img = denoise_tv_bregman(img, weight)
    img_n = img
    ci = np.zeros_like(img)
    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_csrad(array=img_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)
    return img_n


def csrad_denoise(img, steps, step_size):
    img = utils.normalize_0_1(img).astype(np.float32)
    img_n = img
    ci = np.zeros_like(img)
    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_csrad(array=img_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)
    return img_n


def srad_denoise(img, steps, step_size):
    img_n = utils.normalize_0_1(img).astype(np.float32)

    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_srad(array=img_n.clip(1e-8), iter=n, step=step_size)
    return img_n


def hmf_denoise(img):
    img = utils.normalize_0_1(img).astype(np.float32)
    return hybrid_median_filtering(img).base


if __name__ == '__main__':
    print()
    img = utils.get_example_img()
    utils.plot_image_g(img)
    utils.plot_image_g(hmf_denoise(img))
