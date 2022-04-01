import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import utils
import noise_filtering
from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad, cy_srad
from noise_filtering.combined_method import combined_method
from noise_filtering.HMF import hybrid_median_filtering
from noise_filtering import wavelet_denoise
from skimage.restoration import denoise_tv_bregman


def wavelet_denoise_w(img, sigma, mode):
    if mode not in ['visu', 'bayes']:
        raise ValueError("Mode must be either 'visu' or 'bayes'")
    img = utils.normalize_0_1(img).astype(np.float32)

    return wavelet_denoise.wavelet_denoise(img, sigma, mode)


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


def get_settings_dict():
    return {
        'wavelet_visu': {'sigma': 0.02, 'mode': 'visu'},
        'wavelet_bayes': {'sigma': 0.15, 'mode': 'bayes'},
        'tv': {'weight': 0.3},
        'csrad': {'steps': 150, 'step_size': 0.1},
        'srad': {'steps': 150, 'step_size': 0.1},
        'tv_csrad': {'steps': 150, 'step_size': 0.05, 'weight': 0.3},
        'combine': {'steps': 100, 'step_size': 0.05, 'weight': 0.5},
        'hmf': {'none': None}
    }


def get_denoise_lambda_dict():
    settings_dict = get_settings_dict()
    return {
        'wavelet_visu': lambda img: wavelet_denoise_w(img, **settings_dict['wavelet_visu']),
        'wavelet_bayes': lambda img: wavelet_denoise_w(img, **settings_dict['wavelet_bayes']),
        'tv': lambda img: tv_denoise(img, **settings_dict['tv']),
        'combine': lambda img: combined_method_denoise(img, **settings_dict['combine']),
        'tv_csrad': lambda img: tv_csrad_denoise(img, **settings_dict['tv_csrad']),
        'csrad': lambda img: csrad_denoise(img, **settings_dict['csrad']),
        'srad': lambda img: srad_denoise(img, **settings_dict['srad']),
        'hmf': lambda img: hmf_denoise(img)
    }


if __name__ == '__main__':
    print()
    img = utils.get_example_img()
    utils.plot_image_g(img)
    utils.plot_image_g(hmf_denoise(img))
