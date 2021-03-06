import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import utils
import noise_filtering
import cv2
from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad, cy_srad
from noise_filtering.combined_method import combined_method
from noise_filtering.HMF import hybrid_median_filtering
from noise_filtering import wavelet_denoise
from skimage.restoration import denoise_tv_bregman
import skimage


def wavelet_denoise_w(img, sigma, mode):
    if mode not in ['visu', 'bayes']:
        raise ValueError("Mode must be either 'visu' or 'bayes'")

    return wavelet_denoise.wavelet_denoise(img, sigma, mode)


def tv_denoise(img, weight, max_iter=100, eps=0.001):
    return denoise_tv_bregman(img, weight, max_iter, eps)


def combined_method_denoise(img, steps, step_size, weight, max_iter=100, eps=0.001):
    if img.dtype != np.float32:
        raise TypeError(f'Image is {img.dtype}, needs to be float32')
    return combined_method(img, steps, step_size, weight, max_iter=max_iter, eps=eps)


def tv_csrad_denoise(img, steps, step_size, weight, max_iter=100, eps=0.001):
    if img.dtype != np.float32:
        raise TypeError(f'Image is {img.dtype}, needs to be float32')
    img = denoise_tv_bregman(img, weight, max_iter=max_iter, eps=eps)
    img_n = img
    ci = np.zeros_like(img)
    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_csrad(array=img_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)
    return img_n


def csrad_denoise(img, steps, step_size):
    if img.dtype != np.float32:
        raise TypeError(f'Image is {img.dtype}, needs to be float32')
    img_n = img
    ci = np.zeros_like(img)
    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_csrad(array=img_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)
    return img_n


def srad_denoise(img, steps, step_size):
    if img.dtype != np.float32:
        raise TypeError(f'Image is {img.dtype}, needs to be float32')
    img_n = img
    for n in range(steps):
        img_n = np.abs(img_n)
        img_n, ci, di = cy_srad(array=img_n.clip(1e-8), iter=n, step=step_size)
    return img_n


def hmf_denoise(img):
    return hybrid_median_filtering(img).base


def get_settings_dict(denoise_strength='strong'):
    if denoise_strength not in ['strong', 'weak', 'strong2']:
        raise ValueError("Strength must be either 'strong' or 'weak'")

    denoise_settings = {
        'strong': {
            'VisuShrink': {'sigma': 0.015, 'mode': 'visu'},
            'BayesShrink': {'sigma': 0.1, 'mode': 'bayes'},
            'TV': {'weight': 0.3, 'max_iter': 250, 'eps': 0.001},
            'CSRAD': {'steps': 150, 'step_size': 0.1},
            'SRAD': {'steps': 150, 'step_size': 0.05},
            # 'TV_CSRAD': {'steps': 150, 'step_size': 0.05, 'weight': 0.3},
            'Combined': {'steps': 50, 'step_size': 0.1, 'weight': 0.7, 'max_iter': 100, 'eps': 0.001},
            'HMF': {'none': None}
        },

        'strong2': {
            'VisuShrink': {'sigma': 0.015, 'mode': 'visu'},
            'BayesShrink': {'sigma': 0.13, 'mode': 'bayes'},
            'TV': {'weight': 0.6, 'max_iter': 250, 'eps': 0.001},
            'CSRAD': {'steps': 150, 'step_size': 0.05},
            'SRAD': {'steps': 150, 'step_size': 0.05},
            # 'TV_CSRAD': {'steps': 150, 'step_size': 0.05, 'weight': 0.3},
            'Combined': {'steps': 50, 'step_size': 0.001, 'weight': 0.9, 'max_iter': 100, 'eps': 0.005},
            'HMF': {'none': None}
        },

        'weak': {
            'VisuShrink': {'sigma': 0.01, 'mode': 'visu'},
            'BayesShrink': {'sigma': 0.08, 'mode': 'bayes'},
            'TV': {'weight': 0.9, 'max_iter': 100, 'eps': 0.05},
            'CSRAD': {'steps': 200, 'step_size': 0.015},
            'SRAD': {'steps': 200, 'step_size': 0.015},
            # 'TV_CSRAD': {'steps': 25, 'step_size': 0.005, 'weight': 0.9, 'max_iter': 20, 'eps': 0.001},
            'Combined': {'steps': 50, 'step_size': 0.015, 'weight': 0.9, 'max_iter': 100, 'eps': 0.1},
            'HMF': {'none': None}

        }
    }

    return denoise_settings[denoise_strength]


def get_denoise_lambda_dict(denoise_strength='strong'):
    settings_dict = get_settings_dict(denoise_strength)
    return {
        'VisuShrink': lambda img: wavelet_denoise_w(img, **settings_dict['VisuShrink']),
        'BayesShrink': lambda img: wavelet_denoise_w(img, **settings_dict['BayesShrink']),
        'TV': lambda img: tv_denoise(img, **settings_dict['TV']),
        'Combined': lambda img: combined_method_denoise(img, **settings_dict['Combined']),
        # 'TV_CSRAD': lambda img: tv_csrad_denoise(img, **settings_dict['TV_CSRAD']),
        'CSRAD': lambda img: csrad_denoise(img, **settings_dict['CSRAD']),
        'SRAD': lambda img: srad_denoise(img, **settings_dict['SRAD']),
        'HMF': lambda img: hmf_denoise(img)
    }


def gamma_noise_repeats(img, noise_map_dim=(256, 256), shape=50, scale=1):
    """
    Add gamma multiplicative noise, repeats the noise map to match img dimensions.
    """
    # rescale factors
    r_x = img.shape[0] / noise_map_dim[0]
    r_y = img.shape[1] / noise_map_dim[1]
    noise = np.repeat(np.repeat(np.random.gamma(shape, scale, size=noise_map_dim), r_x, axis=0), r_y, axis=1)
    return img * noise


def gamma_noise_gaps(img, noise_map_dim=(256, 256), shape=50, scale=1):
    """
    Add gamma multiplicative noise, inserts ones in the noise gap to match img dimensions.
    """
    # repeat count
    r_x = img.shape[0] // noise_map_dim[0]
    r_y = img.shape[1] // noise_map_dim[1]
    noise = np.random.gamma(shape, scale, size=noise_map_dim)
    noise = np.insert(noise, list(range(noise.shape[0])) * r_x, 1, axis=0)
    noise = np.insert(noise, list(range(noise.shape[1])) * r_y, 1, axis=1)
    while img.shape != noise.shape:
        if img.shape[0] != noise.shape[0]:
            noise = np.insert(noise, 0, 1, axis=0)
        if img.shape[1] != noise.shape[1]:
            noise = np.insert(noise, 0, 1, axis=1)

    return img * noise


def gamma_noise_resize(img, noise_map_dim=(256, 256), shape=3, scale=0.01):
    r_x = img.shape[0]
    r_y = img.shape[1]

    noise = np.random.gamma(shape, scale, size=noise_map_dim)
    noise = cv2.resize(noise, (r_x, r_y), cv2.INTER_LINEAR) * 10

    import scipy.special as sps
    count, bins, ignored = plt.hist(noise.flatten(), 50, density=True)
    y = bins ** (shape - 1) * (np.exp(-bins / scale) /
                               (sps.gamma(shape) * scale ** shape))
    plt.plot(bins, y, linewidth=2, color='r')
    plt.show()

    return utils.normalize_0_1(img * noise)


def log_tranform_gauss_noise(img, noise_map_dim=(256, 256), shape=1, var=0.04):
    noise = np.random.normal(shape, var, size=noise_map_dim)
    if noise_map_dim != img.shape:
        noise = cv2.resize(noise, img.shape, cv2.INTER_LINEAR)

    return np.exp(np.log(img) + np.log(noise))


if __name__ == '__main__':
    print()
    img = utils.get_example_img()
    utils.plot_image_g(img)
    utils.plot_image_g(hmf_denoise(img))
