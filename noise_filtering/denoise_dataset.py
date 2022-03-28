import dl.dataloader
import utils
import os
import albumentations as A
from skimage.io import imread, imsave
import numpy as np


def dataset_convert(folder_save_name, img_preprocess_func=None):
    """
    Resizes and saves the dataset into PNG format with optional additional transformation with a function.
    """
    resize = A.Resize(256, 256)
    dataset_path = utils.get_project_root() + '/dataset/' + 'training/'
    converted_path = utils.get_project_root() + '/dataset/' + folder_save_name + '/'

    img_paths, seg_paths = dl.dataloader.get_image_paths(dataset_path)

    folders = os.listdir(dataset_path)
    for folder in folders:
        os.makedirs(converted_path + folder)

    for img_path, seg_path in zip(img_paths, seg_paths):
        image = imread(img_path)[0]
        mask = imread(seg_path)[0]
        if img_preprocess_func is not None:
            image = img_preprocess_func(image / 255.)

        data = resize(image=image, mask=mask)
        imsave(converted_path + img_path[31:-3] + 'png', data['image'])
        imsave(converted_path + seg_path[31:-3] + 'png', data['mask'])


def convert_test(img_preprocess_func):
    resize = A.Resize(256, 256)

    dataset_path = utils.get_project_root() + '/dataset/' + 'training/'

    img_paths, seg_paths = dl.dataloader.get_image_paths(dataset_path)

    for img_path, seg_path in zip(img_paths, seg_paths):
        image = imread(img_path)[0]
        mask = imread(seg_path)[0]
        image = img_preprocess_func(image / 255.)
        data = resize(image=image, mask=mask)
        utils.plot_image_g(data['image'])
        return


def wavelet_convert(mode='visu'):
    from noise_filtering.wavelet_denoise import wavelet_denoise

    if mode not in ['visu', 'bayes']:
        raise ValueError("Mode must be either 'visu' or 'bayes'")

    sigma = {'visu': 0.015, 'bayes': 0.08}

    dataset_convert(f'camus_wavelet_sigma{sigma[mode]}_{mode}',
                    lambda img: wavelet_denoise(img, sigma[mode], mode))


def hmf_convert():
    from noise_filtering.HMF import hybrid_median_filtering

    dataset_convert(f'camus_hmf', lambda img: hybrid_median_filtering(img))


def csrad_convert(steps, step_size):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad
    def csrad_denoise(image, steps, step_size):
        image_n = image
        ci = np.zeros_like(image)
        for n in range(steps):
            image_n = np.abs(image_n)
            image_n, ci, di = cy_csrad(array=image_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
            ci = np.asarray(ci.base)
        return image_n

    convert_test(lambda img: csrad_denoise(img, steps, step_size))
    # dataset_convert(f'camus_csrad', lambda img: csrad_denoise(img, 10, 0.1))


def tv_convert(weight):
    from skimage.restoration import denoise_tv_bregman
    convert_test(lambda img: denoise_tv_bregman(img, weight))


def combine_method_convert(steps, step_size, weight):
    from noise_filtering.combined_method import combined_method
    convert_test(lambda img: combined_method(img, steps, step_size, weight))


def tv_csrad_convert(steps, step_size, weight):
    from skimage.restoration import denoise_tv_bregman
    from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad

    def tv_csrad(image):
        image = denoise_tv_bregman(image, weight)
        image_n = image
        ci = np.zeros_like(image)
        for n in range(steps):
            image_n = np.abs(image_n)
            image_n, ci, di = cy_csrad(array=image_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
            ci = np.asarray(ci.base)
        return image_n

    convert_test(lambda img: tv_csrad(img))


def gamma_noise_repeats(img, noise_map_dim=(128, 128)):
    """
    Add gamma multiplicative noise, repeats the noise map to match img dimensions.
    """
    # rescale factors
    r_x = img.shape[0] / 128
    r_y = img.shape[1] / 128
    noise = np.repeat(np.repeat(np.random.gamma(100, scale=1, size=noise_map_dim), r_x, axis=0), r_y, axis=1)
    return img * noise


def gamma_noise_gaps(img, noise_map_dim=(128, 128)):
    """
    Add gamma multiplicative noise, inserts ones in the noise gap to match img dimensions.
    """
    # repeat count
    r_x = img.shape[0] // 128
    r_y = img.shape[1] // 128
    noise = np.random.gamma(100, scale=1, size=noise_map_dim)
    noise = np.insert(noise, list(range(len(noise))) * r_x, 1, axis=0)
    noise = np.insert(noise, list(range(len(noise))) * r_y, 1, axis=1)
    while img.shape != noise.shape:
        if img.shape[0] != noise.shape[0]:
            noise = np.insert(noise, 0, 1, axis=0)
        if img.shape[1] != noise.shape[1]:
            noise = np.insert(noise, 0, 1, axis=1)

    return img * noise


if __name__ == '__main__':
    # wavelet_convert('visu')
    # utils.load_patient()
    csrad_convert(20, 0.1)
