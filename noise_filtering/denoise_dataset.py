import os

import albumentations as A
import numpy as np
from skimage.io import imread, imsave

import denoise
import dl.dataloader
import utils


def dataset_convert(folder_save_name, img_preprocess_func=None, size=(256, 256), og_dataset_folder='training',
                    extension='.mhd'):
    """
    Resizes and saves the dataset into PNG format with optional additional transformation with a function.
    """
    if size is not None:
        resize = A.Resize(size[0], size[1])
    else:
        resize = None

    dataset_path = f'{utils.get_project_root()}/dataset/{og_dataset_folder}/'
    converted_path = f'{utils.get_project_root()}/dataset/{folder_save_name}/'

    img_paths, seg_paths = dl.dataloader.get_image_paths(dataset_path, extension=extension)

    folders = os.listdir(dataset_path)
    for folder in folders:
        os.makedirs(converted_path + folder, exist_ok=True)

    for img_path, seg_path in zip(img_paths, seg_paths):
        img_savename = f'{converted_path}{img_path[-35:-3]}png'
        seg_savename = f'{converted_path}{seg_path[-38:-3]}png'
        if os.path.exists(img_savename) and os.path.exists(seg_savename):
            continue

        image = imread(img_path)
        mask = imread(seg_path)

        if image.ndim > 2:
            image = image[0]
        if mask.ndim > 2:
            mask = mask[0]

        if img_preprocess_func is not None:
            image = img_preprocess_func(image)

        if resize is not None:
            data = resize(image=image, mask=mask)
            imsave(img_savename, data['image'])
            imsave(seg_savename, data['mask'])
        else:
            imsave(img_savename, image)
            imsave(seg_savename, mask)


def convert_test(img_preprocess_func):
    resize = A.Resize(256, 256)

    dataset_path = utils.get_project_root() + '/dataset/' + 'training/'

    img_paths, seg_paths = dl.dataloader.get_image_paths(dataset_path)

    for img_path, seg_path in zip(img_paths, seg_paths):
        image = imread(img_path)[0]
        mask = imread(seg_path)[0]
        image = img_preprocess_func(image)
        data = resize(image=image, mask=mask)
        utils.plot_image_g(data['image'])
        return


def wavelet_convert(mode, sigma):
    dataset_convert(f'camus_wavelet_sigma{sigma}_{mode}',
                    lambda img: denoise.wavelet_denoise(img, sigma, mode))


def hmf_convert(test_set=False):
    # convert_test(lambda img: denoise.hmf_denoise(img))
    foldername = 'camus_hmf'
    og_dataset_folder = 'camus_png'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'camus_png_test'
    dataset_convert(foldername, lambda img: denoise.hmf_denoise(img), size=None,
                    og_dataset_folder=og_dataset_folder,
                    extension='.png')


def csrad_convert(steps, step_size, test_set=False):
    foldername = 'camus_csrad'
    og_dataset_folder = 'training'
    if test_set:
        foldername += 'test'
        og_dataset_folder = 'test'
    dataset_convert(f'{foldername}_{steps}-{step_size}', lambda img: denoise.csrad_denoise(img, steps, step_size),
                    og_dataset_folder=og_dataset_folder)


def tv_convert(weight, test_set=False):
    foldername = 'camus_tv'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(f'{foldername}_{weight}', lambda img: denoise.tv_denoise(img, weight),
                    og_dataset_folder=og_dataset_folder)


def combine_method_convert(steps, step_size, weight, test_set=False):
    foldername = 'camus_combined'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(f'{foldername}_{steps}-{step_size}_weight-{weight}',
                    lambda img: denoise.combined_method_denoise(img, steps, step_size, weight),
                    og_dataset_folder=og_dataset_folder)


def tv_csrad_convert(steps, step_size, weight, test_set=False):
    foldername = 'camus_tv_csrad'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(f'{foldername}_{steps}-{step_size}_weight-{weight}',
                    lambda img: denoise.tv_csrad_denoise(img, steps, step_size, weight),
                    og_dataset_folder=og_dataset_folder)


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
    denoise_settings = denoise.get_settings_dict()
    # hmf_convert()
    tv_convert(**denoise_settings['tv'], test_set=False)
    tv_convert(**denoise_settings['tv'], test_set=True)
    csrad_convert(**denoise_settings['csrad'], test_set=False)
    csrad_convert(**denoise_settings['csrad'], test_set=True)

    # tv_csrad_convert(**denoise_settings['tv_csrad'])
    # combine_method_convert(**denoise_settings['combine'])
