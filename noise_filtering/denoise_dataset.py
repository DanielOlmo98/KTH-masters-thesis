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

        image = utils.normalize_0_1(image).astype(np.float32)

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


def wavelet_convert(mode, sigma, test_set=False):
    foldername = 'camus_csrad'
    if test_set:
        foldername += '_test'

    dataset_convert(foldername,
                    lambda img: denoise.wavelet_denoise_w(img, sigma, mode))


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
    foldername = f'camus_csrad_{steps}-{step_size}'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(foldername, lambda img: denoise.csrad_denoise(img, steps, step_size),
                    og_dataset_folder=og_dataset_folder)


def tv_convert(weight, max_iter, eps, test_set=False):
    foldername = f'camus_tv_w{weight}_eps{eps}'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(foldername, lambda img: denoise.tv_denoise(img, weight, max_iter, eps),
                    og_dataset_folder=og_dataset_folder)


def combine_method_convert(steps, step_size, weight, max_iter, eps, test_set=False):
    foldername = f'camus_combined_{steps}-{step_size}_w{weight}_eps{eps}'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(foldername,
                    lambda img: denoise.combined_method_denoise(img, steps, step_size, weight, max_iter, eps),
                    og_dataset_folder=og_dataset_folder)


def tv_csrad_convert(steps, step_size, weight, test_set=False):
    """UNUSED"""
    foldername = f'camus_tv_csrad_{steps}-{step_size}_weight-{weight}'
    og_dataset_folder = 'training'
    if test_set:
        foldername += '_test'
        og_dataset_folder = 'test'
    dataset_convert(foldername,
                    lambda img: denoise.tv_csrad_denoise(img, steps, step_size, weight),
                    og_dataset_folder=og_dataset_folder)


if __name__ == '__main__':
    denoise_settings = denoise.get_settings_dict('strong')
    # hmf_convert()
    # tv_convert(**denoise_settings['TV'], test_set=False)
    # tv_convert(**denoise_settings['TV'], test_set=True)
    # csrad_convert(**denoise_settings['CSRAD'], test_set=False)
    csrad_convert(**denoise_settings['CSRAD'], test_set=True)

    # tv_csrad_convert(**denoise_settings['tv_csrad'])
    # combine_method_convert(**denoise_settings['Combined'])
    # combine_method_convert(**denoise_settings['Combined'], test_set=True)
