import numpy as np
import dl.dataloader
import utils
import os
import albumentations as A
from skimage.io import imread, imsave
import denoise


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
            image = img_preprocess_func(image)

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
        image = img_preprocess_func(image)
        data = resize(image=image, mask=mask)
        utils.plot_image_g(data['image'])
        return


def wavelet_convert(mode='visu'):
    sigma = {'visu': 0.020, 'bayes': 0.15}

    dataset_convert(f'camus_wavelet_sigma{sigma[mode]}_{mode}',
                    lambda img: denoise.wavelet_denoise(img, sigma[mode], mode))


def hmf_convert():
    # convert_test(lambda img: denoise.hmf_denoise(img))
    dataset_convert(f'camus_hmf', lambda img: denoise.hmf_denoise(img))


def csrad_convert(steps, step_size):
    dataset_convert(f'camus_csrad_{steps}-{step_size}', lambda img: denoise.csrad_denoise(img, steps, step_size))


def tv_convert(weight):
    dataset_convert(f'camus_tv_{weight}', lambda img: denoise.tv_denoise(img, weight))


def combine_method_convert(steps, step_size, weight):
    dataset_convert(f'camus_combined_{steps}-{step_size}_weight-{weight}',
                    lambda img: denoise.combined_method_denoise(img, steps, step_size, weight))


def tv_csrad_convert(steps, step_size, weight):
    dataset_convert(f'camus_tv_csrad_{steps}-{step_size}_weight-{weight}',
                    lambda img: denoise.tv_csrad_denoise(img, steps, step_size, weight))


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
    hmf_convert()
    # tv_convert(0.3)
    # csrad_convert(150, 0.1)
    # tv_csrad_convert(150, 0.05, 0.3)
    # combine_method_convert(100, 0.1, 0.5)
