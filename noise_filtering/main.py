import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color, img_as_float32
from noise_filtering.denoise import sitk_noisefilter
from noise_filtering.wavelet_denoise import wavelet_exp
from noise_filtering.dct import dct_exp
from utils import get_project_root


# TODO: implement SRAD
# TODO: total variation optimization problem denoise
def load_images(path):
    file_list = os.listdir(path)
    images = []
    for file in file_list:
        common_file_path = path + file
        if not os.path.isfile(common_file_path):
            continue

        image = imread(common_file_path)
        image = image[80:550, 80:720]
        images.append(img_as_float32(color.rgb2gray(image)))
    return images


if __name__ == '__main__':
    path = get_project_root() + '/image/'
    images = load_images(path)
    image = images[0]
    # for image in images:
    plt.imshow(image, cmap='gray')
    plt.title("Before")
    plt.show()

    # image = wavelet_exp(image, plot=True)
    # image = sitk_noisefilter(image)
    # image = dct_exp(image, plot=True)
    # plt.imshow(image, cmap='gray')
    # plt.title("After")
    # plt.show()
    #
    # noise = images[0] - image
    # plt.imshow(noise, cmap='gray')
    # plt.title("Noise")
    # plt.show()
