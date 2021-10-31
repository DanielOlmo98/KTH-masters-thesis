import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color, img_as_float32
from noise_filtering.denoise import sitk_noisefilter
from noise_filtering.wavelet_denoise import wavelet_exp
from noise_filtering.dct import dct_exp
from utils import get_project_root, plot_image_g, heightmap


# TODO: debug SRAD
# TODO: implement CSRAD (Coefficient-Tracking Speckle Reducing Anisotropic Diffusion)
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
    # image = image[130:300, 200:450]
    epsilon = 10e-5
    image += epsilon

    from PyRAD_SRAD import cy_srad

    steps = 50
    step_size = 0.05
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
