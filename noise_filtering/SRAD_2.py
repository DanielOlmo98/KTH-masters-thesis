from odl import uniform_discr
from odl.discr.diff_ops import Gradient, Laplacian, Divergence
import odl
import matplotlib.pyplot as plt
import numpy as np
from utils import get_project_root, plot_image_g, normalize_0_1, normalize_neg1_to_1
from noise_filtering.main import load_images
from scipy.stats import variation


def SRAD(img, step_size, step):
    width, height = img.shape
    c_i_j = np.zeros_like(img)
    q0 = np.exp(-step_size * step / 6)  # speckle scale function

    print('l1')
    for i in range(width):
        for j in range(height):
            # 4 Neighbourhood in cardinal directions (center, north, east, south, west)
            img_C = img[i, j]
            img_N = img[i, j + 1] if j != height - 1 else img[i, j]
            img_E = img[i + 1, j] if i != width - 1 else img[i, j]
            img_S = img[i, j - 1] if height != 0 else img[i, j]
            img_W = img[i - 1, j] if width != 0 else img[i, j]

            # Gradient estimation
            grad_R = [img_E - img_C, img_N - img_C]
            grad_L = [img_C - img_W, img_C - img_S]
            grad_magn = grad_R[0] ** 2 + grad_R[1] ** 2 + grad_L[0] ** 2 + grad_L[1] ** 2

            lapl = img_E + img_W + img_N + img_S - 4 * img_C

            # diffusion coeff
            q = np.sqrt(
                (0.5 * (grad_magn / img_C) ** 2 - 0.0625 * (lapl / img_C) ** 2) / (1 + 0.25 * lapl / img_C) ** 2)
            c_i_j[i, j] = 1 / (1 + (q ** 2 - q0 **2) / ((q0 ** 2)* (1 + q0 **2)))

    print('l2')

    res = np.zeros_like(c_i_j)
    for i in range(width):
        for j in range(height):
            # 4 Neighbourhood in cardinal directions (center, north, east, south, west)
            img_C = img[i, j]
            img_N = img[i, j + 1] if j != height - 1 else img[i, j]
            img_E = img[i + 1, j] if i != width - 1 else img[i, j]
            img_S = img[i, j - 1] if height != 0 else img[i, j]
            img_W = img[i - 1, j] if width != 0 else img[i, j]

            c_C = c_i_j[i, j]
            c_N = c_i_j[i, j + 1] if j != height - 1 else c_i_j[i, j]
            c_E = c_i_j[i + 1, j] if i != width - 1 else c_i_j[i, j]
            c_S = c_i_j[i, j - 1] if height != 0 else c_i_j[i, j]
            c_W = c_i_j[i - 1, j] if width != 0 else c_i_j[i, j]

            d = c_E*(img_E - img_C) + c_C*(img_W - img_C) + c_N*(img_N - img_C) + c_C*(img_S - img_C)

            res[i, j] = img_C + 0.25 * step_size * d

    return res


if __name__ == '__main__':
    path = get_project_root() + '/image/'
    images = load_images(path)
    image = images[0]
    image = image[130:300, 200:450]

    epsilon = 10e-10
    image += epsilon
    steps = 10
    img_n = image
    for n in range(steps):
        img_n = SRAD(img_n, step_size=0.05, step=n)
        if n % 5 == 0:
            plot_image_g(img_n)
    print()
