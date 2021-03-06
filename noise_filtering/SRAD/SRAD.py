from odl import uniform_discr
from odl.discr.diff_ops import Gradient, Laplacian, Divergence
import odl
import matplotlib.pyplot as plt
import numpy as np
from utils import get_project_root, plot_image_g, normalize_0_1, normalize_neg1_to_1, heightmap, symmetric_threshold
from noise_filtering.denoise_tests import load_images
from scipy.stats import variation
from scipy.ndimage.filters import gaussian_filter

'''
Speckle Reducing Anisotropic Diffusion (SRAD)
https://ieeexplore.ieee.org/document/1097762
https://www.mdpi.com/2072-4292/11/23/2768/htm
https://www.researchgate.net/publication/221472052_Coefficient-Tracking_Speckle_Reducing_Anisotropic_Diffusion
 
'''


def ICOV(image, space):
    gradient_op = Gradient(space)
    grad_magnitude = np.squeeze(np.linalg.norm(gradient_op(image), keepdims=True, axis=0))
    laplacian_op = Laplacian(space)
    laplacian = laplacian_op(image)
    numerator = (0.5 * (grad_magnitude ** 2)) - (0.0625 * (laplacian ** 2))
    denominator = (image + 0.25 * laplacian)
    q = np.sqrt(np.abs(numerator)) / denominator

    # plot_image_g(laplacian)
    # plot_image_g(grad_magnitude)
    # plot_image_g(q)

    # heightmap(q, title="ICOV")

    return q


def ICOV_0(image):
    """
    Wrong implementation, unused
    """

    width, height = image.shape
    patch_size = 30
    x, y = width // 2, height // 2
    image = image[x:x + patch_size, y:y + patch_size]
    variance = np.var(image)
    mean = np.mean(image)
    q_0 = np.sqrt(variance) / mean

    return q_0


def diffusion_coef(image, space, step, step_size):
    q = ICOV(image, space)
    q_0 = np.exp(-step_size * step / 6)
    c = 1 / (1 + ((q ** 2 - q_0 ** 2) / ((q_0 ** 2) * (1 + (q_0 ** 2)))))

    plot_image_g(c)
    heightmap(c)

    # c = normalize_neg1_to_1(c)

    # thresh = 0.3
    # zero_height = 0.75
    # c_bool = np.logical_or(c.data < zero_height-thresh, c.data > zero_height+thresh)
    # c.data[~c_bool] = zero_height

    # heightmap(c, title="c")

    return c


def pde(image, space, n, step_size):
    div_op = Divergence(range=space)
    gradient_op = Gradient(space)
    diff_coef = diffusion_coef(image, space, step=n, step_size=step_size)
    grad_img = gradient_op(image)
    grad_img = normalize_neg1_to_1(grad_img)

    # for grad_part in grad_img.parts:
    #     thresh = -5
    #     grad_part.data[grad_part.data < thresh] = 0
    #     # g_bool = np.logical_or(grad_part.data < -thresh, grad_part.data > thresh)
    #     # grad_part.data[~g_bool] = -5

    # heightmap(diff_coef)
    d_n = div_op(diff_coef * grad_img)

    # blurred_d_n = gaussian_filter(d_n.data, sigma=0.5)
    # heightmap(d_n, title="pde")
    # heightmap(blurred_d_n, title='blurred')
    return d_n


def numeric_solve(image, iter, d_t, plot):
    epsilon = 10e-10
    image += epsilon
    width, height = image.shape
    space = uniform_discr([-1, -1], [1, 1], [width, height])
    I = [image]
    # d_t = 0.1
    for n in range(iter):
        d_n = pde(I[n], space, n, d_t).data

        d_n = normalize_neg1_to_1(d_n)
        bool_borders = np.full_like(d_n, True, dtype='bool')
        bool_borders[1:-1, 1:-1] = False
        d_n[bool_borders] = epsilon

        # d_n = symmetric_threshold(d_n, 0.05, zero_point=0, invert=True)
        i_n = I[n] + 0.25 * d_t * d_n
        i_n[i_n < 0] = epsilon
        I.append(i_n)
        if plot and n % 50 == 0:
            fig = plt.figure(figsize=(6, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, projection='3d')
            plot_image_g(I[n], ax=ax1)
            heightmap(d_n, ax=ax2)
            plt.show()
            # plot_image_g(d_n)

    return I[-1]


if __name__ == '__main__':
    images = load_images()
    image = images[0]
    image = image[130:300, 200:450]

    # epsilon = 10e-10
    # image += epsilon
    # ICOV(image)
    res_img = numeric_solve(image, 100, 0.05, plot=True)
    plot_image_g(image, title='Original')
    plot_image_g(res_img, title='Final SRAD')
