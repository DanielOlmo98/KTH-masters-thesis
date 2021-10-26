from odl import uniform_discr
from odl.discr.diff_ops import Gradient, Laplacian, Divergence
import odl
import matplotlib.pyplot as plt
import numpy as np
from utils import get_project_root, plot_image_g, normalize_0_1, normalize_neg1_to_1
from noise_filtering.main import load_images
from scipy.stats import variation
from scipy.ndimage.filters import gaussian_filter



def ICOV(image, space):
    gradient_op = Gradient(space)
    grad_x, grad_y = gradient_op(image).parts
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    laplacian_op = Laplacian(space)
    laplacian = laplacian_op(image)
    numerator = 0.5 * ((grad_magnitude / image) ** 2) - 0.0625 * (laplacian ** 2)
    denominator = (image + 0.25 * laplacian)
    q = np.sqrt(np.abs(numerator)) / denominator

    # plot_image_g(laplacian)
    # plot_image_g(grad_x)
    # plot_image_g(grad_y)
    # plot_image_g(grad_magnitude)
    # plot_image_g(q)

    return q


def ICOV_0(image):
    variance = np.var(image)
    mean = np.mean(image)
    q_0 = np.sqrt(variance) / mean
    return q_0


def diffusion_coef(image, space):
    q = ICOV(image, space)
    q_0 = ICOV_0(image)
    c = 1 / (1 + ((q ** 2 - q_0 ** 2) / ((q_0 ** 2) * (1 + (q_0 ** 2)))))
    # heightmap(c)

    # c = normalize_neg1_to_1(c)

    # thresh = 0.3
    # zero_height = 0.75
    # c_bool = np.logical_or(c.data < zero_height-thresh, c.data > zero_height+thresh)
    # c.data[~c_bool] = zero_height

    # heightmap(c)

    return c


def pde(image, space):
    div_op = Divergence(range=space)
    gradient_op = Gradient(space)
    diff_coef = diffusion_coef(image, space)
    grad_img = gradient_op(image)

    for grad_part in grad_img.parts:
        thresh = -5
        grad_part.data[grad_part.data < thresh] = 0
        # g_bool = np.logical_or(grad_part.data < -thresh, grad_part.data > thresh)
        # grad_part.data[~g_bool] = -5

    d_n = div_op(diff_coef * grad_img)

    blurred_d_n = gaussian_filter(d_n.data, sigma=0.5)
    # heightmap(d_n)
    # heightmap(blurred_d_n)
    return blurred_d_n


def numeric_solve(image, iter, d_t, plot):
    epsilon = 10e-10
    image += epsilon
    width, height = image.shape
    space = uniform_discr([-1, -1], [1, 1], [width, height])
    I = [image]
    # d_t = 0.1
    for n in range(iter):
        d_n = np.abs(pde(I[n], space).data)
        d_n = normalize_0_1(d_n)

        # thresh = 0.05
        # d_n_bool = np.logical_or(d_n < -thresh, d_n > thresh)
        # d_n = np.where(d_n_bool, d_n, 0)

        I.append((I[n] + 0.25 * d_t * d_n) + epsilon)
        if plot and n % 25 == 0:
            fig = plt.figure(figsize=(6, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, projection='3d')
            plot_image_g(I[n], ax=ax1)
            heightmap(d_n, ax=ax2)
            plt.show()
            # plot_image_g(d_n)

    return I[-1]


def heightmap(array, ax=None):
    height, width = array.shape
    x = np.arange(0, height, 1)
    y = np.arange(0, width, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.transpose(array.data)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='hot')
        plt.show()
        return
    else:
        ax.plot_surface(X, Y, Z, cmap='hot')
        return ax


if __name__ == '__main__':
    path = get_project_root() + '/image/'
    images = load_images(path)
    image = images[0]
    image = image[130:300, 200:450]

    # epsilon = 10e-10
    # image += epsilon
    # ICOV(image)
    res_img = numeric_solve(image, 100, 0.05, plot=True)
    plot_image_g(res_img)
