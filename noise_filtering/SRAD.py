from odl import uniform_discr
from odl.discr.diff_ops import Gradient, Laplacian, Divergence
import odl
import matplotlib.pyplot as plt
import numpy as np
from utils import get_project_root, plot_image_g, normalize
from noise_filtering.main import load_images
from scipy.stats import variation


def ICOV(image):
    width, height = image.shape
    space = uniform_discr([-1, -1], [1, 1], [width, height])
    gradient_op = Gradient(space)
    grad_x, grad_y = gradient_op(image).parts
    grad_magnitude = np.sqrt(np.abs(grad_x ** 2) + np.abs(grad_y ** 2))
    laplacian_op = Laplacian(space)
    laplacian = laplacian_op(image)
    numerator = 0.5 * ((grad_magnitude / image) ** 2) - 0.0625 * ((laplacian / image) ** 2)
    denominator = (1 + 0.25 * (laplacian / image)) ** 2
    q = np.sqrt(np.abs(numerator) / denominator)

    # plot_image_g(laplacian)
    # plot_image_g(grad_x)
    # plot_image_g(grad_y)
    # plot_image_g(grad_magnitude)
    # plot_image_g(q)

    return q


def ICOV_0(image):
    variance = np.var(image)
    mean = np.mean(image)
    return np.sqrt(variance) / mean


def diffusion_coef(image):
    q = ICOV(image)
    q_0 = ICOV_0(image)
    c = 1 / (1 + (q ** 2 - q_0 ** 2) / ((q_0 ** 2) * (1 + (q_0 ** 2))))
    return c


def pde(image):
    epsilon = 10e-10
    image += epsilon
    width, height = image.shape
    space = uniform_discr([-1, -1], [1, 1], [width, height])
    dom = odl.ProductSpace(space, space.ndim)
    div_op = Divergence(range=space)
    gradient_op = Gradient(space)
    grad_x, grad_y = gradient_op(image).parts
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    diff_coef = diffusion_coef(image)
    # cxI = diff_coef * grad_magnitude
    f = div_op.domain.element([diff_coef, grad_magnitude])
    d_n = div_op(f)
    return d_n


def numeric_solve(image):
    epsilon = 10e-10
    image += epsilon
    I = [image]
    d_t = 0.05
    for n in range(100):
        d_n = pde(I[n])
        d_n = normalize(d_n)
        I.append(I[n] + 0.25 * d_t * d_n)
        if n % 10 == 0:
            plot_image_g(I[n])
    return I


if __name__ == '__main__':
    path = get_project_root() + '/image/'
    images = load_images(path)
    image = images[0]

    numeric_solve(image)
