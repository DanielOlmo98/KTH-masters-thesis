import numpy as np

import utils
from utils import plot_image_g, binarize
from scipy import interpolate, ndimage, special, signal, misc
import cv2


# https://ieeexplore.ieee.org/document/4228562
# https://ieeexplore.ieee.org/document/7967056

def simulate_noise(image, sample_dimension, angle, d_min, d_max, b, sigma):
    I_p = radial_polar_sampling_gen(image, d_max, angle, d_min, sample_dimension)
    plot_image_g(I_p)
    sector_mask = radial_polar_sampling_gen(image, d_max, angle, d_min)
    sampling_mask = radial_polar_sampling_gen(sector_mask, d_max, angle, d_min, sample_dimension)
    grid = rectification(I_p, sample_dimension, angle, d_min, d_max)
    # plot_image_g(grid)
    noisy_sample = noise_gen(grid, b, sigma)
    # plot_image_g(noisy_sample)
    final_img = interpolate_noise(noisy_sample, angle, sample_dimension, d_min, d_max, image, sampling_mask,
                                  sector_mask)
    final_img = utils.normalize_0_1(final_img)
    plot_image_g(final_img)
    return final_img


def radial_polar_sampling_gen(image, d_max, angle, d_min, sample_dimension=None):
    img_h, img_w = np.shape(image)
    if sample_dimension is None:
        grid_h, grid_w = img_h, img_w
        I_p = np.zeros_like(image, dtype=np.bool)
    else:
        grid_h, grid_w = sample_dimension
        I_p = np.zeros_like(image)

    for i in range(grid_w):
        theta = (3 * np.pi - angle) / 2 + i * angle / grid_w
        for j in range(grid_h):
            d = d_min + j * (d_max - d_min) / grid_h
            x = int((-d * np.sin(theta)))
            y = int((d * np.cos(theta) + img_w / 2))

            if sample_dimension is None:
                I_p[x, y] = True
            else:
                I_p[x, y] = image[x, y]

    return ndimage.median_filter(I_p, size=(4, 4)) if sample_dimension is None else I_p


def rectification(sampled_points, sample_dimension, angle, d_min, d_max):
    grid_h, grid_w = sample_dimension
    img_h, img_w = np.shape(sampled_points)
    rect_grid = np.zeros(shape=(grid_h, grid_w))
    for i in range(grid_w):
        theta = (3 * np.pi - angle) / 2 + i * angle / grid_w
        for j in range(grid_h):
            d = d_min + j * (d_max - d_min) / grid_h
            x = int((-d * np.sin(theta)))
            y = int((d * np.cos(theta) + img_w / 2))

            rect_grid[j, i] = sampled_points[x, y]

    return rect_grid


def noise_gen(sampled_points, b, sigma):
    noised = np.zeros_like(sampled_points)
    img_h, img_w = np.shape(sampled_points)
    for x in range(img_h):
        for y in range(img_w):
            intensity = sampled_points[x, y]
            A = np.real(np.sqrt(intensity))
            rand_int = np.random.randint(0, b)
            for i in range(rand_int):
                u = np.random.normal(intensity, sigma, rand_int)
                v = np.random.normal(intensity, sigma, rand_int)
                # u, v = np.random.multivariate_normal(mean=mean, cov=cov, size=rand_int).T
                A = A + u[i] + 1j * v[i]
            noised[x, y] = np.abs(A)
    return noised


def interpolate_noise(noisy_sample, angle, sample_dimension, d_min, d_max, image, sampling_mask, sector_mask):
    img_h, img_w = np.shape(image)
    grid_h, grid_w = sample_dimension
    img_final = np.zeros((img_h, img_w))
    i = 0
    for theta in np.linspace((3 * np.pi - angle) / 2, (3 * np.pi + angle) / 2, grid_w):
        j = 0
        for d in np.linspace(d_min, d_max, grid_h):
            x = int((-d * np.sin(theta)))
            y = int((d * np.cos(theta) + img_w / 2))
            img_final[x, y] = noisy_sample[j, i]
            j += 1
        i += 1

    for h in range(15,img_h-15):
        if np.count_nonzero(sector_mask[h, :]) == 0:
            continue
        img_final[h, :] = ndimage.convolve1d(img_final[h, :], lanc_kernel(4, np.count_nonzero(sector_mask[h, :])))
        img_final[h - 5:h + 5,:] = ndimage.convolve1d(img_final[h - 5:h + 5,:],
                                                    lanc_kernel(3, np.count_nonzero(sector_mask[h, :])), axis=0)

    img_final[~sector_mask] = 0
    return img_final

    # # big_grid = cv2.resize(noisy_sample, dsize=(grid_w * scaling, grid_h * scaling), interpolation=cv2.INTER_LANCZOS4)
    # i = 0
    # for h in range(img_h):
    #     # for y in range(img_w):
    #     # if sector_mask[x, :]:
    #     sector_line = sector_mask[h, :]
    #     noisy_line = noisy_sample[i, :]
    #     x = np.linspace(-3, 3, np.count_nonzero(sector_line))
    #     lanc_kernel = (special.sinc(x) * special.sinc(x / 3))
    #     resample = signal.convolve(lanc_kernel, noisy_line, mode='same')
    #     for index, value in zip(np.argwhere(sector_line), resample):
    #         img_final[h, index] = value
    #     if h // int(img_h / grid_h) == 0:
    #         i += 1
    #         # temp = np.put_along_axis(img_final[h, :], np.argwhere(sector_line), resample, axis=0)
    #     # img_final[h, :] = temp
    #     # img_final[x, y] = noisy_big_flat[i]
    #     # if i < len(noisy_big_flat) - 1:
    #     #     i += 1

    # x = np.arange(-3, 3, .1)
    # y = np.arange(-3, 3, 1)
    # lanc_x = (special.sinc(x) * special.sinc(x / 3))
    # lanc_y = (special.sinc(y) * special.sinc(y / 3))
    # # lanc = np.outer(lanc, lanc.T)
    # # img_final = img_final + noisy_sample
    # img_final = ndimage.convolve1d(img_final, lanc_x, axis=1)
    # img_final = ndimage.convolve1d(img_final, lanc_y, axis=0)
    # # img_final = lanczos_interp(noisy_sample, 3)


def lanc_kernel(a, steps):
    x = np.linspace(-a, a, steps)
    return special.sinc(x) * special.sinc(x / a)


def lanczos_interp(rectified_noise, window_size):
    def lanczos_kernel(window):
        window = np.reshape(window, (-1, 7))
        x = np.arange(-3, 3, .5)
        lanc = (special.sinc(x) * special.sinc(x / 3))

        np.multiply(window, lanc)

        return np.sum(np.sum(window))

    L_xy = ndimage.generic_filter(rectified_noise, lanczos_kernel, size=(window_size * 2 + 1, 1))
    # np.sum(rectified_noise[x, y] * L_xy)
    return L_xy


if __name__ == '__main__':
    from utils import load_images

    images = load_images()
    image = images[0]
    plot_image_g(image)
    #                                         h    w
    sampling_settigns = {'sample_dimension': (100, 40),
                         'angle': np.radians(90),
                         'd_min': 1,
                         'd_max': 450,
                         'b': 10,
                         'sigma': 0.7
                         }

    simulate_noise(image=image, **sampling_settigns)
