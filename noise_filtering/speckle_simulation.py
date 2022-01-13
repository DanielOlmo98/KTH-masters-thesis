import numpy as np
from utils import plot_image_g
from scipy import interpolate


# https://ieeexplore.ieee.org/document/4228562
# https://ieeexplore.ieee.org/document/7967056

def simulate_noise(image, sample_dimension, angle, d_min, d_max, b, sigma):
    I_p = radial_polar_sampling_gen(image, sample_dimension, angle, d_min, d_max)
    plot_image_g(I_p)
    grid = rectification(I_p, sample_dimension, angle, d_min, d_max)
    plot_image_g(grid)
    noisy_sample = noise_gen(grid, b, sigma)
    plot_image_g(noisy_sample)
    final_img = interpolate_noise(noisy_sample, angle, sample_dimension, d_min, d_max, image)
    plot_image_g(final_img)


def radial_polar_sampling_gen(image, sample_dimension, angle, d_min, d_max):
    img_h, img_w = np.shape(image)
    grid_h, grid_w = sample_dimension
    I_p = np.zeros_like(image)
    for i in range(grid_w):
        theta = (3 * np.pi - angle) / 2 + i * angle / grid_w
        for j in range(grid_h):
            d = d_min + j * (d_max - d_min) / grid_h
            x = int((-d * np.sin(theta)))
            y = int((d * np.cos(theta) + img_w / 2))

            I_p[x, y] = image[x, y]

    return I_p


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


def interpolate_noise(noisy_sample, angle, sample_dimension, d_min, d_max, image):
    img_h, img_w = np.shape(image)
    grid_h, grid_w = sample_dimension
    img_temp = np.zeros(sample_dimension)
    img_final = np.zeros((img_h, img_w))
    i = 0
    for theta in np.linspace((3 * np.pi - angle) / 2, (3 * np.pi + angle) / 2, grid_w):
        j = 0
        for d in np.linspace(d_min, d_max, grid_h):
            x = int((-d * np.sin(theta)))
            y = int((d * np.cos(theta) + img_w / 2))
            img_final[x, y] = noisy_sample[j, i]
            img_temp[j, i] = noisy_sample[j, i]
            j += 1
        i += 1

    mask = img_final == 0
    points = mask.nonzero()
    values = img_final[points]
    gridcoords = np.meshgrid[:img_h, :img_w]

    img_final = interpolate.griddata(points, values, gridcoords, method='nearest')  # or method='linear', method='cubic'
    # img_final = interpolate.interp2d

    return img_final
