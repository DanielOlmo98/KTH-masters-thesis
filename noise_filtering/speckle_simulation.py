import numpy as np


def radial_polar_sampling_gen(sample_img_size, angle, d_min, d_max, image):
    height, width = np.shape(image)
    sizex, sizey = sample_img_size
    I_p = np.zeros_like(image)
    for i in range(sizex):
        theta = (3 * np.pi - angle) / 2 + i * angle / sizex
        for j in range(sizey):
            d = d_min + j * (d_max - d_min) / sizey
            x = int((d * np.cos(theta) + width / 2))
            y = int((-d * np.sin(theta)))

            I_p[y, x] = image[y, x]

    return I_p
