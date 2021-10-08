from scipy.fft import fft2, dct, ifft2
import matplotlib.pyplot as plt
import numpy as np


def fft_exp(img):
    fft_im = fft2(img)
    from matplotlib.colors import LogNorm
    plt.figure()
    plt.imshow(np.abs(fft_im), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')
    plt.show()

    keep_fraction = 0.1
    im_fft2 = fft_im.copy()
    r, c = im_fft2.shape

    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0

    im_new = ifft2(im_fft2).real

    plt.figure()
    plt.imshow(im_new, cmap='gray')
    plt.title('Reconstructed Image')
    plt.show()

    return
