import pywt
import numpy as np
import matplotlib.pyplot as plt

import utils

'''
Adapted from:
https://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise_wavelet.html
'''


def wavelet_denoise(img, sigma=0.1, mode='visu'):
    wavelet = 'db1'
    max_lvl = pywt.dwtn_max_level(img.shape, wavelet)
    coeffs = pywt.wavedec2(img, wavelet, level=max_lvl - 3)
    if mode == 'visu':
        threshold = sigma * np.sqrt(2 * np.log(img.size))
    for i, level in enumerate(coeffs[1:]):
        coeffs[i + 1] = []
        for coeff in level:
            if mode == 'bayes':
                var = sigma ** 2
                dvar = np.mean(coeff * coeff)
                threshold = var / np.sqrt(max(dvar - var, 1e-7))
            coeffs[i + 1].append(pywt.threshold(coeff, threshold, mode='soft'))

    img_dn = pywt.waverec2(coeffs, wavelet)
    return img_dn


def wavelet_plot(img):
    titles = ['LL', 'LH',
              'HL', 'HH']
    wavelet = 'db1'
    coeffs = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs
    img_t = pywt.idwt2(coeffs, wavelet)

    fig2 = plt.figure(figsize=(5, 5))
    for i, a in enumerate([img_t, LH, HL, HH]):
        ax = fig2.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    image = utils.load_images()[-1]
    denoised = wavelet_denoise(image, sigma=0.05, mode='bayes')
    # utils.plot_image_g(image, title='Original')
    utils.plot_image_g(denoised, title='Denoised')
