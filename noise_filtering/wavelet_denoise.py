import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_wavelet


def skimage_wavelet_denoise(img):
    return denoise_wavelet(img, method='VisuShrink', mode='soft', rescale_sigma=True, sigma=0.04)


def wavelet_exp(img, sigma=0.1, plot=False):
    titles = ['Thresholded', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    wavelet = 'db1'
    max_lvl = pywt.dwtn_max_level(img.shape, wavelet)
    coeffs = pywt.wavedec2(img, wavelet, level=max_lvl - 3)
    threshold = sigma * np.sqrt(2 * np.log(img.size))
    for i, level in enumerate(coeffs[1:]):
        coeffs[i + 1] = []
        for coeff in level:
            # threshold = 50
            threshold = sigma * np.sqrt(2 * np.log(coeff.size))
            coeffs[i + 1].append(pywt.threshold(coeff, threshold, mode='soft'))

    img_dn = pywt.waverec2(coeffs, wavelet)
    return img_dn
    # LL, (LH, HL, HH) = coeffs
    # LH = pywt.threshold(LH, threshold, mode='soft')
    # HL = pywt.threshold(HL, threshold, mode='soft')
    # HH = pywt.threshold(HH, threshold, mode='soft')
    # coeffs_t = (LL, (LH, HL, HH))
    # img_t = pywt.idwt2(coeffs_t, wavelet)

    # if plot:
    #     fig2 = plt.figure(figsize=(12, 12))
    #     for i, a in enumerate([img_t, LH, HL, HH]):
    #         ax = fig2.add_subplot(2, 2, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap='gray')
    #         ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     fig2.tight_layout()
    #     plt.show()

    # return img_t


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
