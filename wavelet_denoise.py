import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def wavelet_exp(img):
    titles = ['Denoised', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    # fig = plt.figure(figsize=(12, 12))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(2, 2, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap='gray')
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # fig.tight_layout()
    # plt.show()

    upper = 0.11
    lower = 0.0001
    l = 0.01
    LH = pywt.threshold_firm(LH, lower, upper)
    HL = pywt.threshold_firm(HL, lower, upper)
    HH = pywt.threshold_firm(HH, lower, upper)
    coeffs_t = (LL, (LH, HL, HH))
    img_t = pywt.idwt2(coeffs_t, 'bior1.3')
    fig2 = plt.figure(figsize=(12, 12))
    for i, a in enumerate([img_t, LH, HL, HH]):
        ax = fig2.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.tight_layout()
    plt.show()
