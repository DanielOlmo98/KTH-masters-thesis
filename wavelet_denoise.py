import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def wavelet_exp(img, plot=False):
    titles = ['Thresholded', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2

    l = 0.2
    LH = LH - pywt.threshold(LH, l, mode='soft', substitute=0)
    HL = HL - pywt.threshold(HL, l, mode='soft', substitute=0)
    HH = pywt.threshold(HH, l, mode='soft', substitute=0)

    coeffs_t = (LL, (LH, HL, HH))
    img_t = pywt.idwt2(coeffs_t, 'haar')

    if plot:
        fig2 = plt.figure(figsize=(12, 12))
        for i, a in enumerate([img_t, LH, HL, HH]):
            ax = fig2.add_subplot(2, 2, i + 1)
            ax.imshow(a, interpolation="nearest", cmap='gray')
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig2.tight_layout()
        plt.show()

    return img_t
