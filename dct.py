from scipy.fft import dct
import matplotlib.pyplot as plt
import numpy as np


def dct2(image, inverse=False):
    type = 3 if inverse else 2
    return dct(dct(image, axis=0, norm='ortho', type=type), axis=1, norm='ortho', type=type)


def dct_exp(img, plot=False):
    dct_image = dct2(img)
    # dct_image_thresh = sitk.GetArrayFromImage(
    #     sitk.Threshold(sitk.GetImageFromArray(dct_image[400:,600:]), lower=0, upper=100))

    dct_image[200:, 0:200] = 0
    recon_img = dct2(dct_image, inverse=True)
    dct_im2 = dct2(img)
    dct_im2[0:200, 200:] = 0

    if plot:
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.imshow(recon_img, cmap='gray')
        plt.show()
        plt.imshow(dct2(dct_im2, inverse=True), cmap='gray')
        plt.show()

    return
