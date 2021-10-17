from scipy.fft import dct, dctn
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize


def dct2(image, inverse=False):
    type = 3 if inverse else 2
    return dct(dct(image, axis=0, norm='ortho', type=type), axis=1, norm='ortho', type=type)


def dct_exp(img, plot=False):
    dct_image = dctn(img)
    # dct_image_thresh = sitk.GetArrayFromImage(
    #     sitk.Threshold(sitk.GetImageFromArray(dct_image[400:,600:]), lower=0, upper=100))

    recon_img = dct2(dct_image, inverse=True)
    norml = normalize(dct_image)
    if plot:
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.show()
        plt.imshow(norml, cmap='gray')
        plt.title("DCT")
        plt.show()
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed")
        plt.show()
    # from imageio import imwrite
    # imwrite('outfile.jpg', dct_image)
    return dct_image
