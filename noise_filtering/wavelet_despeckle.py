import utils
import pywt
import matplotlib.pyplot as plt
from noise_filtering.main import load_images
import cv2
import numpy as np
import odl

"""
Wavelet DecompositionYBased Speckle
Reduction Method for Ultrasound Images by
Using Speckle-Reducing Anisotropic Diffusion
and Hybrid Median
http://dx.doi.org/10.1097/JCE.0000000000000300
"""


# Modified from:
# https://www.researchgate.net/publication/332574579_Image_Processing_Course_Project_Image_Filtering_with_Wiener_Filter_and_Median_Filter
def HMF(wavelet_coeffs):
    for arr in wavelet_coeffs:
        x_size, y_size = np.shape(arr)
        temp = []
        data_final = np.zeros((len(arr), len(arr[0])))
        kernel_size = 5
        indexer = kernel_size // 2
        for i in range(x_size):
            for j in range(y_size):
                for z in range(kernel_size):
                    if i + z - indexer < 0 or i + z - indexer > len(arr) - 1:
                        for c in range(kernel_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(arr[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(kernel_size):
                                temp.append(arr[i + z - indexer][j + k - indexer])
                while len(temp) < 25:
                    temp.append(0)

                diag = np.median(np.take(temp, [0, 4, 6, 8, 16, 18, 20, 24, 12]))
                cross = np.median(np.take(temp, [2, 7, 17, 22, 10, 11, 13, 14, 12]))
                center = np.take(temp, 12)
                data_final[i][j] = np.median(np.array([diag, cross, center]))
                temp = []

        plt.imshow(arr, cmap='gray')
        plt.show()
        plt.imshow(data_final, cmap='gray')
        plt.show()
    return


def wavelet_despeckle(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    HMF((LH, HL, HH))
    # HMF(img)


if __name__ == '__main__':
    path = utils.get_project_root() + '/image/'
    images = load_images(path)
    image = images[0]
    wavelet_despeckle(image)
