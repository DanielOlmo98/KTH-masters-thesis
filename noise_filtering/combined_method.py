import numpy as np
import matplotlib.pyplot as plt
from utils import *
from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad
from noise_filtering.HMF import hybrid_median_filtering
import pywt

'''
Wavelet Decomposition-Based Speckle
Reduction Method for Ultrasound Images by
Using Speckle-Reducing Anisotropic Diffusion
and Hybrid Median 
https://www.researchgate.net/publication/44262468_Wavelet_Decomposition_Based_Speckle_Reduction_Method_for_Ultrasound_Images_by_Using_Speckle_Reducing_Anisotropic_Diffusion
'''


def csrad(image, steps, step_size):
    image_n = image
    ci = np.zeros_like(image)
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_csrad(array=image_n.clip(1e-9), ci_1=ci.clip(1e-9), iter=n, step=step_size)
        ci = np.asarray(ci.base)

    return image_n


def wavelet_HMF(coeffs):
    res = []
    for coeff in coeffs:
        res.append(hybrid_median_filtering(coeff).base)
    return res[0], res[1], res[2]


def combined_method(image, steps, step_size):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, wavelet_coeffs = coeffs2

    LL = csrad(LL, steps=steps, step_size=step_size)
    LH, HL, HH = wavelet_HMF(wavelet_coeffs)
    recon_img = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return recon_img[:, 0:-1]


if __name__ == '__main__':
    images = load_images()
    image = images[0]
    # image = image[130:300, 200:450]
    steps = 30
    step_size = 0.05
    denoised = combined_method(image, steps, step_size)
    plot_image_g(image, title='Original')
    plot_image_g(denoised, title='Denoised')
