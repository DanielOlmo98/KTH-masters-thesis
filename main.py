import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
from scipy.fft import fft2, dct, ifft2


def binarize(img):
    img[img > 0] = 1
    return img


def dct2(image, inverse=False):
    type = 3 if inverse else 2
    return dct(dct(image, axis=0, norm='ortho', type=type), axis=1, norm='ortho', type=type)


def dct_exp(images):
    for img in images:
        dct_image = dct2(img)
        # dct_image_thresh = sitk.GetArrayFromImage(
        #     sitk.Threshold(sitk.GetImageFromArray(dct_image[400:,600:]), lower=0, upper=100))

        dct_image[200:, 0:200] = 0

        plt.imshow(img, cmap='gray')
        plt.show()
        recon_img = dct2(dct_image, inverse=True)
        plt.imshow(recon_img, cmap='gray')
        plt.show()
        dct_im2 = dct2(img)
        dct_im2[0:200, 200:] = 0
        plt.imshow(dct2(dct_im2, inverse=True), cmap='gray')
        plt.show()

    return


def fft_exp(images):
    for img in images:
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


def sitk_noisefilter(images):
    for img in images:
        img_sitk = sitk.GetImageFromArray(img)
        noise_filter = sitk.NoiseImageFilter()
        out = noise_filter.Execute(img_sitk)

        out = sitk.Threshold(out, 0.1, 1)

        # minmaxfilter = sitk.MinimumMaximumImageFilter()
        # minmaxfilter.Execute(out)
        # print(minmaxfilter.GetMaximum())
        # print(minmaxfilter.GetMinimum())

        noise = sitk.GetArrayFromImage(out)
        noise = binarize(noise)
        denoised = np.subtract(img, noise)
        denoised[denoised < 0.1] = 0

        gauss_filter = sitk.DiscreteGaussianImageFilter()

        denoised = sitk.GetArrayFromImage(gauss_filter.Execute(sitk.GetImageFromArray(denoised)))
        plt.imshow(noise, cmap='gray')
        plt.show()
        plt.imshow(denoised, cmap='gray')
        plt.show()
        plt.imshow(img, cmap='gray')
        plt.show()

    return


file_list = os.listdir('image/')

images = []
for file in file_list:
    common_file_path = 'image/' + file
    if not os.path.isfile(common_file_path):
        continue

    image = imread(common_file_path)
    image = image[80:550, 80:720]
    images.append(color.rgb2gray(image))

sitk_noisefilter(images)
