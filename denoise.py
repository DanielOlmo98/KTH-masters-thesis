import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from utils import binarize


def sitk_noisefilter(img):
    img_sitk = sitk.GetImageFromArray(img)
    noise_filter = sitk.NoiseImageFilter()
    out = noise_filter.Execute(img_sitk)

    out = sitk.Threshold(out, 0.2, 1)

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

    fig = plt.figure(figsize=(6, 12))
    titles = ['Image', 'Noise', 'Denoised Image']
    for i, a in enumerate([img, noise, denoised]):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray', vmin=a.min(), vmax=a.max())
        ax.set_title(titles[i], fontsize=10)
    fig.tight_layout()
    plt.show()

    return denoised
