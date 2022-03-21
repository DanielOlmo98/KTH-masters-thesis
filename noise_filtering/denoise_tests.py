import os
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color, img_as_float32
from skimage.transform import resize
from noise_filtering.denoise import sitk_noisefilter
from noise_filtering.wavelet_denoise import wavelet_denoise, wavelet_plot
from noise_filtering.dct import dct_exp
import cv2
import utils


def srad_test(image, steps, step_size, overlay=None):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_srad

    image_n = image
    ci = np.zeros_like(image)
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_srad(array=image_n.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)

    title = 'Final cy_SRAD ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    # fig = plt.figure(figsize=(6, 10))
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212, projection='3d')
    # utils.heightmap(ci, ax=ax2, azim=10, elev=40)
    # plt.show()
    utils.plot_image_g(image_n, title=title, overlay_img=overlay)
    return image_n


def csrad_test(image, steps, step_size, overlay=None):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad

    image_n = image
    ci = np.zeros_like(image)
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_csrad(array=image_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)

    title = 'Final cy_CSRAD ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    # fig = plt.figure(figsize=(6, 10))
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212, projection='3d')
    # plot_image_g(image_n, ax=ax1, title=title)
    # heightmap(ci, ax=ax2, azim=10, elev=40)
    # plt.show()
    utils.plot_image_g(image_n, title=title, overlay_img=overlay)
    return image_n


def hmf_test(image, overlay=None):
    from noise_filtering.HMF import hybrid_median_filtering
    import pywt

    coeffs = pywt.dwt2(image, 'haar')
    LL, wavelet_coeffs = coeffs

    res = []

    for coeff in wavelet_coeffs:
        res.append(hybrid_median_filtering(coeff).base)

    recopn_img = pywt.idwt2((LL, (res[0], res[1], res[2])), 'haar')
    utils.plot_image_g(recopn_img, overlay_img=overlay)


def combined_test(image, steps, step_size, weight, overlay=None):
    from noise_filtering.combined_method import combined_method
    image = combined_method(image, steps, step_size, weight)
    title = 'Combined method ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    utils.plot_image_g(image, title=title, overlay_img=overlay)
    return image


def total_variation_test(image, weight, overlay=None):
    from skimage.restoration import denoise_tv_bregman
    image = denoise_tv_bregman(image, weight)
    title = f"Total variation {weight} weight"
    utils.plot_image_g(image, title=title, overlay_img=overlay)
    return image


def tv_csrad(image, steps, step_size, weight, overlay=None):
    from skimage.restoration import denoise_tv_bregman
    from noise_filtering.SRAD.PyRAD_SRAD import cy_csrad
    image = denoise_tv_bregman(image, weight)
    image_n = image
    ci = np.zeros_like(image)
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_csrad(array=image_n.clip(1e-8), ci_1=ci.clip(1e-8), iter=n, step=step_size)
        ci = np.asarray(ci.base)
    title = f"TV + CSRAD, {steps} steps, {step_size} step size, {weight} weight"
    utils.plot_image_g(image_n, title=title, overlay_img=overlay)
    return image_n


def noise_test(img):
    #                                         h    w
    sampling_settigns = {'sample_dimension': (100, 40),
                         'angle': np.radians(60),
                         'd_min': 1,
                         'd_max': 548,
                         'b': 10,
                         'sigma': 0.7
                         }

    from noise_filtering.speckle_simulation import simulate_noise

    noise = simulate_noise(image=img, **sampling_settigns)
    utils.plot_image_g(noise * img + denoised2)


def timetest():
    import timeit

    import_module = """import random
    import numpy as np"""
    testcode = '''randarr = np.random.randint(0, 25, 25)
    diag = np.median(np.take(randarr, [0, 4, 6, 8, 12, 16, 18, 20, 24]))
    cross = np.median(np.take(randarr, [2, 7, 10, 11, 12, 13, 14, 17, 22]))
    center = randarr[12]
    np.median(np.array([diag, cross, center]))
        '''

    testcode2 = '''randarr = np.random.randint(0, 25, 25)
    diag = np.take(randarr, [0, 4, 6, 8, 12, 16, 18, 20, 24])
    cross = np.take(randarr, [2, 7, 10, 11, 12, 13, 14, 17, 22])
    center = randarr[12]
    np.median(np.array([diag[0], cross[0], center]))
            '''

    testcode3 = '''randarr = np.random.randint(0, 25, 25)
    diag = np.median(randarr)
    cross = np.median(randarr)
    center = randarr[12]
    np.median(np.array([diag, cross, center]))
                '''

    testcode4 = '''randarr = np.random.randint(0, 25, 25)
    diag = np.sort(np.take(randarr, [0, 4, 6, 8, 12, 16, 18, 20, 24]))[4]
    cross = np.sort(np.take(randarr, [2, 7, 10, 11, 12, 13, 14, 17, 22]))[4]
    center = randarr[12]
    np.median(np.array([diag, cross, center]))
                '''

    print(timeit.timeit(stmt=testcode4, setup=import_module, number=500 * 500))


if __name__ == '__main__':
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

    segmentation = None
    ground_truth = utils.normalize_0_1(utils.load_images()[0])
    image = utils.load_images()[-1]

    gt_h, gt_w = ground_truth.shape
    img_h, img_w = image.shape

    img_h = int(img_h * (gt_w / img_w))

    image = cv2.resize(image, (gt_w, img_h))

    image = cv2.copyMakeBorder(
        image, ground_truth.shape[0] - image.shape[0], 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)

    utils.plot_image_g( # add gamma multiplicative noise
        image * np.repeat(np.repeat(np.random.gamma(100, scale=1, size=[128, 128]), 2, axis=0), 2, axis=1))

    # wavelet_plot(image)
    utils.plot_image_g(image, title='Noised', overlay_img=segmentation)
    # utils.plot_image_g(ground_truth, title='Original', overlay_img=segmentation)
    utils.plot_image_g(wavelet_denoise(image, sigma=0.08, mode='bayes'), title='Wavelet denoise bayes')
    # utils.plot_image_g(wavelet_denoise(image, sigma=0.015, mode='visu'), title='Wavelet denoise visu')
    # utils.plot_image_g(skimage_wavelet_denoise(image))
    #
    # step_list = [150]
    # step_size_list = [0.05]
    # denoised_list = []
    # for steps in step_list:
    #     for step_size in step_size_list:
    #         denoised_list.append(utils.normalize_0_1(srad_test(image, steps=steps, step_size=step_size, overlay=None)))
    #         denoised_list.append(utils.normalize_0_1(csrad_test(image, steps=steps, step_size=step_size, overlay=None)))
    #         denoised_list.append(cv2.copyMakeBorder(utils.normalize_0_1(
    #             combined_test(image, steps=steps // 2, step_size=step_size, overlay=None, weight=0.3)), 0, 0, 0, 1,
    #             borderType=cv2.BORDER_CONSTANT, value=0)
    #         )
    #         denoised_list.append(total_variation_test(image, weight=0.3, overlay=None))
    #         denoised_list.append(tv_csrad(image, steps=steps, step_size=step_size, weight=0.3, overlay=None))
    #
    #         print(f"Steps: {steps}, Step size: {step_size}")
    #         for denoised in denoised_list:
    #             ssim = structural_similarity(ground_truth, denoised)
    #             psnr = peak_signal_noise_ratio(ground_truth, denoised)
    #             print("SSIM: {:.3f}".format(ssim))
    #             print("PSNR: {:.3f}".format(psnr))
    #             print()
