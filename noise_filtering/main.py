import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color, img_as_float32
from noise_filtering.denoise import sitk_noisefilter
from noise_filtering.wavelet_denoise import wavelet_exp
from noise_filtering.dct import dct_exp
import utils


# TODO: total variation optimization problem denoise

def srad_test(image, steps, step_size):
    from noise_filtering.SRAD.PyRAD_SRAD import cy_srad

    image_n = image
    ci = []
    for n in range(steps):
        image_n = np.abs(image_n)
        image_n, ci, di = cy_srad(array=image_n.clip(1e-10), iter=n, step=step_size)

    ci = np.asarray(ci.base)
    title = 'Final cy_SRAD ' + str(steps) + ' steps, ' + str(step_size) + ' step size'
    fig = plt.figure(figsize=(6, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, projection='3d')
    utils.plot_image_g(image_n, ax=ax1, title=title)
    utils.heightmap(ci, ax=ax2, azim=10, elev=40)
    plt.show()


def csrad_test(image, steps, step_size):
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
    utils.plot_image_g(image_n, title=title)
    return image_n


def hmf_test(image):
    from noise_filtering.HMF import hybrid_median_filtering
    import pywt

    coeffs = pywt.dwt2(image, 'haar')
    LL, wavelet_coeffs = coeffs

    res = []

    for coeff in wavelet_coeffs:
        res.append(hybrid_median_filtering(coeff).base)

    recopn_img = pywt.idwt2((LL, (res[0], res[1], res[2])), 'haar')
    utils.plot_image_g(recopn_img)


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
    images = utils.load_images()
    image = images[0]

    # image = utils.normalize_0_1(np.squeeze(load_test_img()).astype(dtype='float32'))

    # image = image[130:300, 200:450]

    # epsilon = 1e-9
    # image = image.clip(epsilon)
    # hmf_test(image)
    utils.plot_image_g(image, title='Original')

    steps = 100
    step_size = 0.1
    denoised = csrad_test(image, steps=steps, step_size=step_size)

    #                                         h    w
    sampling_settigns = {'sample_dimension': (100, 40),
                         'angle': np.radians(90),
                         'd_min': 1,
                         'd_max': 450,
                         'b': 10,
                         'sigma': 0.7
                         }

    from noise_filtering.speckle_simulation import simulate_noise
    noise = simulate_noise(image=denoised, **sampling_settigns)
    utils.plot_image_g(noise*denoised+denoised)

    # srad_test(image, steps=steps, step_size=step_size)

    # from skimage.segmentation import chan_vese
    #
    # # segmented = chan_vese(image, mu=2, lambda1=0.8, lambda2=0.8, tol=1e-3,
    # #                       max_iter=100, dt=0.5, init_level_set="checkerboard",
    # #                       extended_output=False)
    # # plot_image_g(segmented, title='Segmented')
    #
    # segmented2 = chan_vese(denoised, mu=0.5, lambda1=0.8, lambda2=0.8, tol=1e-3,
    #                        max_iter=100, dt=0.8, init_level_set="checkerboard",
    #                        extended_output=False)
    # utils.plot_image_g(segmented2, title='Segmented CSRAD')
