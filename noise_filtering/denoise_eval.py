import random

import skimage.util

import utils
from noise_filtering import denoise
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision
from scipy.misc import face
import albumentations as A
import albumentations.pytorch


def center_crop(img, border_size):
    return img[border_size:-border_size, border_size:-border_size]


def eval_denoise(img, gt, dn_strength='strong'):
    dn_lambda_dict = denoise.get_denoise_lambda_dict(dn_strength)
    dn_settings_dict = denoise.get_settings_dict(dn_strength)
    denoised_images = {}
    df_list = []

    border_crop_size = 16
    gt = center_crop(gt, border_crop_size)

    noised_img_croped = center_crop(img, border_crop_size)
    psnr = peak_signal_noise_ratio(image_true=gt, image_test=noised_img_croped)
    mse = mean_squared_error(gt, noised_img_croped)
    ssim = structural_similarity(gt, noised_img_croped)
    df_list.append(pd.DataFrame(
        data={'PSNR': psnr, 'MSE': mse, 'SSIM': ssim, 'Params': 'None'},
        index=['Noisy image']
    ))
    for method, dn_lambda in dn_lambda_dict.items():
        # print(method)
        dn_img = dn_lambda(img)[0:img.shape[0], 0:img.shape[1]]
        dn_img = center_crop(dn_img, border_crop_size)

        # dn_img = utils.normalize_0_1(dn_img)
        denoised_images[method] = dn_img
        dn_img = (dn_img - np.min(dn_img)) / (np.max(dn_img) - np.min(dn_img)).clip(1e-8)

        psnr = peak_signal_noise_ratio(image_true=gt, image_test=dn_img)
        mse = mean_squared_error(gt, dn_img)
        ssim = structural_similarity(gt, dn_img)
        df_list.append(pd.DataFrame(
            data={'PSNR': psnr, 'MSE': mse, 'SSIM': ssim, 'Params': str(dn_settings_dict[method])},
            index=[method]
        ))

    plot_denoise(img, gt, denoised_images, show_hist=False)
    # plot_denoise(img, gt,
    #              {method: np.abs(gt - dn_image) for method, dn_image in denoised_images.items()},
    #              cmap='hot')
    eval_frame = pd.concat(df_list).rename_axis('Method')
    return eval_frame
    # return eval_frame.sort_values(by=['SSIM'], ascending=False)


def plot_denoise(img, gt, denoised_images, cmap='gray', show_hist=False, bins=256):
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    axs.flat[0].imshow(img, cmap=cmap)
    axs.flat[0].set_title('Noisy')
    axs.flat[0].axis('off')
    # axs.flat[-1].imshow(gt, cmap=cmap)
    # axs.flat[-1].set_title('Ground truth')
    # axs.flat[-1].axis('off')

    if show_hist:
        hist_ax = fig.add_axes(axs.flat[0].get_position(), frameon=False)
        hist_ax.hist(img.ravel(), bins=bins, histtype='step', color='black')
        hist_ax.set_xlim(0, 1)
        hist_ax.axis('off')
        hist_ax = fig.add_axes(axs.flat[-1].get_position(), frameon=False)
        hist_ax.hist(img.ravel(), bins=bins, histtype='step', color='black')
        hist_ax.set_xlim(0, 1)
        hist_ax.axis('off')

    for i, (method_name, dn_img) in enumerate(denoised_images.items()):
        axs.flat[i + 1].imshow(dn_img, cmap=cmap)
        axs.flat[i + 1].set_title(method_name)
        axs.flat[i + 1].axis('off')
        if show_hist:
            hist_ax = fig.add_axes(axs.flat[i + 1].get_position(), frameon=False)
            hist_ax.hist(dn_img.ravel(), bins=bins, histtype='step', color='black')
            hist_ax.set_xlim(0, 1)
            hist_ax.axis('off')

    plt.show()


def denoise_eval_synthetic_img():  # img = utils.get_example_img(png=False).squeeze()
    images = utils.load_images()
    gt = images[0]
    # img_shape = gt.shape
    img_shape = (256, 256)
    gt = cv2.resize(gt, img_shape, cv2.INTER_LINEAR)
    high_noise = cv2.resize(images[1], img_shape, cv2.INTER_LINEAR)
    low_noise = cv2.resize(images[2], img_shape, cv2.INTER_LINEAR)
    lower_noise = cv2.resize(images[3], img_shape, cv2.INTER_LINEAR)

    eval_frame = eval_denoise(high_noise, gt)
    eval_frame.to_csv(f'denoise_eval_high_noise.csv')
    print(eval_frame.loc[:, 'PSNR':'MSE'])

    eval_frame = eval_denoise(low_noise, gt)
    eval_frame.to_csv(f'denoise_eval_low_noise.csv')
    print(eval_frame.loc[:, 'PSNR':'MSE'])

    eval_frame = eval_denoise(lower_noise, gt)
    eval_frame.to_csv(f'denoise_eval_lower_noise.csv')
    print(eval_frame.loc[:, 'PSNR':'MSE'])


def denoise_eval_gammanoise(img, noise_map_size=(256, 256)):
    img_noised_resize = denoise.gamma_noise_resize(img, noise_map_size).astype(np.float32)
    eval_frame = eval_denoise(img_noised_resize, img, dn_strength='weak')
    print(eval_frame.loc[:, 'PSNR':'MSE'])

    return eval_frame


def denoise_eval_gauss_log(img, noise_map_size=(256, 256), shape=1, var=0.08):
    img_noised = denoise.log_tranform_gauss_noise(img, noise_map_size, shape, var).astype(np.float32)
    eval_frame = eval_denoise(img_noised, img, dn_strength='weak')
    print(eval_frame.loc[:, ['PSNR', 'MSE', 'SSIM']])
    return eval_frame


def denoise_specklenoise(img, mean=0, var=0.06, dn_strength='weak'):
    img_noised = skimage.util.random_noise(img, mode='speckle', mean=mean, var=var).astype(np.float32)
    eval_frame = eval_denoise(img_noised, img, dn_strength=dn_strength)
    return eval_frame


def augment_noise_eval(img, var_lim=(10, 50)):
    transformer = [A.GaussNoise(p=1., var_limit=var_lim)]
    transformer = A.Compose(transformer)
    img_noised = transformer(image=img)['image']
    psnr = (peak_signal_noise_ratio(image_true=img, image_test=img_noised))
    # ssim = structural_similarity(img, img_noised)
    utils.plot_image_g(img)
    utils.plot_image_g(img_noised)

    return psnr


def evaluate_image(eval_func=denoise_specklenoise, img_index=8, noise_mean=0, noise_var=0.03):
    images = utils.load_images()
    img_size = (256, 256)
    gt = cv2.resize(images[img_index], img_size, cv2.INTER_LINEAR)
    img = cv2.resize(images[img_index], img_size, cv2.INTER_LINEAR)
    dn_strength = 'weak '

    frame_list = []
    for _ in range(10):
        eval_frame = eval_func(img, mean=noise_mean, var=noise_var, dn_strength=dn_strength)
        # with pd.option_context('precision', 4):
        #     print(eval_frame.loc[:, ['PSNR', 'MSE', 'SSIM']])
        frame_list.append(eval_frame)

    metrics = pd.concat(frame_list)
    with pd.option_context('precision', 4):
        print(metrics.mean(level=0))
    with pd.option_context('precision', 3):
        print(metrics.std(level=0))


if __name__ == '__main__':
    print()
    psnr_list = []
    ed_es = ['ES', 'ED']
    ch_2_4 = [2, 4]
    for _ in range(2):
        patient_number = random.randint(1, 405)

        img, _ = utils.load_patient(patient=f'{patient_number:04}',
                                    ED_or_ES=random.choice(ed_es),
                                    CH=random.choice(ch_2_4),
                                    dataset_path='camus_combined_50-0.1_w0.7_eps0.001',
                                    png=True)
        psnr_list.append(augment_noise_eval(img))

    print(f'PSNR average: {np.average(psnr_list)}')

    # for idx, row in metrics.iterrows():
    #     print(idx)
    #     print(row)
    #     # print(row.mean())
    #     # print(row.std())

    # eval_frame = eval_denoise(img, gt, dn_strength)
    # eval_frame.to_csv(f'denoise_eval_synthetic_lower_noise.csv')
