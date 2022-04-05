import utils
from noise_filtering import denoise
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.misc import face


def eval_denoise(img, gt, dn_strength='strong'):
    dn_lambda_dict = denoise.get_denoise_lambda_dict(dn_strength)
    dn_settings_dict = denoise.get_settings_dict(dn_strength)
    denoised_images = {}
    df_list = []

    psnr = peak_signal_noise_ratio(image_true=gt, image_test=img)
    mse = mean_squared_error(gt, img)
    df_list.append(pd.DataFrame(
        data={'PSNR': psnr, 'MSE': mse, 'Params': 'None'},
        index=['Noisy image']
    ))
    for method, dn_lambda in dn_lambda_dict.items():
        dn_img = dn_lambda(img)[0:img.shape[0], 0:img.shape[1]]
        dn_img = utils.normalize_0_1(dn_img)
        denoised_images[method] = dn_img
        psnr = peak_signal_noise_ratio(image_true=gt, image_test=dn_img)
        mse = mean_squared_error(gt, dn_img)
        df_list.append(pd.DataFrame(
            data={'PSNR': psnr, 'MSE': mse, 'Params': str(dn_settings_dict[method])},
            index=[method]
        ))

    plot_denoise(utils.normalize_0_1(img), gt, denoised_images)
    # plot_denoise(img, gt,
    #              {method: np.abs(gt - dn_image) for method, dn_image in denoised_images.items()},
    #              cmap='hot')
    eval_frame = pd.concat(df_list).rename_axis('Method')
    return eval_frame.sort_values(by=['MSE'])


def plot_denoise(img, gt, denoised_images, cmap='gray'):
    fig, axs = plt.subplots(5, 2, figsize=(4, 10))
    axs.flat[0].imshow(img, cmap=cmap)
    axs.flat[0].set_title('Noisy')
    axs.flat[0].axis('off')
    axs.flat[-1].imshow(gt, cmap=cmap)
    axs.flat[-1].set_title('Ground truth')
    axs.flat[-1].axis('off')
    for i, (method_name, dn_img) in enumerate(denoised_images.items()):
        axs.flat[i + 1].imshow(dn_img, cmap=cmap)
        axs.flat[i + 1].set_title(method_name)
        axs.flat[i + 1].axis('off')

    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
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


def denoise_eval_gauss_log(img, noise_map_size=(256, 256), shape=1, scale=0.08):
    img_noised = denoise.log_tranform_gauss_noise(img, noise_map_size, shape, scale).astype(np.float32)
    eval_frame = eval_denoise(img_noised, img, dn_strength='weak')
    print(eval_frame.loc[:, 'PSNR':'MSE'])
    return eval_frame


if __name__ == '__main__':
    images = utils.load_images()
    img_size = (256, 256)
    img = utils.normalize_0_1(images[8]).astype(np.float32)
    img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
    shape, scale = 1, 0.08
    eval_frame = denoise_eval_gauss_log(img, noise_map_size=img_size, shape=shape, scale=scale)
    eval_frame.to_csv(f'denoise_eval_low_noise.csv')
