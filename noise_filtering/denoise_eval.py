import utils
from noise_filtering import denoise
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def eval_denoise(img, gt):
    dn_lambda_dict = denoise.get_denoise_lambda_dict()
    dn_settings_dict = denoise.get_settings_dict()
    denoised_images = {}
    df_list = []
    for method, dn_lambda in dn_lambda_dict.items():
        dn_img = dn_lambda(img)[0:img.shape[0], 0:img.shape[1]]
        denoised_images[method] = dn_img
        psnr = peak_signal_noise_ratio(image_true=gt, image_test=dn_img)
        mse = mean_squared_error(gt, dn_img)
        df_list.append(pd.DataFrame(
            data={'PSNR': psnr, 'MSE': mse, 'Params': str(dn_settings_dict[method])},
            index=[method]
        ))

    plot_denoise(img, gt, denoised_images)
    plot_denoise(img, gt,
                 {method: np.abs(gt - dn_image) for method, dn_image in denoised_images.items()},
                 cmap='hot')
    eval_frame = pd.concat(df_list).rename_axis('Method')
    return eval_frame.sort_values(by=['MSE'])


def plot_denoise(img, gt, denoised_images, cmap='gray'):
    fig, axs = plt.subplots(5, 2, figsize=(4, 10))
    axs.flat[0].imshow(img, cmap=cmap)
    axs.flat[0].set_title('Original')
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


if __name__ == '__main__':
    # img = utils.get_example_img(png=False).squeeze()
    images = utils.load_images()
    gt = images[0]
    gt_w, gt_h = gt.shape
    high_noise = images[1][0:gt_w, 0:gt_h]
    low_noise = images[2][0:gt_w, 0:gt_h]
    lower_noise = images[3][0:gt_w, 0:gt_h]
    eval_frame = eval_denoise(low_noise, gt)
    eval_frame.to_csv(f'denoise_eval_low_noise.csv')
    print(eval_frame)
