import utils
from noise_filtering import denoise
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt


def eval_denoise(img, settings_dict):
    denoised_images = {
        'wavelet_visu': denoise.wavelet_denoise(img, **settings_dict['wavelet_visu']),
        'wavelet_bayes': denoise.wavelet_denoise(img, **settings_dict['wavelet_bayes']),
        'tv': denoise.tv_denoise(img, **settings_dict['tv']),
        'combine': denoise.combined_method_denoise(img, **settings_dict['combine']),
        'tv_csrad': denoise.tv_csrad_denoise(img, **settings_dict['tv_csrad']),
        'csrad': denoise.csrad_denoise(img, **settings_dict['csrad']),
        'srad': denoise.srad_denoise(img, **settings_dict['srad']),
        'hmf': denoise.hmf_denoise(img)}

    df_list = []
    for method, dn_img in denoised_images.items():
        psnr = peak_signal_noise_ratio(image_true=img, image_test=dn_img)
        mse = mean_squared_error(img, dn_img)
        df_list.append(pd.DataFrame(
            data={'PSNR': psnr, 'MSE': mse, 'Params': settings_dict[method]},
            index=[method]
        ))

    plot_denoise(img, denoised_images)

    return pd.concat(df_list)


def plot_denoise(img, denoised_images):
    fig, axs = plt.subplots(4, 2)
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original')
    for i, (method_name, dn_img) in enumerate(denoised_images.items()):
        axs.flat[i + 1].imshow(dn_img)
        axs.flat[i + 1].set_title(method_name)
    plt.show()


if __name__ == '__main__':
    img = utils.get_example_img()
    eval_frame = eval_denoise(img, denoise.get_settings_dict())

    print()
