from dl.dataloader import dataset_convert
from noise_filtering.wavelet_denoise import wavelet_denoise


def wavelet_convert():
    sigma = 0.015
    mode = 'visu'

    # sigma = 0.08
    # mode = 'bayes'

    def denoise_wavelet_func(image, mask):
        image = wavelet_denoise(image, sigma, mode)
        return image, mask

    dataset_convert(f'camus_wavelet_sigma{sigma}_{mode}', denoise_wavelet_func)


if __name__ == '__main__':
    wavelet_convert()
