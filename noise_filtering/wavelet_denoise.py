import pywt
import numpy as np
import matplotlib.pyplot as plt
import utils

'''Copyright (C) 2011, the scikit-image team All rights reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the 
following conditions are met: 

Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
following disclaimer in the documentation and/or other materials provided with the distribution. Neither the name of 
skimage nor the names of its contributors may be used to endorse or promote products derived from this software 
without specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 


Adapted from:
https://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise_wavelet.html
'''


def wavelet_denoise(img, sigma=0.1, mode='visu'):
    if mode not in ['visu', 'bayes']:
        raise ValueError("Mode must be either 'visu' or 'bayes'")
    wavelet = 'db1'
    max_lvl = pywt.dwtn_max_level(img.shape, wavelet)
    coeffs = pywt.wavedec2(img, wavelet, level=max_lvl - 3)
    if mode == 'visu':
        threshold = sigma * np.sqrt(2 * np.log(img.size))
    for i, level in enumerate(coeffs[1:]):
        coeffs[i + 1] = []
        for coeff in level:
            if mode == 'bayes':
                var = sigma ** 2
                dvar = np.mean(coeff * coeff)
                threshold = var / np.sqrt(max(dvar - var, 1e-7))
            coeffs[i + 1].append(pywt.threshold(coeff, threshold, mode='soft'))

    img_dn = pywt.waverec2(coeffs, wavelet)
    return img_dn


def wavelet_plot(img):
    titles = ['LL', 'LH',
              'HL', 'HH']
    wavelet = 'db1'
    coeffs = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs
    img_t = pywt.idwt2(coeffs, wavelet)

    fig2 = plt.figure(figsize=(5, 5))
    for i, a in enumerate([img_t, LH, HL, HH]):
        ax = fig2.add_subplot(2, 2, i + 1)
        ax.imshow(a, interpolation="nearest", cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    image = utils.load_images()[-1]
    denoised = wavelet_denoise(image, sigma=0.05, mode='bayes')
    # utils.plot_image_g(image, title='Original')
    utils.plot_image_g(denoised, title='Denoised')
