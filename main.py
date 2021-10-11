import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
from denoise import sitk_noisefilter
from wavelet_denoise import wavelet_exp

file_list = os.listdir('image/')

images = []
for file in file_list:
    common_file_path = 'image/' + file
    if not os.path.isfile(common_file_path):
        continue

    image = imread(common_file_path)
    image = image[80:550, 80:720]
    images.append(color.rgb2gray(image))

image = images[0]
# for image in images:
plt.imshow(image, cmap='gray')
plt.title("Before")
plt.show()

image = wavelet_exp(image, plot=True)
image = sitk_noisefilter(image)

plt.imshow(image, cmap='gray')
plt.title("After")
plt.show()

noise = images[0] - image
plt.imshow(noise, cmap='gray')
plt.title("Noise")
plt.show()
