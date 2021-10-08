import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color

def binarize(img):
    img[img > 0] = 1
    return img
