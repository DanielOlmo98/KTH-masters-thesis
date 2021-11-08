import numpy as np
import cython
from utils import plot_image_g
"""
Wavelet DecompositionYBased Speckle
Reduction Method for Ultrasound Images by
Using Speckle-Reducing Anisotropic Diffusion
and Hybrid Median
http://dx.doi.org/10.1097/JCE.0000000000000300
"""


# Modified from:
# https://www.researchgate.net/publication/332574579_Image_Processing_Course_Project_Image_Filtering_with_Wiener_Filter_and_Median_Filter
@cython.boundscheck(False)
@cython.wraparound(False)
def hybrid_median_filtering(float [:, :] arr, int kernel_size = 5):
    # cdef float [:, :] data_final = np.zeros((y_size,x_size), dtype='f')
    cdef float [:, :] data_final = np.zeros_like(arr)
    cdef int i, j, k, z, c, indexer, temp_index, y_size ,x_shape
    cdef float diag, cross, center
    cdef float [:] temp = np.zeros((kernel_size*kernel_size,), dtype='f')


    x_size, y_size = np.shape(arr)
    temp_index = 0
    indexer = kernel_size // 2
    for i in range(x_size):
        for j in range(y_size):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > x_size - 1:
                    for c in range(kernel_size):
                        temp_index += 1

                else:
                    if j + z - indexer < 0 or j + indexer > y_size - 1:
                        temp_index += 1

                    else:
                        for k in range(kernel_size):
                            temp[temp_index] = arr[i + z - indexer][j + k - indexer]
                            temp_index += 1

            diag = np.sort(np.take(temp, [0, 4, 6, 8, 12, 16, 18, 20, 24]))[4]
            cross = np.sort(np.take(temp, [2, 7, 10, 11, 12, 13, 14, 17, 22]))[4]
            center = temp[12]

            data_final[i,j] = np.sort(np.array([diag, cross, center]))[1]
            temp_index = 0
            temp = np.zeros_like(temp)

    return data_final
