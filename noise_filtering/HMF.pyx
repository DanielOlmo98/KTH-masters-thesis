import numpy as np
import cython
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
    cdef float [:, :] result = np.zeros_like(arr)
    cdef float [:, :] data_final = np.zeros_like(arr)
    cdef int i, j, k, z, c, x_size, y_size, indexer, diag, cross, center
    cdef float [:] temp = np.zeros((kernel_size*kernel_size,), dtype='f')


    x_size, y_size = np.shape(arr)
    indexer = kernel_size // 2
    for i in range(x_size):
        for j in range(y_size):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(arr) - 1:
                    for c in range(kernel_size):
                        np.append(temp, 0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(arr[0]) - 1:
                                                np.append(temp, 0)

                    else:
                        for k in range(kernel_size):
                            np.append(temp, arr[i + z - indexer][j + k - indexer])
            while len(temp) < kernel_size*kernel_size:
                np.append(temp, 0)

            diag = np.median(np.take(temp, [0, 4, 6, 8, 16, 18, 20, 24, 12]))
            cross = np.median(np.take(temp, [2, 7, 17, 22, 10, 11, 13, 14, 12]))
            center = np.take(temp, 12)
            data_final[i,j] = np.median(np.array([diag, cross, center]))

    return data_final
