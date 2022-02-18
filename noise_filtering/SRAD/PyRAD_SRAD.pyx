import cython
import numpy as np
from libc.math cimport sqrt, exp, fabs


"""
Modified from:
https://github.com/birgander2/PyRAT
"""
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_srad(float [:, :] array,float step=0.05, int iter=0):
    cdef float p1, p2, p3, p4, sip, sim, sjp, sjm
    cdef float aim, aip, ajm, ajp
    cdef int i, j
    cdef float d2i, qp1, qp2, q2
    cdef float dn

    cdef int ny = array.shape[0]
    cdef int nx = array.shape[1]

    cdef q0 = exp(-step*iter/6.0)
    cdef q02 = q0 * q0

    cdef float [:, :] ci = np.zeros_like(array, dtype='f4')
    cdef float [:, :] di = np.zeros_like(array, dtype='f4')
    cdef float [:, :] out = np.zeros_like(array)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # 4 neighbourhood pixels
            p1 = array[i+1, j] - array[i, j] # d_N
            p2 = array[i, j+1] - array[i, j] # d_E
            p3 = array[i, j] - array[i-1, j] # d_S
            p4 = array[i, j] - array[i, j-1] # d_W
            d2I = (array[i+1, j] + array[i-1, j] + array[i, j+1] + array[i, j-1] - 4.0*array[i, j]) # ∇^2
            qp1 = sqrt(p1**2 + p2**2 + p3**2 + p4**2) / array[i, j]
            qp2 = d2I / array[i, j]
            q = ((qp1*qp1/2.0 - qp2*qp2/16.0) / (1.0 + qp2/4.0)**2)
            ci[i, j] = 1/(1+(q - q02)/(q02*(1.0+q02)))

    # boundary conditions 1

    for i in [0, ny-1]:
        for j in [0, nx-1]:
            sip = array[i+1, j] if i != ny-1 else array[i, j]
            sim = array[i-1, j] if i != 0 else array[i, j]
            sjp = array[i, j+1] if j != nx-1 else array[i, j]
            sjm = array[i, j-1] if j != 0 else array[i, j]
            p1 = sip - array[i, j]
            p2 = sjp - array[i, j]
            p3 = array[i, j] - sim
            p4 = array[i, j] - sjm
            d2I = (sip + sim + sjp + sjm - 4.0*array[i, j])
            qp1 = sqrt(p1**2 + p2**2 + p3**2 + p4**2) / array[i, j]
            qp2 = d2I / array[i, j]
            q = ((qp1*qp1/2.0 - qp2*qp2/16.0) / (1.0 + qp2/4.0)**2)
            ci[i, j] = 1/(1+(q - q02)/(q02*(1.0+q02)))

    # update

    for i in range(1, ny-1):
        for j in range(1, nx-1):
                    d = ( ci[i+1, j]*(array[i+1, j]-array[i, j]) # d_S*∇S_U
                        + ci[i, j]  *(array[i-1, j]-array[i, j]) # d_c*∇N_U
                        + ci[i, j+1]*(array[i, j+1]-array[i, j]) # d_E*∇E_U
                        + ci[i, j]  *(array[i, j-1]-array[i, j]) # d_c*∇W_U
                        )
                    out[i, j] = array[i, j] + step/4.0*d
                    di[i,j] = d

    # boundary conditions 2

    for i in [0, ny-1]:
        for j in [0, nx-1]:
            cip = ci[i+1, j] if i != ny-1 else ci[i, j]
            cjp = ci[i, j+1] if j != nx-1 else ci[i, j]

            aip = array[i+1, j] if i != ny-1 else array[i, j]
            aim = array[i-1, j] if i != 0 else array[i, j]
            ajp = array[i, j+1] if j != nx-1 else array[i, j]
            ajm = array[i, j-1] if j != 0 else array[i, j]
            out[i, j] = array[ i, j] + step/4.0*(
                      cip*(aip-array[i, j])
                    + ci[i, j]*(aim-array[ i, j])
                    + cjp*(ajp-array[ i, j])
                    + ci[i, j]*(ajm-array[ i, j]))


    return np.asarray(out), ci, di

'''
https://www.researchgate.net/publication/221472052_Coefficient-Tracking_Speckle_Reducing_Anisotropic_Diffusion
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_csrad(float [:, :] array,float [:, :] ci_1,float step=0.05, int iter=0,float alpha_c = 1.0, float alpha_t = 0.8, float alpha_t_1 = 0.2):
    cdef float p1, p2, p3, p4, sip, sim, sjp, sjm
    cdef float aim, aip, ajm, ajp
    cdef int i, j
    cdef float d2i, qp1, qp2, q2
    cdef float dn
    # cdef float alpha_t, alpha_t_1, alpha_c

    cdef int ny = array.shape[0]
    cdef int nx = array.shape[1]

    cdef q0 = exp(-step*iter/6.0)
    cdef q02 = q0 * q0

    cdef float [:, :] ci = np.zeros_like(array, dtype='f4')
    cdef float [:, :] di = np.zeros_like(array, dtype='f4')
    cdef float [:, :] out = np.zeros_like(array)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            p1 = array[i+1, j] - array[i, j]
            p2 = array[i, j+1] - array[i, j]
            p3 = array[i, j] - array[i-1, j]
            p4 = array[i, j] - array[i, j-1]
            d2I = (array[i+1, j] + array[i-1, j] + array[i, j+1] + array[i, j-1] - 4.0*array[i, j])
            qp1 = sqrt(p1**2 + p2**2 + p3**2 + p4**2) / array[i, j]
            qp2 = d2I / array[i, j]
            q = ((qp1*qp1/2.0 - qp2*qp2/16.0) / (1.0 + qp2/4.0)**2)
            ci[i, j] = 1/(1+(q - q02)/(q02*(1.0+q02))) # diffusion coeff

    # boundary conditions 1

    for i in [0, ny-1]:
        for j in [0, nx-1]:
            sip = array[i+1, j] if i != ny-1 else array[i, j]
            sim = array[i-1, j] if i != 0 else array[i, j]
            sjp = array[i, j+1] if j != nx-1 else array[i, j]
            sjm = array[i, j-1] if j != 0 else array[i, j]
            p1 = sip - array[i, j]
            p2 = sjp - array[i, j]
            p3 = array[i, j] - sim
            p4 = array[i, j] - sjm
            d2I = (sip + sim + sjp + sjm - 4.0*array[i, j])
            qp1 = sqrt(p1**2 + p2**2 + p3**2 + p4**2) / array[i, j]
            qp2 = d2I / array[i, j]
            q = ((qp1*qp1/2.0 - qp2*qp2/16.0) / (1.0 + qp2/4.0)**2)
            ci[i, j] = 1/(1+(q - q02)/(q02*(1.0+q02)))

    # update

    for i in range(1, ny-1):
        for j in range(1, nx-1):
                    d = ( sqrt(alpha_c*ci[i, j]*fabs(alpha_t_1*ci_1[i+1, j]-alpha_t*ci[i+1, j]))*(array[i+1, j]-array[i, j])  # d_S*∇S_U
                        + sqrt(alpha_c*ci[i, j]*fabs(alpha_t_1*ci_1[i, j]-alpha_t*ci[i, j] ))*(array[i-1, j]-array[i, j])     # d_c*∇N_U
                        + sqrt(alpha_c*ci[i, j]*fabs(alpha_t_1*ci_1[i, j+1]-alpha_t*ci[i, j+1] ))*(array[i, j+1]-array[i, j]) # d_E*∇E_U
                        + sqrt(alpha_c*ci[i, j]*fabs(alpha_t_1*ci_1[i, j-1]-alpha_t*ci[i, j-1] ))*(array[i, j-1]-array[i, j]) # d_c*∇W_U
                        )
                    out[i, j] = array[i, j] + step/4.0*d
                    di[i,j] = d

    # boundary conditions 2

    for i in [0, ny-1]:
        for j in [0, nx-1]:
            cip = ci[i+1, j] if i != ny-1 else ci[i, j]
            cjp = ci[i, j+1] if j != nx-1 else ci[i, j]

            aip = array[i+1, j] if i != ny-1 else array[i, j]
            aim = array[i-1, j] if i != 0 else array[i, j]
            ajp = array[i, j+1] if j != nx-1 else array[i, j]
            ajm = array[i, j-1] if j != 0 else array[i, j]
            out[i, j] = array[ i, j] + step/4.0*(
                    cip*(aip-array[i, j])
                  + ci[i, j]*(aim-array[ i, j])
                  + cjp*(ajp-array[ i, j])
                  + ci[i, j]*(ajm-array[ i, j]))


    return np.asarray(out), ci, di
