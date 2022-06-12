# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:35:23 2021

@author: yasar
"""
import numpy as np
global AVOID_DIV_BY_ZERO
AVOID_DIV_BY_ZERO = 1e-15


# Functions for WMMSE algorithm
def WMMSE_algorithm_single(N, 
                           H, 
                           Pmax, 
                           var_noise, 
                           priorities,
                           cell_mapping = None):
    # start_WMMSE_time = time.time()
    vnew = 0
    # random initialization gives much lower performance.
    b = np.sqrt(Pmax) *np.ones(N) # np.random.rand(N) # #
    if cell_mapping is not None:
        for k in range(max(cell_mapping)):
            idxs = np.where(cell_mapping == k)
            I_k = idxs[-1].shape[-1]
            b[idxs] = b[idxs] / I_k
            
    f = np.zeros(N)
    w = np.zeros(N)
    for i in range(N):
        f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]), np.square(b)) + var_noise)
        w[i] = 1.0 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log2(w[i])

#    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(N):
            btmp = priorities[i]* w[i] * f[i] * H[i, i] / (AVOID_DIV_BY_ZERO + sum(priorities * w * np.square(f) * np.square(H[:, i])))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(N):
            f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]) ,np.square(b)) + var_noise)
            w[i] = 1.0 / (AVOID_DIV_BY_ZERO + 1 - f[i] * b[i] * H[i, i])
            vnew = vnew + np.log2(w[i])

        if vnew - vold <= 0.01:
            break
        
    p_opt = np.square(b)
    # Make sure that per BS constraint is satisfied if cell_mapping is specified.
    if cell_mapping is not None:
        norm_vect = np.ones(N)
        for k in range(max(cell_mapping)):
            idxs = np.where(cell_mapping == k)
            if sum(p_opt[idxs]) > Pmax:
                norm_vect[idxs] = sum(p_opt[idxs]) / Pmax
        p_opt = p_opt / norm_vect
    # end_time = time.time() - start_WMMSE_time
    # end_statistics = [end_time, iter]
    return p_opt, iter#, end_statistics

def WMMSE_algorithm(H, 
                    Pmax, 
                    noise_var,
                    priorities = None,
                    cell_mapping = None):
    N = H.shape[0]
    M = H.shape[-1]
    if priorities is None: priorities = np.ones(N)

    p = np.zeros((N, M))
    iters = 0
    for m in range(M):
        p[:,m], iter = WMMSE_algorithm_single(N, 
                                              H[:,:,m], 
                                              Pmax, 
                                              noise_var,
                                              priorities, 
                                              cell_mapping)
        iters += iter

    # Return optimum result after convergence
    return p, iters#, end_statistics