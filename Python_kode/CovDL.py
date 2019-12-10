# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:34:43 2019

@author: trine
"""
import numpy as np
import data_generation
import dictionary_learning


def reverse_vec(x):
    m = int(np.sqrt(x.size*2))
    if (m*(m+1))//2 != x.size:
        print("reverse_vec fail")
        return None
    else:
        R, C = np.tril_indices(m)
        out = np.zeros((m, m), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out


def Cov_DL1(Y, A, X, m, n, cov_seg, L, k, ):
    """ 
    """
    ## internal segmetation for training examples
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, cov_seg, L)
    
    ## Covariance Domain Tranformation
    
    Y_big = np.zeros([int(m*(m+1)/2.),n_seg])
    for i in range(n_seg):              # loop over all segments
        
        # Transformation to covariance domain and vectorization
        Y_cov = np.cov(Ys[i])                      # covariance 
        X_cov = np.cov(Xs[i])                      # NOT diagonl ??
        print(X_cov)
        # Vectorization of lower tri, row wise  
        vec_Y = Y_cov[np.tril_indices(m)]
        
        sigma = np.diagonal(X_cov)
    
        Y_big.T[i] = vec_Y
    
    ## Dictionary Learning on Transformed System
    L = n_seg 
    
    Y_big_rec, D, sigma = dictionary_learning.DL(Y_big.T, n=len(sigma), k=k)

    Y_err = np.linalg.norm(Y_big - Y_big_rec.T)

    print('cov domain: reconstruction error %f'%(Y_err))
    
    ## Find Mixing Matrix From D 
   
    A_rec = np.zeros(([m, n]))
    for j in range(n):
        d = D.T[j]
        matrix_d = reverse_vec(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(np.abs(E))     # should it be abselut here?
        index = np.where(np.abs(E) == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_rec.T[j] = temp.T

    A_err = np.linalg.norm(A-A_rec)
        
    print('Dictionary error %f'%(A_err))
    
    return A_rec, A_err