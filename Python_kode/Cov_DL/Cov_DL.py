# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:27:46 2019

@author: trine
"""
#import os
#data_path = os.getcwd()[:-6]
#import sys
#sys.path.append(data_path)  # make the right path

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dictionary_learning import K_SVD
from dictionary_learning import DL
from sklearn.datasets import make_sparse_coded_signal
import data_generation


np.random.seed(1)

""" INITIALISING PARAMETERS """

m = 8               # number of sensors
n = 32            # number of sources
non_zero = 20       # max number of non-zero coef. in rows of X
n_samples = 100      # number of sampels


""" DATA GENERATION """

#Y, A_real, X_real = data_generation.mix_signals_det(n_samples, 8, non_zero)
Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, m, n, non_zero)
A_real = A_real/np.linalg.norm(A_real, ord=2, axis=0, keepdims=True)
#Y, A_real, X_real = data_generation.random_sparse_data(m, n, non_zero, n_samples)

""" SEGMENTATION """
L = 5               # number of sampels in one segment 
Ys, Xs, n_seg = data_generation.segmentation_split(Y, X_real, L, n_samples)


""" Cov - DL """
# jeg har fundet i teksten at vi laver dictionary over alle segmenterne 
# som hver har en vector her, giver det mening ift stationaitet??  
  
Y_big = np.zeros([int(m*(m+1)/2.),n_seg])

for i in range(n_seg):              # loop over all segments
    # Transformation to covariance domain and vectorization
    Y_cov = np.cov(Ys[i])                      # covariance 
    X_cov = np.cov(Xs[i])                      # NOT diagonl ??
    
    # Vectorization of lower tri, row wise  
    vec_Y = np.array(list(Y_cov[np.tril_indices(m)]))
    vec_X = np.array(list(X_cov[np.tril_indices(n)]))
    sigma = np.diagonal(X_cov)
    
    Y_big.T[i] = vec_Y


# Y_big is now considered as a "new" observation vector, 
# and dictionary learning is performed      

n_samples = n_seg 

# Dictionary learning
#D, sigma, iter_ = K_SVD(Y_big, n=len(sigma), m=len(Y_big),
#                                 non_zero=non_zero, n_samples=n_samples,
#                                 max_iter=1000)


Y_big_rec, D, sigma = DL(Y_big.T, n=len(sigma), k=non_zero)

Y_err = np.linalg.norm(Y_big - Y_big_rec.T)

print('cov domain: reconstruction error %f'%(Y_err))


# Find A approximative
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

 
#A_rec = A_rec/np.linalg.norm(A_rec, ord=2, axis=0, keepdims=True)
A_err = np.linalg.norm(A_real-A_rec)
    
print('Dictionary error %f'%(A_err))

    
""" prediction of X """
#
#