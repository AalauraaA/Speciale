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
from sklearn.datasets import make_sparse_coded_signal
np.random.seed(0)

""" Random generated data """
# INITIALISING PARAMETERS
m = 6               # number of sensors
n = 8               # number of sources
non_zero = 6        # max number of non-zero coef. in rows of X
n_samples = 20      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n,
                                   n_features=m,
                                   n_nonzero_coefs=non_zero,
                                   random_state=0)


""" First segmentation of observations (easy way - split) """
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples

S = 5                                       # Samples pr segment
n_seg = int(n_samples/S)                    # Number of segments

Ys = np.split(Y, n_seg, axis=1)            # Matrixs with segments in axis=0
Xs = np.split(X_real, n_seg, axis=1)


""" Cov - DL """

# make covariance vector from each segment and then group these into segments
vec_Y = np.zeros([int(m*(m+1)/2), n_seg])
vec_X = np.zeros([int(n*(n+1)/2), n_seg])
for i in range(n_seg):
    # Transformation to covariance domain and vectorization
    Ys_cov = np.cov(Ys[i])                      # covariance 
    Xs_cov = np.cov(Xs[i])                      # NOT diagonl ??
    
    # Vectorization of lower tri, row wise  
    vec_Y.T[i] = np.array(list(Ys_cov[np.tril_indices(m)]))
    vec_X.T[i] = np.array(list(Xs_cov[np.tril_indices(n)]))
    
#segmentation of vec_Y 
n_seg = 2
Y_new = np.split(vec_Y, n_seg, axis=1)            # Matrixs with segments in axis=0
X_new = np.split(vec_X, n_seg, axis=1)


#
## Dictionary learning for each segment of serveral cov vectors
A_rec = np.zeros([n_seg,m,n]) 
for j in range(n_seg):
    D, sigma, iter_ = K_SVD(Y_new[j], n=int(n*(n+1)/2), m=np.shape(Y_new)[1],
                             non_zero=np.shape(Y_new)[1]-16, n_samples=np.shape(Y_new)[0],
                             max_iter=100)
    print(D.shape,sigma.shape)
    # results for the large system
    Ys_cov_rec = np.matmul(D,sigma)
    Y_err = np.linalg.norm(Y_new[j] - Ys_cov_rec)
    print('reconstruction error %f \nnumber of iterations %i'%(Y_err, iter_))


    # Find A approximative
    def reverse_vec(x):
        m = int(np.sqrt(x.size*2))
        if (m*(m+1))/2 != x.size:
            print("reverse_vec fail")
            return None
        else:
            R, C = np.tril_indices(m)
            out = np.zeros((m, m), dtype=x.dtype)
            out[R, C] = x
            out[C, R] = x
        return out
    
    A_app = np.zeros(([m, n]))
    for u in range(n):
        d = D.T[u]
        matrix_d = reverse_vec(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(E)
        index = np.where(E == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_app.T[u] = temp.T

    A_rec[j] = A_app
    A_err = np.linalg.norm(A_real-A_rec[j])
        
    print('dictionary error %f'%( A_err))

#
#
#""" prediction of X """
#
#
#
#
#""" Plot """
#plt.figure(1)
#plt.subplot(3, 1, 1)
#plt.plot(seg_time[0], Ys[0].T)
#plt.xlabel('[sec.]')
#plt.title("Measurements")
#
#plt.subplot(3, 1, 3)
#plt.plot(seg_time[0], Xs[0].T)
#plt.xlabel('[sec.]')
#plt.title("real sources")

#plt.subplot(3,1,3)
#plt.plot(predicted_X)
#plt.title("predicted sources")
#plt.show()
