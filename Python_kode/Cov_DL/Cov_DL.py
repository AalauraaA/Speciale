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
np.random.seed(0)

""" generation of data - Linear Mixture Model Ys = A * Xs """
#n_samples = 100
#duration = 8                                # duration in seconds
#time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
#s1 = np.sin(2 * time)                       # sinusoidal
#s2 = np.sign(np.sin(3 * time))              # square signal
#s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
#s4 = np.sin(4 * time)                       # another sinusoidal
#zero_row = np.zeros(n_samples)

# Column concatenation
#X_real = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T
#n = len(X_real)
#m = 6
#non_zero = 6
#A_real = np.random.random((m, n))                 # Random mix matrix
#Y = np.dot(A_real, X_real)                        # Observed signal


" Rossler Data"
from Rossler import Generate_Rossler    # import rossler here
X1, X2, X3, X4, X5, X6 = Generate_Rossler()

#Subtract the 6 sensors/sources from the solution space
X01 = X1.T[0]
X02 = X1.T[1]
X03 = X1.T[2]
X04 = X1.T[3]
X05 = X1.T[4]
X06 = X1.T[5]

# Måske ikke den rigtig duration (Rossler er på 50 sec før reducering)
n_samples = len(X01)                        # 1940 samples
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
zero_row = np.zeros(n_samples)

#Generate Y Data
X_real = np.c_[X01, zero_row, X02, zero_row, zero_row, X03, X04, zero_row, X05, X06].T      # Original X sources - 40 x 6
n = len(X_real)
m = 6
non_zero = 6
A_real = np.random.random((m, n))                 # Random mix matrix
Y = np.dot(A_real, X_real)                               # Observed signal Y - 40 x 6

""" Segmentation of observations (easy way - split) """
fs = n_samples/duration                     # Samples pr second
S = 5                                       # Samples pr segment
n_seg = int(n_samples/S)                    # Number of segments

Ys = np.split(Y, n_seg, axis=1)            # Matrixs with segments in axis=0
Xs = np.split(X_real, n_seg, axis=1)

seg_time = np.split(time, n_seg)


""" Cov - DL """
# now solv for 1 segment only..

A_rec = np.zeros([n_seg,m,n])
for i in range(n_seg):
    print(i)
    # Transformation to covariance domain and vectorization
    Ys_cov = np.cov(Ys[i])                      # covariance 
    Xs_cov = np.cov(Xs[i])                      # NOT diagonl ??
    
    # Vectorization of lower tri, row wise  
    vec_Y = np.array(list(Ys_cov[np.tril_indices(m)]))
    vec_X = np.array(list(Xs_cov[np.tril_indices(n)]))
    sigma = np.diag(Xs_cov)
    
    n_samples = 1 
    vec_Y = vec_Y.reshape(len(vec_Y),n_samples)
    # Dictionary learning
    D, sigma, iter_, err = K_SVD(vec_Y, n=len(vec_X), m=len(vec_Y),
                                 non_zero=len(vec_Y), n_samples=n_samples,
                                 max_iter=100)

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
    
    A_app = np.zeros(([m, n]))
    for j in range(n):
        d = D.T[j]
        matrix_d = reverse_vec(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(E)
        index = np.where(E == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_app.T[j] = temp.T
    
    A_rec[i] = A_app
    
""" prediction of X """




""" Plot """
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(seg_time[0], Ys[0].T)
plt.xlabel('[sec.]')
plt.title("Measurements")

plt.subplot(3, 1, 3)
plt.plot(seg_time[0], Xs[0].T)
plt.xlabel('[sec.]')
plt.title("real sources")

#plt.subplot(3,1,3)
#plt.plot(predicted_X)
#plt.title("predicted sources")
#plt.show()
