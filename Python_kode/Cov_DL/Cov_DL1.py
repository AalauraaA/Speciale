# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:27:46 2019

@author: trine
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
np.random.seed(0)

""" generation of data - Linear Mixture Model Ys = A * Xs """

n_samples = 100
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
s1 = np.sin(2 * time)                       # sinusoidal
s2 = np.sign(np.sin(3 * time))              # square signal
s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
s4 = np.sin(4 * time)                       # another sinusoidal
zero_row = np.zeros(n_samples)

X_real = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T                     # Column concatenation
m = len(X_real)
n = 6
non_zero = 4
A_real = np.random.random((n,m))                 # Mix matrix
Y = np.dot(A_real, X_real)                       # Observed signal



# Segmentation of observations (easy way - split)
fs = n_samples/duration                     # Samples pr second
S = 10                                      # Samples pr segment
nr_seg = n_samples/S                        # Number of segments

Ys = np.split(Y, nr_seg, axis=1)            # Matrixs with segments in axis=0
Xs = np.split(X_real, nr_seg, axis=1)

seg_time = np.split(time, nr_seg)

""" Cov - DL """
# now solv for 1 segment only..

# Transformation to covariance domain and vectorization
Ys_cov = np.cov(Ys[0])                      # covariance size(3 x 3)
Xs_cov = np.cov(Xs[0])                      # NOT diagonl ??

vec_Y = np.array(list(Ys_cov[np.tril_indices(m)]))  # Vectorization og lower tri, row wise  
sigma = np.diag(Xs_cov)

# Dictionary learning






# just a random D
D = np.random.normal(size=(len(vec_Y), n))

# Find A approximative


def reverse_vec(x):
    n = int(np.sqrt(x.size*2))
    if (n*(n+1))//2 != x.size:
        print("reverse_vec fail")
        return None
    else:
        R, C = np.tril_indices(n)
        out = np.zeros((n, n), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out

A_app = np.zeros(([M, N]))
for i in range(N):
    d = D.T[i]
    matrix_d = reverse_vec(d)
    E = np.linalg.eig(matrix_d)[0]
    V = np.linalg.eig(matrix_d)[1]
    max_eig = np.max(E)
    index = np.where(E == max_eig)
    max_vec = V[:, index[0]]        # using the column as eigen vector here
    temp = np.sqrt(max_eig)*max_vec
    A_app.T[i] = temp.T

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
