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

n_samples = 200
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
s1 = np.sin(2 * time)                       # sinusoidal
s2 = np.sign(np.sin(3 * time))              # square signal
s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
s4 = np.sin(4 * time)                       # another sinusoidal

X = np.c_[s1, s2, s3, s4].T                 # Column concatenation
A = np.array(([[1, 1, 1, 1], [0.5, 2, 1.0, 2], [1.5, 1.0, 2.0, 1]]))  # Mix matrix
Y = np.dot(A, X)                            # Observed signal

M = len(Y)
N = len(X)


# Segmentation of observations (easy way - split)
fs = n_samples/duration                     # Samples pr second
S = 10                                      # Samples pr segment
nr_seg = n_samples/S                        # Number of segments

Ys = np.split(Y, nr_seg, axis=1)            # Matrixs with segments in axis=0
Xs = np.split(X, nr_seg, axis=1)

seg_time = np.split(time, nr_seg)

""" Cov - DL """
# now solv for 1 segment only.. Ys[0] size(10 x 3)

# Transformation to covariance domain and vectorization
Ys_cov = np.cov(Ys[0])                      # covariance size(3 x 3)
Xs_cov = np.cov(Xs[0])                      # NOT diagonl ??

vec_Y = np.array(list(Ys_cov[np.tril_indices(M)]))  # Vectorization og lower tri, row wise  
sigma = np.diag(Xs_cov)

# Dictionary learning

# just a random D
D = np.random.normal(size=(len(vec_Y), N))

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
