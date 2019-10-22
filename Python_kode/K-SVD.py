# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:23:32 2019

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

X = np.c_[s1, s2, s3].T                     # Column concatenation
A = np.array(([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]))  # Mix matrix
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
Xs_cov = np.cov(Xs[0])                      # NOT diagonl??

vec_Y = np.array(list(Ys_cov[np.tril_indices(M)]))  # Vectorization og lower tri, row wise  
sigma = np.diag(Xs_cov)



""" Dictionary learning K-SVD """







