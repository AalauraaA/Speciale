# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:23:32 2019

@author: trine
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
np.random.seed(0)

""" generation of data - Linear Mixture Model Ys = A * Xs """

#n_samples = 200
#duration = 8                                # duration in seconds
#time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
#s1 = np.sin(2 * time)                       # sinusoidal
#s2 = np.sign(np.sin(3 * time))              # square signal
#s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
#
#X = np.c_[s1, s2, s3].T                     # Column concatenation
#A = np.array(([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]))  # Mix matrix
#Y = np.dot(A, X)                            # Observed signal
#
#M = len(Y)
#N = len(X)
#
## Segmentation of observations (easy way - split)
#fs = n_samples/duration                     # Samples pr second
#S = 10                                      # Samples pr segment
#nr_seg = n_samples/S                        # Number of segments
#
#Ys = np.split(Y, nr_seg, axis=1)            # Matrixs with segments in axis=0
#Xs = np.split(X, nr_seg, axis=1)
#
#seg_time = np.split(time, nr_seg)
#
#""" Cov - DL """
## now solv for 1 segment only.. Ys[0] size(10 x 3)
#
## Transformation to covariance domain and vectorization
#Ys_cov = np.cov(Ys[0])                      # covariance size(3 x 3)
#Xs_cov = np.cov(Xs[0])                      # NOT diagonl??
#
#vec_Y = np.array(list(Ys_cov[np.tril_indices(M)]))  # Vectorization og lower tri, row wise  
#sigma = np.diag(Xs_cov)
#


""" Dictionary learning K-SVD """

# First a basis pursiut method to solve for initial X 
# OMP algorithm form book, using clase from sklearn
# lav toy eksemple 

# Define parameters 

n_samples = 200
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
s1 = np.sin(2 * time)                       # sinusoidal
s2 = np.sign(np.sin(3 * time))              # square signal
s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal

X_real = np.c_[s1, s2, s3].T                     # Column concatenation
A_real = np.array(([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))  # Mix matrix
Y = np.dot(A_real, X_real)                            # Observed signal


# Initialtion
K = 100  # possible iterations k

# let A = A_0 and normalizeing the columns 
A = np.random.random(A_real.shape)
A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)  # normalizing the columns
X = np.zeros(Y.shape)

# update X 
def X_update(Y, A_real): 
    for i in range(len(Y.T)):
        omp = OrthogonalMatchingPursuit().fit(A, Y.T[i])  # create object from class 
        x = omp.coef_
        X.T[i] = x
    return X

X = X_update(Y, A_real)
    








        
        




 








