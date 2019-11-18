# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:03:14 2019

@author: Laura
"""
import numpy as np
from scipy import signal
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt

from Cov_DL.dictionary_learning import K_SVD
np.random.seed(1)
"""
Algorithm Summary:
Given Y and a dictionary A 
1. Initialise gamma (gamma = 1)
2. Compute Sigma and Mu (posterior moments)
3. Update gamma with EM rule or fast fixed-point
4. Iterate step 2 and 3 until convergence to a fixed point gamma* 
5. Assuming a point estimate is desired for the unknown weights X_gen,
   choose X_m-sbl = Mu* = X_gen, with Mu* = E[X|Y ; gamma*]
6. Given that gamma* is sparse, the resultant estimator Mu* 
   will necessarily be row sparse.

Threshold = 10E-16
"""
# =============================================================================
# Import data and dictionary
# =============================================================================
#A = A   # Found from COV-DL -- size M x N
#Y = Y   # Data -- size M x L

" Signals "
m = 6               # number of sensors
n = 8               # number of sources
non_zero = 5        # max number of non-zero coef. in rows of X
n_samples = 20      # number of sampels
duration = 8        # duration in seconds

" Another type signal "
# RANDOM GENERATION OF SPARSE DATA
Y, A, X = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n,
                                   n_features=m,
                                   n_nonzero_coefs=non_zero,
                                   random_state=0)


" One type of signals "
#time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
#s1 = np.sin(2 * time)                       # sinusoidal
#s2 = np.sign(np.sin(3 * time))              # square signal
#s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
#s4 = np.sin(4 * time)                       # another sinusoidal
#zero_row = np.zeros(n_samples)
#
#X_real = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T                     # Column concatenation
#n = len(X_real)
#m = 6
#non_zero = 6
#A = np.random.random((m,n))                 # Random mix matrix
#Y = np.dot(A, X_real)                       # Observed signal

# =============================================================================
# Algorithm
# =============================================================================
def M_SBL(A, Y, iterations = 10):
    M = len(Y)
    L = len(Y.T)
    N = len(A.T)
    gamma = np.ones((N,1))            # size N x iterations
    g = np.ones((N,1))
    Gamma = np.diag(gamma)            # size iterations x 1
    lam = np.zeros(N)                 # size N x 1
    mean = np.zeros(N)
    for k in range(iterations):       # the k iterations
        Mu = 0                            # size N x L
        Sigma = 0                         # size N x L        
        for i in range(N):           
            " Making Sigma and Mu "
            inv = np.linalg.inv(lam[i] * np.identity(M) + (A.dot(Gamma * A.T)))
            Sigma = Gamma - Gamma * (A.T.dot(inv)).dot(A) * Gamma
            Mu = Gamma * (A.T.dot(inv)).dot(Y)
            mean = Mu
            
            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
            lam_num = 1/L * np.linalg.norm(Y - A.dot(Mu), ord = 'fro')  # numerator
            lam_for = 0
            for j in range(N):
                lam_for += Sigma[j][j] / gamma[j][k]
            lam_den = M - N + lam_for                      # denominator
            lam[i] =  lam_num / lam_den
       
            " Update gamma with EM and with M being Fixed-Point"
            gam_num = 1/L * np.linalg.norm(Mu[i])
            gam_den = 1 - gamma[i][k] * Sigma[i][i]
            g[i] = gam_num/gam_den
        gamma = np.vstack((gamma.T,g.T))
        gamma = gamma.T
            
    return gamma, mean

gam, X_new = M_SBL(A, Y)
X_err = np.zeros(len(X))
for i in range(len(X)):
    X_err[i] = np.linalg.norm(X[i] - X_new[i])


#
#plt.figure(1)
#plt.plot(X[0])
#plt.plot(X_new[0])
#plt.show
#
#plt.figure(2)
#plt.plot(X[5])
#plt.plot(X_new[5])
#
#plt.figure(3)
#plt.plot(X)
#plt.plot(X_new)