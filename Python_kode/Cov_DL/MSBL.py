# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:03:14 2019

@author: Laura
"""
import numpy as np
from scipy import signal
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt

from dictionary_learning import K_SVD
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
m = 30               # number of sensors
n = 60               # number of sources
non_zero = 40       # max number of non-zero coef. in rows of X
n_samples = 25     # number of sampels
duration = 8        # duration in seconds

" Another type signal "
# RANDOM GENERATION OF SPARSE DATA
#Y, A, X = make_sparse_coded_signal(n_samples=n_samples,
#                                   n_components=n,
#                                   n_features=m,
#                                   n_nonzero_coefs=non_zero,
#                                   random_state=0)
#

" One type of signals "
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
s1 = np.sin(2 * time)                       # sinusoidal
s2 = np.sign(np.sin(3 * time))              # square signal
s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
s4 = np.sin(4 * time)                       # another sinusoidal
zero_row = np.zeros(n_samples)

X = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T                     # Column concatenation
n = len(X)
m = 6
non_zero = 7
A = np.random.random((m,n))                 # Random mix matrix
Y = np.dot(A, X)                       # Observed signal

# =============================================================================
# Algorithm
# =============================================================================
iterations = 1000
#def M_SBL(A, Y, m, n, n_samples):
gamma = np.ones([iterations, n,1])            # size n x 1
lam = np.zeros(n)                 # size N x 1
mean = np.zeros(n)
Sigma = 0                         # size N x L   
for k in range(iterations):
    Gamma = np.diag(gamma[k])            # size iterations x 1
    for i in range(n):          
        " Making Sigma and Mu "
        sig = lam[i] * np.identity(m) + (A.dot(Gamma * A.T))
        inv = np.linalg.inv(sig)
        Sigma = Gamma - Gamma * (A.T.dot(inv)).dot(A) * Gamma
        mean = Gamma * (A.T.dot(inv)).dot(Y)
        
        " Making the noise variance/trade-off parameter lambda of p(Y|X)"
        lam_num = 1/n_samples * np.linalg.norm(Y - A.dot(mean), 
                                               ord = 'fro')  # numerator
        lam_for = 0
        for j in range(n):
            lam_for += Sigma[j][j] / gamma[k][j]
        lam_den = m - n + lam_for                      # denominator
        lam[i] =  lam_num / lam_den
       
        " Update gamma with EM and with M being Fixed-Point"
        gam_num = 1/n_samples * np.linalg.norm(mean[i])
        gam_den = 1 - gamma[k][i] * Sigma[i][i]
        gamma[k][i] = gam_num/gam_den
    
support = np.zeros(non_zero)
gam = gamma[-1]
for l in range(non_zero):
    if gam[np.argmax(gam)] != 0:
        support[l] = np.argmax(gam)
        gam[np.argmax(gam)] = 0

New_mean = np.zeros((n,n_samples))
for i in support:
    New_mean[int(i)] = mean[int(i)]

X_err = np.linalg.norm(X - New_mean)
#    return mean

#X_rec = M_SBL(A, Y, m, n, n_samples)
# =============================================================================
# Segmenteret M-SBL
# =============================================================================
def M_SBL_Seg(A, Y, m, n, n_seg, non_zero, n_samples):
    gamma = np.ones([n_seg, n, 1])        # size n_seg x N x 1
    lam = np.ones(n)                      # size N x 1
    mean = np.zeros([n_seg, n, n_samples])        # size n_seg x N x L
    
    for seg in range(n_seg):
        Gamma = np.diag(gamma[seg])       # size 1 x 1
        Sigma = 0                         # size N x L        
        for i in range(n):   
            " Making Sigma and Mean "
            sig = lam[i] * np.identity(m) + (A[seg].dot(Gamma * A[seg].T))
            inv = np.linalg.inv(sig)
            Sigma = Gamma - Gamma * (A[seg].T.dot(inv)).dot(A[seg]) * Gamma
            mean[seg] = Gamma * (A[seg].T.dot(inv)).dot(Y[seg])
            
            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
            lam_num = 1/n_samples * np.linalg.norm(Y[seg] - A[seg].dot(mean[seg]), 
                                           ord = 'fro')  # numerator
            lam_for = 0
            for j in range(n):
                lam_for += Sigma[j][j] / gamma[seg][j]
            lam_den = m - n + lam_for                    # denominator
            lam[i] =  lam_num / lam_den
       
            " Update gamma with EM and with M being Fixed-Point"
            gam_num = 1/n_samples * np.linalg.norm(mean[seg][i])
            gam_den = 1 - gamma[seg][i] * Sigma[i][i]
            gamma[seg][i] = gam_num/gam_den 
            
        support = np.zeros([n_seg, non_zero])
        gam = gamma[seg][-1]
        for l in range(non_zero):
            if gam[np.argmax(gam)] != 0:
                support[seg][l] = np.argmax(gam)
                gam[np.argmax(gam)] = 0

        New_mean = np.zeros([n_seg, n, n_samples])
        for i in support[seg]:
            New_mean[seg][int(i)] = mean[seg][int(i)]
    
        return New_mean, mean

X_rec, X_old = M_SBL_Seg(A, Y, m, n, n_seg, non_zero, n_samples)

plt.figure(1)
plt.plot(X.T[5])
plt.plot(New_mean.T[5])
plt.show

plt.figure(2)
plt.plot(X.T[5])
plt.plot(mean.T[5])

#plt.figure(3)
#plt.plot(X)
#plt.plot(X_new)