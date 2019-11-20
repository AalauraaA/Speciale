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
import data_generation
np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 15               # number of sensors
n = 35               # number of sources
non_zero = 20        # max number of non-zero coef. in rows of X
n_samples = 20       # number of sampels
duration = 8

" Random Signals Generation - Sparse X "
Y_ran, A_ran, X_ran = data_generation.random_sparse_data(m, n, non_zero, n_samples)

"Mixed Signals Generation - Sinus, sign, saw tooth and zeros"
Y_mix, A_mix, X_mix = data_generation.mix_signals(n_samples, duration)

# =============================================================================
# Without Segmentation M-SBL Algorithm
# =============================================================================
iterations = 1000
def M_SBL(A, Y, m, n, n_samples):
    gamma = np.ones([iterations, n,1])   # size iterations x n x 1
    lam = np.zeros(n)                    # size N x 1
    mean = np.zeros(n)
    Sigma = 0                            # size N x L   
    for k in range(iterations):
        Gamma = np.diag(gamma[k])        # size N x 1
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
            lam_den = m - n + lam_for                            # denominator
            lam[i] =  lam_num / lam_den
           
            " Update gamma with EM and with M being Fixed-Point"
            gam_num = 1/n_samples * np.linalg.norm(mean[i])
            gam_den = 1 - gamma[k][i] * Sigma[i][i]
            gamma[k][i] = gam_num/gam_den
    
    " Finding the support set "    
    support = np.zeros(non_zero)
    gam = gamma[-1]
    for l in range(non_zero):
        if gam[np.argmax(gam)] != 0:
            support[l] = np.argmax(gam)
            gam[np.argmax(gam)] = 0
   
    " Create new mean with support set "
    New_mean = np.zeros((n,n_samples))
    for i in support:
        New_mean[int(i)] = mean[int(i)]

    return New_mean

m_mix = len(Y_mix)
n_mix = len(A_mix.T)

X_Rec_mix = M_SBL(A_mix, Y_mix, m_mix, n_mix, n_samples)
X_Rec_ran = M_SBL(A_ran, Y_ran, m, n, n_samples)

plt.figure(1)
plt.plot(X_mix.T[5])
plt.plot(X_Rec_mix.T[5])
plt.show

plt.figure(2)
plt.plot(X_ran.T[5])
plt.plot(X_Rec_ran.T[5])
plt.show

# =============================================================================
# Segmenteret M-SBL
# =============================================================================
#def M_SBL_Seg(A, Y, m, n, n_seg, non_zero, n_samples):
#    gamma = np.ones([n_seg, n, 1])        # size n_seg x N x 1
#    lam = np.ones(n)                      # size N x 1
#    mean = np.zeros([n_seg, n, n_samples])        # size n_seg x N x L
#    
#    for seg in range(n_seg):
#        Gamma = np.diag(gamma[seg])       # size 1 x 1
#        Sigma = 0                         # size N x L        
#        for i in range(n):   
#            " Making Sigma and Mean "
#            sig = lam[i] * np.identity(m) + (A[seg].dot(Gamma * A[seg].T))
#            inv = np.linalg.inv(sig)
#            Sigma = Gamma - Gamma * (A[seg].T.dot(inv)).dot(A[seg]) * Gamma
#            mean[seg] = Gamma * (A[seg].T.dot(inv)).dot(Y[seg])
#            
#            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
#            lam_num = 1/n_samples * np.linalg.norm(Y[seg] - A[seg].dot(mean[seg]), 
#                                           ord = 'fro')  # numerator
#            lam_for = 0
#            for j in range(n):
#                lam_for += Sigma[j][j] / gamma[seg][j]
#            lam_den = m - n + lam_for                    # denominator
#            lam[i] =  lam_num / lam_den
#       
#            " Update gamma with EM and with M being Fixed-Point"
#            gam_num = 1/n_samples * np.linalg.norm(mean[seg][i])
#            gam_den = 1 - gamma[seg][i] * Sigma[i][i]
#            gamma[seg][i] = gam_num/gam_den 
#            
#        support = np.zeros([n_seg, non_zero])
#        gam = gamma[seg][-1]
#        for l in range(non_zero):
#            if gam[np.argmax(gam)] != 0:
#                support[seg][l] = np.argmax(gam)
#                gam[np.argmax(gam)] = 0
#
#        New_mean = np.zeros([n_seg, n, n_samples])
#        for i in support[seg]:
#            New_mean[seg][int(i)] = mean[seg][int(i)]
#    
#        return New_mean, mean
#
#X_rec, X_old = M_SBL_Seg(A, Y, m, n, n_seg, non_zero, n_samples)