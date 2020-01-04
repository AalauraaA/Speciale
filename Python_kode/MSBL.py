# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:03:14 2019

@author: Laura
"""
import numpy as np
np.random.seed(1)
import data_generation

m = 16               # number of sensors
n = 40               # number of sources
non_zero = 5        # max number of non-zero coef. in rows of X
n_samples = 100       # number of sampels
#duration = 8
iterations = 500

Y, A, X = data_generation.generate_AR_v2(n, m, n_samples, non_zero)

# =============================================================================
# Without Segmentation M-SBL Algorithm
# =============================================================================
def M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise):
    if noise == False: 
        gamma = np.ones([iterations+2, n,1])   
        Gamma = np.ones([iterations+1, 1, 1])
        mean = np.ones([iterations+1, n, n_samples])
        Sigma = np.zeros([iterations+1, n, n]) 
        k = 0                                                       
        while gamma[k].any() >= 10E-16:
            Gamma[k] = np.diag(gamma[k])        # size 1 x 1
            for i in range(n):   
                " Making Sigma and Mu "
                Sigma[k] = (np.identity(n) - np.sqrt(Gamma[k]) * np.matmul(np.linalg.pinv(A * np.sqrt(Gamma[k])),A)) * Gamma[k]
                mean[k] = np.sqrt(Gamma[k]) * np.matmul(np.linalg.pinv(A * np.sqrt(Gamma[k])), Y)
                
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den
                    
            if k == iterations:
                break            
            k += 1
                
    elif noise == True:
        gamma = np.ones([iterations+2, n,1])   
        Gamma = np.ones([iterations+1, 1, 1])
        mean = np.ones([iterations+1, n, n_samples])
        Sigma = np.zeros([iterations+1, n, n]) 
        k = 0 
        lam = np.ones([iterations+1, n, 1])     # size N x 1
        while gamma[k].any() >= 10E-16:
            for i in range(n):   
                " Making Sigma and Mu "
                sig = lam[k][i] * np.identity(n) + (A * Gamma[k]).dot(A.T)
                inv = np.linalg.inv(sig)
                Sigma[k] = Gamma[k] - Gamma[k] * (A.T.dot(inv)).dot(A) * Gamma[k]
                mean[k] = Gamma[k] * (A.T.dot(inv)).dot(Y)
                
                " Making the noise variance/trade-off parameter lambda of p(Y|X)"
                lam_num = 1/n_samples * np.linalg.norm(Y - A.dot(mean[k]), 
                                                       ord = 'fro')  # numerator
                lam_for = 0
                for j in range(n):
                    lam_for += Sigma[k][j][j] / gamma[k][j]
                lam_den = m - n + lam_for                            # denominator
                lam[k][i] =  lam_num / lam_den
               
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den
                    
                if k == iterations:
                    break           
                k += 1

    " Finding the support set "    
    support = np.zeros(non_zero)
    H = gamma[-2]
    for l in range(non_zero):
        if H[np.argmax(H)] != 0:
            support[l] = np.argmax(H)
            H[np.argmax(H)] = 0
           
    " Create new mean with support set "
    New_mean = np.zeros([n,n_samples])
    for i in support:
        New_mean[int(i)] = mean[-1][int(i)]

    return New_mean
