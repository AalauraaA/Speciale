# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:03:14 2019

@author: Laura
"""
import numpy as np
np.random.seed(1)
#from Cov_DL import data_generation

#m = 3               # number of sensors
#n = 4               # number of sources
#non_zero = 4        # max number of non-zero coef. in rows of X
#n_samples = 20       # number of sampels
##duration = 8
#iterations = 100
#
#" Random Signals Generation - Sparse X "
#Y, A, X_ran = data_generation.random_sparse_data(m, n, non_zero, n_samples)
#
#"Mixed Signals Generation - Sinus, sign, saw tooth and zeros"
##Y_mix, A_mix, X_mix = data_generation.mix_signals(n_samples, duration, m)

#X = data_generation.generate_AR(205)
#A = np.random.randn(m,n)
#Y = np.dot(A, X)

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

#X_rec = M_SBL(A, Y, m, n, n_samples, non_zero, iterations)
## =============================================================================
## Segmenteret M-SBL
## =============================================================================
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
