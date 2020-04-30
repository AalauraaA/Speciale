# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:17:44 2020

@author: trine
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:03:14 2019

@author: Laura
"""
import numpy as np

# =============================================================================
# Without Segmentation M-SBL Algorithm
# =============================================================================
def M_SBL(A, Y, m, n, non_zero, iterations, noise):
    n_samples = Y.shape[1]
    tol=0.0001
    if noise is False:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])   
        Gamma = np.ones([iterations+1, n, n])
        mean = np.ones([iterations+1, n, n_samples-2])
        Sigma = np.zeros([iterations+1, n, n])
        k = 1
        while k < 3 or any((gamma[k]-gamma[k-1]) > tol):
            print(k)
            print(gamma[k])
            Gamma[k] = np.diag(np.reshape(gamma[k],(n)))        # size 1 x 1
            
            " Making Sigma and Mu "
            Sigma[k] = np.dot((np.identity(n) - np.linalg.multi_dot(
                    [np.sqrt(Gamma[k]), np.linalg.pinv(
                            np.dot(A, np.sqrt(Gamma[k]))), A])), Gamma[k])
            mean[k] = np.linalg.multi_dot(
                    [np.sqrt(Gamma[k]), np.linalg.pinv(np.dot(A,
                                                    np.sqrt(Gamma[k]))), Y])
            for i in range(n):
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - ((1/(gamma[k][i])) * Sigma[k][i][i])
                gamma[k+1][i] = gam_num/gam_den
            if k == iterations:
                break
            k += 1

    elif noise is True:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])
        Gamma = np.ones([iterations+1, n, n])
        mean = np.ones([iterations+1, n, n_samples-2])
        Sigma = np.zeros([iterations+1, n, n])
        lam = np.ones([iterations+2,1])     # size N x 1
        k = 1
        while k < 3 or any((gamma[k]-gamma[k-1]) > tol):
            Gamma[k] = np.diag(np.reshape(gamma[k],(n)))
            
            " Making Sigma and Mu "
            sig = lam[k] * np.identity(m) +  np.linalg.multi_dot([A,Gamma[k],A.T])
            inv = np.linalg.pinv(sig)
            Sigma[k] = (Gamma[k]) - (np.linalg.multi_dot([Gamma[k],A.T,inv,A,Gamma[k]]))
            mean[k] = np.linalg.multi_dot([Gamma[k],A.T,inv,Y])

            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
            lam_num = 1/n_samples * np.linalg.norm(Y - A.dot(mean[k]),
                                                   ord='fro')**2  # numerator
            lam_for = 0
            for j in range(n):
                lam_for += Sigma[k][j][j] / gamma[k][j]
                
            lam_den = m - n + lam_for                        # denominator
            lam[k+1] = lam_num / lam_den
                
            for i in range(n):
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - ((1/(gamma[k][i])) * Sigma[k][i][i])
                gamma[k+1][i] = gam_num/gam_den

                if k == iterations:
                    break
                k += 1

    " Finding the support set "

    #print(gamma[0],gamma[1],gamma[2],gamma[3])
    #print(Sigma[2])
    support = np.zeros(non_zero)
    H = gamma[k-1]
    for l in range(non_zero):
        if H[np.argmax(H)] != 0:
            support[l] = np.argmax(H)
            H[np.argmax(H)] = 0

    " Create new mean with support set "
    New_mean = np.zeros([n, n_samples-2])
#    print('support shape {}'.format(support.shape))
    for i in support:
        New_mean[int(i)] = mean[k-1][int(i)]
    return New_mean
