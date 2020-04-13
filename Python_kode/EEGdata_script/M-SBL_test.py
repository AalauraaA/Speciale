# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:05:22 2020

@author: Laura
"""
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import numpy as np

def M_SBL_with_k(A, Y, m, n, non_zero, iterations, noise):
    n_samples = Y.shape[1]
    if noise is False:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])   
        Gamma = np.ones([iterations+1, 1, 1])
        mean = np.ones([iterations+1, n, n_samples-2])
        Sigma = np.zeros([iterations+1, n, n])
        k = 0
        while gamma[k].any() >= 10E-16:
            Gamma[k] = np.diag(gamma[k])        # size 1 x 1
            for i in range(n):
                " Making Sigma and Mu "
                Sigma[k] = (np.identity(n) - np.sqrt(Gamma[k]) *
                            np.matmul(np.linalg.pinv(A * np.sqrt(Gamma[k])),
                                      A)) * Gamma[k]
                
                mean[k] = np.sqrt(Gamma[k]) * np.matmul(np.linalg.pinv(A *
                                                        np.sqrt(Gamma[k])), Y)

                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den

            if k == iterations:
                break
            k += 1

    elif noise is True:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])
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
                
                Sigma[k] = (Gamma[k] - Gamma[k] * (A.T.dot(inv)).dot(A) *
                            Gamma[k])
                mean[k] = Gamma[k] * (A.T.dot(inv)).dot(Y)

                " Making the noise variance/trade-off parameter lambda of p(Y|X)"
                lam_num = 1/n_samples * np.linalg.norm(Y - A.dot(mean[k]),
                                                       ord='fro')  # numerator
                lam_for = 0
                
                for j in range(n):
                    lam_for += Sigma[k][j][j] / gamma[k][j]
                
                lam_den = m - n + lam_for                        # denominator
                lam[k][i] = lam_num / lam_den

                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den

                if k == iterations:
                    break
                k += 1

    " Finding the support set "
    support = np.zeros(non_zero)
    H = gamma[-1]
    for l in range(non_zero):
        if H[np.argmax(H)] != 0:
            support[l] = np.argmax(H)
            H[np.argmax(H)] = 0

    " Create new mean with support set "
    New_mean = np.zeros([n, n_samples-2])
    for i in support:
        New_mean[int(i)] = mean[-1][int(i)]

    return New_mean, gamma, non_zero


def M_SBL_without_k(A, Y, m, n, iterations, noise):
    n_samples = Y.shape[1]
    if noise is False:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])   
        Gamma = np.ones([iterations+1, 1, 1])
        mean = np.ones([iterations+1, n, n_samples-2])
        Sigma = np.zeros([iterations+1, n, n])
        k = 0
        while gamma[k].any() >= 10E-16:
            Gamma[k] = np.diag(gamma[k])        # size 1 x 1
            for i in range(n):
                " Making Sigma and Mu "
                Sigma[k] = (np.identity(n) - np.sqrt(Gamma[k]) *
                            np.matmul(np.linalg.pinv(A * np.sqrt(Gamma[k])),
                                      A)) * Gamma[k]
                
                mean[k] = np.sqrt(Gamma[k]) * np.matmul(np.linalg.pinv(A *
                                                        np.sqrt(Gamma[k])), Y)

                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den

            if k == iterations:
                break
            k += 1

    elif noise is True:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, n, 1])
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
                
                Sigma[k] = (Gamma[k] - Gamma[k] * (A.T.dot(inv)).dot(A) *
                            Gamma[k])
                mean[k] = Gamma[k] * (A.T.dot(inv)).dot(Y)

                " Making the noise variance/trade-off parameter lambda of p(Y|X)"
                lam_num = 1/n_samples * np.linalg.norm(Y - A.dot(mean[k]),
                                                       ord='fro')  # numerator
                lam_for = 0
                
                for j in range(n):
                    lam_for += Sigma[k][j][j] / gamma[k][j]
                
                lam_den = m - n + lam_for                        # denominator
                lam[k][i] = lam_num / lam_den

                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/n_samples * np.linalg.norm(mean[k][i])
                gam_den = 1 - gamma[k][i] * Sigma[k][i][i]
                gamma[k+1][i] = gam_num/gam_den

                if k == iterations:
                    break
                k += 1

    " Finding the support set "
    gamma_copy = np.array(gamma[-1], copy=True)
    for i in range(len(gamma_copy)):
        if gamma_copy[i] >= 0.03:
            gamma_copy[i] = 0
    non_zero = np.count_nonzero(gamma_copy)
    support = np.zeros(non_zero)
    
    H = gamma[-1]
    for l in range(non_zero):
        if H[np.argmax(H)] != 0:
            support[l] = np.argmax(H)
            H[np.argmax(H)] = 0

    " Create new mean with support set "
    New_mean = np.zeros([n, n_samples-2])
    for i in support:
        New_mean[int(i)] = mean[-1][int(i)]

    return New_mean, gamma, non_zero

# =============================================================================
# 
# =============================================================================
np.random.seed(4523)

M = 8
N = 16
k = 11
L = 1000 
n_seg = 1

Y, A_real, X_real = generate_AR(N, M, L, k)
X_real = X_real.T[:-2]
X_real = X_real.T

X_rec1, gamma1, non_zero1 = M_SBL_with_k(A_real, Y, M, N, k, iterations=1000, noise=False)
X_rec2, gamma2, non_zero2 = M_SBL_without_k(A_real, Y, M, N, iterations=1000, noise=False)



#plot_seperate_sources_comparison(X_real,X_result[0],M,N,k,L)

X_mse1 = MSE_one_error(X_real,X_rec1)
X_mse2 = MSE_one_error(X_real,X_rec2)



