# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:35:27 2019

@author: trine
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp
from sklearn.decomposition import DictionaryLearning


def K_SVD(Y, n, m, non_zero, n_samples, max_iter=100, stop=0.005):
    """
    Discribtion: 
        use k-svd algorithm to learn the dictionary A and representation X, 
        from a training database of n_sampels sampels Y. considering the 
        optimisation problem 
                                min A,X in Y = AX 
                                st. non_zero <= m
        
        
    INPUT:      Y,              array of size m x n_sampels
                m,              number of sensors 
                n,              number of sources
                non_zero,       sparsity constraint to ensure solution, < m 
                n_sampels,      number of sampels/training database 
        
    OUTPUT:     A,              array of size m x n 
    """
    # initialisation
    A = np.random.random((m,n))  # let A = A0, random chosen
    X = np.zeros((n,n_samples))  # let X = X0, all zero
    
    A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)  # column normalisation 
    
    for k in range(max_iter):
        # UPDATE X
        for i in range(n_samples):
            x = orthogonal_mp(A, Y.T[i], n_nonzero_coefs=non_zero, tol=None,
                              precompute=False, copy_X=True,
                              return_path=False, return_n_iter=False)
            X.T[i] = x
        
        # UPDATE A
        for j in range(n): 
            # identify non-zeros entries in j'te row of X 
            W = np.nonzero(X[j])
            W = W[0]
            
            if W.size == 0: # if the column is all zero let it stay all zero
                X[j] = 0
                
            else:
                # make G = AX without the contribution from the j0 column of A 
                G = np.zeros(np.shape(Y))
                idx = np.arange(n)
                idx = np.delete(idx,j)
                for i in idx:
                    G = G + (np.outer(A.T[i], X[i]))
                
                # make error matrix E, indicates the contribution from j0 
                E = Y - G
                
                # make matrix P from W
                P = np.zeros([len(X[0]), len(W)])
                for i in range(len(W)):
                    P.T[i][W[i]] = 1
                
                # restrict E, by remove columns with no contribution
                E_r = np.matmul(E,P)
                
                # apply SVD to E_r
                u,d,vh = np.linalg.svd(E_r)
                
                # update a_j0 og x^T_j0
                A.T[j] = u.T[0]
                x_r = d[0]*vh[0] 
                # transform length of x_r back to m by inseting zeros
                x = np.matmul(x_r,P.T) 
                
                X[j] = x
        
        err = np.linalg.norm(Y-(np.matmul(A,X)))
        if err < stop:
            break
        
    iter_ = k
    return A, X, iter_

def DL(Y,n,k,iter_=1000):
    dct = DictionaryLearning(n_components=n,transform_algorithm='omp',transform_n_nonzero_coefs=k, max_iter=iter_)
    dct.fit(Y)
    A_new = dct.components_
    X_new = dct.transform(Y)
    Y_new = np.matmul(A_new.T,X_new.T)
    
    Y_new = Y_new.T
    A_new = A_new.T
    X_new = X_new.T
    return Y_new, A_new, X_new
    
    
    
    
    
    