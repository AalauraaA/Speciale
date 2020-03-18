# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:34:43 2019

@author: trine
"""
import numpy as np
import data_generation
import dictionary_learning
from sklearn.decomposition import PCA


def reverse_vec(x):
    m = int(np.sqrt(x.size*2))
    if (m*(m+1))//2 != x.size:
        print("reverse_vec fail")
        return None
    else:
        R, C = np.tril_indices(m)
        out = np.zeros((m, m), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out


def Cov_DL1(Y, A, X, m, n, cov_seg, L, k):
    """ 
    """
    ## internal segmetation for training examples
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, cov_seg, L)
    
    ## Covariance Domain Tranformation
    
    Y_big = np.zeros([int(m*(m+1)/2.),n_seg])
    for i in range(n_seg):              # loop over all segments
        
        # Transformation to covariance domain and vectorization
        Y_cov = np.corrcoef(Ys[i])                      # covariance 
        X_cov = np.corrcoef(Xs[i])                      # NOT diagonl ??
        
        # Vectorization of lower tri, row wise  
        vec_Y = Y_cov[np.tril_indices(m)]
        
        sigma = np.diagonal(X_cov)
    
        Y_big.T[i] = vec_Y
    
    ## Dictionary Learning on Transformed System
    L = n_seg 
    
    Y_big_rec, D, sigma = dictionary_learning.DL(Y_big.T, n=len(sigma), k=k)

    Y_err = np.linalg.norm(Y_big - Y_big_rec.T)

    print('cov domain: reconstruction error %f'%(Y_err))
    
    ## Find Mixing Matrix From D 
   
    A_rec = np.zeros(([m, n]))
    for j in range(n):
        d = D.T[j]
        matrix_d = reverse_vec(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(np.abs(E))     # should it be abselut here?
        index = np.where(np.abs(E) == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_rec.T[j] = temp.T

    A_err = data_generation.MSE_one_error(A,A_rec)
        
    print('Dictionary error %f'%(A_err))
    
    return A_rec, A_err


## funktion til brug i DL2:



def Cov_DL2(Y, A, X, m, n, cov_seg, L, k):
    """ 
    """
    ## internal segmetation for training examples
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, cov_seg, L)
    
    ## Covariance Domain Tranformation
    
    Y_big = np.zeros([int(m*(m+1)/2.),n_seg])
    for i in range(n_seg):              # loop over all segments
        
        # Transformation to covariance domain and vectorization
        Y_cov = np.cov(Ys[i])                      # covariance 
        X_cov = np.cov(Xs[i])                      # NOT diagonl ??
#        print(X_cov)
        # Vectorization of lower tri, row wise  
        vec_Y = Y_cov[np.tril_indices(m)]
        
        sigma = np.diagonal(X_cov)
    
        Y_big.T[i] = vec_Y
    
    ## Dictionary Learning on Transformed System
    L = n_seg  
    
    pca = PCA(n_components=n, svd_solver='randomized',
          whiten=True) 
    pca.fit(Y_big.T)
    U = pca.components_.T
    
#    A = np.random.random((m,n)) # random initial A 
    A = np.random.randn(m,n)  # Gaussian initial A
    a = np.reshape(A,(A.size)) # vectorization of initial A
    
    def D_(a):
        D = np.zeros((int(m*(m+1)/2),n))
        for i in range(n):
            A_tilde = np.outer(a[m*i:m*i+m],a[m*i:m*i+m].T)
            D.T[i] = A_tilde[np.tril_indices(m)]
        return D
    
    def D_term(a):
        return np.dot(np.dot(D_(a),(np.linalg.inv(np.dot(D_(a).T,D_(a))))),D_(a).T)
    
    def U_term():
        return np.dot(np.dot(U,(np.linalg.inv(np.dot(U.T,U)))),U.T)
    
    def cost1(a):
        return np.linalg.norm(D_term(a)-U_term())**2
        
    
    # predefined optimization method, without defined the gradient og the cost. 
    from scipy.optimize import minimize
    res = minimize(cost1, a, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True})
    a_new = res.x
    A_rec = np.reshape(a_new,(m,n)) 
    print(A_rec)
    A_err = data_generation.MSE_one_error(A,A_rec)
    return A_rec, A_err