# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:36:46 2020

@author: trine
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:34:43 2019

@author: trine
"""
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import PCA
import numpy as np

def _dictionarylearning(Y, n, k, iter_=100000):
    dct = DictionaryLearning(n_components=n, transform_algorithm='lars',
                             transform_n_nonzero_coefs=k, max_iter=iter_)
    dct.fit(Y.T)
    A_new = dct.components_
    return A_new.T

def _inversevectorization(x):
    """
    input:  vector of size m(m+1)/2
    output: matrix og´f size mxm
    """
    m = int(np.sqrt(x.size*2))
    if (m*(m+1))//2 != x.size:
        print("inverse vectorization fail")
        return None
    else:
        R, C = np.tril_indices(m)
        out = np.zeros((m, m), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out

def _vectorization(X, m):
    vec_X = X[np.tril_indices(m)]
    return vec_X

def _A(D, m, n):
    """
    determine A from D
    """
    A_rec = np.zeros(([m, n]))
    for j in range(n):
        d = D.T[j]
        matrix_d = _inversevectorization(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(np.abs(E))     # should it be abselut here?
        index = np.where(np.abs(E) == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_rec.T[j] = temp.T
    return A_rec

def _covdomain(Y, L, L_covseg, M):
    n_seg = int(L/L_covseg)               # number of segments
    Y = Y.T[:n_seg*L_covseg].T              # remove last segment if to small
    Ys = np.split(Y, n_seg, axis=1)       # list of all segments in axis 0
    Y_big = np.zeros([int(M*(M+1)/2.), n_seg])
    for j in range(n_seg):                   # loop over all segments
        Y_cov = np.cov(Ys[j])           # normalised covariance mtrix is np.corrcoef(Ys[j])
        Y_big.T[j] = _vectorization(Y_cov, M)
    return Y_big

def Cov_DL1(Y_big, M, N, k):
    """
    """
    print('using Cov_DL1')
    D = _dictionarylearning(Y_big, N, k)
    A_rec = _A(D, M, N)
    print('Estimation of A is done')
    return A_rec

# funktion til brug i DL2:

def Cov_DL2(Y_big, m, n, k, A_real):
    """ 
    """
    print('Using Cov_DL2 \n')
    #np.random.seed(12)
    # Dictionary Learning on Transformed System
    pca = PCA(n_components=n, svd_solver='randomized', whiten=True)
    pca.fit(Y_big.T)
    U = pca.components_.T
#    A = np.random.random((m,n))    # random initial A
    A = np.random.randn(m, n)       # Gaussian initial A
#    A = np.random.randint(-5,5,(m,n))
    a = np.reshape(A, (A.size),order='F')     # normal vectorization of initial A

    def D_(a):
        D = np.zeros((int(m*(m+1)/2), n))
        for i in range(n):
            A_tilde = np.outer(a[m*i:m*i+m], a[m*i:m*i+m].T)
            D.T[i] = A_tilde[np.tril_indices(m)]
        return D

    def D_term(a):
        return np.dot(np.dot(D_(a), (np.linalg.inv(np.dot(D_(a).T, D_(a))))),
                      D_(a).T)

    def U_term():
        return np.dot(np.dot(U, (np.linalg.inv(np.dot(U.T, U)))), U.T)

    def cost1(a):
        return np.linalg.norm(D_term(a)-U_term())**2
    # predefined optimization method, without defineing the gradient og the cost.
    from scipy.optimize import minimize
    res = minimize(cost1, a, method='BFGS',# BFGS, Nelder-Mead
                  options={'maxiter': 10000, 'disp': True})
    a_new = res.x
    A_rec = np.reshape(a_new, (m, n), order='F')
    
    print('\nCost(A_init) = {}'.format(np.round_(cost1(a), decimals=4)))
    print('Cost(A_estimte) = {}'.format(np.round_(cost1(a_new), decimals=4)))
    print('Cost(A_true) = {}'.format(np.round_(cost1(np.reshape(A_real, (A_real.size),order='F')),decimals=4)))
    
    return A_rec, A
