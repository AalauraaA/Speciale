# -*- coding: utf-8 -*-
"""
@Author: Mattek10 

Definitions used for the Covariance-Domain Dictionary Learning method
to recover the mixing matrix A from Y = AX.
"""
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import PCA
import numpy as np

def _dictionarylearning(Y, N, k, iter_=1000):
    """
    Perform dictionary learning
    ------------------------------------------
    Input
        Y: measurement matrix of size M x L
        N: number of sources
        k: number of active sources
        
    Output
        A_new: the matrix D of sixe M x N
    """
    dct = DictionaryLearning(n_components=N, transform_algorithm='omp',
                             transform_n_nonzero_coefs=k, max_iter=iter_)
    dct.fit(Y.T)
    A_new = dct.components_
    return A_new.T

def _inversevectorization(x):
    """
    Input  
        x: vector of size M(M+1)/2
    Output 
        out: matrix of size M x M
    """
    M = int(np.sqrt(x.size*2))
    if (M*(M+1))//2 != x.size:
        print("inverse vectorization fail")
        return None
    else:
        R, C = np.tril_indices(M)
        out = np.zeros((M, M), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out

def _vectorization(X, M):
    """
    Vectorization of the source matrix
    -----------------------------------
    Input
        X: source matrix of size N x L
        M: number of sensors
    
    Output
        vec_X: vectorized source matrix of size N x M(M+1)/2
    """
    vec_X = X[np.tril_indices(M)]
    return vec_X

def _A(D, M, N):
    """
    Determine the mixing matrix A from the matrix D
    -----------------------------------------------
    Input
        D: a matrix of size M x N
        M: number of sensors
        N: number of sources
        
    Output
        A_rec: recovered mixing matrix A of size M x N
    """
    A_rec = np.zeros(([M, N]))
    
    for j in range(N):
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
    """
    Transformation into the covariance domain
    ------------------------------------------
    Input
        Y: measurement matrix of size M x L
        L: numnber of samples
        L_covseg: number of samples within one segment
        M: number of sensors
    """
    n_seg = int(L/L_covseg)                  # number of segments
    Y = Y.T[:n_seg*L_covseg].T               # remove last segment if to small
    Ys = np.split(Y, n_seg, axis=1)          # list of all segments in axis 0
    Y_big = np.zeros([int(M*(M+1)/2.), n_seg])
    
    for j in range(n_seg):                   # loop over all segments
        Y_cov = np.cov(Ys[j])           # covariance matrix
        Y_big.T[j] = _vectorization(Y_cov, M)
    return Y_big

def Cov_DL1(Y_big, M, N, k):
    """
    Recovering of mixing matrix A when D is under-determined
    --------------------------------------------------------
    Input
        Y_big: transformed measurement matrix Y
        M: number of sensors
        N: number of sources
        k: number of active sources
        
    Output
        A_rec: recovered mixing matrix A
    """
#    np.random.seed(12)
    D = _dictionarylearning(Y_big, N, k)
    A_rec = _A(D, M, N)
    return A_rec

def Cov_DL2(Y_big, M, N, k):
    """ 
    Recovering of mixing matrix A when D is over-determined
    -------------------------------------------------------
    Input
        Y_big: transformed measurement matrix Y
        M: number of sensors
        N: number of sources
        k: number of active sources
        
    Output
        A_rec: recovered mixing matrix A
    """
#    np.random.seed(12)
    
    " Dictionary Learning on Transformed System "
    pca = PCA(n_components=N, svd_solver='randomized', whiten=True)
    pca.fit(Y_big.T)
    U = pca.components_.T
    A = np.random.randn(M, N)       # Gaussian initial A
    a = np.reshape(A, (A.size))     # normal vectorization of initial A

    def D_():
        D = np.zeros((int(M*(M+1)/2), N))
        for i in range(N):
            A_tilde = np.outer(a[M*i:M*i+M], a[M*i:M*i+M].T)
            D.T[i] = A_tilde[np.tril_indices(M)]
        return D

    def D_term():
        return np.dot(np.dot(D_(), (np.linalg.inv(np.dot(D_().T, D_())))),
                      D_().T)

    def U_term():
        return np.dot(np.dot(U, (np.linalg.inv(np.dot(U.T, U)))), U.T)

    def cost1(a):
        return np.linalg.norm(D_term()-U_term())**2
    
    " Predefined optimization method "
    from scipy.optimize import minimize
    res = minimize(cost1, a, method='nelder-mead',
                   options={'xatol': 1e-8, 'disp': True})
    a_new = res.x
    A_rec = np.reshape(a_new, (M, N))
    return A_rec
