# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:56:32 2020

@author: Laura

Augmented Lagrange Multiplers (ALM)
"""
import numpy as np

X = np.random.random((3,3))

def f(D,U):
    """
    The objective function
    ||D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F
    where d_i = vec(a_i a_i^T) and U is basic vector from PCA
    """
    D_ = D*np.linalg.inv(D.T.dot(D)) * D.T
    U_ = U*np.linalg.inv(U.T.dot(U)) * U.T
    return np.linalg.norm(D_ - U_)

def h(m, n, a):
    """
    The contrained function (subject to)
    d_i = vec(a_i a_i^T)
    where a_i is the columns in A we wish to find
    """
    d = np.zeros((m,n))
    for i in range(n):
        d[i] = np.vectorize(a[i].dot(a[i].T))
    return d

def Lagrange(D, U, a, Y, m, n, mu):
    f_ = f(D,U)
    h_ = h(m, n, a)
    L = np.zeros((m,n))
    for i in range(n):
        L[i] = f_ + np.inner(Y[i], h_[i]) + mu/2 * np.linalg.norm(h_[i])**2
    return L


# =============================================================================
# 
# =============================================================================
Y = X # Covariance-domian Y
B = np.zeros(11)
mu = np.matrix((0.3/np.linalg.norm(Y, ord=2),0.3/np.linalg.norm(Y, ord=2)))
rho = 1.1 + 2.5
k = 0
np.array()
A = np.ones(11)
while k <= 10:
    U, S, V = np.linalg.svd(Y + np.linalg.inv(mu[k]) * B[k])
    S_ = 1 # S_mu[k]^-1 (S)
    A[k+1] = U * S_ * V.T
    B[k+1] = B[k] + mu[k] * (Y - A[k+1])
    mu[k+1] = mu[k] * rho
    k += 1

    
    

    
    

