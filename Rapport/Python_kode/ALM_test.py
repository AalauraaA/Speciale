# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:56:32 2020

@author: Laura

Augmented Lagrange Multiplers (ALM)
"""
import numpy as np

# =============================================================================
# Functions - f(x), h(x) and Lagrange
# =============================================================================
def f_norm(D,U,m):
    """
    The objective function
    ||D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F
    where d_i = vec(a_i a_i^T) and U is basic vector from PCA
    """
    norm = np.zeros(m)
    for i in range(m):
        D_ = D*np.linalg.inv(D.T.dot(D)) * D.T
        U_ = U*np.linalg.inv(U.T.dot(U)) * U.T
        norm[i] = np.linalg.norm(D_ - U_)
    return norm
#
#def f_sum(d,u,m,n):
#    """
#    The objective function
#    ||D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F
#    where d_i = vec(a_i a_i^T) and U is basic vector from PCA
#    """
#    d_sum = np.zeros(n)
#    u_sum = np.zeros(n)
#    for i in range(n):
#        d_sum[i] = d[i] * (d.T[i] * d[i]) * d.T[i]
#        u_sum[i] = u[i] * (u.T[i] * u[i]) * u.T[i]
#    return np.linalg.norm(d_sum - u_sum)

def h(m, n, A):
    """
    The contrained function (subject to)
    d_i = vec(a_i a_i^T)
    where a_i is the columns in A we wish to find
    """
    D = np.zeros((m,n))
    for i in range(n):
        D[i] = A[i] * A[i].T
    return D

def Lagrange_v1(A, D, U, lam, m, n, mu):
    f_ = f_norm(D,U,n)
    h_ = h(m, n, A)
    L = np.zeros((m,n))
    for i in range(n):
        L[i] = f_ + np.inner(lam, h_[i]) + mu/2 * np.linalg.norm(h_[i])**2
    return L
#
#def Lagrange_v2(d, u, A, m, n, mu, lam):
#    f_ = f_sum(d,u,m,n)
#    h_ = h(m, A)
#    L = np.zeros((m,n))
#    for i in range(n):
#        for j in range(n):
#            L[i] = f_ + mu[i]/2. * h_[j]**2 + lam[j] * h_[j]
#    return L

from scipy.optimize import minimize

A = np.random.random((3,3))
D = np.random.random((3,3))
U = np.random.random((3,3))
m = 3
n = 3
mu = 1
lam = np.random.random(3)


#Lagrange = Lagrange_v1(A, D, U, lam, m, n, mu)
#
#res = minimize(Lagrange, A, method='nelder-mead',
#            options={'xatol': 1e-8, 'disp': True})

res = minimize(Lagrange_v1, A, args=(D, U, lam, m, n, mu), method='nelder-mead',
            options={'xatol': 1e-8, 'disp': True})

res.A

# =============================================================================
# 
## =============================================================================
#Y = X # Covariance-domian Y
#B = np.zeros(11)
#mu = np.matrix((0.3/np.linalg.norm(Y, ord=2),0.3/np.linalg.norm(Y, ord=2)))
#rho = 1.1 + 2.5
#k = 0
#np.array()
#A = np.ones(11)
#while k <= 10:
#    U, S, V = np.linalg.svd(Y + np.linalg.inv(mu[k]) * B[k])
#    S_ = 1 # S_mu[k]^-1 (S)
#    A[k+1] = U * S_ * V.T
#    B[k+1] = B[k] + mu[k] * (Y - A[k+1])
#    mu[k+1] = mu[k] * rho
#    k += 1

# =============================================================================
# Augmented Lagrangian Method-Equality Constraints
## =============================================================================
#mu0 = 1
#tau0 = 1
#x_initial = 0
#lam0 = 0
#for k in range(1,10): # 1-9 with indices 0-8
#    if convergence == True:
#        break
#    lam[k+1] = lam[k] - mu[k] * c
#    mu[k+1] = mu[k]*1.5
#    x_initial[k+1] = x[k]
#    tau[k+1] = 2

    

    
    

