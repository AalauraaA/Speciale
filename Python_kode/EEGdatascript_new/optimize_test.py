# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:03:41 2020

@author: trine
"""
import scipy
import scipy.optimize as so
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(123)
#
#A_init = np.random.randint(0,10,(4,4))
#a_init = np.reshape(A_init, (A_init.size))
#B = np.random.randint(0,10,(4,4))
#b = np.reshape(B, (B.size))
#
#def cost(A):    
#    return np.linalg.norm(np.reshape(A, (A.size))-np.reshape(B, (B.size)))**2
#
#res = so.minimize(cost, a_init, method='BFGS',
#                  options={'maxiter': 1000000, 'disp': True})
#
#a_new = res.x
#A_rec = np.reshape(a_new, (A_init.shape))
#
#plt.figure(1)
#plt.plot(a_new, '-or', label = 'estimate a' )
#plt.plot(a_init, '-ob', label = 'initial a' )
#plt.plot(b, '-og', label = 'true b' )
#plt.legend()
#plt.show
#
#fun_val = cost(A_rec)

#########################################################
n = 4
m = 3
L = 100
L_covseg = 10

Y = np.random.random((m,L))

def _vectorization(X, m):
    vec_X = X[np.tril_indices(m)]
    return vec_X

def _covdomain(Y, L, L_covseg, M):
    n_seg = int(L/L_covseg)               # number of segments
    Y = Y.T[:n_seg*L_covseg].T              # remove last segment if to small
    print(Y.shape)
    Ys = np.split(Y, n_seg, axis=1)       # list of all segments in axis 0
    Y_big = np.zeros([int(M*(M+1)/2.), n_seg])
    for j in range(n_seg):                   # loop over all segments
        Y_cov = np.cov(Ys[j])           # normalised covariance mtrix is np.corrcoef(Ys[j])
        Y_big.T[j] = _vectorization(Y_cov, M)
    return Y_big

Y_big = _covdomain(Y,L,L_covseg, m)
y_big = np.reshape(Y_big, (Y_big.size),order='F') 
pca = PCA(n_components=n, svd_solver='randomized', whiten=True)
pca.fit(Y_big.T)
U = pca.components_.T
A = np.random.randint(0,5,(m, n))       # Gaussian initial A
a = np.reshape(A, (A.size),order='F')             # n

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

def cost(a):
    val =  np.linalg.norm(D_term(a)-U_term())**2
    return val


fun_init = cost(a)

res = so.minimize(cost, a, method='BFGS',# BFGS, Nelder-Mead
                  options={'maxiter': 1000000, 'disp': True})

a_new = res.x
A_rec = np.reshape(a_new, (A.shape),order='F')
fun_slut = cost(a_new)

plt.figure(1)
plt.plot(a_new, '-or', label = 'estimate a' )
plt.plot(a, '-ob', label = 'initial a' )
#plt.plot(y_big, '-og', label = 'true b' )
plt.legend()
plt.show