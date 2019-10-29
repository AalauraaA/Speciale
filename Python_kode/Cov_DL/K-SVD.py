# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:23:32 2019

@author: trine
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import orthogonal_mp
from sklearn.datasets import make_sparse_coded_signal

np.random.seed(1)


""" Dictionary learning K-SVD """

# First a basis pursiut method to solve for initial X 
# OMP algorithm form book, using clase from sklearn
# lav toy eksemple 
# Define parameters 
#
n_samples = 100
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
s1 = np.sin(2 * time)                       # sinusoidal
s2 = np.sign(np.sin(3 * time))              # square signal
s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
s4 = np.sin(4 * time)                       # another sinusoidal
zero_row = np.zeros(n_samples)

X_real = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T                     # Column concatenation
m = len(X_real)
n = 6
A_real = np.random.random((n,m))                 # Mix matrix
Y = np.dot(A_real, X_real)                       # Observed signal
non_zero = 4



def X_update(Y, A, X, non_zero): 
    for i in range(n_samples):
        x = orthogonal_mp(A, Y.T[i], n_nonzero_coefs=non_zero, tol=None,
                          precompute=False, copy_X=True, return_path=False,
                          return_n_iter=False)
        X.T[i] = x
    return X

## funktionen virker ikke som funktion nu!!!
def K_SVD(Y, n, m, n_sampels, non_zero, stop=0.005, max_iter=10):
 
    # INITIALIZATION
    A = np.random.random((n,m))         # let A = A_0 
    X = np.zeros((m,n_samples))         # let X = X_0 
    
    # Normalisation og A and Y (tjeck results with and without) normalising A is required from the alg in book
    A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)  # normalizing the columns
    #Y = Y/np.linalg.norm(Y, ord=2, axis=0, keepdims=True)
    
    for k in range(max_iter): 
        print (k)
        # UPDATE X
        X = X_update(Y, A, X, non_zero)
        
        # UPDATE A and corresponding X row
        for j in range(m): # run over number of columns in A
            
            # identify non-zeros entries in j'te row of X 
            W = np.nonzero(X[j])
            W = W[0]
            
            if W.size == 0:
                X[j] = 0
                
            else:
                # make AX without the contribution from j0 
                G = np.zeros(np.shape(Y))
                idx = np.arange(m)
                idx = np.delete(idx,j)
                for i in idx:
                    G = G + (np.outer(A.T[i],X[i]))
                
                # make E 
                E = Y - G
                
                # makes P from O to restrict E
                P = np.zeros([len(X[0]),len(W)])
                for i in range(len(W)):
                    P.T[i][W[i]]=1
                
                # restrict E
                E_r = np.matmul(E,P)
                
                # apply SVD to E_r
                u,d,vh = np.linalg.svd(E_r)
                
                # update a og x
                A.T[j] = u.T[0]
                
                x_r = d[0]*vh[0] 
                x_r = np.matmul(x_r,P.T)
                
                X[j] = x_r
            
        err_A = np.linalg.norm(Y-(np.matmul(A,X)))
        if err_A < stop:
            break
        
    iter_ = k

    return(A, X, iter_ )
    
        



A, X, iter_= K_SVD(Y, A_real, X_real, n, m, n_samples, non_zero) 

# Automatic generated data
#m = 8
#n = 6
#non_zero = 4
#n_samples = 100
#
#n_sources, n_sensors = m, n
#n_nonzero_coefs = non_zero
#
#Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
#                                   n_components=m,
#                                   n_features=n,
#                                   n_nonzero_coefs=non_zero,
#                                   random_state=0)



      
plt.figure(3)
plt.plot(X_real[5],'-b')    
plt.plot(X[5],'-r')  

#X = X_update(Y, A)
#
#plt.figure(4)
#plt.plot(X_real[5],'-b')    
#plt.plot(X[5],'-r')  
    
    
    
        
    
        
    








        
        




 








