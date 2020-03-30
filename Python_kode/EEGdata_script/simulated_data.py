# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:43:00 2020

@author: trine
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:19:05 2019

@author: mathi
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import mean_squared_error
from scipy import signal

# consider Y = AX 
# 
# m -> number of measurements, dim of y
# n -> number of sources, dim of x
# k -> number og non-zeros entries in x, aktive sources 
# n_samples -> number of samples 

def MSE_all_errors(real,estimate):
    """
    Mean Squared Error (MSE) - m or n errors
    ----------------------------------------------------------------------
    Info:
        A small value -- close to zero -- is a good estimation.
        The inputs must be transponet to perform the action rowwise
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error = mean_squared_error(real.T, estimate.T, multioutput='raw_values')
    return error

def MSE_one_error(real,estimate):
    """
    Mean Squared Error (MSE) - One Error
    ----------------------------------------------------------------------
    Info:
        A small value -- close to zero -- is a good estimation.
        The inputs must be transponet to perform the action rowwise
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error = mean_squared_error(real.T, estimate.T)
    return error

def mix_signals(n_samples, duration, m, n, non_zero):
    """ 
    Generation of 4 independent signals, united in X with zero rows in 
    between for sparsity. 
    Generation of random mixing matrix A and corresponding Y, such that Y = AX
    
    where A is (8 x 6), X is (8 x n_samples), Y is (6 x n_sampels) 
    

    RETURN: Y, A, X,
    
    """
    time = np.linspace(0, duration, n_samples)  # list of time index 
    
    s1 = np.sin(2 * time)                       # sinusoidal
    s2 = np.sign(np.sin(3 * time))              # square signal
    s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
    s4 = np.sin(4 * time)                       # different sinusoidal
    S = np.vstack((s1,s2,s3,s4))
    #    zero_row = np.zeros(n_samples)
    
    # Column concatenation
    X = np.zeros((n,n_samples))
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,n)
    #    print('temp old:', temp)
        while temp in ind:
            temp = np.random.randint(0,n)
    #        print(i)
    #        if temp not in ind:
    #            break
        ind[i] = temp
    #    print('temp new:', temp)
    
    for j in ind:
        k = np.random.choice(len(S))
        X[int(j)] = S[k]
           
    
    #    X = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T
    #    X = np.c_[s1, s2, s3, s4].T
    n = len(X)
    A = np.random.randn(m, n)                 # Random mix matrix
    A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    Y = np.dot(A, X)                             # Observed signal
    
    return Y, A, X

def gaussian_signals(m, n, n_samples, non_zero, long=True):
    """ 

    RETURN: Y, A, X,
    
    """
    X = np.zeros((n,n_samples))
    zero_row = np.zeros(n_samples)
    
    for i in range(n):
        X[i] = np.random.normal(0,1,n_samples)

    A = np.random.randn(m, n)                 # Random mix matrix
    Y = np.dot(A, X)                       # Observed signal
    
    if long==True:
        X = np.zeros((n*2,n_samples))
        for i in np.arange(0,n+1,2):
            X[i] = np.random.normal(0,1,n_samples)
            X[i+1] = zero_row
            
        n = len(X)
        A = np.random.randn(m, n)                 # Random mix matrix
        A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)
        Y = np.dot(A, X)    
    return Y, A, X


def generate_AR(N, M, L, non_zero):
    """
    Generate sources from an AR process
    
    Input:
        N: size of the rows (amount of sources)
        L: size of the columns (amount of samples)
        
    Output:
        X: Source matrix of size N x L     
    """
    np.random.seed(123)
    X = np.zeros([N, L+2])
    
    for i in range(N):
        ind = np.random.randint(1,4)
        W = np.random.randn(N, L)
        A = np.random.uniform(-1,1, (N,L))
        for j in range(2,L):
            if ind == 1:
                X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i][j-2] + W[i][j]
            
            elif ind == 2: 
                X[i][j] = A[i][j-1] * X[i-1][j-1] + A[i][j-2] * X[i][j-1] + W[i][j]
                
            elif ind == 3:
                X[i][j] = A[i][j] * X[i-2][j-1] + A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i-1][j-1] + W[i][j]
            
            elif ind == 4:
                X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i-3][j-1] + W[i][j]
    
    " Making zero and non-zero rows "
    Real_X = np.zeros([N, L+2])
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,N)
        while temp in ind:
            temp = np.random.randint(0,N)
        ind[i] = temp
    
    for j in ind:
        Real_X[int(j)] = X[int(j)]
    
    Real_X = Real_X.T[2:].T  

    " System "
    A_Real = np.random.randn(M,N)
    Y_Real = np.dot(A_Real, Real_X)    
    return Y_Real, A_Real, Real_X

def segmentation_split(Y, X, Ls, n_sampels):
    """
    Segmentation of data by split into segments of length Ls. 
    The last segment is removed if too small.  
    
    OUTPUT:
        Ys -> array of size (n_seg, m, Ls), with segments in axis 0 
        Xs -> array of size (n_seg, n, Ls), with segments in axis 0 
        n_seg -> number of segments
    """ 
    n_seg = int(n_sampels/Ls)               # Number of segments
    X = X.T[:n_seg*Ls]                        # remove last segement if too small
    Y = Y.T[:n_seg*Ls]
    
    Ys = np.split(Y.T, n_seg, axis=1)        # Matrixs with segments in axis=0
    Xs = np.split(X.T, n_seg, axis=1)
    
    return Ys, Xs, n_seg


