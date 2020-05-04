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

def MSE_segments(X_rec, X_true):
    """
    inputs are arrays of size (n_seg, sources, Ls)
    """
    mse_array = np.zeros(len(X_rec))
    for i in range(len(X_rec)):
        mse_array[i] = MSE_one_error(X_rec[i], X_true[i])
    average_mse = np.average(mse_array)
    return mse_array, average_mse

def mix_signals(n_samples, m, version=None, duration=4):
    """ 
    Generation of 4 independent signals, united in X with manuel zero rows in 
    between for sparsity. 
    Generation of random mixing matrix A and corresponding Y, such that Y = AX
            M=3
    version none -> N=5, k=4
    version 0 -> N=4, k=4
    version 1 -> N=8, k=4    -> cov_dl1 
            M=6
    version 2 -> N=8, k=8
    version 3 -> N=12, k=8
    version 4 -> N=21, k=8   -> cov_dl1

    RETURN: Y, A, X,
    """
    #np.random.seed(1234)
    time = np.linspace(0, duration, n_samples)  # list of time index 
    
    s1 = np.sin(2 * time)                       # sinusoidal
    s2 = np.sign(np.sin(3 * time))              # square signal
    s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
    s4 = np.sin(4 * time)                       # different sinusoidal
    s5 = np.cos(2 * time)                       # cosinus
    s6 = np.sign(np.sin(4 * time))              # square signal
    s7 = signal.sawtooth(5 * np.pi * time)      # saw tooth signal
    s8 = np.sin(8 * time)                       # different sinusoidal
    
    zero_row = np.zeros(n_samples)
    
    X = np.c_[s1, zero_row, s3, s4, s2].T 
    if version == 'test':
        X = np.c_[s1, s2, s3].T
    if version == 0:
        X = np.c_[s1, s3, s4, s2].T
    if version == 1:
        X = np.c_[zero_row, s1, zero_row, s3, zero_row, zero_row, s4, s2].T
    if version == 2:
        X = np.c_[s1, s2, s3, s4, s5, s6, s7, s8].T
    if version == 3:
        X = np.c_[s1, zero_row, s2, s3, zero_row, s4, zero_row, s5, s6, zero_row, s7, s8].T
    if version == 4:
        X = np.c_[s1, zero_row, zero_row, s2, s3, zero_row,  zero_row,
                  zero_row, s4, zero_row,  zero_row, zero_row, s5, zero_row,
                  s6, zero_row, s7,  zero_row,  zero_row, s8,  zero_row].T
        
    n = len(X)
    A = np.random.randn(m,n)                   # Random mix matrix
    #A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
    #A = np.array([[1,2,3,4,5],
#                  [6,7,8,9,10],
#                  [11,12,13,14,15]])
#    A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    Y = np.dot(X.T, A)                            # Observed signal
    return Y.T, A, X


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
        Y:
        A:
        X: Source matrix of size N x L 
    """
    #np.random.seed(123)
    X = np.zeros([N, L+2])
    
    for i in range(N):
        ind = np.random.randint(1,4)
        for j in range(2,L):
            if ind == 1:
                sig = np.random.uniform(-1,1,(2))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + np.random.randn(1)
            
            elif ind == 2: 
                sig = np.random.uniform(-1,1,(3))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + sig[2] * X[i][j-3] + np.random.randn(1)
                
            elif ind == 3:
                sig = np.random.uniform(-1,1,(2))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + np.random.randn(1)
            
            elif ind == 4:
                sig = np.random.uniform(-1,1,(4))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + sig[2] * X[i][j-3] + sig[3] * X[i][j-4]+ np.random.randn(1)
                
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


