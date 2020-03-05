# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:19:05 2019

@author: mathi
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from scipy import signal

# consider Y = AX 
# 
# m -> number of measurements, dim of y
# n -> number of sources, dim of x
# k -> number og non-zeros entries in x, aktive sources 
# n_samples -> number of samples 

def norm_mse(X_real,X_rec):
    X_real1 = X_real/np.max(X_real)
    X_rec1 = X_rec/np.max(X_rec)
    
#    X_real = X_real/np.linalg.norm(X_real, ord=2, axis=0, keepdims=True)
#    X_rec = X_rec/np.linalg.norm(X_rec, ord=2, axis=0, keepdims=True)
    
    temp = np.sum(((np.abs(X_real1-X_rec1))**2),axis=0)/len(X_rec1)

#    if np.any(temp == 0) or np.any(np.isnan(temp)):
#        raise SystemExit('temp = 0')
    nmse = np.average(temp)
    return nmse

def random_sparse_data(n_measurement, n_source, n_nonzero, n_samples):
    """
    Generate a signal as a sparse combination of dictionary elements.

    Returns a matrix Y = DX, 
    such as D is (n_measurement, n_source), 
    X is (n_source, n_samples) 
    and each column of X has exactly n_nonzero non-zero elements.
    
    INPUT:
    n_measurement   -> column dim of y
    n_source        -> column dim of x
    n_nonzero       -> number of non-zeros entries in x, aktive sources 
    n_samples       -> number of samples, number of columns in Y and X 
    
    RETURN: Y, A, X
    """
    
    Y, A, X = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n_source,
                                   n_features=n_measurement,
                                   n_nonzero_coefs=n_nonzero,
                                   random_state=0)

    return Y, A, X

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

def mix_signals_det(n_samples, duration, non_zero, long=True):
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
    zero_row = np.zeros(n_samples)

    X = np.c_[s1, s2, s3, s4].T
    n = len(X)
    m = len(X)
    A = np.random.randn(m, n)                 # Random mix matrix
    #A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    Y = np.dot(A, X)                       # Observed signal
    
    if long==True:
        X = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T
        n = len(X)
        m = 3
        A = np.random.randn(m, n)                 # Random mix matrix
        A = A/np.linalg.norm(A, ord=2, axis=0, keepdims=True)
        Y = np.dot(A, X)    
    return Y, A, X

def rossler_data(n_sampels=1940, ex = 1, m=8):
    """
    - denne skal måske opdaters så vi kan vi kan ændre dimensioner 
    
    Generates rossler data with 
    ex = 1 
    m = 8, n = 16, k = 6
    
    ex = 2 
    m = 8, n = 16, k = 10 
    
    INPUT: n_sampels -> max value is 1940
    """
    from Rossler_generation import Generate_Rossler  # import rossler here
    X1, X2, X3, X4, X5, X6 = Generate_Rossler()
    # we use only one of these dataset which are 1940 x 6
    X1 = X1[:n_sampels] # input max 1940 samples
    
    # Include zero rows to make n larger
    zero_row = np.zeros(n_sampels)
    
    if ex == 1:
        X = np.c_[X1.T[0], zero_row, X1.T[1], zero_row, zero_row, zero_row,
                  X1.T[2], X1.T[3], zero_row, zero_row, X1.T[4], zero_row, 
                  zero_row, X1.T[5], zero_row, zero_row,].T
        n = len(X)
        k = 6
        
    
    if ex == 2:
        X = np.c_[X1.T[0], X1.T[0], zero_row, X1.T[1], zero_row, X1.T[5], 
                  zero_row, X1.T[2], X1.T[3], zero_row, X1.T[4], X1.T[5], 
                  zero_row, X1.T[2], X1.T[4], zero_row ].T
        n = len(X)      
        k = 10      
              
    n = len(X)
    m = 8  
    # Generate A and Y 
    A = np.random.randn(m, n)            # Random mix matrix
    Y = np.dot(A, X)                        # Observed signal
    
    return Y, A, X, k

def generate_AR_v1(N, M, L, non_zero):
    """
    Generate sources from an AR process
    
    Input:
        N: size of the rows (amount of sources)
        L: size of the columns (amount of samples)
        
    Output:
        X: Source matrix of size N x L     
    """
    #np.random.seed(123)
    A = np.random.uniform(-1,1, (N,L))
    X = np.zeros([N, L+2])
    W = np.random.randn(N, L)
    for i in range(N):
        for j in range(2,L):
            X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i][j-2] + W[i][j]
    
    " Making zero and non-zero rows "
    Real_X = np.zeros([N, L+2])
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,N)
        while temp in ind:
            temp = np.random.randint(0,N)
        ind[i] = temp
    
    for j in ind:
        k = np.random.choice(len(X))
        Real_X[int(j)] = X[k]
    
    Real_X = Real_X.T[2:].T  
    
    " System "
    A_Real = np.random.randn(M,N)
    Y_Real = np.dot(A_Real, Real_X)    
    return Y_Real, A_Real, Real_X

def generate_AR_v2(N, M, L, non_zero):
    """
    Generate sources from an AR process
    
    Input:
        N: size of the rows (amount of sources)
        L: size of the columns (amount of samples)
        
    Output:
        X: Source matrix of size N x L     
    """
    #np.random.seed(123)
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


