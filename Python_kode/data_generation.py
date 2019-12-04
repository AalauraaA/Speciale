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

#def generate_AR(N):
#    """
#    Generate sources from an AR process
#    
#    Input:
#        N: size of the columns
#        
#    Output:
#        X: Source matrix of size 4 x N -- 4 can be change by adding more
#           AR processes
#    
#    """
##    A = np.array([[0.05, 0, 0, 0, 0, 0, 0, 0, 0],
##                  [0, -0.05, 0, 0, 0, 0, 0, 0, 0],
##                  [0, 0, 0.3, 0, 0, 0, 0, 0, 0],
##                  [0, 0, 0, 0.4, 0, 0, 0, 0, 0],
##                  [0, 0, 0, 0, 0.8, 0, 0, 0, 0],
##                  [0, 0, 0, 0, 0, 0.06, 0, 0, 0],
##                  [0, 0, 0, 0, 0, 0, 0.2, 0, 0],
##                  [0, 0, 0, 0, 0, 0, 0, 0.2, 0],
##                  [0, 0, 0, 0, 0, 0, 0, 0, 0.05]])
##    A = MixingMatrix(N,N)    
#    XX1 = np.zeros(N)
#    XX2 = np.zeros(N)
#    XX3 = np.zeros(N)
#    XX4 = np.zeros(N)
#    
#    LP = 200
#
#    " Generating Synthetic AR Data "
#    w = np.random.randn(4, N)
#    for j in range(2,N):
##        for i in range(A.shape[0]):
##            XX1[j-2] = A[0][0] * XX1[j-1] - A[1][1] * XX1[j-2] + w[0, j-2]
##            XX2[j-2] = A[2][2] * XX1[j-1] + A[3][3] * XX3[j-1] + w[1, j-2]
##            XX3[j-2] = A[4][4] * XX1[j-1]**2 + A[5][5] * XX3[j-1] + A[6][6] * XX2[j-1] + w[2, j-2]
##            XX4[j-2] = A[7][7] * XX4[j-1] + A[8][8] * XX1[j-1] + w[3, j-2]
#            XX1[j-2] = 0.05 * XX1[j-1] - (-0.05) * XX1[j-2] + w[0, j-2]
#            XX2[j-2] = 0.3 * XX1[j-1] + 0.4 * XX3[j-1] + w[1, j-2]
#            XX3[j-2] = 0.8 * XX1[j-1]**2 + 0.06 *XX3[j-1] + 0.2 * XX2[j-1] + w[2, j-2]
#            XX4[j-2] = 0.2 * XX4[j-1] + 0.05 * XX1[j-1] + w[3, j-2]
#    XX1 = XX1[LP+1 : -1]
#    XX2 = XX2[LP+1 : -1]
#    XX3 = XX3[LP+1 : -1]
#    XX4 = XX4[LP+1 : -1]
#    
#    X = np.vstack([XX1, XX2, XX3, XX4])
#    return X

def segmentation_split(Y, X, Ls, n_sampels):
    """
    Segmentation of data by split into segments of length L. 
    The last segment is removed if too small.  
    
    OUTPUT:
        Ys -> array of size (n_seg, m, L), with segments in axis 0 
        Xs -> array of size (n_seg, n, L), with segments in axis 0 
        n_seg -> number of segments
    """ 
    n_seg = int(n_sampels/Ls)               # Number of segments
    X = X.T[:n_seg*Ls]                        # remove last segement if too small
    Y = Y.T[:n_seg*Ls]
    
    Ys = np.split(Y.T, n_seg, axis=1)        # Matrixs with segments in axis=0
    Xs = np.split(X.T, n_seg, axis=1)
    
    return Ys, Xs, n_seg


