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

def mix_signals(n_samples, duration):
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
    
    # Column concatenation
    X = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T
    n = len(X)
    m = 6
    A = np.random.random((m, n))                 # Random mix matrix
    Y = np.dot(A, X)                             # Observed signal
    
    return Y, A, X

def rossler_data(n_sampels=1940):
    """
    - denne skal måske opdaters så vi kan vi kan ændre dimensioner 
    
    Generate rossler data with 
    m = 8, n = 10, k = 6
    
    INPUT: n_sampels -> max value is 1940
    """
    from Rossler_copy import Generate_Rossler  # import rossler here
    X1, X2, X3, X4, X5, X6 = Generate_Rossler()
    # we use only one of these dataset which are 1940 x 6
    X1 = X1[:n_sampels] # max 1940 samples
    
    # Include zero rows to make n larger
    zero_row = np.zeros(n_sampels)
    X = np.c_[X1.T[0], zero_row, X1.T[1], zero_row, zero_row, X1.T[2],
                   X1.T[3], zero_row, X1.T[4], X1.T[5]].T 
    
    n = len(X)
    m = 8  
    # Generate A and Y 
    A = np.random.random((m, n))            # Random mix matrix
    Y = np.dot(A, X)                        # Observed signal
    
    return Y, A, X

def segmentation_split(Y, X, L, n_sampels):
    """
    Segmentation of data by split into segments of length L. 
    The last segment is removed if too small.  
    
    OUTPUT:
        Ys -> array of size (n_seg, m, L), with segments in axis 0 
        Xs -> array of size (n_seg, n, L), with segments in axis 0 
        n_seg -> number of segments
    """ 
    n_seg = int(n_sampels/L)               # Number of segments
    X = X[:n_seg*L]                        # remove last segement if too small
    Y = Y[:n_seg*L]
    
    Ys = np.split(Y, n_seg, axis=1)        # Matrixs with segments in axis=0
    Xs = np.split(X, n_seg, axis=1)
    
    return Ys, Xs, n_seg


