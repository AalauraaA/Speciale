# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:18:47 2020

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# =============================================================================
# Functions and derivatives
# =============================================================================
def g1(x):
    """
    Equation 8.31
    """
    a = 1
    return np.tanh(a*x)

def g2(x):
    """
    Equation 8.32
    """
    return x * np.exp(-x**2/2)

def g3(x):
    """
    Equation 8.33
    """
    return x**2

def g_der1(x):
    """
    Equation 8.44
    """
    a = 1
    return a * (1 - g1(a*x)**2)

def g_der2(x):
    """
    Equation 8.45
    """
    return (1 - x**2) * np.exp(-x**2/2)

def g_der3(x):
    """
    Equation 8.46
    """
    return 3*x**2

# =============================================================================
# Algorithm
# =============================================================================
def preprocessing(X):
    """
    Subtract the mean from the signal observed signal X to make the
    mean of the data X zero
    """
    X = np.array(X)
    mean = X.mean(axis = 1, keepdims = True)
    
    X_center = X - mean   # X becomes zero mean (centred data)
    
    """
    Whitening the observed signal X to removed possibly correlations
    between the components.

    Use the eigenvalue decomposition (EVD) to whitening    
    """
    cov = np.cov(X_center)   # covariance matrix of X
    
    # EVD
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)                    # Matrix with eigenvalues in the diagonal
    D_inv = np.sqrt(np.linalg.inv(D)) # D^(-1/2)
    
    # Whitening
    X_white = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    
    return X_white
    
def ica(X, iterations, tolerance=1e-5):
    """
    The ICA algorithm
    fittet to segmented input
    input X has dim (n_seg, sources, Ls )
    """
    
    X = preprocessing(X) # Subtract mean from observed signal and whitening the observed signal
        
    components_nr = X.shape[0] # The amount of components
    
    W = np.zeros((components_nr, components_nr)) # Mixing/whiting matrix
        
    " Used the Gradient Algorithm to update W "
    for i in range(components_nr):
        w = np.random.rand(components_nr)  # Normal distribution - initial value
        
        for j in range(iterations): # Update the mixing matrix elementwise
            w_new = (X * g1(np.dot(w.T, X))).mean(axis=1) - g_der1(np.dot(w.T, X)).mean() * w
            w_new /= np.linalg.norm(w_new, ord=2) # Normalisation of w
            
            if i >= 1: # Components_nr is greater than 1
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            # (w * w_new).sum() want to be approx 1
            w = w_new
            
            if distance < tolerance:
                break
                
        W[i, :] = w # Update w
        
    S = np.dot(W, X) # Source signal  
    return S, W


def ica_segments(X, iterations):
    n_seg = len(X)
    N = X[0].shape[0]
    L = X[0].shape[1]
    
    S_ICA = np.zeros((n_seg, N, L-2))
    for i in range(len(X)):
        print('ICA on segment {}'.format(i))
        S, W = ica(X[i],iterations)
        S_ICA[i] = S.T[:L-2].T
    return S_ICA, W