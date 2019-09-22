# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:40:41 2019

@author: Laura
Url: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e

See chapter 8 in ICA book
"""
import numpy as np
np.random.seed(0)

def center(X):
    """
    Subtract the mean from the signal observed signal X
    """
    X = np.array(X)
    mean = X.mean(axis = 1, keepdims = True)
    
    return X - mean

def whitening(X):
    """
    Whitening the observed signal X to removed possibly correlations
    between the components.

    Use the eigenvalue decomposition (EVD) to whitening    
    """
    cov = np.cov(X)
    
    # EVD
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)                   # Matrix with eigenvalues in the diagonal
    D_inv = np.sqrt(np.linalg.inv(D)) 
    
    # Whitening
    X_white = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    
    return X_white

def g(x):
    """
    Equation 8.31, with a = 1
    """
    return np.tanh(x)

def g_der(x):
    return 1 - g(x) * g(x)

def new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum()) # Normalisation of w
    
    return w_new


def ica(X, iterations, tolerance=1e-5):
    """
    The ICA algorithmen
    """
    X = center(X)     # Subtract mean from observed signal
    X = whitening(X)  # Whitening the observed signal
        
    components_nr = X.shape[0] # The amount of components
    
    W = np.zeros((components_nr, components_nr), dtype=X.dtype) # Mixing matrix
    
    
    " Used the Gradient Algorithm to update W "
    for i in range(components_nr):
        w = np.random.rand(components_nr)  # Normal distribution - initial value
        
        for j in range(iterations): # Update the mixing matrix elementwise
            w_new = new_w(w, X)
            
            if i >= 1: # Components_nr is greater than 1
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            # (w * w_new).sum() want to be approx 1
            
            w = w_new
            
            if distance < tolerance:
                break
                
        W[i, :] = w # Update w
        
    S = np.dot(W, X) # Source signal  
    return S
