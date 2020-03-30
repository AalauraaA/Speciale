# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:40:41 2019

@author: Laura
Url: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e

See chapter 8 in ICA book
-----------------------------------------------------------------------
We have the ICA data model
    x = As
Estimating the independent component s can be done as
    s = A^{-1} x
We must introduce whitening which must be done before ICA
    z = Wx = WAs

"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation

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
    return S

# =============================================================================
# Generating Data
# =============================================================================
m = 3                # number of sensors
n = 3                # number of sources
non_zero = 2         # max number of non-zero coef. in rows of X
n_samples = 100       # number of sampels
iterations = 1000


Y, A, X = data_generation.generate_AR_v2(n, m, n_samples, non_zero)


#n_samples = 2000
#time = np.linspace(0, 8, n_samples)
#s1 = np.sin(2 * time)  # sinusoidal
#s2 = np.sign(np.sin(3 * time))  # square signal
#s3 = signal.sawtooth(2 * np.pi * time)  # saw tooth signal
#
#X = np.c_[s1, s2, s3]
#A = np.array(([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])) #mix matrix
#Y = np.dot(X, A) # Observed signal
#Y = Y.T

S = ica(Y, iterations)

mse = data_generation.MSE_one_error(X, S)

" Plots "
plt.figure(1)
plt.subplot(2, 1, 1)
for x in X:
    plt.plot(x)
plt.title("real sources")

plt.subplot(2,1,2)
for s in S:
    plt.plot(s)
plt.title("predicted sources")
plt.show()
