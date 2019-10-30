# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:49:13 2019

@author: trine
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt

from dictionary_learning import K_SVD
np.random.seed(1)

# INITIALISING PARAMETERS
m = 8               # number of sources
n = 6               # number of sensors
non_zero = n        # max number of non-zero koef. in rows of X
n_samples = 100     # number of sampels 

# RANDOM GENERATION OF SPARSE DATA
Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=m,
                                   n_features=n,
                                   n_nonzero_coefs=4,
                                   random_state=0)


# PERFORM DICTIONARY LEARNING
A, X, iter_, err = K_SVD(Y, n, m, non_zero, n_samples, max_iter=100)


plt.figure(1)
plt.plot(X_real.T[3].T,'-b')    
plt.plot(X.T[3].T,'-r')

plt.figure(2)
test = np.matmul(A,X.T[:5].T)
plt.plot(Y.T[:5].T, '-b')
plt.plot(test, '-r')

