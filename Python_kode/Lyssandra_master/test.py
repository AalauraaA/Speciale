# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:43:43 2019

@author: Laura
"""
import os
os.getcwd()

import numpy as np
from sklearn.linear_model import orthogonal_mp
from sklearn.datasets import make_sparse_coded_signal

from lyssa.dict_learning import ksvd

""" Random generated data """
# INITIALISING PARAMETERS
m = 6               # number of sensors
n = 8               # number of sources
non_zero = 6        # max number of non-zero coef. in rows of X
n_samples = 60      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n,
                                   n_features=m,
                                   n_nonzero_coefs=non_zero,
                                   random_state=0)

A = np.random.random((m,n))  # let A = A0, random chosen
X = np.zeros((n,n_samples))  # let X = X0, all zero
for i in range(n_samples):
    x = orthogonal_mp(A, Y.T[i], n_nonzero_coefs=non_zero, tol=None,
                      precompute=False)
    X.T[i] = x


K = ksvd.ksvd_dict_learn(X, n_atoms=len(A.T), init_dict='data')
