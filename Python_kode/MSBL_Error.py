# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:42:35 2019

@author: Laura


M-SBL Error Testing:
    - Varying L samples
    - Varying M sensors
    - Varying N sources
    - Varying k active sources
"""
import MSBL
import data_generation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(1)

# =============================================================================
# Initial Conditions
# =============================================================================
m = 3                # number of sensors
n = 50                # number of sources
non_zero = 35         # max number of non-zero coef. in rows of X
n_samples = 1000     # number of sampels
duration = 8
iterations = 100
noise = False

# =============================================================================
# Known Signal Matrices
# =============================================================================
Y, A, X = data_generation.mix_signals(n_samples, duration, m, n, non_zero)
#X_rec = MSBL.M_SBL(A, Y, m, n, 1000, non_zero, iterations, noise)

# =============================================================================
# Finding X - Varying samples
# =============================================================================
#n_samples_vary = np.linspace(1,2000,201) # increase with 10
#
#X_rec_L = np.zeros([len(n_samples_vary), len(X), len(X.T)])
#for i in range(len(n_samples_vary)):
#    X_rec_L[i] = MSBL.M_SBL(A, Y, m, n, int(n_samples_vary[i]), non_zero, iterations, noise)

## =============================================================================
## Finding X - Varying M sensors
## =============================================================================
#m_vary = np.linspace(1,32,17)
#
#X_rec_M = np.zeros([len(m_vary), len(X), len(X.T)])
#for i in range(len(m_vary)):
#    X_rec_M[i] = MSBL.M_SBL(A, Y, int(m_vary[i]), n, n_samples, non_zero, iterations, noise)
#
#for j in range(len(X_rec_M)):
#    plt.plot(X_rec_M[j][5])

# =============================================================================
# Finding X - Varying M sensors
# =============================================================================
n_vary = np.linspace(1,50,26)
m = 25                # number of sensors
non_zero = 35         # max number of non-zero coef. in rows of X
n_samples = 100     # number of sampels
duration = 8
iterations = 1000
noise = False

X_rec_N = np.zeros([len(n_vary), len(X), len(X.T)])
for j in range(len(n_vary)):
    for i in n_vary:
        X_rec_N[j] = MSBL.M_SBL(A, Y, m, int(i), n_samples, non_zero, iterations, noise)

#for j in range(len(X_rec_N)):
#    plt.plot(X_rec_N[j][5])



