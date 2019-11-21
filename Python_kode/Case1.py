# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 1 - Simpel Sparse Signal
m = 30
n = 80
L = n_samples = 20
k = non-zero = 40 (halvdelen af X)
Y = random signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt sparse signal - n x L 

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
from Cov_DL import data_generation
from Cov_DL import MSBL

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 30               # number of sensors
n = 80               # number of sources
non_zero = 40       # max number of non-zero coef. in rows of X
n_samples = 20       # number of sampels
iterations = 1000

" Random Signals Generation - Sparse X "
Y_ran, A_ran, X_ran = data_generation.random_sparse_data(m, n, non_zero, n_samples)

# =============================================================================
# M-SBL
# =============================================================================
X_Rec_ran = MSBL.M_SBL(A_ran, Y_ran, m, n, n_samples, non_zero, iterations)

# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.title('Plot of row 1 of X - Random Sparse Data')
plt.plot(X_ran[1], 'r',label='Real X')
plt.plot(X_Rec_ran[1],'g', label='Computed X')
plt.legend()
plt.show

plt.figure(2)
plt.title('Plot of column 1 of X - Random Sparse Data')
plt.plot(X_ran.T[1], 'r',label='Real X')
plt.plot(X_Rec_ran.T[1],'g', label='Computed X')
plt.legend()
plt.show
