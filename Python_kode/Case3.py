# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 3 - oscillation
m = 3
n = 4
L = n_samples = 20
k = non-zero = 4 
Y = signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt signal - n x L 

Full recovery because k = n and k < m

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation
from Cov_DL import MSBL

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 8                  # number of sensors
n_samples = 1940       # number of sampels
iterations = 1000

"Mixed Signals Generation - Sinus, sign, saw tooth and zeros"
Y, A, X, non_zero = data_generation.rossler_data(n_sampels=1940, ex = 1, m=8)
n = len(X)
# =============================================================================
# M-SBL
# =============================================================================
X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations)

# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.title('Plot of row 2 of X - Mixed Signal Data')
plt.plot(X[5], 'r',label='Real X')
plt.plot(X_rec[5],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case3_1.png')

plt.figure(2)
plt.title('Plot of column 2 of X - Mixed Signal Data')
plt.plot(X.T[5], 'r',label='Real X')
plt.plot(X_rec.T[5],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case3_2.png')

for i in range(4):
    plt.figure(3)
    plt.title('Plot of row of X - Known X')
    plt.plot(X[i], label=i)
    plt.legend()
    plt.savefig('case3_3.png')
    plt.figure(4)
    plt.title('Plot of row of X - Recovered X')
    plt.plot(X_rec[i], label=i)
    plt.legend()
    plt.savefig('case3_4.png')
