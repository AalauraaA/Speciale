# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 2 - mixed Signals
m = 7
n = 8
L = n_samples = 20
k = non-zero = 4 
Y = signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt signal - n x L 

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation
from Cov_DL import MSBL
from sklearn.metrics import mean_squared_error

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 10              # number of sensors
n = 15               # number of sources
non_zero = 2         # max number of non-zero coef. in rows of X
n_samples = 20       # number of sampels
duration = 8
iterations = 1000

"Mixed Signals Generation - Sinus, sign, saw tooth and zeros"
Y, A, X = data_generation.mix_signals(n_samples, duration, m, n, non_zero)

# =============================================================================
# M-SBL
# =============================================================================
X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations)

mse = mean_squared_error(X, X_rec)
print("This is the error of X: ", mse)
# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.title('Plot of row 2 of X - Mixed Signal Data')
plt.plot(X[9], 'r',label='Real X')
plt.plot(X_rec[9],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case2_1.png')

plt.figure(2)
plt.title('Plot of column 2 of X - Mixed Signal Data')
plt.plot(X.T[5], 'r',label='Real X')
plt.plot(X_rec.T[5],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case2_2.png')

plt.figure(3)
plt.title('Plot of row 2 of X - Mixed Signal Data')
plt.plot(X[11], 'r',label='Real X')
plt.plot(X_rec[11],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case2_3.png')

plt.figure(4)
plt.title('Plot of column 2 of X - Mixed Signal Data')
plt.plot(X.T[11], 'r',label='Real X')
plt.plot(X_rec.T[11],'g', label='Recovered X')
plt.legend()
plt.show
plt.savefig('case2_4.png')

"This plot is not neccessary as we only have 2 non zeros"
#for i in range(4):
#    plt.figure(3)
#    plt.title('Plot of row of X - Known X')
#    plt.plot(X[i], label=i)
#    plt.legend()
#    plt.savefig('case2_3.png')
#    plt.figure(4)
#    plt.title('Plot of row of X - Recovered X')
#    plt.plot(X_rec[i], label=i)
#    plt.legend()
#    plt.savefig('case2_4.png')
