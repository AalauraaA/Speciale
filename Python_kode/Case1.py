# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 1 - mixed Signals
m = 3
n = 8
L = n_samples = 20
k = non-zero = 4 
Y = signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt signal - n x L 

Full recovery because k < m

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
#from Cov_DL import data_generation
#from Cov_DL import MSBL
from sklearn.metrics import mean_squared_error
import data_generation
import MSBL

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 3               # number of sensors
n = 8              # number of sources
non_zero = 4         # max number of non-zero coef. in rows of X
n_samples = 100       # number of sampels
duration = 8
iterations = 1000

" Random Signals Generation - Sparse X "
#Y, A, X = data_generation.mix_signals_det(n_samples, duration, non_zero, long=False)
#m = len(Y)
#n = len(X)

Y, A, X = data_generation.mix_signals(n_samples, duration, m, n, non_zero)

# =============================================================================
# M-SBL
# =============================================================================
X_rec_noise = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise=True)

X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise=False)

mse_noise = mean_squared_error(X, X_rec_noise)
print("This is the error of X with noise: ", mse_noise)

mse = mean_squared_error(X, X_rec)
print("This is the error of X without: ", mse)
# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.subplot(4, 1, 1)
plt.title('Comparison of each active source in X and corresponding reconstruction ')
plt.plot(X[0], 'r',label='Real X')
plt.plot(X_rec_noise[0],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')

plt.subplot(4, 1, 2)
#plt.title('Plot of row 2 of X - Mixed Signal Data')
plt.plot(X[1], 'r',label='Real X')
plt.plot(X_rec_noise[1],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')

plt.subplot(4, 1, 3)
#plt.title('Plot of row 4 of X - Mixed Signal Data')
plt.plot(X[4], 'r',label='Real X')
plt.plot(X_rec_noise[4],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')


plt.subplot(4, 1, 4)
#plt.title('Plot of row 8 of X - Mixed Signal Data')
plt.plot(X[7], 'r',label='Real X')
plt.plot(X_rec_noise[7],'g', label='Recovered X')
plt.legend()
plt.show
#plt.savefig('case1_1.png')

plt.figure(2)
plt.subplot(4, 1, 1)
plt.title('Comparison of each active source in X and corresponding reconstruction ')
plt.plot(X[0], 'r',label='Real X')
plt.plot(X_rec[0],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')

plt.subplot(4, 1, 2)
#plt.title('Plot of row 2 of X - Mixed Signal Data')
plt.plot(X[1], 'r',label='Real X')
plt.plot(X_rec[1],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')

plt.subplot(4, 1, 3)
#plt.title('Plot of row 4 of X - Mixed Signal Data')
plt.plot(X[4], 'r',label='Real X')
plt.plot(X_rec[4],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_1.png')


plt.subplot(4, 1, 4)
#plt.title('Plot of row 8 of X - Mixed Signal Data')
plt.plot(X[7], 'r',label='Real X')
plt.plot(X_rec[7],'g', label='Recovered X')
plt.legend()
plt.show
#plt.savefig('case1_1.png')


#plt.figure(5)
#plt.title('Plot of column 2 of X - Mixed Signal Data')
#plt.plot(X.T[2], 'r',label='Real X')
#plt.plot(X_rec.T[2],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_2.png')
#

#for i in range(8):
#    plt.figure(3)
#    plt.title('Plot of row of X - Known X')
#    plt.plot(X[i], label=i)
#    plt.legend()
#    plt.savefig('case1_3.png')
#    plt.figure(4)
#    plt.title('Plot of row of X - Recovered X')
#    plt.plot(X_rec[i], label=i)
#    plt.legend()
#    plt.savefig('case1_4.png')
    

