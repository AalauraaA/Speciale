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
from sklearn.metrics import mean_squared_error
import MSBL

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
plt.title('Comparison of each active source in X and corresponding reconstruction ')
for i in range(n+1)[1:6]: 
    print(i)
    plt.subplot(5, 1, i)
    plt.plot(X[i-1], 'r',label='Real X')
    plt.plot(X_rec_noise[i-1],'g', label='Recovered X (noise)')

plt.legend()
plt.show
#plt.savefig('case_rossler1.png')


plt.figure(2)
plt.title('Comparison of each active source in X and corresponding reconstruction ')
for i in range(n+1)[1:6]: 
    print(i)
    plt.subplot(5, 1, i)
    plt.plot(X[i-1], 'r',label='Real X')
    plt.plot(X_rec[i-1],'g', label='Recovered X (without noise)')

plt.legend()
plt.show

#plt.figure(2)
#for i in range(n+1)[1:6 ]: 
#    plt.subplot(5, 1, i)
#    plt.plot(X[i+4], 'r',label='Real X')
#    plt.plot(X_rec[i+4],'g', label='Recovered X')
#
#plt.legend()
#plt.show
##plt.savefig('case_rossler2.png')
#
#plt.figure(5)
#plt.title('Plot of column 2 of X - Mixed Signal Data')
#plt.plot(X.T[2], 'r',label='Real X')
#plt.plot(X_rec.T[2],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case1_2.png')
#
#


#plt.figure(1)
#plt.title('Plot of row 2 of X - Mixed Signal Data')
#plt.plot(X[5], 'r',label='Real X')
#plt.plot(X_rec[5],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case3_1.png')
#
#plt.figure(2)
#plt.title('Plot of column 2 of X - Mixed Signal Data')
#plt.plot(X.T[5], 'r',label='Real X')
#plt.plot(X_rec.T[5],'g', label='Recovered X')
#plt.legend()
#plt.show
#plt.savefig('case3_2.png')
#
#for i in range(4):
#    plt.figure(3)
#    plt.title('Plot of row of X - Known X')
#    plt.plot(X[i], label=i)
#    plt.legend()
#    plt.savefig('case3_3.png')
#    plt.figure(4)
#    plt.title('Plot of row of X - Recovered X')
#    plt.plot(X_rec[i], label=i)
#    plt.legend()
#    plt.savefig('case3_4.png')
