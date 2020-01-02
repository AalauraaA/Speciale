# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:42:35 2019

@author: Laura


M-SBL Error Testing:
    - Varying L samples
    - Varying N sources
    - Varying k active sources
"""
import MSBL
import data_generation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(2)

# =============================================================================
# Initial Conditions and known signals
# =============================================================================
#m = 8                # number of sensors
#n = 8                # number of sources
#non_zero = 4        # max number of non-zero coef. in rows of X
#n_samples = 100   # number of sampels
##duration = 8
#iterations = 1000
#noise = False
#
#Y, A, X = data_generation.generate_AR_v1(n, m, n_samples, non_zero)
#X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise)
#
#norm = mean_squared_error(X/len(X), X_rec/len(X_rec))

#plt.figure(1)
#plt.subplot(4, 1, 1)
#plt.title('Comparison of X and Xrec')
#plt.plot(X[1], 'r',label='Real')
#plt.plot(X_rec[1],'g', label='Rec')
#
#plt.subplot(4, 1, 2)
#plt.plot(X[3], 'r',label='Real')
#plt.plot(X_rec[3],'g', label='Rec')
#
#plt.subplot(4, 1, 3)
#plt.plot(X[4], 'r',label='Real')
#plt.plot(X_rec[4],'g', label='Rec')
#
#plt.subplot(4, 1, 4)
#plt.plot(X[5], 'r',label='Real')
#plt.plot(X_rec[5],'g', label='Rec')
#plt.legend()
#plt.show
#
#Y2, A2, X2 = data_generation.generate_AR_v2(n, m, n_samples, non_zero)
#X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise)
#
#plt.figure(2)
#plt.subplot(4, 1, 1)
#plt.title('Comparison of X and Xrec')
#plt.plot(X[2], 'r',label='Real')
#plt.plot(X_rec[2],'g', label='Rec')
#
#plt.subplot(4, 1, 2)
#plt.plot(X[5], 'r',label='Real')
#plt.plot(X_rec[5],'g', label='Rec')
#
#plt.subplot(4, 1, 3)
#plt.plot(X[6], 'r',label='Real')
#plt.plot(X_rec[6],'g', label='Rec')
#
#plt.subplot(4, 1, 4)
#plt.plot(X[7], 'r',label='Real')
#plt.plot(X_rec[7],'g', label='Rec')
#plt.legend()
#plt.show
#



# =============================================================================
# Finding X - Varying samples
# =============================================================================
#samples_vary = np.linspace(1,50,11)
#m = 25
#n = 50
#non_zero = 16
#duration = 8
#iterations = 10
#noise = False
#
#mse = np.zeros(len(samples_vary))
#norm = np.zeros(len(samples_vary))
#for i in range(len(samples_vary)):
#    Y, A, X = data_generation.generate_AR_v2(n, m, int(samples_vary[i]), non_zero)
#    X_rec_sample = MSBL.M_SBL(A, Y, m, n, int(samples_vary[i]), non_zero, iterations, noise)
#    mse[i] = mean_squared_error(X, X_rec_sample)
#    norm[i] = np.linalg.norm(X - X_rec_sample)/len(samples_vary)
#    
#plt.figure(1)
##plt.plot(samples_vary, mse)
#plt.plot(samples_vary, norm)
#plt.title('Varying L - M = 25, N = 50, k = 16, iterations = 10')
#plt.xlabel('Samples')
#plt.ylabel('norm')
#plt.savefig('varying_samples.png')
#plt.show()

## =============================================================================
## Finding X - Varying N sources
## =============================================================================
#n_vary = np.linspace(1,50,26)
#m = 25
#n_samples = 10
#duration = 8
#iterations = 10
#noise = False
#    
#mse = np.zeros(len(n_vary))
#norm = np.zeros(len(n_vary))
#for i in range(len(n_vary)):
#    non_zero = int(n_vary[i]/3) * 2
#    Y, A, X = data_generation.generate_AR_v2(int(n_vary[i]), m, n_samples, non_zero)
#    X_rec_N = MSBL.M_SBL(A, Y, m, int(n_vary[i]), n_samples, non_zero, iterations, noise)
#    mse[i] = mean_squared_error(X, X_rec_N)
#    norm[i] = np.linalg.norm(X - X_rec_N)/len(n_vary)
##    temp = X - X_rec_N
##    norm[i] = temp/np.max(temp)
#
#plt.figure(2)
##plt.plot(n_vary, mse)
#plt.plot(n_vary, norm)
#plt.title('Varying N and k - M = 25, L = 10, iterations = 10')
#plt.xlabel('Sources')
#plt.ylabel('norm')
#plt.savefig('varying_sources.png')
#plt.show()

## =============================================================================
## Finding X - Varying k non-zero rows
## =============================================================================
k_vary = np.linspace(1,50,26)
m = 25
n = 50
n_samples = 10
duration = 8
iterations = 10
noise = False
mse = np.zeros(len(k_vary))
norm = np.zeros(len(k_vary))
for i in range(len(k_vary)):
    #print(i)
    Y, A, X = data_generation.generate_AR_v2(m, n, n_samples, int(k_vary[i]))
    X_rec_k = MSBL.M_SBL(A, Y, m, n, n_samples, int(k_vary[i]), iterations, noise)
    mse[i] = mean_squared_error(X, X_rec_k)
    norm[i] = np.linalg.norm(X - X_rec_k)/len(k_vary)

plt.figure(3)
#plt.plot(k_vary, mse)
plt.plot(k_vary, norm)
plt.title('Varying k - M = 25, N = 50, L = 10, iteration = 10')
plt.xlabel('non-zeros')
plt.ylabel('Norm')
plt.savefig('varying_non_zeros.png')
plt.show()