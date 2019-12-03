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

np.random.seed(1)

# =============================================================================
# Initial Conditions and known signals
# =============================================================================
#m = 32                # number of sensors
#n = 50                # number of sources
#non_zero = 49        # max number of non-zero coef. in rows of X
#n_samples = 2   # number of sampels
#duration = 8
#iterations = 10
#noise = False

#Y, A, X = data_generation.mix_signals(n_samples, duration, m, n, non_zero)
#X_rec = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations, noise)

# =============================================================================
# Finding X - Varying samples
# =============================================================================
samples_vary = np.linspace(1,50,11)
m = 25
n = 50
non_zero = 16
duration = 8
iterations = 10
noise = False

mse = np.zeros(len(samples_vary))
for i in range(len(samples_vary)):
    Y, A, X = data_generation.mix_signals(int(samples_vary[i]), duration, m, n, non_zero)
    X_rec_sample = MSBL.M_SBL(A, Y, m, n, int(samples_vary[i]), non_zero, iterations, noise)
    mse[i] = mean_squared_error(X, X_rec_sample)
    
plt.figure(1)
plt.plot(samples_vary, mse)
plt.title('Varying L - M = 25, N = 50, k = 16, iterations = 10')
plt.xlabel('Samples')
plt.ylabel('MSE')
plt.savefig('varying_samples.png')
plt.show()

# =============================================================================
# Finding X - Varying N sources
# =============================================================================
n_vary = np.linspace(1,50,26)
m = 25
n_samples = 10
duration = 8
iterations = 10
noise = False

mse = np.zeros(len(n_vary))
for i in range(len(n_vary)):
    non_zero = int(n_vary[i]/3) * 2
    Y, A, X = data_generation.mix_signals(n_samples, duration, m, int(n_vary[i]), non_zero)
    X_rec_N = MSBL.M_SBL(A, Y, m, int(n_vary[i]), n_samples, non_zero, iterations, noise)
    mse[i] = mean_squared_error(X, X_rec_N)

plt.figure(2)
plt.plot(n_vary, mse)
plt.title('Varying N and k - M = 25, L = 10, iterations = 10')
plt.xlabel('Sources')
plt.ylabel('MSE')
plt.savefig('varying_sources.png')
plt.show()

# =============================================================================
# Finding X - Varying k non-zero rows
# =============================================================================
k_vary = np.linspace(1,50,26)
m = 25
n = 50
n_samples = 10
duration = 8
iterations = 10
noise = False
mse = np.zeros(len(k_vary))
for i in range(len(k_vary)):
    #print(i)
    Y, A, X = data_generation.mix_signals(n_samples, duration, m, n, int(k_vary[i]))
    X_rec_k = MSBL.M_SBL(A, Y, m, n, n_samples, int(k_vary[i]), iterations, noise)
    mse[i] = mean_squared_error(X, X_rec_k)

plt.figure(3)
plt.plot(k_vary, mse)
plt.title('Varying k - M = 25, N = 50, L = 10, iteration = 10')
plt.xlabel('non-zeros')
plt.ylabel('MSE')
plt.savefig('varying_non_zeros.png')
plt.show()