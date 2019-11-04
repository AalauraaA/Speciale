# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:49:13 2019

@author: trine
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt
from scipy import signal

from dictionary_learning import K_SVD
np.random.seed(1)

# INITIALISING PARAMETERS
#m = 6               # number of sensors
#n = 8               # number of sources
#non_zero = m        # max number of non-zero coef. in rows of X
#n_samples = 1     # number of sampels

# RANDOM GENERATION OF SPARSE DATA
#Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
#                                   n_components=n,
#                                   n_features=m,
#                                   n_nonzero_coefs=4,
#                                   random_state=0)

# GENERATION OF SIGNAL FROM DETERMNISTIC SIGNAL MIXTURE
#duration = 8                                # duration in seconds
#time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
#s1 = np.sin(2 * time)                       # sinusoidal
#s2 = np.sign(np.sin(3 * time))              # square signal
#s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
#s4 = np.sin(4 * time)                       # another sinusoidal
#zero_row = np.zeros(n_samples)
#
#X_real = np.c_[s1, zero_row, zero_row, s2, zero_row, s3, zero_row, s4].T                     # Column concatenation
#n = len(X_real)
#m = 6
#non_zero = 6
#A_real = np.random.random((m,n))                 # Random mix matrix
#Y = np.dot(A_real, X_real)                       # Observed signal

# TEST DATA
#Y = np.array([0.70509564, 0.57942377, 0.47660506, 0.70332399, 0.57813733,
#       0.71132446, 0.66494341, 0.54655971, 0.67665673, 0.64543953,
#       0.63693084, 0.5235195 , 0.63882834, 0.60544199, 0.57661413,
#       0.27021547, 0.22219083, 0.26392632, 0.24708834, 0.24210971,
#       0.10689917])
#
#Y = Y.reshape(21,1)
#
#n = 36 
#m = len(Y)
#n_samples = 1
#non_zero = m

" Rossler Data"
from Rossler import Generate_Rossler    # import rossler here
X1, X2, X3, X4, X5, X6 = Generate_Rossler()

#Subtract the 6 sensors/sources from the solution space
X01 = X1.T[0]
X02 = X1.T[1]
X03 = X1.T[2]
X04 = X1.T[3]
X05 = X1.T[4]
X06 = X1.T[5]

# Måske ikke den rigtig duration (Rossler er på 50 sec før reducering)
n_samples = len(X01)                        # 1940 samples
duration = 8                                # duration in seconds
time = np.linspace(0, duration, n_samples)  # 8 seconds, with n_samples
zero_row = np.zeros(n_samples)

#Generate Y Data
X_real = np.c_[X01, zero_row, X02, zero_row, zero_row, X03, X04, zero_row, X05, X06].T      # Original X sources - 40 x 6
n = len(X_real)
m = 6
non_zero = 6
A_real = np.random.random((m, n))                 # Random mix matrix
Y = np.dot(A_real, X_real)                               # Observed signal Y - 40 x 6


# PERFORM DICTIONARY LEARNING
A, X, iter_, err = K_SVD(Y, n, m, non_zero, n_samples, max_iter=100)

Y_rec = np.matmul(A,X)

plt.figure(1)
plt.plot(X_real.T[3].T,'-b')    
plt.plot(X.T[3].T,'-r')

plt.figure(2)
plt.plot(Y.T, '-b')
plt.plot(Y_rec.T, '-r')

