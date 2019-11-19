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
m = 6               # number of sensors
n = 8               # number of sources
non_zero = 5        # max number of non-zero coef. in rows of X
n_samples = 20      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
Y, A_real, X_real = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n,
                                   n_features=m,
                                   n_nonzero_coefs=non_zero,
                                   random_state=0)

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

##" Rossler Data"
#from Rossler_copy import Generate_Rossler  # import rossler here
#X1, X2, X3, X4, X5, X6 = Generate_Rossler()
#
## we use only one of these dataset which are 1940 x 6
#X1 = X1[:50] # only 50 samples
#
#X_real = X1.T
#
#n_samples = len(X_real.T)                        # 1940 samples
#
## Include zero row to make n larger
#zero_row = np.zeros(n_samples)
#X_real = np.c_[ zero_row, X1.T[1], zero_row, zero_row, X1.T[2],
#               X1.T[3], zero_row, X1.T[4], X1.T[5]].T 
#
#n = len(X_real)
#m = 6
#non_zero = 6
## Generate A and Y 
#A_real = np.random.random((m, n))                 # Random mix matrix
#Y = np.dot(A_real, X_real)                        # Observed signal Y - 40 x 6
#
#
## PERFORM DICTIONARY LEARNING
A, X, iter_= K_SVD(Y, n, m, non_zero, n_samples, max_iter=1000)

Y_rec = np.matmul(A,X)

Y_err = np.linalg.norm(Y-Y_rec)
A_err = np.linalg.norm(A_real-A)
X_err = np.linalg.norm(X_real-X)


print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f, \nnumber of iterations %i'%(Y_err, A_err, X_err, iter_))

plt.figure(1)
plt.title("comparison of source signals (column)")
plt.plot(X_real.T[3].T,'-b', label="orig.")    
plt.plot(X.T[3].T,'-r', label="rec.")
plt.legend()

plt.figure(2)
plt.title("comparison of measurements and reconstructed signal")
plt.plot(Y[0], '-b', label='orig.')
plt.plot(Y_rec[0], '-r', label='rec.')
#plt.legend()

