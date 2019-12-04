# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:49:13 2019

@author: trine
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import dict_learning 
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt
from scipy import signal
import data_generation
from sklearn.metrics import mean_squared_error


from dictionary_learning import K_SVD
from dictionary_learning import DL
np.random.seed(1)


# INITIALISING PARAMETERS
m = 50               # number of sensors
n = 25               # number of sources
k = 10       # max number of non-zero coef. in rows of X
L = 100      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
#Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, 8)
#m = len(Y)
#n = len(X_real)
#non_zero = 4

#Y, A, X = data_generation.random_sparse_data(m, n, k, L)
 
Y, A, X = data_generation.mix_signals(L, 10, m, n, k)
n = len(X)
m = len(Y)

Y = Y.T

### PERFORM DICTIONARY LEARNING
#A, X, iter_= K_SVD(Y, n, m, non_zero, n_samples, max_iter=1000)
Y_rec, A_rec, X_rec = DL(Y,n,k)
#A_rec = np.zeros(np.shape(A))

A_err = mean_squared_error(A,A_rec)
X_err = mean_squared_error(X,X_rec)
Y_err = mean_squared_error(Y,Y_rec)

#
print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f'%(Y_err, A_err, X_err))

plt.figure(2)
plt.title("comparison of source signals (column)")
plt.plot(X[11],'-b', label="orig.")    
plt.plot(X_rec[11],'-r', label="rec.")
plt.legend()

plt.figure(3)
plt.title("comparison of measurements and reconstructed signal")
plt.plot(Y[0], '-b', label='orig.')
plt.plot(Y_rec[0], '-r', label='rec.')
#plt.legend()



