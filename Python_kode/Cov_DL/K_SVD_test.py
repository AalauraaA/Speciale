# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:49:13 2019

@author: trine
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import dict_learning 
import matplotlib.pyplot as plt
from scipy import signal
import data_generation


from dictionary_learning import K_SVD
np.random.seed(1)

# INITIALISING PARAMETERS
m = 4               # number of sensors
n = 4               # number of sources
non_zero = 4        # max number of non-zero coef. in rows of X
n_samples = 20      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
#Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, 8)
#m = len(Y)
#n = len(X_real)
#non_zero = 4

Y, A_real, X_real = data_generation.random_sparse_data(m, n, non_zero, n_samples)


## PERFORM DICTIONARY LEARNING
A, X, iter_= K_SVD(Y, n, m, non_zero, n_samples, max_iter=1000)

Y_rec = np.matmul(A,X)

Y_err = np.linalg.norm(Y-Y_rec)
A_err = np.linalg.norm(A_real-A)
X_err = np.linalg.norm(X_real-X)


print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f, \nnumber of iterations %i'%(Y_err, A_err, X_err, iter_))
#
#plt.figure(1)
#plt.title("comparison of source signals (column)")
#plt.plot(X_real.T[3].T,'-b', label="orig.")    
#plt.plot(X.T[3].T,'-r', label="rec.")
#plt.legend()
#
#plt.figure(2)
#plt.title("comparison of measurements and reconstructed signal")
#plt.plot(Y[0], '-b', label='orig.')
#plt.plot(Y_rec[0], '-r', label='rec.')
##plt.legend()
#

X_D, A_D, err, _iter = dict_learning(Y.T,n,4,return_n_iter=True, random_state=1)  
Y_rec = np.matmul(A_D,X_D)

Y_err = np.linalg.norm(Y-Y_rec)
A_err = np.linalg.norm(A_real-A_D)
X_err = np.linalg.norm(X_real-X_D)

print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f, \nnumber of iterations %i'%(Y_err, A_err, X_err, _iter))


# toy eksempel 
A = np.matrix([[2,4,2],[2,4,1],[4,2,1]])


