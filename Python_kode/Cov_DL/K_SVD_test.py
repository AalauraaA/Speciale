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


from dictionary_learning import K_SVD
np.random.seed(1)

# INITIALISING PARAMETERS
m = 5               # number of sensors
n = 8               # number of sources
non_zero = 4       # max number of non-zero coef. in rows of X
n_samples = 100      # number of sampels

# RANDOM GENERATION OF SPARSE DATA
#Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, 8)
#m = len(Y)
#n = len(X_real)
#non_zero = 4

#Y, A_real, X_real = data_generation.random_sparse_data(m, n, non_zero, n_samples)

Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, m)
n = len(X_real)
m = len(Y)
non_zero = 4

### PERFORM DICTIONARY LEARNING
A, X, iter_= K_SVD(Y, n, m, non_zero, n_samples, max_iter=1000)

Y_rec = np.matmul(A,X)

Y_err = (np.square(Y-Y_rec)).mean(axis=None)
A_err = (np.square(A_real-A)).mean(axis=None)
X_err = (np.square(X_real-X)).mean(axis=None)

#
print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f, \nnumber of iterations %i'%(Y_err, A_err, X_err, iter_))
plt.figure(1)
plt.plot(Y.T)
plt.figure(2)
plt.plot(Y_rec.T)

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

#X_D, A_D, err, _iter = dict_learning(Y.T,n,4,return_n_iter=True, random_state=1)  
#Y_rec = np.matmul(A_D,X_D.T)
#
#Y_err = np.linalg.norm(Y-Y_rec)
#A_err = np.linalg.norm(A_real-A_D)
#X_err = np.linalg.norm(X_real-X_D.T)
#
#print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f, \nnumber of iterations %i'%(Y_err, A_err, X_err, _iter))


# toy eksempel 
#A_real = np.matrix([[2,4,2,1],[2,4,1,1],[4,2,1,1]])
#X_real = np.array([0, 0, 3, 2])
#Y = np.matmul(A_real,X_real)

temp = DictionaryLearning(n,non_zero)
A = temp.fit(Y.T).components_
X = temp.transform(Y.T)

#A = np.zeros([n,m])

Y_rec = np.matmul(A.T,X.T)

Y_err = (np.square(Y-Y_rec)).mean(axis=None)
A_err = (np.square(A_real-A.T)).mean(axis=None)
X_err = (np.square(X_real-X.T)).mean(axis=None)

print('reconstruction error %f,\ndictionary error %f,\nrepresentation error %f'%(Y_err, A_err, X_err))

plt.figure(5)
plt.plot(Y.T)
plt.figure(6)
plt.plot(Y_rec.T)