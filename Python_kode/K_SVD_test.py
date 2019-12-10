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

# vary parameters
list_ = np.arange(10,40,2)
#list_ = np.array([20])
err_listA = np.zeros((len(list_),2))
for i in range(len(list_)): 
    print(i)
    # INITIALISING PARAMETERS
    m = list_[i]              # number of sensors
    n = 50               # number of sources
    k = 25       # max number of non-zero coef. in rows of X
    L = 1000      # number of sampels
    
    # RANDOM GENERATION OF SPARSE DATA
    #Y, A_real, X_real = data_generation.mix_signals(n_samples, 10, 8)
    #m = len(Y)
    #n = len(X_real)
    #non_zero = 4
    
    #Y, A, X = data_generation.random_sparse_data(m, n, k, L)
     
    Y, A, X = data_generation.mix_signals(L, 10, m, n, k)
    n = len(X)
    m = len(Y)
    
    ### PERFORM DICTIONARY LEARNING
    A_rec, X_rec, iter_= K_SVD(Y, n, m, k, L, max_iter=1000)
    Y_rec = np.dot(A_rec,X_rec)
    
    A_err = mean_squared_error(A,A_rec)
    X_err = mean_squared_error(X,X_rec)
    Y_err = mean_squared_error(Y,Y_rec)
    
    #
    print('ksvd reconstruction error %f,\ndictionary error %f,\nrepresentation error %f'%(Y_err, A_err, X_err))
    err_listA[i,0] = A_err
    
    Y = Y.T
    Y_rec, A_rec, X_rec = DL(Y,n,k)
    
    #A_rec = np.zeros(np.shape(A))
    
    A_err_sk = mean_squared_error(A,A_rec)
    X_err_sk = mean_squared_error(X,X_rec)
    Y_err_sk = mean_squared_error(Y,Y_rec)
    
    #
    print('sklearn reconstruction error %f,\ndictionary error %f,\nrepresentation error %f'%(Y_err_sk, A_err_sk, X_err_sk))
    err_listA[i,1] = A_err_sk

#plt.figure(2)
#plt.title("comparison of source signals (column)")
#plt.plot(X[11],'-b', label="orig.")    
#plt.plot(X_rec[11],'-r', label="rec.")
#plt.legend()

#plot mse of A 
plt.figure(1)
plt.plot(list_,err_listA.T[0], '-r', label='k-svd error')
plt.plot(list_,err_listA.T[1], '-b', label='sklearn error')
plt.title('Varying M - k = 30, N = 50, L = 1000, iteration = 500')
plt.xlabel('non-zeros')
plt.ylabel('MSE')
plt.legend()
plt.savefig('Resultater/Dictionary Learning/comparison/varyM_mix.png')
plt.show()



