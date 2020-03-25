# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:19:33 2019

@author: trine
"""
from sklearn.decomposition import DictionaryLearning
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import data_generation
import dictionary_learning
np.random.seed(1)
  
def test_DictionaryLearning(Y,n_com,non_zero):
        '''
        test the DL method
        :return: None
        '''
        dct = DictionaryLearning(n_components=n_com,transform_algorithm='omp',transform_n_nonzero_coefs=non_zero, max_iter=1000)
        dct.fit(Y)
        return dct

#list_ = np.array([2]) # for a single konstant
list_ = np.arange(2,10,1)   # k vary
#list_ = np.arange(15,60+1,5)  # n vary
#list_ = np.arange(4,32+1,4)   # m vary

err_listA = np.zeros(len(list_))

for i in range(len(list_)):
    print(list_[i]) 
    L = 1000
  
    m = 10
    n = 10
    k = list_[i]

    
    
    """ Generate AR data and Dividing in Segments """
    Y, A, X = data_generation.generate_AR_v2(n, m, L, k) 
    
    Ls = 100             # number of sampels per segment (L -> no segmentation) 
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, Ls, L)
                          # return list of arrays -> segments in axis = 0
    
    sum_A = 0
    for j in range(len(Ys)):
        Y = Ys[j]
        X = Xs[j]
        
        Y_new, A_new, X_new = dictionary_learning.DL(Y.T,n,k,iter_=500)
        
        sum_A += data_generation.norm_mse(A,A_new)
    
    avg_err_A = sum_A/len(Ys) 
    err_listA[i] = avg_err_A

plt.figure(1)
plt.plot(list_,err_listA)
plt.title('Varying k - M = 10, N = 10, Ls = 100, L = 1000, iter = 500 ')
plt.xlabel('k')
plt.ylabel('norm MSE of dictionary')
plt.savefig('Resultater/Dictionary_1_omp.png')
plt.show()
    
