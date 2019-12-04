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


# vary parameters
#list_ = np.arange(2,60,2)
list_ = np.array([6])
err_listA = np.zeros(len(list_))
for i in range(len(list_)):
    print(list_[i])
    
    m = 6
    n = 10
    k = list_[i]
    L = 100
    
    # import data
#    Y, A, X = make_sparse_coded_signal(n_samples=L,
#                                       n_components=n,
#                                       n_features=m,
#                                       n_nonzero_coefs=k,
#                                       random_state=0)
    
    #Y, A, X = data_generation.rossler_data(L, 1, m )[:3]
    
    Y, A, X = data_generation.mix_signals(L, 10, m)
    n = len(X)
    m = len(Y)

    
    Y = Y.T
    
    def test_DictionaryLearning(Y,n_com,non_zero):
        '''
        test the DL method
        :return: None
        '''
        dct=DictionaryLearning(n_components=n_com,transform_algorithm='omp',transform_n_nonzero_coefs=non_zero, max_iter=500)
        dct.fit(Y)
        return dct
    
    dic = test_DictionaryLearning(Y,n,k)
    
    A_new = dic.components_
    #A_new = np.zeros(np.shape(A)).T
    X_new = dic.transform(Y)
    Y_new = np.matmul(A_new.T,X_new.T)
    
    err_A = mean_squared_error(A,A_new.T)
    err_X = mean_squared_error(X,X_new.T)
    err_Y = mean_squared_error(Y.T,Y_new)
#     
    
    
    err_listA[i] = err_A
#
    
#plt.figure(1)
#    plt.plot(Y.T[0])
#    #plt.figure(2)
#    plt.plot(Y_new[0])

plt.figure(1)
plt.plot(list_,err_listA)
plt.title('Varying k - M = 50, N = 100, L = 100, iteration = 500')
plt.xlabel('non-zeros')
plt.ylabel('MSE')
plt.savefig('Resultater/K-SVD/3.png')
plt.show()
    



#def test_DictionaryLearning():
#    '''
#    test the DL method
#    :return: None
#    '''
#    X=np.array([[1,2,3,4,5],
#       [6,7,8,9,10],
#       [10,9,8,7,6,],
#       [5,4,3,2,1] ])
#    print("before transform:",X)
#    dct=DictionaryLearning(n_components=3)
#    dct.fit(X)
#    print("components is :",dct.components_)
#    print("after transform:",dct.transform(X)) 
#    return dct,X
#dic,X = test_DictionaryLearning()
#
#X_new = dic.transform(X)
#A_new = dic.components_
#
#err_Y = mean_squared_error(X.T,np.matmul(A_new.T,X_new.T))
#
#    