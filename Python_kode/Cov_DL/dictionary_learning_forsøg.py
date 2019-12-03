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
#
#A =np.array([[1,2,3,4],
#  [6,7,8,9],
#  [10,9,8,4],])
#x = np.array([[1,0,1,0],[2,0,3,0]])
#y = np.matmul(A,x.T) 

k_list = np.array([10])
err_listA = np.zeros(len(k_list))
for i in range(len(k_list)):
    print(k_list[i])
    Y, A, X = make_sparse_coded_signal(n_samples=50,
                                       n_components=50,
                                       n_features=25,
                                       n_nonzero_coefs=k_list[i],
                                       random_state=0)
    
    
    Y = Y.T
    
    def test_DictionaryLearning(Y,n_com,non_zero):
        '''
        test the DL method
        :return: None
        '''
        #print("before transform:",Y)
        dct=DictionaryLearning(n_components=n_com,transform_algorithm='omp',transform_n_nonzero_coefs=non_zero)
        dct.fit(Y)
#        print("components is :",dct.components_)
#        print("after transform:",dct.transform(Y)) 
        return dct
    
    dic = test_DictionaryLearning(Y,50,k_list[i])
    
    A_new = dic.components_
    #A_new = np.zeros(np.shape(A)).T
    X_new = dic.transform(Y)
    Y_new = np.matmul(A_new.T,X_new.T)
    
    err_A = mean_squared_error(A,A_new.T)
    err_X = mean_squared_error(X,X_new.T)
    err_Y = mean_squared_error(Y.T,Y_new)
#     
    plt.figure(1)
    plt.plot(Y.T[0])
    #plt.figure(2)
    plt.plot(Y_new[0])
    
    err_listA[i] = err_A
#
#plt.figure(2)
#plt.plot(k_list,err_listA)


    



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