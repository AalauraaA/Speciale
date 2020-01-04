# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:04:55 2020

@author: trine
"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation
from sklearn.metrics import mean_squared_error
import CovDL
import MSBL

np.random.seed(1)

""" Lists for variation """
#list_ = np.array([6]) # for a single konstant
list_ = np.arange(5,40,5)   # k vary
#list_ = np.arange(15,60+1,5)  # n vary
#list_ = np.arange(4,32+1,4)   # m vary

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))
err_listY = np.zeros(len(list_))

for i in range(len(list_)):
    print(list_[i]) 
    L = 1000
    
    " Case 1 "
    k = list_[i]
    m = 16
    n = 40
   
    " Case 3 "
#    n = list_[i]        # Case 3
#    m = 16
#    k = int((list_[i]/3)*2)  # Case 3
  #  k = 4        # Case 3
  #  k = 20       # Case 3
    
    " Case 4 "
#    m = list_[i]        # Case 4
#    n = 64       # Case 4
#    k = 20       # Case 4

    
    """ Generate AR data and Dividing in Segments """
    Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k) 
    
    Ls = 100              # number of sampels per segment (L -> no segmentation) 
    Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0
    
    sum_X = 0
    for k in range(len(Ys)): # loop over segments 
        Y_real = Ys[k]
        X_real = Xs[k]
    
        cov_seg = 100
#        
#        if n <= (m*(m+1))/2.:
#            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
#          #  raise SystemExit('D is over-determined')
## input later        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
#            
#        elif k <= (m*(m+1))/2.:
#            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
#        
#        elif k > (m*(m+1))/2.:
#            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
#           # raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
#        
#     
        X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=500, noise=False)
        print(X_rec)
        print(np.count_nonzero(X_rec))
        sum_X += data_generation.norm_mse(X_real, X_rec)
        print(sum_X)
    
    
    avg_err_X = sum_X/len(Ys) 
    print(avg_err_X)
    
#    
#    err_listA[i] = err_A
    err_listX[i] = avg_err_X

# print - change title ! 
    
plt.figure(1)

plt.plot(list_, err_listX)
plt.title('X error- vary k, N = 40, M = 16, Ls = 100, n_seg = 10')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered X')
plt.savefig('Resultater/X_varying_k.png')
plt.show()

