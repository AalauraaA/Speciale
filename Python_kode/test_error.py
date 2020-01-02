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
list_ = range(1,50+1)   # k vary
#list_ = range(16,60+1)  # n vary
#list_ = range(4,32)   # m vary

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))

for i in list_:
    print(i) 
    L = 1000
    
    " Case 1 "
    k = i
    m = 32
    n = 64

    " Case 2 "
#    k = i
#    m = 32
#    n = 64
#    
    " Case 3 "
#    n = i        # Case 3
#    m = 16
#    k = (i/3)*2  # Case 3
#    k = 4        # Case 3
#    k = 20       # Case 3
    
    " Case 4 "
#    m = i        # Case 4
#    n = 64       # Case 4
#    k = 20       # Case 4

    
    """ Generate AR data and Dividing in Segments """
    Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k) 
    
    Ls = 100              # number of sampels per segment (L -> no segmentation) 
    Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0
    
    
    for i in range(len(Ys)): # loop over segments 
        Y_real = Ys[i]
        X_real = Xs[i]
    
        cov_seg = 100
        
        if n <= (m*(m+1))/2.:
            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
          #  raise SystemExit('D is over-determined')
# input later        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
            
        elif k <= (m*(m+1))/2.:
            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
        
        elif k > (m*(m+1))/2.:
            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
           # raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
#        
#     
#        X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=1000, noise=False)
#        
#        mse_X = mean_squared_error (X_real, X_rec)
#    
#    #Y_rec = np.matmul(A_rec,X_rec)
#    
#    err_A = mean_squared_error(A_real,A_rec.T)
#    err_X = mean_squared_error(X_real,X_rec.T)
#    #err_Y = mean_squared_error(Y_real.T,Y_rec)
#    
#    err_listA[i] = err_A
#    err_listX[i] = err_X