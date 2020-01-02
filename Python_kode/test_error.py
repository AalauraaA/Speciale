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


# Lists for variation 
#list_ = np.arange(1,10,1)
list_ = np.array([6]) # for a single konstant

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))
for i in list_:
    print(i)
    
    m = 6
    n = 10
    k = i
    L = 100
    
    # generate AR data 
    Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k) 
    
    Ls = L                  # number of sampels per segment (L -> no segmentation) 
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
        
     
        X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=1000, noise=False)
        
        mse_X = mean_squared_error (X_real, X_rec)
    
    #Y_rec = np.matmul(A_rec,X_rec)
    
    err_A = mean_squared_error(A_real,A_rec.T)
    err_X = mean_squared_error(X_real,X_rec.T)
    #err_Y = mean_squared_error(Y_real.T,Y_rec)
    
    err_listA[i] = err_A