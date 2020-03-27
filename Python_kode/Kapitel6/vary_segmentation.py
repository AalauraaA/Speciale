# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura

Testing different samples sizes and number of segments
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation

import CovDL
import MSBL

""" DATA GENERATION """
m = 8                          # number of sensors
k = 16                         # number of active sources
L = 1000                       # number of sampels
duration = 10

list_ = [10, 20, 30, 40, 50, 60]

err_listA = np.zeros(10)
err_listX = np.zeros(10)

Amse = np.zeros(len(list_))
Xmse = np.zeros(len(list_))

for s in range(len(list_)):
    print(list_[s])
    for ite in range(10):
        print(ite)
        
#        Y_real, A_real, X_real = data_generation.mix_signals(L, duration, m, k, k)
        Y_real, A_real, X_real = data_generation.generate_AR_v2(k, m, L, k)

        """ SEGMENTATION - OVER ALL """       
        Ls = L                  # number of sampels per segment (L -> no segmentation) 
        
        Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                                # return list of arrays -> segments in axis = 0        
        """ COV-DL and M-SBL """
        for i in range(len(Ys)): # loop over segments 
            Y_real = Ys[i]
            X_real = Xs[i]
            
            cov_seg = list_[s]
        
            """ Cov-DL """
            if k <= (m*(m+1))/2.:
                A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, k, cov_seg, L, k)
                
            elif k <= (m*(m+1))/2.:
                A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, k, cov_seg, L, k)
            
            elif k > (m*(m+1))/2.:
                raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
                
            """ M-SBL """
            X_rec = MSBL.M_SBL(A_rec, Y_real, m, k, Ls, k, iterations=1000, noise=False)
            X_real = X_real.T[:-2]
            X_real = X_real.T
            
            """ Calculating errors between true and estimated A and X """
            err_listA[ite] = A_err
            err_listX[ite] = data_generation.MSE_one_error(X_real, X_rec)
    
    Amse[s] = np.average(err_listA)
    Xmse[s] = np.average(err_listX)
    

""" PLOTS """
plt.figure(1)
plt.plot(Amse, '-r', label = 'A')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')
plt.plot(2, Amse[2], 'ro')
plt.plot(3, Amse[3], 'ro')
plt.plot(4, Amse[4], 'ro')
plt.plot(5, Amse[5], 'ro')


plt.plot(Xmse, '-b', label = 'X')
plt.plot(0, Xmse[0], 'bo')
plt.plot(1, Xmse[1], 'bo')
plt.plot(2, Xmse[2], 'bo')
plt.plot(3, Xmse[3], 'bo')
plt.plot(4, Xmse[4], 'bo')
plt.plot(5, Xmse[5], 'bo')


plt.title('MSE of A and X for variyng segments')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/AR_Error_vary_seg_m8_k16_L1000.png')
plt.show()
