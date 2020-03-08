# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation

import CovDL
import MSBL


#np.random.seed(154)

""" Lists for variation """
list_ = np.arange(5,16,2)  # k vary


err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))
errA = np.zeros(len(list_))
errX = np.zeros(len(list_))

for l in range(len(list_)):
    for ite in range(len(list_)):
        print(l)
        m = 8                        # number of sensors
        n = 16                       # number of sources
        k = list_[l]                 # max number of non-zeros
        L = 1000                     # number of sampels
        k_true = list_[l] + 1        # true number of non-zeros
    
        Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 
        
        """ SEGMENTATION - OVER ALL """
        
        Ls = L                  # number of sampels per segment (L -> no segmentation) 
        
        Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                                # return list of arrays -> segments in axis = 0
        
        """ COV-DL and M-SBL """
        for i in range(len(Ys)): # loop over segments 
            Y_real = Ys[i]
            X_real = Xs[i]
           
            def cov_seg_max(n, L):
                """
                Give us the maximum number of segment within the margin.
                """
                n_seg = 1
                while int(n) > n_seg:
                    n_seg += 1
                return int(L/n_seg)    # Samples within one segment   
           
            
#            cov_seg = cov_seg_max(n, L)
            cov_seg = 30
        
            """ Cov-DL """
            if n <= (m*(m+1))/2.:
                A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
                
            elif k <= (m*(m+1))/2.:
                A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
            
            elif k > (m*(m+1))/2.:
                raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
                
            """ M-SBL """
            X_rec = MSBL.M_SBL(A_rec, Y_real, m, n, Ls, k, iterations=1000, noise=False)
            X_real = X_real.T[:-2]
            X_real = X_real.T
            
            """ Calculating errors between true and estimated A and X """
            err_listA[ite] = A_err
            err_listX[ite] = data_generation.norm_mse(X_real, X_rec)
    
    errA[l] = np.average(err_listA)
    errX[l] = np.average(err_listX)
    

""" PLOTS """
plt.figure(1)
plt.plot(list_, errX)
plt.title('X error- vary k, n = 16, m = 8, true k = k+1')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered X')
plt.savefig('Resultater/X_varying_k_lower_n16_m8_L1000_covseg30.png')
plt.show()

plt.figure(2)
plt.plot(list_, errA)
plt.title('A error- vary k, n = 16, m = 8, true k = k+1')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered A')
plt.savefig('Resultater/A_varying_k_lower_n16_m8_L1000_covseg30.png')
plt.show()



