# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura

Testing different samples sizes and number of segments -- small system
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation

import CovDL
import MSBL

""" Lists for variation """
#list_ = [62, 30, 10]      # for L = 1000
#list_ = [156, 75, 10]    # for L = 2500
list_ = [312, 150, 10]      # for L = 5000

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))
errA = np.zeros(len(list_))
errX = np.zeros(len(list_))

for l in range(len(list_)):
    for ite in range(len(list_)):
        print(l)
        m = 8             # number of sensors
        n = 16             # number of sources
        k = 16             # max number of non-zeros
#        L = 1000          # number of sampels
#        L = 2500          # number of sampels
        L = 5000          # number of sampels        
        k_true = 16        # true number of non-zeros
    
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
           
            
            cov_seg = list_[l]
        
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
plt.title('X error- vary cov_seg, n = 16, m = 8, true k = 16, L = 5000')
plt.xlabel('cov_seg')
plt.ylabel('Norm. mse of recovered X')
#plt.savefig('Resultater/X_varying_covseg_L1000_n16_m8_k16.png')
#plt.savefig('Resultater/X_varying_covseg_L2500_n16_m8_k16.png')
plt.savefig('Resultater/X_varying_covseg_L5000_n16_m8_k16.png')
plt.show()

plt.figure(2)
plt.plot(list_, errA)
plt.title('A error- vary cov_seg, n = 16, m = 8, true k = 16, L = 5000')
plt.xlabel('cov_seg')
plt.ylabel('Norm. mse of recovered A')
#plt.savefig('Resultater/A_varying_covseg_L1000_n16_m8_k16.png')
#plt.savefig('Resultater/A_varying_covseg_L2500_n16_m8_k16.png')
plt.savefig('Resultater/A_varying_covseg_L5000_n16_m8_k16.png')
plt.show()
