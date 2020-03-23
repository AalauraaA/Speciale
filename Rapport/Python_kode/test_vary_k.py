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

""" Lists for variation """
list_ = [7, 10, 12, 14, 16]  # k vary


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
        k_true = list_[l]            # true number of non-zeros
    
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
            cov_seg = 10
        
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
            err_listA[ite] = data_generation.MSE_one_error(A_real, A_rec)
            err_listX[ite] = data_generation.MSE_one_error(X_real, X_rec)
    
    errA[l] = np.average(err_listA)
    errX[l] = np.average(err_listX)
    

""" PLOTS """
#plt.figure(1)
#plt.plot(list_, errX)
#plt.title('X error- vary k, n = 16, m = 8, true k = k')
#plt.xlabel('k')
#plt.ylabel('Norm. mse of recovered X')
#plt.savefig('Resultater/X_varying_k_n16_m8_L1000_covseg10.png')
#plt.show()
#
#plt.figure(2)
#plt.plot(list_, errA)
#plt.title('A error- vary k, n = 16, m = 8, true k = k')
#plt.xlabel('k')
#plt.ylabel('Norm. mse of recovered A')
#plt.savefig('Resultater/A_varying_k_n16_m8_L1000_covseg10.png')
#plt.show()


plt.figure(2)
plt.title('m = 8, n = 16, k = 7, true k = 7, L = 1000, covseg = 10')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[i]!=0) or np.any(X_rec[i]!=0):
        
        nr_plot += 1
        plt.subplot(k*2, 1, nr_plot)
       
        plt.plot(X_real[i], 'r',label='Real X')
        plt.plot(X_rec[i],'g', label='Recovered X')

       
plt.legend()
plt.xlabel('sample')
plt.show
#plt.savefig('13-03-2020_3.png')

