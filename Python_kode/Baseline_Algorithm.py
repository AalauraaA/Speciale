# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:24:55 2019

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation
import CovDL
import MSBL


np.random.seed(123)

# choose datageneration method ...
""" DATA GENERATION - AUTO-REGRESSIVE SIGNAL """

m = 8                         # number of sensors
n = 16                         # number of sources
k = 16                         # max number of non-zero coef. in rows of X
L = 1000                 # number of sampels
k_true = 16 

Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 

""" DATA GENERATION - MIX OF DETERMINISTIC SIGNALS """
#m = 6                         # number of sensors
#n = 6                         # number of sources
#k = 6                         # max number of non-zero coef. in rows of X
#L = 100                       # number of sampels
#k_true = 6
#
#Y_real, A_real, X_real = data_generation.mix_signals(L, 10, m, n, k_true)

#""" DATA GENERATION - ROSSLER DATA """
#
#L = 1940                        # number of sampels, max value is 1940
#
#Y_real, A_real, X_real, k = data_generation.rossler_data(n_sampels=1940, 
#                                                         ex = 1, m=8)
#m = len(Y_real)                      # number of sensors
#n = len(X_real)                      # number of sources
#k = np.count_nonzero(X_real.T[0])    # number of non-zero coef. in rows of X


""" SEGMENTATION - OVER ALL """
#
Ls = L                  # number of sampels per segment (L -> no segmentation) 
Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0

""" COV - DL and M-SBL """
#
for i in range(len(Ys)): # loop over segments 
    Y_real = Ys[i]
    X_real = Xs[i]
   
    def cov_seg_max(n, L):
        """
        Give us the maximum number of segment within the margin.
        For some parameters (low) you can add one more segment.
        """
        n_seg = 1
        while int(n) > n_seg:
            n_seg += 1
        return int(L/n_seg)    # Samples within one segment   
    

#    cov_seg = cov_seg_max(n,L)
    cov_seg = 10


   
    if n <= (m*(m+1))/2.:
        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
        
    elif k <= (m*(m+1))/2.:
        A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
    
    elif k > (m*(m+1))/2.:
        raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
        
     
    X_rec = MSBL.M_SBL(A_rec, Y_real, m, n, Ls, k, iterations=1000, noise=False)
    X_real = X_real.T[:-2]
    X_real = X_real.T
    
    Amse = data_generation.MSE_one_error(A_real, A_rec)
    print("Representation error (without noise) for A: ", Amse)  
    print("Representation error (without noise) for A: ", A_err)
    
    Xmse = data_generation.MSE_one_error(X_real, X_rec)
    print("Representation error (without noise) for X: ", Xmse) 

""" PLOTS """
 
#plt.figure(2)
#plt.title('m = 8, n = 16, k = 7, true k = 7, L = 1000, covseg = 10')
#nr_plot=0
#for i in range(len(X_real.T[0])):
#    if np.any(X_real[i]!=0) or np.any(X_rec[i]!=0):
#        
#        nr_plot += 1
#        plt.subplot(k*2, 1, nr_plot)
#       
#        plt.plot(X_real[i], 'r',label='Real X')
#        plt.plot(X_rec[i],'g', label='Recovered X')
#
#       
#plt.legend()
#plt.xlabel('sample')
#plt.show
#plt.savefig('13-03-2020_3.png')
