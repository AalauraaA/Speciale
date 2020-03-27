# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:24:55 2019

@author: Mattek10
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation
import CovDL
import MSBL


np.random.seed(4523)

""" DATA GENERATION - MIX OF DETERMINISTIC SIGNALS """
m = 3                         # number of sensors
n = 4                         # number of sources
k = 4                        # max number of non-zero coef. in rows of X
L = 100                       # number of sampels
k_true = 4

Y_real, A_real, X_real = data_generation.mix_signals(L, 10, m, n, k_true)

""" SEGMENTATION - OVER ALL """
Ls = L                  # number of sampels per segment (L -> no segmentation) 
Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0

""" COV - DL and M-SBL """
for i in range(len(Ys)): # loop over segments 
    Y_real = Ys[i]
    X_real = Xs[i] 

    cov_seg = 10
 
    if n <= (m*(m+1))/2.:
        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
        
    elif k <= (m*(m+1))/2.:
        A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
    
    elif k > (m*(m+1))/2.:
        raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
        
     
    X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=1000, noise=False)
    X_real = X_real.T[:-2]
    X_real = X_real.T
 
    print("Representation error (without noise) for A: ", A_err)
    Xmse = data_generation.MSE_one_error(X_real, X_rec)
    print("Representation error (without noise) for X: ", Xmse) 

""" PLOTS """
plt.figure(1)
plt.title('m = 3, k = 4, L = 100, covseg = 10')
nr_plot = 0
for i in range(len(X_real.T[0])):
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        plt.plot(X_real[i], 'r',label='Real X')
        plt.plot(X_rec[i],'g', label='Recovered X')
      
plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('figures/test_of_algo_mix_data_realA.png')

