# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:24:55 2019

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation
from sklearn.metrics import mean_squared_error
import CovDL
import MSBL


np.random.seed(1)

# choose datageneration method ... 

""" DATA GENERATION - MIX OF DETERMINISTIC SIGNALS """

m = 8                         # number of sensors
n = 50                        # number of sources
k = 4                         # max number of non-zero coef. in rows of X
L = 10000                     # number of sampels


Y_real, A_real, X_real = data_generation.mix_signals(L, 10, m, n, k)



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

for i in range(len(Ys)): # loop over segments 
    Y_real = Ys[i]
    X_real = Xs[i]
    
    cov_seg = 100
    
    if n <= (m*(m+1))/2.:
        raise SystemExit('D is over-determined')
# input        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
        
    elif k <= (m*(m+1))/2.:
        A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
    
    elif k > (m*(m+1))/2.:
        raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
        
     
    X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=1000, noise=False)
    
    mse = mean_squared_error(X_real, X_rec)
    print("Representation error (without noise): ", mse)   
    
    

""" PLOTS """
 
plt.figure(1)
plt.subplot(4, 1, 1)
plt.title('Comparison of each active source in X and corresponding reconstruction ')
plt.plot(X_real[0], 'r',label='Real X')
plt.plot(X_rec[0],'g', label='Recovered X')

plt.subplot(4, 1, 2)
plt.plot(X_real[1], 'r',label='Real X')
plt.plot(X_rec[1],'g', label='Recovered X')

plt.subplot(4, 1, 3)
plt.plot(X_real[15], 'r',label='Real X')
plt.plot(X_rec[15],'g', label='Recovered X')

plt.subplot(4, 1, 4)
plt.plot(X_real[16], 'r',label='Real X')
plt.plot(X_rec[16],'g', label='Recovered X')
plt.legend()
plt.show
#plt.savefig('case1_1.png')





