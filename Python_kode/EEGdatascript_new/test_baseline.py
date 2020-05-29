# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:14:19 2020

@author: trine
"""

import matplotlib.pyplot as plt
import simulated_data
import numpy as np
import ICA
import data
np.random.seed(321)

def A_ICA(Y, request):
    if len(Y.shape) is 3:
        X_ica, A = ICA.ica_segments(Y, 1000)
    if len(Y.shape) is 2:
        X_ica, A = ICA.ica(Y, 1000)
    
    A_r = data._reduction(A, request)   
    
    return A_r

def Main_Algorithm_test(N, k, Y, M, L, n_seg, A_real, L_covseg = 10): ## OBS remove A_real as input 
    """
    input:
        L -> the length of one segments
    """
    import numpy as np
    import M_SBL

    #################################  Cov-DL  ####################################

#    A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
    X_result = np.zeros((n_seg, N, L-2))
    
    A_fix = np.random.normal(0,2,(M,N))
    
    for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
        print(i)

        X_rec = M_SBL.M_SBL(A_fix, Y[i], M, N, k, iterations=1000, noise=False)
        print('efter X')
        X_result[i] = X_rec

    return A_fix, X_result

##################### The test with vary specifications ######################
M = 8 

""" List for variation of N """
#list_ = np.arange(M+1,36+1,2)  # vary N 
list_ = np.array([M+1])

average = 500  # number og times each systems is solved, before the average is computed
list_A = np.zeros(len(list_))
list_X = np.zeros(len(list_))
for j in range(len(list_)): 
    N = list_[j]
    k = N
    L = 1000
    n_seg = 1
    print('list_ value = {}'.format(N))
    
    avg_A = np.zeros(average)
    avg_X = np.zeros(average)
    for p in range(average):
        print(p)
        Y, A_real, X_real = simulated_data.generate_AR(N, M, L, k)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        A_result, X_result = Main_Algorithm_test(N, k, Y, M, L, n_seg, A_real, L_covseg=10)
        
        mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
        A_mse = simulated_data.MSE_one_error(A_real,A_result)
            
        avg_A[p] = A_mse
        avg_X[p] = mse_avg
   
    list_A[j] = np.mean(avg_A)
    list_X[j] = np.mean(avg_X) 
#    list_A[j] = A_mse
#    list_X[j] = mse_array

plt.figure(3)
plt.title(r'$MSE(\mathbf{X},\hat{\mathbf{X}})$ for varying N - using $\hat{\mathbf{A}}_{norm2}$')
plt.plot( list_, list_X, '-ob')
plt.xlabel('N')
plt.ylabel('MSE')
plt.show()
plt.savefig('figures/varyN2.png')
#
#plt.figure(4)
#plt.subplot(1, 2, 1)
#plt.title('Outliers included')
#plt.boxplot(avg_X,labels=['$N = M+1$'])
#plt.ylabel('MSE')
#plt.subplot(1, 2, 2)
#plt.title('Outliers excluded')
#plt.boxplot(avg_X,showfliers=False,labels=['$N = M+1$'])
#plt.ylabel('MSE')
#plt.tight_layout()
