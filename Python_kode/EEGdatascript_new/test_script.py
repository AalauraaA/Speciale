# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:14:19 2020

@author: trine
"""

import matplotlib.pyplot as plt
import simulated_data
import numpy as np
np.random.seed(321)

def Main_Algorithm_test(N, k, Y, M, L, n_seg, A_real, L_covseg = 10): ## OBS remove A_real as input 
    """
    input:
        L -> the length of one segments
    """
    import numpy as np
    import Cov_DL
    import M_SBL

    #################################  Cov-DL  ####################################

    A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
    X_result = np.zeros((n_seg, N, L-2))
    
    for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
        print(i)
        Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain
        print('shape of Y_big{}'.format(Y_big.shape))
        if N <= (M*(M+1))/2.:
            A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
            A_result[i] = A_rec

        elif k <= (M*(M+1))/2.:
            A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
            A_result[i] = A_rec

        elif k > (M*(M+1))/2.:
            raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')

    #################################  M-SBL  #################################

        X_rec = M_SBL.M_SBL(A_real, Y[i], M, N, k, iterations=1000, noise=False) #### OBS CHECK for A_rec
        print('efter X')
        X_result[i] = X_rec


    return A_result, X_result

##################### The test with vary specifications ######################
M = 8 

""" List for variation og N """
list_ = np.arange(M+1,36+1,2)  # vary N 

average = 5  # number og times each systems is solved, before the average is computed
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
        
        mse_array, mse_avg = simulated_data.MSE_segments(X_real,X_result)
        A_mse = simulated_data.MSE_one_error(A_real,A_result[0])
            
        avg_A[p] = A_mse
        avg_X[p] = mse_avg
   
    list_A[j] = np.mean(avg_A)
    list_X[j] = np.mean(avg_X) 
    list_A[j] = A_mse
    list_X[j] = mse_array

plt.figure(3)
plt.title(r'MSE of $\mathbf{X}$ for varying N - using true $\mathbf{A}$')
plt.plot( list_, list_X, '-ob')
plt.xlabel('N')
plt.ylabel('MSE of X')
plt.show()
plt.savefig('figures/varyN_trueA.png')


###############################
    
def Main_Algorithm_test(N, k, Y, M, L, n_seg, A_real, L_covseg = 10): ## OBS remove A_real as input 
    """
    input:
        L -> the length of one segments
    """
    import numpy as np
    import Cov_DL
    import M_SBL

    #################################  Cov-DL  ####################################

    A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
    X_result = np.zeros((n_seg, N, L-2))
    
    for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
        print(i)
        Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain
        print('shape of Y_big{}'.format(Y_big.shape))
        if N <= (M*(M+1))/2.:
            A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
            A_result[i] = A_rec

        elif k <= (M*(M+1))/2.:
            A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
            A_result[i] = A_rec

        elif k > (M*(M+1))/2.:
            raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')

    #################################  M-SBL  #################################

        X_rec = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False) #### OBS CHECK for A_rec
        print('efter X')
        X_result[i] = X_rec


    return A_result, X_result

##################### The test with vary specifications ######################
M = 8 
np.random.seed(435)
""" List for variation og N """
list_ = np.arange(M+1,36+1,2)  # vary N 

average = 5  # number og times each systems is solved, before the average is computed
list_A = np.zeros(len(list_))
list_X2 = np.zeros(len(list_))
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
        
        mse_array, mse_avg = simulated_data.MSE_segments(X_real,X_result)
        A_mse = simulated_data.MSE_one_error(A_real,A_result[0])
        
        avg_A[p] = A_mse
        avg_X[p] = mse_avg
   
    list_A[j] = np.mean(avg_A)
    list_X2[j] = np.mean(avg_X) 

plt.figure(5)
plt.title('MSE of $\mathbf{A}$ and $\mathbf{X}$ for varying N')
plt.plot( list_, list_A,'-or', label=r'$\mathbf{A}_{MSE}$')
plt.plot( list_, list_X2,'-ob', label=r'$\mathbf{X}_{MSE}$')
plt.xlabel('N')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.savefig('figures/varyN_recA.png')
