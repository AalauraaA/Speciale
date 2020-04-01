# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura

Testing different samples sizes and number of segments
"""

import numpy as np
import matplotlib.pyplot as plt

import Cov_DL
import M_SBL

from simulated_data import generate_AR
from simulated_data import mix_signals
from simulated_data import MSE_one_error


""" DATA GENERATION """
M = 8
N = 16
k = 16
L = 1000 
n_seg = 1                     
duration = 10

#Y, A_real, X_real = mix_signals(L, duration, M, N, k)
Y, A_real, X_real = generate_AR(N, M, L, k)

Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))

list_ = [10, 20, 30, 40, 50, 60]   #L_covseg

err_listA = np.zeros(10)
err_listX = np.zeros(10)

Amse = np.zeros(len(list_))
Xmse = np.zeros(len(list_))

for s in range(len(list_)):
    for ite in range(10):       
        def Main_Algorithm(Y, M, L, n_seg, L_covseg = list_[s]):       
            #################################  Cov-DL  ####################################
        
            A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
            X_result = np.zeros((n_seg, N, L-2))
        
            for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
                print(i)
                Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain
                if N <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k > (M*(M+1))/2.:
                    raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
#                print('A_result{}'.format(A_result[0]))
                
            #################################  M-SBL  #####################################
        
                X_rec = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False)
#                print('efter X')
                X_result[i] = X_rec
        
            return A_result, X_result
            
        A_result, X_result = Main_Algorithm(Y, M, L, n_seg)
        
        err_listX[ite] = MSE_one_error(X_real.T[0:X_result[0].shape[1]].T,X_result[0])
        err_listA[ite] = MSE_one_error(A_real,A_result[0])
        print(err_listX[ite])
    
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


plt.title('MSE of A and X for varying segments')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/AR_Error_vary_covseg_m8_k16_L1000.png')
plt.show()
