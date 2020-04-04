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
err_listX1 = np.zeros(10)
err_listX2 = np.zeros(10)

Amse = np.zeros(len(list_))
Xmse1 = np.zeros(len(list_))
Xmse2 = np.zeros(len(list_))

for s in range(len(list_)):
    for ite in range(10):       
        def Main_Algorithm(Y, M, N, k, L, n_seg, L_covseg = list_[s]):       
            #################################  Cov-DL  ####################################
        
            A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
            X_result1 = np.zeros((n_seg, N, L-2))
            X_result2 = np.zeros((n_seg, N, L-2)) 
            
            for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
                Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain
                if N <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k > (M*(M+1))/2.:
                    raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
                
            #################################  M-SBL  #####################################
            X_rec1 = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False)
            X_result1[i] = X_rec1
            
            X_rec2 = M_SBL.M_SBL(A_real, Y[i], M, N, k, iterations=1000, noise=False)
            X_result2[i] = X_rec2
        
            return A_result, X_result1, X_result2
              
        A_result, X_result1, X_result2 = Main_Algorithm(Y, M, N, k, L, n_seg, L_covseg = 10)
        
        err_listX1[ite] = MSE_one_error(X_real.T[0:X_result1[0].shape[1]].T,X_result1[0])
        err_listX2[ite] = MSE_one_error(X_real.T[0:X_result2[0].shape[1]].T,X_result2[0])
        err_listA[ite] = MSE_one_error(A_real,A_result[0])
    
    Amse[s] = np.average(err_listA)
    Xmse1[s] = np.average(err_listX1)
    Xmse2[s] = np.average(err_listX2)
    

""" PLOTS """
plt.figure(1)
plt.plot(Amse, '-r', label = 'A')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')
plt.plot(2, Amse[2], 'ro')
plt.plot(3, Amse[3], 'ro')
plt.plot(4, Amse[4], 'ro')
plt.plot(5, Amse[5], 'ro')


plt.plot(Xmse1, '-b', label = 'X')
plt.plot(0, Xmse1[0], 'bo')
plt.plot(1, Xmse1[1], 'bo')
plt.plot(2, Xmse1[2], 'bo')
plt.plot(3, Xmse1[3], 'bo')
plt.plot(4, Xmse1[4], 'bo')
plt.plot(5, Xmse1[5], 'bo')

plt.title('MSE of A and X for varying segments')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
#plt.savefig('figures/Mix_Error_vary_covseg_m8_k16_L1000.png')
plt.savefig('figures/AR_Error_vary_covseg_m8_k16_L1000.png')
plt.show()

plt.figure(2)
plt.plot(Amse, '-r', label = 'A')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')
plt.plot(2, Amse[2], 'ro')
plt.plot(3, Amse[3], 'ro')
plt.plot(4, Amse[4], 'ro')
plt.plot(5, Amse[5], 'ro')


plt.plot(Xmse2, '-b', label = 'X')
plt.plot(0, Xmse2[0], 'bo')
plt.plot(1, Xmse2[1], 'bo')
plt.plot(2, Xmse2[2], 'bo')
plt.plot(3, Xmse2[3], 'bo')
plt.plot(4, Xmse2[4], 'bo')
plt.plot(5, Xmse2[5], 'bo')

plt.title('MSE of A and X for varying segments')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
#plt.savefig('figures/Mix_Error_vary_covseg_m8_k16_L1000_RealA.png')
plt.savefig('figures/AR_Error_vary_covseg_m8_k16_L1000_RealA.png')
plt.show()
