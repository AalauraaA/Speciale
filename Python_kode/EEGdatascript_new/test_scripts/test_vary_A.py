# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:11:57 2020

@author: Laura


Compare reulst with different mixing matrices instead of Cov-DL A.
"""
from main_A import Main_Algorithm
import vary_A
from simulated_data import generate_AR
from simulated_data import MSE_one_error
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(4523)

M = 8
N = 16
k = 16
L = 1000 
n_seg = 1

Y, A_real, X_real = generate_AR(N, M, L, k)
X_real = X_real.T[:-2]
X_real = X_real.T

A = [vary_A.A_uniform(M,N), vary_A.A_random(M,N), vary_A.A_gaussian(M,N), vary_A.EEG_A(M,N)]

Xmse_list = np.zeros(10)
Amse_list = np.zeros(10)

Xmse = np.zeros(len(A))
Amse = np.zeros(len(A))

for i in range(len(A)):
    print(i)
    for ite in range(10):
        X_result = Main_Algorithm(Y, A[i], M, N, k, L, n_seg)
        Xmse_list[ite] = MSE_one_error(X_real,X_result)  
        Amse_list[ite] = MSE_one_error(A_real,A[i])

        
    
    Xmse[i] = np.average(Xmse_list)  
    Amse[i] = np.average(Amse_list)


""" PLOTS """
plt.figure(1)
plt.plot(Amse, '-r', label = 'A')
plt.plot(Xmse, '-b', label = 'X')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')
plt.plot(2, Amse[2], 'ro')
plt.plot(3, Amse[3], 'ro')
plt.plot(0, Xmse[0], 'bo')
plt.plot(1, Xmse[1], 'bo')
plt.plot(2, Xmse[2], 'bo')
plt.plot(3, Xmse[3], 'bo')
plt.title('MSE of varying A')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
#plt.savefig('figures/AR_Error_vary_A_m8_k16_N16_L1000.png')
plt.show()


"""
A =             [uni,          random,      gaussian,   eeg]
Amse = np.array([1.2558986 ,   4.65224642,  2.08269631,   1.15658083])
Xmse = np.array([ 27.92955353, 8.30111343, 12.95423584, 103.75651538])
"""