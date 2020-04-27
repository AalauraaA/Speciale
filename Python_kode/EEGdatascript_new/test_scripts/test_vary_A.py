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
import data

#np.random.seed(4523)

N = 16  
k = 16
L = 1000 
n_seg = 1

iterationer = 1
Xmse_list = np.zeros((4,iterationer))
Amse_list = np.zeros((4,iterationer))

Xmse = np.zeros(4)
Amse = np.zeros(4)

for ite in range(iterationer):
    print('iteration {}'.format(ite))

    # M=N
    Y_f, A_real_f, X_real = generate_AR(N, N, L, k) #(N,M) is real
    
    #reduce so M<N
    request='remove 1/2'
    Y = data._reduction(Y_f, request)
    A_real = data._reduction(A_real_f, request)
    
    M = len(Y)
    
    X_real = X_real.T[:-2]
    X_real = X_real.T
    
    A = [vary_A.A_uniform(M,N), vary_A.A_random(M,N), vary_A.A_gaussian(M,N), vary_A.A_ICA(Y_f,request)]

    for i in range(len(A)):
        print('A_fix type {}'.format(i))
    
        X_result = Main_Algorithm(Y, A[i], M, N, k, L, n_seg)
        
        Xmse_list[i][ite] = MSE_one_error(X_real,X_result)  
        Amse_list[i][ite] = MSE_one_error(A_real,A[i])

           
for j in range(len(Xmse)):
    Xmse[j] = np.average(Xmse_list[j])  
    Amse[j] = np.average(Amse_list[j])


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