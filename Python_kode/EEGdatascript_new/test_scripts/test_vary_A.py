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

iterationer = 50
Xmse_list = np.zeros((4,iterationer))
Amse_list = np.zeros((4,iterationer))

Xmse = np.zeros(4)
Amse = np.zeros(4)

for ite in range(iterationer):
    print('iteration {}'.format(ite))

    # M=N
    Y_f, A_real_f, X_real = generate_AR(N, N, L, k) #(N,M) is real
    
    #reduce so M<N
    request='remove 1/3'
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
plt.subplot(2, 1, 1)
plt.title('MSE of varying fixed A')
plt.plot(Xmse, 'ob', label = 'X')
plt.xticks([])
plt.ylabel('MSE')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(2, 1, 2)
plt.plot(Amse, 'or', label = 'A')
plt.xticks([0, 1, 2, 3],["$\mathcal{U}(-1,1)$","normal \n $\mu = 0, \sigma^2 = 2$","Gaussian","ICA"])

plt.ylabel('MSE')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()      
plt.savefig('A_fix.png')
plt.show()


"""
A =             [uni,          random,      gaussian,   eeg]
Amse = np.array([1.2558986 ,   4.65224642,  2.08269631,   1.15658083])
Xmse = np.array([ 27.92955353, 8.30111343, 12.95423584, 103.75651538])
"""