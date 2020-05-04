# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:44:57 2020

@author: trine
"""
from main import Main_Algorithm
import matplotlib.pyplot as plt
from simulated_data import generate_AR
from simulated_data import MSE_one_error
import numpy as np
import simulated_data

np.random.seed(1234)
M = 6
N = 8
k = 4
L = 1000
n_seg = 1

Y, A_real, X_real = generate_AR(N, M, L, k)

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_result, X_result= Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
print('A_rec række {}'.format(A_result[0][0]))

mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
A_mse = MSE_one_error(A_real,A_result[0])
print('A mse = {}'.format(A_mse))
print('X mse = {}'.format(mse_avg))

segment = 0

plt.figure(7)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k*2, 1, nr_plot)
       
        plt.plot(X_real[segment][i][0:100], 'g',label='Real X')
        plt.plot(X_result[segment][i][0:100],'r', label='Recovered X')

plt.legend(loc='lower right')
plt.xlabel('sample')
#plt.tight_layout()
plt.show
plt.savefig('figures/M=N_testk=4.png')

############################ analysis


