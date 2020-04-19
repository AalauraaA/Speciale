# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:34:00 2020

@author: trine
"""

# plot file
from main import Main_Algorithm
import matplotlib.pyplot as plt
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import data
import numpy as np
import simulated_data

np.random.seed(1234) 

### COV-DL2
M = 3 
N = 5
k = 5
L = 1000
n_seg = 1

Y, A_real, X_real = simulated_data.mix_signals(L,M,version=None)
#Y, A_real, X_real = generate_AR(N, M, L, k)
print('X søjle {}'.format(X_real.T[1]))
print('Y søjle {}'.format(Y.T[1]))
print('A række {}'.format(A_real[0]))

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_result, X_result,A_init= Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
print('A_rec række {}'.format(A_result[0][0]))
A_mse = MSE_one_error(A_real,A_result[0])
print('A mse = {}'.format(A_mse))

segment = 0
plt.figure(2)
plt.title('Comparison of Mixing Matrix - COV-DL2')
plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-r', label=r'Etimate $\hat{\mathbf{A}}$')
plt.plot(np.reshape(A_init,(A_init.size)), 'o-b',label=r'init $\mathbf{A}$')
plt.legend()
plt.xlabel('index')
plt.show