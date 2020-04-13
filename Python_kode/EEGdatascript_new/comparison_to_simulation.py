# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:59:06 2020

@author: trine
"""

from main import Main_Algorithm
import matplotlib.pyplot as plt
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import data
import numpy as np
import simulated_data

#data_file = 'data/S1_CClean.mat'            # file path
#segment_time = 20                           # length of segmenta i seconds
#Y, M, L, n_seg = data._import(data_file, segment_time, 
#                                  request='remove 2')

M = 3 
N = 5
k = 4
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

A_result, X_result = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
print('A_rec række {}'.format(A_result[0][0]))

mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
A_mse = MSE_one_error(A_real,A_result[0])
print('A mse = {}'.format(A_mse))
print('X mse = {}'.format(mse_avg))

segment = 0

plt.figure(1)
plt.title('Source matrix')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k*2, 1, nr_plot)
       
        plt.plot(X_real[segment][i], 'r',label='Real X')
        plt.plot(X_result[segment][i],'g', label='Recovered X')

plt.legend()
plt.xlabel('sample')
plt.show
#plt.savefig('16-03-2020_3.png')

plt.figure(3)
plt.title('Comparison of Mixing Matrix')
plt.plot(np.reshape(A_real, (A_real.size)), 'o-r',label=r'True $\mathbf{A}$')
plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-g', label=r'Etimate $\hat{\mathbf{A}}$')
plt.legend()
plt.xlabel('index')
plt.show
#plt.savefig('16-03-2020_3.png')


