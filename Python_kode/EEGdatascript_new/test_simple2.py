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
import numpy as np
import simulated_data

np.random.seed(1234) 

### COV-DL2
M = 3 
L = 1000
k = 4
N = 5 
n_seg = 1

Y, A_real, X_real = simulated_data.mix_signals(L,M,version=None)

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_result, X_result = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
A_mse = MSE_one_error(A_real,A_result[0])
mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
print('\nMSE_A = {}'.format(np.round_(A_mse,decimals=4)))
print('MSE_X = {}'.format(np.round_(mse_avg,decimals=4)))

##### plot
segment = 0
#plt.figure(0)
#models = [X_real[segment], Y[segment]]
#names = ['Source Signals, $\mathbf{X}$',
#         'Measurements, $\mathbf{Y}$',
#         ]
#colors = ['red', 'steelblue', 'orange', 'green', 'yellow', 'blue', 'cyan',
#          'purple']
#
#for ii, (model, name) in enumerate(zip(models, names), 1):
#    plt.subplot(2, 1, ii)
#    plt.title(name)
#    for sig, color in zip(model, colors):
#        plt.plot(sig, color=color)
#
#plt.tight_layout()
#plt.xlabel('sample')
#plt.tight_layout()
#plt.show()
#plt.savefig('figures/simple_data.png')

#plt.figure(1)
#plt.title('Comparison of Mixing Matrix - COV-DL2')
#plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
#plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-r', label=r'Estimate $\hat{\mathbf{A}}$')
#plt.plot(np.reshape(A_init,(A_init.size)), 'o-b',label=r'init $\mathbf{A}$')
#plt.legend()
#plt.xlabel('index')
#plt.show
#plt.savefig('figures/COV2_simple.png')

plt.figure(2)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
       
        plt.plot(X_real[segment][i], 'g',label='Real X')
        plt.plot(X_result[segment][i],'r',label='Recovered X')


plt.legend()
plt.xlabel('sample')
plt.tight_layout()
plt.show
plt.savefig('figures/M-SBL_simple1.png')
