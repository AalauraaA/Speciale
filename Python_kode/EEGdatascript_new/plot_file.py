# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:54:37 2020

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

#data_file = 'data/S1_CClean.mat'            # file path
#segment_time = 20                           # length of segmenta i seconds
#Y, M, L, n_seg = data._import(data_file, segment_time, 
#                                  request='remove 2')

np.random.seed(1234)  #1235 ,12347 good mse 0.1392151666945144
################ simple data set #####################
#M = 3
#L = 100
#
#Y, A_real, X_real = simulated_data.mix_signals(L, M, long=True)
#
#plt.figure(0)
#segment = 0
#models = [X_real, Y]
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

################ AR data set #####################

#M = 3
#N = 5
#k = 4
#L = 1000
#n_seg = 1
#
#Y, A_real, X_real = simulated_data.generate_AR(N, M, L, k)
#
#plt.figure(1)
#segment = 0
#models = [X_real, Y]
#names = [r'Source Signals, $\mathbf{X}$',
#         'Measurements, $\mathbf{Y}$',
#         ]
#colors = ['red', 'steelblue', 'orange', 'green', 'yellow']
#
#for ii, (model, name) in enumerate(zip(models, names), 1):
#    plt.subplot(2, 1, ii)
#    plt.title(name)
#    for sig, color in zip(model, colors):
#        plt.plot(sig[0:100], color=color)
#
#plt.tight_layout()
#plt.xlabel('sample')
#plt.tight_layout()
#plt.show()
#plt.savefig('figures/AR_data.png')

################ TEST of COV-DL #####################

### COV-DL2
#M = 3 
#N = 5
#k = 4
#L = 1000
#n_seg = 1
#
#Y, A_real, X_real = simulated_data.mix_signals(L,M,version=None)
##Y, A_real, X_real = generate_AR(N, M, L, k)
#print('X søjle {}'.format(X_real.T[1]))
#print('Y søjle {}'.format(Y.T[1]))
#print('A række {}'.format(A_real[0]))
#
#Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
#X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
#X_real = X_real.T[0:L-2].T
#
#A_result, X_result,A_init = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
#print('A_rec række {}'.format(A_result[0][0]))
#A_mse = MSE_one_error(A_real,A_result[0])
#print('A mse = {}'.format(A_mse))
#
#segment = 0
#plt.figure(2)
#plt.title('Comparison of Mixing Matrix - COV-DL2')
#plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
#plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-r', label=r'Etimate $\hat{\mathbf{A}}$')
#plt.plot(np.reshape(A_init,(A_init.size)), 'o-b',label=r'init $\mathbf{A}$')
#plt.legend()
#plt.xlabel('index')
#plt.show
#plt.savefig('figures/COV2_simple.png')

### COV-DL1
#M = 3 
#N = 8
#k = 4
#L = 1000
#n_seg = 1
#
#Y, A_real, X_real = simulated_data.mix_signals(L,M,version=1)
##Y, A_real, X_real = generate_AR(N, M, L, k)
#print('X søjle {}'.format(X_real.T[1]))
#print('Y søjle {}'.format(Y.T[1]))
#print('A række {}'.format(A_real[0]))
#
#Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
#X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
#X_real = X_real.T[0:L-2].T
#
#A_result, X_result= Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
#print('A_rec række {}'.format(A_result[0][0]))
#A_mse = MSE_one_error(A_real,A_result[0])
#print('A mse = {}'.format(A_mse))
#
#segment = 0
#plt.figure(3)
#plt.title('Comparison of Mixing Matrix - COV-DL1')
#plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
#plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-r', label=r'Etimate $\hat{\mathbf{A}}$')
#plt.legend()
#plt.xlabel('index')
#plt.show
#plt.savefig('figures/COV1_simple.png')

#################### test of M-SBL ####################

### set A_rec = A_real in main algorithm !!!!
np.random.seed(1234)
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

A_result, X_result, A_init = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
print('A_rec række {}'.format(A_result[0][0]))

mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
A_mse = MSE_one_error(A_real,A_result[0])
print('A mse = {}'.format(A_mse))
print('X mse = {}'.format(mse_avg))

segment = 0

plt.figure(4)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
       
        plt.plot(X_real[segment][i], 'g',label='Real X')
        plt.plot(X_result[segment][i],'r', label='Recovered X')

plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('figures/M-SBL_simple1.png')

### more sparse 
np.random.seed(1235)
M = 3 
N = 8
k = 4
L = 1000
n_seg = 1

Y, A_real, X_real = simulated_data.mix_signals(L,M,version=1)
#Y, A_real, X_real = generate_AR(N, M, L, k)
print('X søjle {}'.format(X_real.T[1]))
print('Y søjle {}'.format(Y.T[1]))
print('A række {}'.format(A_real[0]))

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

plt.figure(5)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k+1, 1, nr_plot)
       
        plt.plot(X_real[segment][i], 'g',label='Real X')
        plt.plot(X_result[segment][i],'r', label='Recovered X')

plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('figures/M-SBL_simple2.png')
