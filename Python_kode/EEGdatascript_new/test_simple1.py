# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:12:07 2020

@author: trine
"""

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

np.random.seed(12345) 

# Cov-DL1
M = 3
L = 1000
k = 4
N = 8
n_seg = 1

<<<<<<< HEAD
## Cov-DL2
=======
# Cov-DL2
>>>>>>> b317e570c50fcc9e9273915747489a3d771b93cf
#M = 3
#L = 1000
#k = 4
#N = 5 
#n_seg = 1

<<<<<<< HEAD
Y, A_real, X_real = simulated_data.mix_signals(L,M,version=1)
=======
Y, A_real, X_real = simulated_data.mix_signals(L,M,version=None)
>>>>>>> b317e570c50fcc9e9273915747489a3d771b93cf

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_result, X_result = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)
A_mse = MSE_one_error(A_real,A_result[0])
A_mse0 = MSE_one_error(A_real,np.zeros(A_real.shape))
mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
print('\nMSE_A = {}'.format(np.round_(A_mse,decimals=3)))
print('MSE_A_0 = {}'.format(np.round_(A_mse0,decimals=3)))
print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))


#### plot
segment = 0
plt.figure(0)
models = [X_real[segment], Y[segment]]
names = ['Source Signals, $\mathbf{X}$',
         'Measurements, $\mathbf{Y}$',
         ]
colors = ['red', 'steelblue', 'orange', 'green', 'yellow', 'blue', 'cyan',
          'purple']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, ii)
    plt.title(name)
    for sig, color in zip(model, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.xlabel('sample')
plt.tight_layout()
plt.show()
plt.savefig('figures/simple_data2.png')

plt.figure(1)
plt.title('Comparison of Mixing Matrix - COV-DL1')
plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
plt.plot(np.reshape(A_result[segment], (A_result[segment].size)),'o-r', label=r'Estimate $\hat{\mathbf{A}}$')
plt.legend()
plt.xlabel('index')
plt.show
#plt.savefig('figures/COV1_simple.png')

plt.figure(2)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_result[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(X_real[segment][i], 'g',label='Real X')
        plt.plot(X_result[segment][i],'r', label='Recovered X')

plt.legend(loc='lower right')
plt.xlabel('sample')
plt.tight_layout()
plt.show
plt.savefig('figures/M-SBL_simple2.png')

