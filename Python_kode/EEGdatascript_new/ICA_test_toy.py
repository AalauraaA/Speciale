# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:19:03 2020

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from main import Main_Algorithm
import simulated_data
import ICA

########################## Given example ################################## 
# Generate sample data from example 
#np.random.seed(0)
#n_samples = 2000
#time = np.linspace(0, 8, n_samples)
#
#s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
#s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
#s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
#
#X = np.c_[s1, s2, s3]
#X += 0.2 * np.random.normal(size=X.shape)  # Add noise
#X /= X.std(axis=0)  #Standardize data
#
## Mix data
#A_real = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
#X_real = X.T
#Y = np.dot(X, A_real.T).T  # Generate observations

M = 2
N = M
k = N
L = 2000
n_seg = 1

# generate own signal 
#Y, A_real, X_real = simulated_data.mix_signals(L,5,M,N,k)
#

Y, A_real, X_real = simulated_data.generate_AR(N, M, L, k)
Y = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))


A_result, X_result, A_real = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg = 20)

## our ICA

X_ica, A = ICA.ica_segments(Y, 1000) 


mse, average_mse = simulated_data.MSE_segments(X_result,X_ica)

print('MSE = {}'.format(average_mse))

### Plot results

plt.figure()
segment = 0
models = [Y[segment].T, X_real.T, X_ica[segment].T, X_result[segment].T]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'Baseline recovered signals']
colors = ['red', 'steelblue', 'orange', 'green']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig[:100], color=color)

plt.tight_layout()
plt.show()



