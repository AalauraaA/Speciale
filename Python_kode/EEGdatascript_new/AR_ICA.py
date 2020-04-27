# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:53:15 2020

@author: trine
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from main import Main_Algorithm
import simulated_data
import ICA
import data

# simulated AR data 
M = 8
N = 8
k = 8
L = 2000
n_seg = 1
Y, A_real, X_real = simulated_data.generate_AR(N, M, L, k)
Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))

# perform ICA on full datasetX_base
X_ica, A = ICA.ica_segments(Y, 1000)

# perform baseline in full dataset
A_base_full, X_base_full = Main_Algorithm(Y, M, L, n_seg, L_covseg=10) 

# remove sensors from data
n_remove = 2                                    # have to correspond to request
Y_ = np.zeros((len(Y),M-n_remove,L))
for i in range(len(Y)):
    Y_[i] = data._reduction(Y[i],'remove 2')
M = Y_.shape[1]

# perform baseline on reduced dataset
A_base, X_base = Main_Algorithm(Y_, M, L, n_seg, L_covseg=10)  

# mse between ICA sources and baseline sources calculated for each segment
mse_true, average_mse_true = simulated_data.MSE_segments(X_ica, X_real.T[:L-2].T)
print('MSE between ica and true sources = {}'.format(average_mse_true))

mse_truebase, average_mse_truebase = simulated_data.MSE_segments(X_base, X_real.T[:L-2].T)
print('MSE between base and true sources = {}'.format(average_mse_truebase))

mse, average_mse = simulated_data.MSE_segments(X_base, X_ica)
print('MSE between ica and reduced base = {}'.format(average_mse))

mse_icafull, average_mse_icafull = simulated_data.MSE_segments(X_base_full, X_ica)
print('MSE between ica and full base = {}'.format(average_mse_icafull))

mse_base, average_mse_base = simulated_data.MSE_segments(X_base, X_base_full)
print('MSE between full base and reduced base = {}'.format(average_mse_base))
