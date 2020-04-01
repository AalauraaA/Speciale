# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:24:55 2019

@author: Mattek10
"""
from main import Main_Algorithm
from simulated_data import generate_AR
from simulated_data import mix_signals
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import numpy as np

#data_file = 'data/S1_CClean.mat'            # file path
#segment_time = 20                           # length of segmenta i seconds
#Y, M, L, n_seg = EEG_data._import(data_file, segment_time, 
#                                  request='remove 1/2')

np.random.seed(4523)

M = 3
N = 4
k = 4
L = 100
duration = 10
n_seg = 1

Y, A_real, X_real = mix_signals(L, duration, M, N, k)

print('A_real {}'.format(A_real))
Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))
A_result, X_result = Main_Algorithm(Y, M, L, n_seg)

plot_seperate_sources_comparison(X_real,X_result[0],M,N,k,L)

X_mse = MSE_one_error(X_real.T[0:X_result[0].shape[1]].T,X_result[0])
A_mse = MSE_one_error(A_real,A_result[0])
