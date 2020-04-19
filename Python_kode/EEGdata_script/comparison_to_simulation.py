# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:59:06 2020

@author: trine
"""

from main import Main_Algorithm
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import EEG_data
import numpy as np

#data_file = 'data/S1_CClean.mat'            # file path
#segment_time = 20                           # length of segmenta i seconds
#Y, M, L, n_seg = EEG_data._import(data_file, segment_time, 
#                                  request='remove 1/2')

np.random.seed(4523)

M = 8
N = 16
k = 16
L = 1000 
n_seg = 1
Y, A_real, X_real = generate_AR(N, M, L, k)

#print('A_real {}'.format(A_real))
Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))
A_result1, A_result2, X_result1, X_result2 = Main_Algorithm(Y, M, L, n_seg)

plot_seperate_sources_comparison(X_real,X_result1[0],M,N,k,L)
plot_seperate_sources_comparison(X_real,X_result2[0],M,N,k,L)

X_mse1 = MSE_one_error(X_real.T[0:X_result1[0].shape[1]].T,X_result1[0])
X_mse2 = MSE_one_error(X_real.T[0:X_result2[0].shape[1]].T,X_result2[0])

A_mse1 = MSE_one_error(A_real,A_result1[0])
A_mse2 = MSE_one_error(A_real,A_result2)

# Noter:
# jo længere segmenter jo længere tager det at finde X 