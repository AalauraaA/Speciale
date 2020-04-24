# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from main import Main_Algorithm_EEG
import simulated_data
import ICA
import data
from plot_functions import plot_seperate_sources_comparison


data_file = 'data/S1_CClean.mat'            # file path
segment_time = 5                           # length of segmenta i seconds


""" ICA """
# perform ICA on full dataset
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')

X_ica, A_ica = ICA.ica_segments(Y_ica, 1000)
X_ica_new = []
for i in range(len(Y_ica)):
    if i >= 17:
        print(i)
        X_ica_new.append(X_ica[i,:,:-1])
    else:
       X_ica_new.append(X_ica[i]) 

# replacing small values with zero and find k for each segment
k = np.zeros(len(X_ica_new))
#X_ica_sparse = np.array(X_ica_new, copy=True)
for i in range(len(X_ica_new)):
    # Looking at one segment at time
    for j in range(len(X_ica_new[i])):
        # Looking at on row of one segment at the time
        for h in range(len(X_ica_new[i][j])):
            # Looking at one element of one row of one segment at the time
            if X_ica_new[i][j][h] <= 10E-15:  # if smaller than 
                X_ica_new[i][j][h] = 0        # replace by zero
#        if X_ica_sparse[i][j].all() == 0:
#            X_ica_minus_0 = np.delete(X_ica_minus_0, i, 1)
    k[i] = np.count_nonzero(X_ica_new[i].T[0]) # count the number of non-zeros rows in one segment


""" Main Algorithm """
# remove sensors and the same sources from dataset - every third
Y, M, L, n_seg = data._import(data_file, segment_time, request='remove 1/3')
A_ica = np.delete(A_ica, np.arange(0, A_ica.shape[0], 3), axis=0)

X_result = []
for i in range(len(k)):
    X_result.append(np.zeros([len(Y), int(k[i])]))

A = []
for i in range(len(k)):
    a = np.array(A_ica, copy=True)
    A.append(a[:,:int(k[i])])



mse = []
for i in range(len(k)):
    mse.append(np.zeros([len(Y), int(k[i])]))
    
average_mse = np.zeros(len(Y))
for i in range(len(Y)):
    # Looking at one time segment
#    A = np.random.normal(0,2,(M,int(k[i])))
    A = np.array(A_ica, copy=True)
    A = A[:,:int(k[i])]
    X_result[i] = Main_Algorithm_EEG(Y[i], A, M, int(k[i]), L)
    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_new[i])


#mse between ICA sources and baseline sources calculated for each segment
#mse, average_mse = simulated_data.MSE_segments(X_result, X_ica)
#print('MSE = {}'.format(average_mse))


plot_seperate_sources_comparison(X_result[1],X_ica[1],M,int(k[1]),int(k[1]),L)

##A_result, X_result = Main_Algorithm(Y, M, L, n_seg)