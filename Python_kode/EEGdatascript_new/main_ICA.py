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

data_file = 'data/S1_CClean.mat'           # file path
segment_time = 5                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
# perform ICA on full dataset
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')
X_ica, A_ica = ICA.ica_segments(Y_ica, 1000)

" Remove the last column from X_ica to match size of X "
X_ica_new = np.array(X_ica, copy = True)
X_ica_array = []
for i in range(len(Y_ica)):
    if i >= 17:
        # From index 17 and to the end Y_ica have one less sample but
        # X_ica do not take this into account        
        X_ica_array.append(X_ica_new[i,:,:-1])
    else:
        # From index 0 to 17 the original X_ica is kept
       X_ica_array.append(X_ica_new[i]) 

" Replacing small values with zero and creating X_ica of size k x samples for each segment "
X_ica_nonzero = []
for i in range(len(X_ica_array)):
    temp = []
#    X_ica_nonzero.append(np.zeros([len(Y_ica), int(k[i])]))
    # Looking at one segment at time
    for j in range(len(X_ica_array[i])):
        # Looking at on row of one segment at the time
        if X_ica_array[i][j].all() >= -10E-15 and X_ica_array[i][j].all() < 0:  # if smaller than 
            X_ica_array[i][j] = 0   # replace by zero
        else:
            temp.append(X_ica_array[i][j])
    X_ica_nonzero.append(temp)

" Finding the number of active sources (k) for each segment "
k = np.zeros(len(X_ica_nonzero))
for i in range(len(X_ica_nonzero)):
    # count the number of non-zeros rows in one segment
    k[i] = len(X_ica_nonzero[i])
 
# =============================================================================
# Main Algorithm with random A
# =============================================================================
# remove sensors and the same sources from dataset - every third
Y, M, L, n_seg = data._import(data_file, segment_time, request='remove 1/3')

X_result = []
mse = []
for i in range(len(k)):
    " Making the right size of X for all segments "
    X_result.append(np.zeros([len(Y), int(k[i])]))
    
    " Making the mse for all sources in all segments "
    mse.append(np.zeros([len(Y), int(k[i])]))
  
average_mse = np.zeros(len(Y))

for i in range(len(Y)):
    # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[i])))
    X_result[i] = Main_Algorithm_EEG(Y[i], A[i], M, int(k[i]), L)
    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_new[i])

#print('MSE = {}'.format(average_mse))

" Plot the all the sources of time segment 1 "
plot_seperate_sources_comparison(X_result[1],X_ica[1],M,int(k[1]),int(k[1]),L)

# =============================================================================
# Main Algorithm with A_ica
# =============================================================================
## remove sensors and the same sources from dataset - every third
#Y, M, L, n_seg = data._import(data_file, segment_time, request='remove 1/3')
#A_ica = np.delete(A_ica, np.arange(0, A_ica.shape[0], 3), axis=0)
#
#X_result = []
#A = []
#mse = []
#for i in range(len(k)):
#    " Making the right size of X for all segments "
#    X_result.append(np.zeros([len(Y), int(k[i])]))
#    
#    " Making the mixing matrix A from ICA for all segments "
#    a = np.array(A_ica, copy=True)
#    A.append(a[:,:int(k[i])])
#    
#    " Making the mse for all sources in all segments "
#    mse.append(np.zeros([len(Y), int(k[i])]))
#
#average_mse = np.zeros(len(Y)) # Average MSE for each segments (size 28 x 1)
#
#for i in range(len(Y)):
#    # Looking at one time segment
#    X_result[i] = Main_Algorithm_EEG(Y[i], A[i], M, int(k[i]), L)
#    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_new[i])
#
##print('MSE = {}'.format(average_mse))
#
#" Plot the all the sources of time segment 1 "
#plot_seperate_sources_comparison(X_result[1], X_ica[1], M, int(k[1]), int(k[1]), L)
