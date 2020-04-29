# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

@author: trine
"""

import numpy as np
from main import Main_Algorithm_EEG
import simulated_data
import ICA
import data
from plot_functions import plot_seperate_sources_comparison

data_name = 'S1_OClean.mat'
data_file = 'data/' + data_name            # file path
segment_time = 1                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" Perform ICA on full dataset "
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')

Y_ica[0] = Y_ica[0].T[:-1]
Y_ica[0] = Y_ica[0].T

X_ica, A_ica = ICA.ica_segments(Y_ica, 1000)

" Remove the last column from X_ica to match size of X "
X_ica_new = np.array(X_ica, copy = True)
X_ica_array = []
for i in range(len(Y_ica)):
    X_ica_array.append(X_ica_new[i])
#    if i == 0:     
#        X_ica_array.append(X_ica_new[i,:,:-1])
#    else:
#       X_ica_array.append(X_ica_new[i]) 

" Replacing small values with zero and creating X_ica of size k x samples for each segment "
X_ica_nonzero = []
tol = 10E-3
for i in range(len(X_ica_array)): # Looking at one segment at time
    temp = [] # temporary variable to store the nonzero array for one segment
    for j in range(len(X_ica_array[i])): # Looking at on row of one segment at the time
        if np.average(X_ica_array[i][j]) < tol and np.average(X_ica_array[i][j]) > -tol:  # if smaller than 
            X_ica_array[i][j] = 0   # replace by zero
        else:
            temp.append(X_ica_array[i][j])
    X_ica_nonzero.append(temp)

" Finding the number of active sources (k) for each segment "
k = np.zeros(len(X_ica_nonzero))
for i in range(len(X_ica_nonzero)):
    # count the number of nonzeros rows in one segment
    k[i] = len(X_ica_nonzero[i])

# =============================================================================
# Main Algorithm with random A
# =============================================================================
request='remove 1/3' # remove sensors and the same sources from dataset - every third
Y, M, L, n_seg = data._import(data_file, segment_time, request=request)
for i in range(23):
    Y[i] = Y[i].T[:-1]
    Y[i] = Y[i].T

#for i in range(len(X_ica_nonzero)):
#    for j in range(len(X_ica_nonzero[i])):
#        X_ica_nonzero[i][j] = X_ica_nonzero[i][j][:-1]

X_result = []
mse = []
for i in range(len(k)):
    " Making the right size of X for all segments "
    X_result.append(np.zeros([len(Y), int(k[i])]))
    
    " Making the mse for all sources in all segments "
    mse.append(np.zeros([len(Y), int(k[i])]))
  
average_mse = np.zeros(len(Y))

for i in range(len(Y)): # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[i])))
    X_result[i] = Main_Algorithm_EEG(Y[i], A, M, int(k[i]), L)
    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_nonzero[i])

" Plot the all the sources of time segment 1 "
for i in range(10):
    figsave = "figures/EEG_third_removed_" + str(data_name) + '_' + str(i) + ".png"
    plot_seperate_sources_comparison(X_result[i],X_ica_nonzero[i],M,int(k[i]),int(k[i]),L,figsave,i)
    print('MSE = {}'.format(average_mse[i]))

# =============================================================================
# Main Algorithm with A_ica
# =============================================================================
# remove sensors and the same sources from dataset - every third
#request='remove 1/3'
#Y, M, L, n_seg = data._import(data_file, segment_time, request=request)
#Y[0] = Y[0].T[:-1]
#Y[0] = Y[0].T

#A_ica_array = np.array(A_ica, copy=True)
#A_ica_array = data._reduction(A_ica_array, request)
##
##for i in range(len(X_ica_nonzero)):
##    for j in range(len(X_ica_nonzero[i])):
##        X_ica_nonzero[i][j] = X_ica_nonzero[i][j][:-1]
#
#X_result = []
#A = []
#mse = []
#for i in range(len(k)):
#    " Making the right size of X for all segments "
#    X_result.append(np.zeros([len(Y), int(k[i])]))
#    
#    " Making the mixing matrix A from ICA for all segments "
#    A.append(A_ica_array[:,:int(k[i])])
#    
#    " Making the mse for all sources in all segments "
#    mse.append(np.zeros([len(Y), int(k[i])]))
#
#average_mse = np.zeros(len(Y)) # Average MSE for each segments (size 28 x 1)
#
#for i in range(len(Y)):
#    # Looking at one time segment
#    X_result[i] = Main_Algorithm_EEG(Y[i], A[i], M, int(k[i]), L)
#    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_nonzero[i])
#
##print('MSE = {}'.format(average_mse))
#
#" Plot the all the sources of time segment 1 "
#plot_seperate_sources_comparison(X_result[1], X_ica_nonzero[1], M, int(k[1]), int(k[1]), L)
