# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:57:52 2020

@author: MATTEK10

This is a script to estimation of (k) active sources from real EEG data.

This script need the following libraries to run:
    - Numpy
    - main 
    - data
    - simulated_data
    - plot_functions
    
The script use the:
    - data library to import the data which has been segmented into time 
    segments of one second
    - main library to perform the recovering process of X given Y and a 
    random mixing matrix A
    - simulated_data library to calculated the MSE of the recovered X
    - plot_functions to plot the sources of X from a given time segment
"""
# =============================================================================
# Import libraries
# =============================================================================
import numpy as np
from main import Main_Algorithm_EEG
import data
import simulated_data
from plot_functions import plot_seperate_sources

np.random.seed(1234)
# =============================================================================
# Import EEG data file
# =============================================================================
data_name = 'S1_CClean.mat'
#data_name = 'S1_OClean.mat'
data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# Main Algorithm with random A
# =============================================================================
request='remove 1/2' # remove sensors and the same sources from dataset - none, every third or every second
Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

k = np.ones([len(Y)]) * (M+4)  # a choice for k -- k = M + 4

if data_name == 'S1_CClean.mat':
    " For S1_CClean.mat remove last sample of first segment "
    Y[0] = Y[0].T[:-1]
    Y[0] = Y[0].T

if data_name == 'S1_OClean.mat':
    " For S1_OClean.mat remove last sample of segment 1 to 22 "
    for i in range(len(Y)):
        if i <= 22:
            Y[i] = Y[i].T[:-1]
            Y[i] = Y[i].T
        else:
            continue

" Recovering of the source matrix X"
X_result = []
for i in range(k.shape[0]): # Making the right size of X for all segments
    X_result.append(np.zeros([len(Y), int(k[i])]))

for i in range(len(Y)): # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[i])))
    X_result[i] = Main_Algorithm_EEG(Y[i], A, M, int(k[i]), L)

" Searching for replicates  "
list_ = np.zeros([len(Y), len(X_result[0])])
for seg in range(len(Y)):
    for i in range(len(X_result[0])):
        rep = 0
        for j in range(len(X_result[0])):
            mse = simulated_data.MSE_one_error(X_result[seg][i], X_result[seg][j])
            if  mse < 1.0:
                rep += 1
        list_[seg][i] = rep

" Plots of segment seg = 9 -- the 10 th segment"
seg = 9
fignr = 1
figsave = "figures/EEG_second_removed_est_k" + str(data_name) + '_' + str(seg) + ".png"
plot_seperate_sources(X_result[seg], M, int(k[seg]), int(k[seg]), L, figsave, fignr)

