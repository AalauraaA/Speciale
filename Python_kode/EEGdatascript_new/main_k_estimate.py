# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:57:52 2020

@author: Laura
"""
import numpy as np
from main import Main_Algorithm_EEG
import data
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
request='none' # remove sensors and the same sources from dataset - every third
Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

k = np.ones([len(Y)]) * M

if data_name == 'S1_CClean.mat':
    " For S1_CClean.mat remove last sample of first segment "
    Y[0] = Y[0].T[:-1]
    Y[0] = Y[0].T

if data_name == 'S1_OClean.mat':
    for i in range(len(Y)):
        if i <= 22:
            Y[i] = Y[i].T[:-1]
            Y[i] = Y[i].T
        else:
            continue

X_result = []
for i in range(k.shape[0]):
    " Making the right size of X for all segments "
    X_result.append(np.zeros([len(Y), int(k[i])]))

for i in range(len(Y)): # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[i])))
    X_result[i] = Main_Algorithm_EEG(Y[i], A, M, int(k[i]), L)

" Plots of second (i = 1) segment "

i = 10
figsave = "figures/EEG_second_removed" + str(data_name) + '_' + str(i) + ".png"
plot_seperate_sources(X_result[i],M,int(k[i]),int(k[i]),L,figsave,1)

i = 45
figsave = "figures/EEG_second_removed" + str(data_name) + '_' + str(i) + ".png"
plot_seperate_sources(X_result[i],M,int(k[i]),int(k[i]),L,figsave,2)


i = 100
figsave = "figures/EEG_second_removed" + str(data_name) + '_' + str(i) + ".png"
plot_seperate_sources(X_result[i],M,int(k[i]),int(k[i]),L,figsave,3)
