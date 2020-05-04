# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

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


data_file = 'data/S1_CClean.mat'            # file path
segment_time = 10                           # length of segmenta i seconds

# perform ICA on full dataset
Y, M, L, n_seg = data._import(data_file, segment_time, request='none')
X_ica = ICA.ica_segments(Y, 1000)

# remove sensors and the same sources again
Y, M, L, n_seg = data._import(data_file, segment_time, request='remove 2')
A_result, X_result = Main_Algorithm(Y, M, L, n_seg, L_covseg=50)

# mse between ICA sources and baseline sources calculated for each segment
mse, average_mse = simulated_data.MSE_segments(X_result, X_ica)
print('MSE = {}'.format(average_mse))
