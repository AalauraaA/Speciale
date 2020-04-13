# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:28:35 2020

@author: trine
"""

import numpy as np
import EEG_data
import Cov_DL
import M_SBL

##############################  Initialization  ###############################

data_file = 'data/S1_CClean.mat'            # file path
segment_time = 0                           # length of segmenta i seconds
Y, M, L, n_seg = EEG_data._import(data_file, segment_time, 
                                  request='remove 1/2')

print('Data information:\n number of sensors \t \t M = {} \n number of segments \t \t n_seg = {} \n number of samples pr segment \t L = {}'.format(M, n_seg, Y[0].shape[1]))

N = int(input("Please enter N: "))               # number of sources
k = int(input("Please enter k: "))               # active sources to be found

#################################  Cov-DL  ####################################

L_covseg = 10                        # len of training example

A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
X_result = np.zeros((n_seg, N, L-2))

for i in range(len(Y)):           # loop over all segments (axis 0 of Y)
    print(i)
    Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain

    if N <= (M*(M+1))/2.:
        A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
        A_result[i] = A_rec

    elif k <= (M*(M+1))/2.:
        A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
        A_result[i] = A_rec

    elif k > (M*(M+1))/2.:
        raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')

#################################  M-SBL  #####################################

    X_rec = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=500, noise=False)
    X_result[i] = X_rec
