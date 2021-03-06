# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:52:54 2020

@author: trine
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:28:35 2020

@author: trine
"""

import EEG_data

##############################  Initialization  ###############################
#
#data_file = 'data/S1_CClean.mat'            # file path
#segment_time = 10                           # length of segmenta i seconds
#Y, M, L, n_seg = EEG_data._import(data_file, segment_time, 
#                                  request='remove 1/2')
#



def Main_Algorithm(Y, M, L, n_seg, L_covseg = 10):
    
    import numpy as np
    import Cov_DL
    import M_SBL
    import EEG_A
    
    print('Data information:\n number of sensors \t \t M = {} \n number of segments \t \t n_seg = {} \n number of samples pr segment \t L = {}'.format(M, n_seg, Y[0].shape[1]))

    N = int(input("Please enter N: "))               # number of sources
    k = int(input("Please enter k: "))               # active sources to be found

    #################################  Cov-DL  ####################################

    A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
    X_result1 = np.zeros((n_seg, N, L-2))
    X_result2 = np.zeros((n_seg, N, L-2))    
    
    A = EEG_A.EEG_A(N)
    A = A[:M]
    
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
        print('A_result{}'.format(A_result[0]))
    #################################  M-SBL  #####################################
        X_rec1 = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False)
        X_result1[i] = X_rec1
        
        X_rec2 = M_SBL.M_SBL(A, Y[i], M, N, k, iterations=1000, noise=False)
        X_result2[i] = X_rec2

    return A_result, A, X_result1, X_result2