# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:41:05 2020

@author: Laura
"""
def Main_Algorithm_EEG(Y, A_real, M, k, L): 
    import M_SBL

    X_result = M_SBL.M_SBL(A_real, Y, M, k, k, iterations=1000, noise=False) #### OBS CHECK for A_rec
    
    return X_result


def X_main(data_name, Y, M, k):
    import numpy as np
    np.random.seed(1234)
    " Perform Main Algorithm on Segmented Dataset "
    X_result = []    # Original recovered source matrix X
    L = Y[0].shape[1]

    for i in range(k.shape[0]):
        " Making the right size of X for all segments "
        X_result.append(np.zeros([len(Y), int(k[i])]))
    
    " Original Recovered Source Matrix X, MSE and Average MSE with X_ica "
    for seg in range(len(Y)): 
        # Looking at one time segment
        A = np.random.normal(0,2,(M,int(k[seg])))
        #print(A[0][0])
        X_result[seg] = Main_Algorithm_EEG(Y[seg], A, M, int(k[seg]), L)

    return X_result, Y
        