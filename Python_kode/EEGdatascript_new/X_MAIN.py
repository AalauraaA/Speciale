# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:41:05 2020

@author: Laura
"""

def X_main(data_name, Y, M, k):
    import numpy as np
    from main import Main_Algorithm_EEG
    " Make the measurement matrit Y the right dimension "
    if data_name == 'S1_CClean.mat':
        # For S1_CClean.mat remove last sample of first segment "
        Y[0] = Y[0].T[:-1]
        Y[0]=Y[0].T

    if data_name == 'S1_OClean.mat':
        for i in range(len(Y)):
            if i <= 22:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue
        
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
        X_result[seg] = Main_Algorithm_EEG(Y[seg], A, M, int(k[seg]), L)

    return X_result, Y
        