# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:31:52 2020

@author: Laura
"""

def X_ica(data_name, Y, M):  
    from sklearn.decomposition import FastICA
    import numpy as np
   
    " Perform FastICA on Segmented Dataset "
    n_seg = len(Y)
    N = Y[0].shape[0]
    L = Y[0].shape[1]
    
    X_ica = np.zeros((n_seg, N, L-2))
    A_ica = np.zeros((n_seg, N, M))
    for i in range(n_seg):   
        X = Y[i].T
        ica = FastICA(n_components=N, max_iter=1000,random_state=123)
        print('ica on segment: {}'.format(i))
        X_ICA = ica.fit_transform(X)  # Reconstruct signals
        A_ICA = ica.mixing_
        X_ica[i] = X_ICA[:L-2].T
        A_ica[i] = A_ICA              # Get estimated mixing matrix

    " Copy the X_ica to an Array "
    X_ica_new = np.array(X_ica, copy = True)
    X_ica_array = []
    for i in range(len(Y)):
        X_ica_array.append(X_ica_new[i])

    " Replacing small values with zero and creating X_ica of size k x samples for each segment "
    X_ica_nonzero = []
    tol = 10E-5
    for seg in range(len(X_ica_array)): 
        # Looking at one segment at time
        temp = []                       # temporary variable to store the nonzero array for one segment
        for i in range(len(X_ica_array[seg])): 
            # Looking at on row of one segment at the time   
            if np.average(X_ica_array[seg][i]) < tol and np.average(X_ica_array[seg][i]) > -tol:  # if smaller than 
                X_ica_array[seg][i] = 0   # replace by zero
            else:
                temp.append(X_ica_array[seg][i])
                
        X_ica_nonzero.append(temp)

    " Finding the number of active sources (k) for each segment "
    k = np.zeros(len(X_ica_nonzero))
    for seg in range(len(X_ica_nonzero)):
        # count the number of nonzeros rows in one segment
        k[seg] = len(X_ica_nonzero[seg])
        
    return X_ica_nonzero, k