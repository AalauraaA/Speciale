# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:24:44 2020

@author: Laura
"""

def Main_Algorithm(Y, A, M, N, k, L, n_seg):
    import M_SBL1
    import M_SBL
    
#    print('Data information:\n number of sensors \t \t M = {} \n number of segments \t \t n_seg = {} \n number of samples pr segment \t L = {}'.format(M, n_seg, Y[0].shape[1]))
#
#    N = int(input("Please enter N: "))               # number of sources
#    k = int(input("Please enter k: "))               # active sources to be found
    
    X_rec = M_SBL.M_SBL(A, Y, M, N, k, iterations=10, noise=True)

    return X_rec