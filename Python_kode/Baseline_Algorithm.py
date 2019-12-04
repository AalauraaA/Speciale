# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:24:55 2019

@author: trine
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation
import dictionary_learning
import MSBL
#import Cov_DL

np.random.seed(1)

# choose datageneration method ... 

""" DATA GENERATION - MIX OF DETERMINISTIC SIGNALS """

m = 10                          # number of sensors
n = 25                          # number of sources
k = 5                           # max number of non-zero coef. in rows of X
L = 10000                       # number of sampels


Y_real, A_real, X_real = data_generation.mix_signals(L, 10, m, n, k)



""" DATA GENERATION - ROSSLER DATA """

L = 1940                        # number of sampels, max value is 1940

Y_real, A_real, X_real, k = data_generation.rossler_data(n_sampels=1940, 
                                                         ex = 1, m=8)
m = len(Y_real)                      # number of sensors
n = len(X_real)                      # number of sources
k = np.count_nonzero(X_real.T[0])    # number of non-zero coef. in rows of X


""" SEGMENTATION - OVER ALL """
#
Ls = L                  # number of sampels per segment (L -> no segmentation) 
Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0

for i in range(len(Ys)):
    Y_real = Ys[i]
    X_real = Xs[i]
    
    
    
    




