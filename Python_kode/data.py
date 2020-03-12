# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:21:34 2020

@author: Laura

This is a script for generating of the simulated data use for the
testing of the baseline algorithm.

There are three type of datasets:
    - Auto-regressive (random) data set 
    - Mixed signal (determinitistic) data set
    - Rossler data set

"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation

# =============================================================================
# Parameters
# =============================================================================
m = 3                         # number of sensors
n = 4                         # number of sources
k = 4                         # max number of non-zero coef. in rows of X
L = 1000                      # number of sampels
k_true = 4 

# =============================================================================
# Data
# =============================================================================
""" DATA GENERATION - AUTO-REGRESSIVE SIGNAL """
#Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 

""" DATA GENERATION - MIX OF DETERMINISTIC SIGNALS """
#Y_real, A_real, X_real = data_generation.mix_signals(L, 10, m, n, k_true)

""" DATA GENERATION - ROSSLER DATA """
Y_real, A_real, X_real, k = data_generation.rossler_data(n_sampels=1940, ex = 1, m=8)

# =============================================================================
# Plots
# =============================================================================
plt.figure(1)
plt.title('One Signal of Rossler Data Set')
plt.plot(Y_real[0])
plt.xlabel('Samples (L)')
plt.savefig('Resultater/Rossler_Data_m8_n16_k6_L1940.png')
plt.show()