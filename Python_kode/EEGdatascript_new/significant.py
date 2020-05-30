# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:59:06 2020

@author:
"""
from main import Main_Algorithm
import matplotlib.pyplot as plt
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison
import data
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
import simulated_data

np.random.seed(11)

M = 3 
N = 5
k = 4
L = 1000
n_seg = 1

Y, A_real, X_real = simulated_data.mix_signals(L,M,version=None)
#Y, A_real, X_real = generate_AR(N, M, L, k)

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))

X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_fix = np.random.normal(0,2,A_real.shape)

A_result, X_result = Main_Algorithm(Y, M, L, n_seg, A_fix, L_covseg=10)

mse_array, mse_avg = simulated_data.MSE_segments(X_real[0],X_result[0])
A_mse = MSE_one_error(A_real,A_result[0])

pear = stats.pearsonr(X_real[0][2], X_result[0][2])

print(stats.ttest_rel(a=X_real.T[0],b=X_result.T[0]))
