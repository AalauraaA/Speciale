# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:52:37 2020

@author: Laura
"""

from main import Main_Algorithm
from simulated_data import generate_AR
from simulated_data import MSE_one_error
from plot_functions import plot_seperate_sources_comparison

import numpy as np
import matplotlib.pyplot as plt

M = 16
N = [16, 32]
k = [16, 32]
L = 1000 
n_seg = 1

err_listA = np.zeros(10)
err_listX = np.zeros(10)

Amse = np.zeros(len(N))
Xmse = np.zeros(len(N))

for n in range(len(N)):
    Y, A_real, X_real = generate_AR(N[n], M, L, k[n])
    Y = np.reshape(Y, (1, Y.shape[0], Y.shape[1]))
    print("Next choice for N")
    for ite in range(10): 
        A_result, X_result = Main_Algorithm(Y, M, N[n], k[n], L, n_seg, L_covseg = 30)

        err_listX[ite] = MSE_one_error(X_real.T[0:X_result[0].shape[1]].T,X_result[0])
        err_listA[ite] = MSE_one_error(A_real,A_result[0])
    Xmse[n] = np.average(err_listX)
    Amse[n] = np.average(err_listA)

""" PLOTS """
plt.figure(1)
plt.plot(Amse, '-r', label = 'A')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')

plt.plot(Xmse, '-b', label = 'X')
plt.plot(0, Xmse[0], 'bo')
plt.plot(1, Xmse[1], 'bo')

plt.title('MSE of A and X for varying N')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/AR_Error_vary_n_m16_L1000.png')
plt.show()

