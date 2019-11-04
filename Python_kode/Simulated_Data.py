# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:53:17 2019

@author: Mattek9b
"""
import numpy as np
import matplotlib.pyplot as plt

from Rossler import Generate_Rossler
from ICA_algorithm import ica

# =============================================================================
# Rossler Data - Generate 6 Source Signals from 6 Sensors
# =============================================================================
X1, X2, X3, X4, X5, X6 = Generate_Rossler()

"Subtract the 6 sensors/sources from the solution space"
X01 = X1.T[0]
X02 = X1.T[1]
X03 = X1.T[2]
X04 = X1.T[3]
X05 = X1.T[4]
X06 = X1.T[5]

#X01 = X2.T[0]
#X02 = X2.T[1]
#X03 = X2.T[2]
#X04 = X2.T[3]
#X05 = X2.T[4]
#X06 = X2.T[5]
#
#X01 = X3.T[0]
#X02 = X3.T[1]
#X03 = X3.T[2]
#X04 = X3.T[3]
#X05 = X3.T[4]
#X06 = X3.T[5]
#
#X01 = X4.T[0]
#X02 = X4.T[1]
#X03 = X4.T[2]
#X04 = X4.T[3]
#X05 = X4.T[4]
#X06 = X4.T[5]
#
#X01 = X5.T[0]
#X02 = X5.T[1]
#X03 = X5.T[2]
#X04 = X5.T[3]
#X05 = X5.T[4]
#X06 = X5.T[5]
#
#X01 = X6.T[0]
#X02 = X6.T[1]
#X03 = X6.T[2]
#X04 = X6.T[3]
#X05 = X6.T[4]
#X06 = X6.T[5]

""" Generate Y Data """
X_ori = np.c_[X01, X02, X03, X04, X05, X06].T      # Original X sources - 40 x 6
A = np.random.randn(X_ori.shape[0],X_ori.shape[0]) # Random mixing matrix A - 6 x 6
Y = np.dot(A, X_ori)                               # Observed signal Y - 40 x 6

X_cal = ica(Y, iterations=1000)                    # Computed X sources - 40 x 6     

""" Plot the Y, X_ori and X_cal Data """
plt.figure(1)
plt.subplot(3, 1, 1)
for y in Y:
    plt.plot(y)
plt.title("Observed Mixtures Y")

plt.subplot(3, 1, 2)
for x in X_ori:
    plt.plot(x)
plt.title("Real Sources X")

plt.subplot(3,1,3)
for s in X_cal:
    plt.plot(s)
plt.title("Calculated Sources X")
plt.show()

# =============================================================================
# AR Data - This is to generating non-linear AR data.
# =============================================================================
#def MixingMatrix(M,N):
#    """
#    Generating a M x N Normal Distributed Matrix.
#    For ICA use select M = N
#    """
#    return np.random.randn(M,N)
#
#def generate_AR(N):
#    """
#    Generate sources from an AR process
#    
#    Input:
#        N: size of the columns
#        
#    Output:
#        X: Source matrix of size 4 x N -- 4 can be change by adding more
#           AR processes
#    
#    """
##    A = np.array([[0.05, 0, 0, 0, 0, 0, 0, 0, 0],
##                  [0, -0.05, 0, 0, 0, 0, 0, 0, 0],
##                  [0, 0, 0.3, 0, 0, 0, 0, 0, 0],
##                  [0, 0, 0, 0.4, 0, 0, 0, 0, 0],
##                  [0, 0, 0, 0, 0.8, 0, 0, 0, 0],
##                  [0, 0, 0, 0, 0, 0.06, 0, 0, 0],
##                  [0, 0, 0, 0, 0, 0, 0.2, 0, 0],
##                  [0, 0, 0, 0, 0, 0, 0, 0.2, 0],
##                  [0, 0, 0, 0, 0, 0, 0, 0, 0.05]])
##    A = MixingMatrix(N,N)    
#    XX1 = np.zeros(N)
#    XX2 = np.zeros(N)
#    XX3 = np.zeros(N)
#    XX4 = np.zeros(N)
#    
#    LP = 200
#
#    " Generating Synthetic AR Data "
#    w = np.random.randn(4, N)
#    for j in range(2,N):
##        for i in range(A.shape[0]):
##            XX1[j-2] = A[0][0] * XX1[j-1] - A[1][1] * XX1[j-2] + w[0, j-2]
##            XX2[j-2] = A[2][2] * XX1[j-1] + A[3][3] * XX3[j-1] + w[1, j-2]
##            XX3[j-2] = A[4][4] * XX1[j-1]**2 + A[5][5] * XX3[j-1] + A[6][6] * XX2[j-1] + w[2, j-2]
##            XX4[j-2] = A[7][7] * XX4[j-1] + A[8][8] * XX1[j-1] + w[3, j-2]
#            XX1[j-2] = 0.05 * XX1[j-1] - (-0.05) * XX1[j-2] + w[0, j-2]
#            XX2[j-2] = 0.3 * XX1[j-1] + 0.4 * XX3[j-1] + w[1, j-2]
#            XX3[j-2] = 0.8 * XX1[j-1]**2 + 0.06 *XX3[j-1] + 0.2 * XX2[j-1] + w[2, j-2]
#            XX4[j-2] = 0.2 * XX4[j-1] + 0.05 * XX1[j-1] + w[3, j-2]
#    XX1 = XX1[LP+1 : -1]
#    XX2 = XX2[LP+1 : -1]
#    XX3 = XX3[LP+1 : -1]
#    XX4 = XX4[LP+1 : -1]
#    
#    F = np.vstack([XX1, XX2, XX3, XX4])
#    return F
#
#" Try to use ICA "
#A = MixingMatrix(4,4)    # 4 x 4 matrix
#X = generate_AR(208)     # 4 x 6 matrix -- N must be 208 or higher or change LP
#
#Y = np.dot(A, X)         # 4 x 6 matrix
#X_pre = ica(Y, iterations=1000)  # 4 x 6 matrix      
#
#plt.figure(1)
#plt.subplot(3, 1, 1)
#for y in Y:
#    plt.plot(y)
#plt.title("Observed Mixtures Y")
#
#plt.subplot(3, 1, 2)
#for x in X:
#    plt.plot(x)
#plt.title("Real Sources X")
#
#plt.subplot(3,1,3)
#for z in X_pre:
#    plt.plot(z)
#plt.title("Calculated Sources X")
#plt.show()