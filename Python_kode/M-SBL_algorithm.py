# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:10:46 2019

@author: Laura
"""
import numpy as np

M = 20  # Sensors
N = (M*(M+1))/2# Sources
L = 1024 # Time samples
n = 1
k = N/5

A = np.zeros(M,N) # Dictionary Matrix
Y = np.zeros(M,L) # Observed Matrix
X = np.zeros(N,L)

gamma = np.zeros(N)
G = np.diag(gamma)
s = 0               # sigma**2
Sigma = A * G * A.T + s * np.identity(N)

""" Fixed Point Update """
for i in range(len(N)):
    gamma[i+1] = (gamma[i])/(np.sqrt(A[i].T * np.linalg.inv(Sigma[i])*A[i]))*(np.linalg.norm(Y.T * np.linalg.inv(Sigma[i]) * A[i], ord=2))/(np.sqrt(n))


S = np.nonzero(gamma)     