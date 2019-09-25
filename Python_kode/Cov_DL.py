# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:18:45 2019

@author: Laura

Topic: Covariance-Domain Dictionary Learning
"""
import numpy as np

Sf = 100            # Sample frequency in Hz
ts = 2              # Time segment lenght in seconds ts \in [2,4], 200 frames
Duration = 66       # Minutes
S = (Duration*60)/2 # Time segment in seconds
 
M = 32              # Sensor
"""
Linear Mixture Model:
    Ys = A * Xs
"""
def Cov_DL(N, M, Ls, X):
    Xs = X # Source matrix for time segment s
    Sigma_X = 1/Ls * Xs * Xs.T       # Source Sample Covariance
    Sigma_Y = np.zeros((M, ts * Sf)) # Sample Data Covariance
    D = 0  # Dictionary of size M(M+1)/2 x N -- BiGAMP-DL
    a = np.zeros(N) #
    for s in range(S):
        for i in range(N):
            delta = np.diag(Sigma_X)
            E, V = np.linalg.eig(np.linalg.inv(D[i])) 
            E1 = E[0] # Largest eigenvalue
            V1 = V[0] # Associated eigenvector
            
            a[i] = np.sqrt(E1) * V1 # Global minimum of optimisation problem
            
            Sigma_Y[s] = delta[i][s] * a[i] * a[i].T + Err[s]
    
    

#    Sigma_Ys = D * delta_s + Es
#    Sigma_Ys = A * Lambda_s * A.T + Es