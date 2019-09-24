# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:18:45 2019

@author: Laura

Topic: Covariance-Domain Dictionary Learning
"""
import numpy as np

Y = 0 #EEG data matrix size M x Nd
S_f = 100 # Sample frequency in Hz
Y_s = 0 # Overlapping segment of EEG data matrix. Size M x ts S_f
ts = 2 # time segment in seconds ts \in [2,4], 200 frames
Nd = 0

# 
"""
Linear Mixture Model Ys = A Xs for all s
with Ys Ys^T = A Xs Xs^T A^T and sample data covariance
Sigma_Ys = 1/Ls Ys Ys^T for each segment s
"""
def Cov_DL(N, M, Ls):
    Xs = 0 # Source matrix for time segment s
    Sigma_Xs = 1/Ls * Xs * Xs.T

    delta_s = np.diag(Sigma_Xs)
    D = 0  # Dictionary of size M(M+1)/2 x N -- BiGAMP-DL
    for i in range(N):
        D[i] = a[i] * a[i].T

    Sigma_Ys = D * delta_s + Es
    
    Lambda = 0
    for i in range(N):
        Sigma_Ys[i] = Lambda_s[i][i] * D[i] + Es[i] # D[i] = a[i] * a[i].T
    
    Sigma_Ys = A * Lambda_s * A.T + Es