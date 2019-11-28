# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:18:49 2019

@author: Laura

Test of M-SBL
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import MSBL
np.random.seed(1)

M = 25 # N =25
N = 50 # M = 50
k = 16 # D =16
L = np.linspace(1,5,5)

A = np.zeros((M,N))
for m in range(M):
    for n in range(N):
        A[m][n] = np.random.uniform(0,1)

" L = 1 "       
X1 = np.zeros((N, int(L[0])))
index = np.random.choice(range(0,N), (k,1))
for i in index:
    if np.count_nonzero(X1[i][0]) == 0:
        X1[i] = np.random.random(int(L[0]))       
Y1 = A.dot(X1)
X1_rec = MSBL.M_SBL(A, Y1, M, N, int(L[0]), k, iterations=10, noise=False)

" L = 2 "
X2 = np.zeros((N, int(L[1])))
index = np.random.choice(range(0,N), (k,1))
for i in index:
    if np.count_nonzero(X2[i][0]) == 0:
        X2[i] = np.random.random(int(L[1]))        
Y2 = A.dot(X2)
X2_rec = MSBL.M_SBL(A, Y2, M, N, int(L[1]), k, iterations=10, noise=False)

" L = 3 "
X3 = np.zeros((N, int(L[2])))
index = np.random.choice(range(0,N), (k,1))
for i in index:
    if np.count_nonzero(X3[i][0]) == 0:
        X3[i] = np.random.random(int(L[2]))       
Y3 = A.dot(X3)
X3_rec = MSBL.M_SBL(A, Y3, M, N, int(L[2]), k, iterations=10, noise=False)

X = [X1, X2, X3]
X_rec = [X1_rec, X2_rec, X3_rec] 
mse = np.zeros((len(X),1))
for i in range(len(X)):
    mse[i] = mean_squared_error(X[i], X_rec[i])
    
#print("This is the error of X without: ", mse)


