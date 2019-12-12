# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:25:41 2019

@author: Laura

Auto-Regressive Processes
"""
import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(1)

N = 10
M = 3
L = 100
non_zero = 5

def generate_AR_v1(N, M, L, non_zero):
    """
    Generate sources from an AR process
    
    Input:
        N: size of the rows (amount of sources)
        L: size of the columns (amount of samples)
        
    Output:
        X: Source matrix of size N x L     
    """
    A = np.random.uniform(-1,1, (N,L))
    X = np.zeros([N, L+2])
    W = np.random.randn(N, L)
    for i in range(N):
        for j in range(2,L):
            X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i][j-2] + W[i][j]
    
    " Making zero and non-zero rows "
    Real_X = np.zeros([N, L+2])
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,N)
        while temp in ind:
            temp = np.random.randint(0,N)
        ind[i] = temp
    
    for j in ind:
        k = np.random.choice(len(X))
        Real_X[int(j)] = X[k]
    
    Real_X = Real_X.T[2:].T  
    
    " System "
    A_Real = np.random.randn(M,N)
    Y_Real = np.dot(A_Real, Real_X)    
    return Y_Real, A_Real, Real_X

def generate_AR_v2(N, M, L, non_zero):
    """
    Generate sources from an AR process
    
    Input:
        N: size of the rows (amount of sources)
        L: size of the columns (amount of samples)
        
    Output:
        X: Source matrix of size N x L     
    """
    A = np.random.uniform(-1,1, (N,L))
    X = np.zeros([N, L+2])
    W = np.random.randn(N, L)
    for i in range(N):
        ind = np.random.randint(1,4)
        for j in range(2,L):
            if ind == 1:
                X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i][j-2] + W[i][j]
            
            elif ind == 2: 
                X[i][j] = A[i][j-1] * X[i-1][j-1] + A[i][j-2] * X[i][j-1] + W[i][j]
                
            elif ind == 3:
                X[i][j] = A[i][j] * X[i-2][j-1] + A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i-1][j-1] + W[i][j]
            
            elif ind == 4:
                X[i][j] = A[i][j-1] * X[i][j-1] + A[i][j-2] * X[i-3][j-1] + W[i][j]
    
    " Making zero and non-zero rows "
    Real_X = np.zeros([N, L+2])
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,N)
        while temp in ind:
            temp = np.random.randint(0,N)
        ind[i] = temp
    
    for j in ind:
        k = np.random.choice(len(X))
        Real_X[int(j)] = X[k]
    
    Real_X = Real_X.T[2:].T  

    " System "
    A_Real = np.random.randn(M,N)
    Y_Real = np.dot(A_Real, Real_X)    
    return Y_Real, A_Real, Real_X

Y1, A1, X1 = generate_AR_v1(N, M, L, non_zero)
Y2, A2, X2 = generate_AR_v2(N, M, L, non_zero)

plt.figure(1)
plt.subplot(4, 1, 1)
plt.title('Comparison of X1 and X2')
plt.plot(X1[0], 'r',label='Fixed Coe')
plt.plot(X2[0],'g', label='Random Coe')

plt.subplot(4, 1, 2)
plt.plot(X1[1], 'r',label='Fixed Coe')
plt.plot(X2[1],'g', label='Random Coe')

plt.subplot(4, 1, 3)
plt.plot(X1[2], 'r',label='Fixed Coe')
plt.plot(X2[2],'g', label='Random Coe')

plt.subplot(4, 1, 4)
plt.plot(X1[3], 'r',label='Fixed Coe')
plt.plot(X2[3],'g', label='Random Coe')
plt.legend()
plt.show

