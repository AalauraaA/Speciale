# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:27:46 2019

@author: trine
"""
import numpy as np

Sf = 100            # Sample frequency in Hz
ts = 2              # Time segment lenght in seconds ts \in [2,4], 200 frames
Duration = 66       # Minutes
S = (Duration*60)/2 # Time segment in seconds
 
M = 8             # Sensor
N = 16 

"""
Linear Mixture Model:
    Ys = A * Xs
"""
### Random generation of data - single measurement
np.random.seed(1234)
x = np.random.randint(10, size=M)
yy = np.random.randint(100, size=N)
y = (yy-(np.sum(yy)/len(yy)))/np.max(yy) # substract mean and normalisation


# covariance transformation
z=np.vstack((x,x))
cov_x = np.cov(z.T)

z=np.vstack((y,y))
cov_y = np.cov(z.T)

cov_dim = int((M*(M+1))/2)
### check for underdetermined system
if N > cov_dim:
    print('N is not < (M*(M+1))/2')

### Dictionary learning 

   
# generate random D 
D = np.random.normal(0,1,([cov_dim,N]))
sigma_x = np.diag(cov_x)


### Find A

#A = np.zeros([M,N]) 
#for i in range(N):
#    temp = 
#    
  

#def Cov_DL(N, M, Ls, X):
#    Xs = X # Source matrix for time segment s
#    Sigma_X = 1/Ls * Xs * Xs.T       # Source Sample Covariance
#    Sigma_Y = np.zeros((M, ts * Sf)) # Sample Data Covariance
#    D = 0  # Dictionary of size M(M+1)/2 x N -- BiGAMP-DL
#    a = np.zeros(N) #
#    for s in range(S):
#        for i in range(N):
#            delta = np.diag(Sigma_X)
#            E, V = np.linalg.eig(np.linalg.inv(D[i])) 
#            E1 = E[0] # Largest eigenvalue
#            V1 = V[0] # Associated eigenvector
#            
#            a[i] = np.sqrt(E1) * V1 # Global minimum of optimisation problem
#            
#            Sigma_Y[s] = delta[i][s] * a[i] * a[i].T + Err[s]