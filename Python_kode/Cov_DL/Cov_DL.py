# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:27:46 2019

@author: trine
"""
#import os
#data_path = os.getcwd()[:-6]
#import sys
#sys.path.append(data_path)  # make the right path

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dictionary_learning import K_SVD
from sklearn.datasets import make_sparse_coded_signal
import data_generation

np.random.seed(1)

""" INITIALISING PARAMETERS """

m = 6               # number of sensors
n = 8               # number of sources
non_zero = 6        # max number of non-zero coef. in rows of X
n_sampels = 20      # number of sampels


""" DATA GENERATION """

Y, A_real, X_real = data_generation.mix_signals(n_sampels, 10)



""" SEGMENTATION """
L = 5               # number of sampels in one segment 
Ys, Xs, n_seg = data_generation.segmentation_split(Y, X_real, L, n_sampels)


""" Cov - DL """
# jeg har fundet i teksten at vi laver dictionary over alle segmenterne 
# som hver har en vector her, giver det mening ift stationaitet??  
  
Y_big = np.zeros([int(m*(m+1)/2.),n_seg])

for i in range(n_seg):              # loop over all segments
    # Transformation to covariance domain and vectorization
    Y_cov = np.cov(Ys[i])                      # covariance 
    X_cov = np.cov(Xs[i])                      # NOT diagonl ??
    
    # Vectorization of lower tri, row wise  
    vec_Y = np.array(list(Y_cov[np.tril_indices(m)]))
    vec_X = np.array(list(X_cov[np.tril_indices(n)]))
    sigma = np.diagonal(X_cov)
    
    Y_big.T[i] = vec_Y

# Y_big is now considered as a "new" observation vector, 
# and dictionary learning is performed      

n_samples = n_seg 

# Dictionary learning
D, sigma, iter_ = K_SVD(Y_big, n=len(sigma), m=len(Y_big),
                                 non_zero=non_zero, n_samples=n_samples,
                                 max_iter=100)

# results for the large system
Y_big_rec = np.matmul(D,sigma)
Y_err = np.linalg.norm(Y_big - Y_big_rec)

print('cov domain: reconstruction error %f \nnumber of iterations %i'%(Y_err, iter_))


# Find A approximative
def reverse_vec(x):
    m = int(np.sqrt(x.size*2))
    if (m*(m+1))//2 != x.size:
        print("reverse_vec fail")
        return None
    else:
        R, C = np.tril_indices(m)
        out = np.zeros((m, m), dtype=x.dtype)
        out[R, C] = x
        out[C, R] = x
    return out
    
A_rec = np.zeros(([m, n]))
for j in range(n):
    d = D.T[j]
    matrix_d = reverse_vec(d)
    E = np.linalg.eig(matrix_d)[0]
    V = np.linalg.eig(matrix_d)[1]
    max_eig = np.max(E)
    index = np.where(E == max_eig)
    max_vec = V[:, index[0]]        # using the column as eigen vector here
    temp = np.sqrt(max_eig)*max_vec
    A_rec.T[j] = temp.T
    

A_err = np.linalg.norm(A_real-A_rec)
    
print('Dictionary error %f'%(A_err))

#    
#""" prediction of X """
#def M_SBL_Seg(A, Y, m, n, n_seg, non_zero, n_samples):
#    gamma = np.ones([n_seg, n, 1])        # size n_seg x N x 1
#    lam = np.ones(n)                      # size N x 1
#    mean = np.zeros([n_seg, n, S])        # size n_seg x N x L
#    
#    for seg in range(n_seg):
#        Gamma = np.diag(gamma[seg])       # size 1 x 1
#        Sigma = 0                         # size N x L        
#        for i in range(n):   
#            " Making Sigma and Mean "
#            sig = lam[i] * np.identity(m) + (A[seg].dot(Gamma * A[seg].T))
#            inv = np.linalg.inv(sig)
#            Sigma = Gamma - Gamma * (A[seg].T.dot(inv)).dot(A[seg]) * Gamma
#            mean[seg] = Gamma * (A[seg].T.dot(inv)).dot(Y[seg])
#            
#            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
#            lam_num = 1/S * np.linalg.norm(Y[seg] - A[seg].dot(mean[seg]), 
#                                           ord = 'fro')  # numerator
#            lam_for = 0
#            for j in range(n):
#                lam_for += Sigma[j][j] / gamma[seg][j]
#            lam_den = m - n + lam_for                    # denominator
#            lam[i] =  lam_num / lam_den
#       
#            " Update gamma with EM and with M being Fixed-Point"
#            gam_num = 1/S * np.linalg.norm(mean[seg][i])
#            gam_den = 1 - gamma[seg][i] * Sigma[i][i]
#            gamma[seg][i] = gam_num/gam_den 

######################## IKKE FÆRDIGT ########################################        
#    support = np.zeros([n_seg, non_zero, 1])
#    gam = gamma[seg]
#    for l in range(non_zero):
#        if gamma[seg][np.argmax(gamma[seg])] != 0:
#            support[seg][l] = np.argmax(gamma[seg])
#        else:
#            gamma[seg][np.argmax(gamma[seg])] = 0
#
#    New_mean = np.zeros([n_seg, n, S])
#    for i in support[seg]:
#        New_mean[seg][int(i)] = mean[seg][int(i)]
##############################################################################
#    return mean
#
#X_rec = M_SBL_Seg(A_rec, Ys, m, n, n_seg, non_zero, S)
#
#""" Plot """
#plt.figure(1)
#plt.subplot(3, 1, 1)
#plt.plot(seg_time[0], Ys[0].T)
#plt.xlabel('[sec.]')
#plt.title("Measurements")
#
#plt.subplot(3, 1, 3)
#plt.plot(seg_time[0], Xs[0].T)
#plt.xlabel('[sec.]')
#plt.title("real sources")
#
#
#plt.figure(2)
#plt.plot(Xs[0].T[2])
#plt.plot(X_rec[0].T[2])
#plt.show
#
#plt.subplot(3,1,3)
#plt.plot(predicted_X)
#plt.title("predicted sources")
#plt.show()
#