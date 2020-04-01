# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 07:42:22 2020

@author: Mattek10

This is a script to test the initial A used in the Cov-DL algorithm when 
solving the optimization problem. See Cov_Dl2.
"""
from simulated_data import generate_AR
from simulated_data import mix_signals
from simulated_data import MSE_one_error
import numpy as np
import matplotlib.pyplot as plt
import Cov_DL
from Cov_DL import _dictionarylearning
from Cov_DL import _A
import M_SBL
from sklearn.decomposition import PCA

""" DATA GENERATION """
M = 8                          # number of sensors
k = 16                         # number of active sources
N = 16
L = 1000                       # number of sampels
n_seg = 1
duration = 10

Y, A_real, X_real = mix_signals(L, duration, M, N, k)
#Y, A_real, X_real = generate_AR(N, M, L, k)

Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))

err_listA = np.zeros(10)
err_listX = np.zeros(10)

Amse = np.zeros(3)
Xmse = np.zeros(3)

for ini in range(3):
    for ite in range(10):
        """ DIFFERENTS INITIAL A'S """
        A_list = [np.random.random((M,k)), np.random.uniform(-1,1,(M,k)), np.random.randn(M,k)]

        def Main_Algorithm(Y, M, L, n_seg, L_covseg = 10):
            
            #################################  Cov-DL  ####################################
        
            A_result = np.zeros((n_seg, M, N))   # matrices to store result of each segment
            X_result = np.zeros((n_seg, N, L-2))
        
            def Cov_DL1(Y_big, M, N, k):
                """
                something
                """
                #np.random.seed(12)
                D = _dictionarylearning(Y_big, N, k)
                A_rec = _A(D, M, N)
                return A_rec
        
            # funktion til brug i DL2:
            
            def Cov_DL2(Y_big, m, n, k):
                """ 
                """
        #        np.random.seed(12)
                # Dictionary Learning on Transformed System
                pca = PCA(n_components=n, svd_solver='randomized', whiten=True)
                pca.fit(Y_big.T)
                U = pca.components_.T
                A = A_list[ini]
        
                a = np.reshape(A, (A.size))     # normal vectorization of initial A
            
                def D_():
                    D = np.zeros((int(m*(m+1)/2), n))
                    for i in range(n):
                        A_tilde = np.outer(a[m*i:m*i+m], a[m*i:m*i+m].T)
                        D.T[i] = A_tilde[np.tril_indices(m)]
                    return D
            
                def D_term():
                    return np.dot(np.dot(D_(), (np.linalg.inv(np.dot(D_().T, D_())))),
                                  D_().T)
            
                def U_term():
                    return np.dot(np.dot(U, (np.linalg.inv(np.dot(U.T, U)))), U.T)
            
                def cost1(a):
                    return np.linalg.norm(D_term()-U_term())**2
                
                # predefined optimization method, without defineing the gradient og the cost.
                from scipy.optimize import minimize
#                print('før')
                res = minimize(cost1, a, method='nelder-mead',
                               options={'xatol': 1e-8, 'disp': True})
#                print('efter')
                a_new = res.x
                A_rec = np.reshape(a_new, (m, n))
                return A_rec
        
            for i in range(len(Y)):           # loop over all segments (axis 0 of Y) 
                print(i)
                Y_big = Cov_DL._covdomain(Y[i], L, L_covseg, M) # tansformation to covariace domain
                if N <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL2(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k > (M*(M+1))/2.:
                    raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
#                print('A_result{}'.format(A_result[0]))
            
            #################################  M-SBL  #####################################
                X_rec = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False)
#                print('efter X')
                X_result[i] = X_rec
        
            return A_result, X_result
        
#        print('A_real {}'.format(A_real))
        
        A_result, X_result = Main_Algorithm(Y, M, L, n_seg)
        
        err_listX[ite] = MSE_one_error(X_real.T[0:X_result[0].shape[1]].T,X_result[0])
        err_listA[ite] = MSE_one_error(A_real,A_result[0])
        print(err_listX[ite])
    
    Amse[ini] = np.average(err_listA)
    Xmse[ini] = np.average(err_listX)


#plot_seperate_sources_comparison(X_real,X_result[0],M,N,k,L)
    
plt.figure(1)
plt.plot(Amse, '-r', label = 'A')
plt.plot(0, Amse[0], 'ro')
plt.plot(1, Amse[1], 'ro')
plt.plot(2, Amse[2], 'ro')
plt.plot(Xmse, '-b', label = 'X')
plt.plot(0, Xmse[0], 'bo')
plt.plot(1, Xmse[1], 'bo')
plt.plot(2, Xmse[2], 'bo')
plt.title('MSE of A and X for variyng initial A')
plt.xticks([])
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/Mix_Error_initial_A_m8_k16_L1000.png')
plt.show()

