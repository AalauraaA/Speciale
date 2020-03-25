# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 07:42:22 2020

@author: Mattek10

This is a script to test the initial A used in the Cov-DL algorithm when 
solving the optimization problem. See Cov_Dl2.
"""
import numpy as np
import data_generation
import dictionary_learning
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import MSBL

#np.random.seed(4590)
# =============================================================================
# Definitions
# =============================================================================
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

def Cov_DL1(Y, A, X, m, n, cov_seg, L, k):
    """ 
    """
    ## internal segmetation for training examples
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, cov_seg, L)
    
    ## Covariance Domain Tranformation
    
    Y_big = np.zeros([int(m*(m+1)/2.),n_seg])
    for i in range(n_seg):              # loop over all segments
        
        # Transformation to covariance domain and vectorization
        Y_cov = np.corrcoef(Ys[i])                      # covariance 
        X_cov = np.corrcoef(Xs[i])                      # NOT diagonl ??
        
        # Vectorization of lower tri, row wise  
        vec_Y = Y_cov[np.tril_indices(m)]
        
        sigma = np.diagonal(X_cov)
    
        Y_big.T[i] = vec_Y
    
    ## Dictionary Learning on Transformed System
    L = n_seg 
    
    Y_big_rec, D, sigma = dictionary_learning.DL(Y_big.T, n=len(sigma), k=k)

    Y_err = np.linalg.norm(Y_big - Y_big_rec.T)

    print('cov domain: reconstruction error %f'%(Y_err))
    
    ## Find Mixing Matrix From D 
   
    A_rec = np.zeros(([m, n]))
    for j in range(n):
        d = D.T[j]
        matrix_d = reverse_vec(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(np.abs(E))     # should it be abselut here?
        index = np.where(np.abs(E) == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigen vector here
        temp = np.sqrt(max_eig)*max_vec
        A_rec.T[j] = temp.T

    A_err = data_generation.MSE_one_error(A,A_rec)
        
    print('Dictionary error %f'%(A_err))
    
    return A_rec, A_err

def Cov_DL2(Y, A, X, m, n, cov_seg, L, k):
    """ 
    """
    ## internal segmetation for training examples
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y, X, cov_seg, L)
    
    ## Covariance Domain Tranformation
    
    Y_big = np.zeros([int(m*(m+1)/2.),n_seg])
    for i in range(n_seg):              # loop over all segments
        
        # Transformation to covariance domain and vectorization
        Y_cov = np.cov(Ys[i])                      # covariance 
        X_cov = np.cov(Xs[i])                      # NOT diagonl ??
#        print(X_cov)
        # Vectorization of lower tri, row wise  
        vec_Y = Y_cov[np.tril_indices(m)]
        
        sigma = np.diagonal(X_cov)
    
        Y_big.T[i] = vec_Y
    
    ## Dictionary Learning on Transformed System
    L = n_seg  
    
    pca = PCA(n_components=n, svd_solver='randomized',
          whiten=True) 
    pca.fit(Y_big.T)
    U = pca.components_.T
    
    a = np.reshape(A,(A.size)) # vectorization of initial A
    
    def D_(a):
        D = np.zeros((int(m*(m+1)/2),n))
        for i in range(n):
            A_tilde = np.outer(a[m*i:m*i+m],a[m*i:m*i+m].T)
            D.T[i] = A_tilde[np.tril_indices(m)]
        return D
    
    def D_term(a):
        return np.dot(np.dot(D_(a),(np.linalg.inv(np.dot(D_(a).T,D_(a))))),D_(a).T)
    
    def U_term():
        return np.dot(np.dot(U,(np.linalg.inv(np.dot(U.T,U)))),U.T)
    
    def cost1(a):
        return np.linalg.norm(D_term(a)-U_term())**2
        
    
    # predefined optimization method, without defined the gradient og the cost. 
    from scipy.optimize import minimize
    res = minimize(cost1, a, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True})
    a_new = res.x
    A_rec = np.reshape(a_new,(m,n)) 
    A_err = data_generation.MSE_one_error(A,A_rec)
    return A_rec, A_err

# =============================================================================
# Testing the A's
# =============================================================================
""" DATA GENERATION """
m = 8                          # number of sensors
k = 16                         # number of active sources
L = 1000                       # number of sampels
#duration = 10

#Y_real, A_real, X_real = data_generation.mix_signals(L, duration, m, k, k)

Y_real, A_real, X_real = data_generation.generate_AR_v2(k, m, L, k) 

err_listA = np.zeros(10)
err_listX = np.zeros(10)
Amse = np.zeros(3)
Xmse = np.zeros(3)

""" SEGMENTATION - OVER ALL """
Ls = L                  # number of sampels per segment (L -> no segmentation) 
Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                        # return list of arrays -> segments in axis = 0
""" COV - DL """
for a in range(3):
    for ite in range(10):
        """ DIFFERENTS INITIAL A'S """
        A = [np.random.random((m,k)), np.random.uniform(-1,1,(m,k)), np.random.randn(m,k)]
        

        for i in range(len(Ys)): # loop over segments 
            Y_real = Ys[i]
            X_real = Xs[i]
           
            def cov_seg_max(n, L):
                """
                Give us the maximum number of segment within the margin.
                For some parameters (low) you can add one more segment.
                """
                n_seg = 1
                while int(n) > n_seg:
                    n_seg += 1
                return int(L/n_seg)    # Samples within one segment   
        
            cov_seg = 10
        
            if k <= (m*(m+1))/2.:
                A_rec, A_err = Cov_DL2(Y_real, A[a], X_real, m, k, cov_seg, L, k)
                
            elif k <= (m*(m+1))/2.:
                A_rec, A_err = Cov_DL1(Y_real, A[a], X_real, m, k, cov_seg, L, k)
            
            elif k > (m*(m+1))/2.:
                raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
                
        X_rec = MSBL.M_SBL(A_rec, Y_real, m, k, Ls, k, iterations=1000, noise=False)
        X_real = X_real.T[:-2]
        X_real = X_real.T
                
        """ ERROR """   
        err_listA[ite] = A_err
        err_listX[ite] = data_generation.MSE_one_error(X_real, X_rec)
    
    Amse[a] = np.average(err_listA)
    Xmse[a] = np.average(err_listX)

    
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
plt.savefig('AR_Error_initial_A_m8_k16_L1000.png')
plt.show()

"""
The average error of A and X are:
    Amse: 1.34135856, 3.66556989, 3.94293637
    Xmse: 21.62930811, 10.23301089,  6.18059169
"""

