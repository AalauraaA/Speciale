# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura

This Python Script test the relationship between n (amount of sources)
and k (amount of active sources).

The tests are performed on the cases:
    - k = 1, n = 5, m = 3, true k = 2  (k < n and k = k true)
    - k = 5, n = 5, m = 3, true k = 2  (k = n and k > k true)
    
    - k = 1, n = 6, m = 4, true k = 5  (k < n and k < k true)
    - k = 6, n = 6, m = 4, true k = 6  (k = n and k = k true)
    
    - k = 1, n = 8, m = 6, true k = 8  (k < n and k < k true)
    - k = 5, n = 8, m = 6, true k = 8  (k < n and k < k true)
    - k = 8, n = 8, m = 6, true k = 8  (k = n and k = k true)
"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation

import CovDL
import MSBL


#np.random.seed(154)

""" Lists for variation """
list_ = [1, 6, 8] # k

m = 3 
#m = 4                       # number of sensors
#m = 6                        # number of sensors   

n = 5                       # number of sources
#n = 6
#n = 8                       # number of sources

k_true = 2        # true number of non-zeros
#k_true = 6        # true number of non-zeros
#k_true = 8        # true number of non-zeros 

L = 100                     # number of sampels

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))

errA = np.zeros(len(list_))
errX = np.zeros(len(list_))

X_rec_list = np.zeros((len(list_), n, L-2))
X_real_list = np.zeros((len(list_), n, L-2))

for l in range(len(list_)):
    print(l)
    
    k = list_[l]                 # max number of non-zeros
    
    Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 
#    Y_real, A_real, X_real = data_generation.mix_signals(L, 0.8, m, n, k_true) 
    
    """ SEGMENTATION - OVER ALL """
    Ls = L                  # number of sampels per segment (L -> no segmentation) 
    
    Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                            # return list of arrays -> segments in axis = 0
    
    """ COV-DL and M-SBL """
    for i in range(len(Ys)): # loop over segments 
        Y_real = Ys[i]
        X_real = Xs[i]
       
        def cov_seg_max(n, L):
            """
            Give us the maximum number of segment within the margin.
            """
            n_seg = 1
            while int(n) > n_seg:
                n_seg += 1
            return int(L/n_seg)    # Samples within one segment   
       
        
#            cov_seg = cov_seg_max(n, L)
        cov_seg = 10
    
        """ Cov-DL """
        if n <= (m*(m+1))/2.:
            A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
            
        elif k <= (m*(m+1))/2.:
            A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
        
        elif k > (m*(m+1))/2.:
            raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
            
        """ M-SBL """
        X_rec = MSBL.M_SBL(A_rec, Y_real, m, n, Ls, k, iterations=1000, noise=False)
        X_real = X_real.T[:-2]
        X_real = X_real.T
        
        """ Calculating errors between true and estimated A and X """
    err_listA[l] = data_generation.MSE_one_error(A_real, A_rec)
    err_listX[l] = data_generation.MSE_one_error(X_real, X_rec)
    
    X_rec_list[l] = X_rec
    X_real_list[l] = X_real

""" PLOTS """
plt.figure(1)
nr_plot=0
for i in range(len(X_real_list[0].T[0])):
    if np.any(X_real_list[0][i]!=0) or np.any(X_rec_list[0][i]!=0):
        
        nr_plot += 1
        plt.subplot(n, 1, nr_plot)
       
        plt.plot(X_real_list[0][i], 'r',label='Real X')
        plt.plot(X_rec_list[0][i],'g', label='Recovered X')         
plt.legend()
plt.xlabel('sample')
plt.suptitle('k = 1, n = 5, m = 3, L = 100, true k = 2')
#plt.suptitle('k = 1, n = 6, m = 4, L = 100, true k = 6')
#plt.suptitle('k = 1, n = 8, m = 6, L = 100, true k = 8')
plt.show
plt.savefig('Resultater/X_comparison_k1_n5_m3_L100_truek2')
#plt.savefig('Resultater/X_comparison_k1_n6_m4_L100_truek6')
#plt.savefig('Resultater/X_comparison_k1_n8_m6_L100_truek8')

plt.figure(2)
nr_plot=0
for i in range(len(X_real_list[1].T[0])):
    if np.any(X_real_list[1][i]!=0) or np.any(X_rec_list[1][i]!=0):
        
        nr_plot += 1
        plt.subplot(n, 1, nr_plot)
       
        plt.plot(X_real_list[1][i], 'r',label='Real X')
        plt.plot(X_rec_list[1][i],'g', label='Recovered X')       
plt.legend()
plt.xlabel('sample')
plt.suptitle('k = 6, n = 5, m = 3, L = 100, true k = 2')
#plt.suptitle('k = 6, n = 6, m = 4, L = 100, true k = 6')
#plt.suptitle('k = 6, n = 8, m = 6, L = 100, true k = 8')
plt.show
plt.savefig('Resultater/X_comparison_k6_n5_m3_L100_truek2')
#plt.savefig('Resultater/X_comparison_k6_n6_m4_L100_truek6')
#plt.savefig('Resultater/X_comparison_k6_n8_m6_L100_truek8')

#plt.figure(3)
#nr_plot=0
#for i in range(len(X_real_list[2].T[0])):
#    if np.any(X_real_list[2][i]!=0) or np.any(X_rec_list[2][i]!=0):
#        
#        nr_plot += 1
#        plt.subplot(n, 1, nr_plot)
#       
#        plt.plot(X_real_list[2][i], 'r',label='Real X')
#        plt.plot(X_rec_list[2][i],'g', label='Recovered X')       
#plt.legend()
#plt.xlabel('sample')
#plt.suptitle('k = 8, n = 8, m = 6, L = 100, true k = 8')
#plt.show
#plt.savefig('Resultater/X_comparison_k8_n8_m6_L100_truek8')


"""
MSE Error Results:
    - k = 1, m = 3, n = 5, k true = 2 --> errX = 1.29 & errA = 10.55
    - k = 6, m = 3, n = 5, k true = 2 --> errX = 1.50 & errA = 10.55
    
    - k = 1, m = 4, n = 6, k true = 6 --> errX = 3.71 & errA = 3.64
    - k = 6, m = 4, n = 6, k true = 6 --> errX = 4.68 & errA = 3.64
    
    - k = 1, m = 6, n = 8, k true = 8 --> errX = 6.04 & errA = 8.63
    - k = 6, m = 6, n = 8, k true = 8 --> errX = 8.50 & errA = 8.63
    - k = 8, m = 6, n = 8, k true = 8 --> errX = 8.74 & errA = 8.63
"""