# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:32:32 2020

@author: Laura

Testing the relationship between n and k
"""

import numpy as np
import matplotlib.pyplot as plt
import data_generation

import CovDL
import MSBL


#np.random.seed(154)

""" Lists for variation """
list_ = [2, 5, 8] # k

m = 3 
#m = 5                       # number of sensors
#m = 6                        # number of sensors   

n = 5                       # number of sources
#n = 6
#n = 8                       # number of sources

k_true = 2        # true number of non-zeros
#k_true = 5        # true number of non-zeros
#k_true = 8        # true number of non-zeros 

L = 100                     # number of sampels

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))

errA = np.zeros(len(list_))
errX = np.zeros(len(list_))

X_rec_list = np.zeros((len(list_), n, L-2))
X_real_list = np.zeros((len(list_), n, L-2))

for l in range(len(list_)):
    for ite in range(len(list_)):
        print(l)
        
        k = list_[l]                 # max number of non-zeros
        
        Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 
        
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
            err_listA[ite] = A_err
            err_listX[ite] = data_generation.norm_mse(X_real, X_rec)
    
    X_rec_list[l] = X_rec
    X_real_list[l] = X_real
    
    errA[l] = np.average(err_listA)
    errX[l] = np.average(err_listX)
    

""" PLOTS """
plt.figure(1)
plt.plot(list_, errX)
plt.title('X error- vary k, n = 5, m = 3, true k = 2')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered X')
plt.savefig('Resultater/X_varying_k_n5_m3_L100_covseg10_ktrue=2.png')
plt.show()

plt.figure(2)
plt.plot(list_, errA)
plt.title('A error- vary k, n = 5, m = 3, true k = 2')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered A')
plt.savefig('Resultater/A_varying_k_n5_m3_L100_covseg10_ktrue=2.png')
plt.show()

plt.figure(3)
nr_plot=0
for i in range(len(X_real_list[0].T[0])):
    if np.any(X_real_list[0][i]!=0) or np.any(X_rec_list[0][i]!=0):
        
        nr_plot += 1
        plt.subplot(n, 1, nr_plot)
        plt.title('k = 2, n = 5, m = 3, L = 100, true k = 2')
       
        plt.plot(X_real_list[1][i], 'r',label='Real X')
        plt.plot(X_rec_list[1][i],'g', label='Recovered X')         
plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('Resultater/X_comparison_k2_n5_m3_L100_truek2')

plt.figure(4)
nr_plot=0
for i in range(len(X_real_list[1].T[0])):
    if np.any(X_real_list[1][i]!=0) or np.any(X_rec_list[1][i]!=0):
        
        nr_plot += 1
        plt.subplot(n, 1, nr_plot)
        plt.title('k = 5, n = 5, m = 3, L = 100, true k = 2')
       
        plt.plot(X_real_list[1][i], 'r',label='Real X')
        plt.plot(X_rec_list[1][i],'g', label='Recovered X')       
plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('Resultater/X_comparison_k5_n5_m3_L100_truek2')

plt.figure(5)
nr_plot=0
for i in range(len(X_real_list[2].T[0])):
    if np.any(X_real_list[2][i]!=0) or np.any(X_rec_list[2][i]!=0):
        
        nr_plot += 1
        plt.subplot(n, 1, nr_plot)
        plt.title('k = 8, n = 5, m = 3, L = 100, true k = 2')
       
        plt.plot(X_real_list[2][i], 'r',label='Real X')
        plt.plot(X_rec_list[2][i],'g', label='Recovered X')       
plt.legend()
plt.xlabel('sample')
plt.show
plt.savefig('Resultater/X_comparison_k8_n5_m3_L100_truek2')

