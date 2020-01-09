# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:04:55 2020

@author: trine
"""
import numpy as np
import matplotlib.pyplot as plt
import data_generation
from sklearn.metrics import mean_squared_error
import CovDL
import MSBL

np.random.seed(1)

""" Lists for variation """
<<<<<<< HEAD
list_ = np.array([16]) # for a single konstant
#list_ = np.arange(5,40,5)   # k vary
#list_ = np.arange(15,60+1,5)  # n vary
#list_ = np.arange(4,32+1,4)   # m vary

L_list = np.arange(10,10000,100)
=======
#list_ = np.array([5]) # for a single konstant
list_ = np.arange(5,40,5)   # k vary
#list_ = np.arange(15,60+1,5)  # n vary
#list_ = np.arange(4,32+1,4)   # m vary
>>>>>>> 285d68cd4f8f25afe4bcaaf5b375fc147d208191

err_listA = np.zeros(len(list_))
err_listX = np.zeros(len(list_))
err_listY = np.zeros(len(list_))

for i in range(len(list_)):
    print(list_[i]) 
<<<<<<< HEAD
    plot_list = np.zeros(len(L_list))
    
    for g in range(len(L_list)):
        sum_L = 0
        for p in range(10):
            L = L_list[g]
            
            " Case 1 "
            k = list_[i]
            m = 8
            n = 64
           
            " Case 3 "
        #    n = list_[i]        # Case 3
        #    m = 16
        #    k = int((list_[i]/3)*2)  # Case 3
        #    k = 4        # Case 3
        #    k = 20       # Case 3 - kan ikke køre den rigtig
            
            " Case 4 "
        #    m = list_[i]        # Case 4
        #    n = 64       # Case 4
        #    k = 20       # Case 4
        
            
            """ Generate AR data and Dividing in Segments """
            Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k) 
            
            Ls = L               # number of sampels per segment (L -> no segmentation) 
            Ys, Xs, n_seg = data_generation.segmentation_split(Y_real, X_real, Ls, L)
                                # return list of arrays -> segments in axis = 0
            
            sum_X = 0
            sum_A = 0
            for j in range(len(Ys)): # loop over segments 
                Y_real = Ys[j]
                X_real = Xs[j]
            
                cov_seg = L
        #        
                if n <= (m*(m+1))/2.:
                   # A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
                    raise SystemExit('D is over-determined')
        # input later        A_rec, A_err = CovDL.Cov_DL2(Y_real, A_real, X_real, m, n, cov_seg, L, k)
                    
                elif k <= (m*(m+1))/2.:
                    A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
                
                elif k > (m*(m+1))/2.:
                   # A_rec, A_err = CovDL.Cov_DL1(Y_real, A_real, X_real, m, n, cov_seg, L, k)
                    raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
                    
                sum_A += A_err
                
                
        ##     
                    
        #        X_rec = MSBL.M_SBL(A_real, Y_real, m, n, Ls, k, iterations=500, noise=False)
        #        X_real = X_real.T[:-2]
        #        X_real = X_real.T
        ##        print(np.count_nonzero(X_rec))
        #        sum_X += data_generation.norm_mse(X_real, X_rec)
        ##        print(sum_X)
        #    
            avg_err_A = sum_A/len(Ys)
            print(avg_err_A)
        #    avg_err_X = sum_X/len(Ys) 
        #    print(avg_err_X)
            sum_L += avg_err_A
    ##    
        err_listA[i] = avg_err_A
     #   err_listX[i] = avg_err_X
=======
    L = 1000
    
    " Case 1 "
    k = list_[i]
    m = 16
    n = 40
#   
    " Case 3 "
#    n = list_[i]        # Case 3
#    m = 16
#    k = int((list_[i]/3)*2)  # Case 3
#    k = 4        # Case 3
#    k = 20       # Case 3 - kan ikke køre den rigtig
    
    " Case 4 "
#    m = list_[i]        # Case 4
#    n = 64       # Case 4
#    k = 20       # Case 4
>>>>>>> 285d68cd4f8f25afe4bcaaf5b375fc147d208191

        plot_list[g] = sum_L/10
# print - change title ! 
    
plt.figure(1)

<<<<<<< HEAD
plt.plot(L_list, plot_list, '-o')
plt.title('A error- vary L, M = 8, N = 64')
plt.xlabel('L')
plt.ylabel('Norm. mse of recovered A')
plt.legend()
plt.savefig('Resultater/A_varying_L.png')
=======
plt.plot(list_, err_listX)
plt.title('X error- vary k, M = 16, N = 40, Ls = 100, n_seg = 10')
plt.xlabel('k')
plt.ylabel('Norm. mse of recovered X')
plt.savefig('Resultater/X_varying_k.png')
>>>>>>> 285d68cd4f8f25afe4bcaaf5b375fc147d208191
plt.show()

