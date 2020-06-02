# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:35:12 2020

@author: trine
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:11:57 2020

@author: Laura


Compare reulst with different mixing matrices instead of Cov-DL A.
"""
from main_A import Main_Algorithm
import vary_A
from simulated_data import generate_AR
from simulated_data import MSE_one_error
import numpy as np
import matplotlib.pyplot as plt
import data

np.random.seed(12345)

M = 6
N = 8  
k = 8
L = 1000 
n_seg = 1

Y, A_real, X_real = generate_AR(N, M, L, k) #(N,M) is real
X_real = X_real.T[:-2]
X_real = X_real.T

iterations = 100 

SNR = np.linspace(2,0.01,iterations)

X_mse = np.zeros(iterations)
A_mse = np.zeros(iterations)
plt.figure(0)
for i in range(iterations):
    #print(i)
    power_signal = np.mean(A_real)
    
    
    for j in range(100):
        # add noise
        mu = 0
        std = np.sqrt(power_signal/SNR[i])
        print(std)
        noise = np.random.normal(mu,std,(A_real.shape))
        
        A = A_real + noise 
    
        X_result = Main_Algorithm(Y, A, M, N, k, L, n_seg)
            
        X_mse[i] += MSE_one_error(X_real,X_result)  
        A_mse[i] += MSE_one_error(A_real,A)
    
    X_mse[i] = X_mse[i]/100
    A_mse[i] = A_mse[i]/100
    
    # plotting noise signal
    plt.plot(np.reshape(A,A.size), label=i)
    plt.legend()
   

plt.show()
""" PLOTS """
plt.figure(1)
plt.title('MSE($\mathbf{X},\hat{\mathbf{X}}}$) for varying SNR of $\hat{\mathbf{A}}$')
plt.plot(SNR,X_mse,'-b', label = 'MSE($\mathbf{X},\hat{\mathbf{X}}$)')
#plt.plot(SNR,A_mse,'-r', label = 'MSE($\mathbf{A},\hat{\mathbf{A}}$)')
plt.ylabel('MSE')
plt.xlabel('SNR')
plt.legend()
#plt.tight_layout()      
plt.savefig('X_func_SNR.png')
plt.show()

plt.figure(2)
plt.title('MSE($\mathbf{A},\hat{\mathbf{A}}}$) for varying SNR of $\hat{\mathbf{A}}$')
#plt.plot(SNR,X_mse,'-b', label = 'MSE($\mathbf{X},\hat{\mathbf{X}}$)')
plt.plot(SNR,A_mse,'-r', label = 'MSE($\mathbf{A},\hat{\mathbf{A}}$)')
plt.ylabel('MSE')
plt.xlabel('SNR')
plt.legend()
#plt.tight_layout()      
plt.savefig('A_func_SNR.png')
plt.show()



"""
A =             [uni,          random,      gaussian,   eeg]
Amse = np.array([1.2558986 ,   4.65224642,  2.08269631,   1.15658083])
Xmse = np.array([ 27.92955353, 8.30111343, 12.95423584, 103.75651538])
"""