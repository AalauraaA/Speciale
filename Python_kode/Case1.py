# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 1 - Simpel Sparse Signal
m = 30
n = 80
L = n_samples = 20
k = non-zero = 40 (halvdelen af X)
Y = random signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt sparse signal - n x L 

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
from Cov_DL import data_generation
from Cov_DL import MSBL

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 30               # number of sensors
n = 80               # number of sources
non_zero = 40       # max number of non-zero coef. in rows of X
n_samples = 20       # number of sampels
iterations = 1000

" Random Signals Generation - Sparse X "
Y, A, X = data_generation.random_sparse_data(m, n, non_zero, n_samples)

# =============================================================================
# M-SBL
# =============================================================================
X_cal = MSBL.M_SBL(A, Y, m, n, n_samples, non_zero, iterations)

# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.title('Plot of row 2 of X - Random Sparse Data')
plt.plot(X[1], 'r',label='Real X')
plt.plot(X_cal[1],'g', label='Calculated X')
plt.legend()
plt.show
plt.savefig('case1_1.png')

plt.figure(2)
plt.title('Plot of column 2 of X - Random Sparse Data')
plt.plot(X.T[1], 'r',label='Real X')
plt.plot(X_cal.T[1],'g', label='Calculated X')
plt.legend()
plt.show
plt.savefig('case1_2.png')

for i in range(4):
    plt.figure(3)
    plt.title('Plot of row of X - Known X')
    plt.plot(X[i], label=i)
    plt.legend()
    plt.savefig('case1_3.png')
    plt.figure(4)
    plt.title('Plot of row of X - Calculated X')
    plt.plot(X_cal[i], label=i)
    plt.legend()
    plt.savefig('case1_4.png')



summation = 0  #variable to store the summation of differences
n = len(X) #finding total number of items in list
for i in range (0,n):  #looping through each element of the list
  difference = X[i] - X_cal[i]  #finding the difference between observed and predicted value
  squared_difference = difference**2  #taking square of the differene 
  summation = summation + squared_difference  #taking a sum of all the differences
MSE = summation/n  #dividing summation by total values to obtain average
print("The Mean Square Error is: " , MSE)