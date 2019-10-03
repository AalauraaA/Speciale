# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:08:54 2019

@author: Laura

ICA Book - Chapter 8 Exercises
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# =============================================================================
# 8.1: Let x(t), t = 1,...,T, be T independent random numbers distributed 
# uniformly on the interval [-1,1], and y = sum_{t=1}^T x(t) their sums. 
# Generate 5000 different realisations of the random variables y for the T = 2,
# T = 4, and T = 12
# =============================================================================
N = 5000

T0 = 2
T1 = 4
T2 = 12

Y0 = np.zeros(N)
Y1 = np.zeros(N)
Y2 = np.zeros(N)
for n in range(N):
    X0 = np.random.uniform(-1, 1, T0)
    X1 = np.random.uniform(-1, 1, T1)
    X2 = np.random.uniform(-1, 1, T2)
    
    Ysum0 = 0
    Ysum1 = 0
    Ysum2 = 0
    
    for i in range(T0):
        Ysum0 += X0[i] 
    
    for i in range(T1):
        Ysum1 += X1[i] 
    
    for i in range(T2):
        Ysum2 += X2[i] 
    
    Y0[n] += Ysum0
    Y1[n] += Ysum1
    Y2[n] += Ysum2

"""
8.1.1: Plot the experimental pdf's of y, and compare it with the gaussian pdf 
having the same (zero) mean and variance. (Hint: you can here estimate the pdf
from the generated samples simply by dividing their value range into small bins
of width 0.1 or 0.05, count the number of samples falling into each bin, and 
divide by the total number of samples. You can compute the difference between 
the respective gaussian and the estimated density to get a better idea of the 
similarity of the two distributions.)
"""
samples = np.random.normal(size=N)
# Compute a histogram of the sample
bins = np.linspace(-5, 5, 30)
histogram, bins = np.histogram(samples, bins=bins, normed=True)
lin = 0.5*(bins[1:] + bins[:-1])

# Compute the PDF on the bin centers from scipy distribution object
pdf = stats.norm.pdf(lin)

plt.figure(1)
plt.subplot(311)
plt.title("T = 2")
count, bins, ignored = plt.hist(Y0, 30, density=True)
plt.plot(lin, pdf, 'k')

plt.subplot(312)
plt.title("T = 4")
count, bins, ignored = plt.hist(Y1, 30, density=True)
plt.plot(lin, pdf, 'b')

plt.subplot(313)
plt.title("T = 12")
count, bins, ignored = plt.hist(Y2, 30, density=True)
plt.plot(lin, pdf, 'r')
plt.show()

"""
8.1.2: Plot the kurtoses in each case. Note that you must normalize all the 
variables to unit variance. What if you don’t normalize?
"""


# =============================================================================
# 8.2: Program the FastICA algorithm in (8.4).
# =============================================================================
"""
8.2.1: Take the data x(t) in the preceding assignement as two independent 
components by splitting the samples in two. Mix them using a random mixing 
matrix, and estimate the model, using one of the nonlinearity in (8.31)
"""








