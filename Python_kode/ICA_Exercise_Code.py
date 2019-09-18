# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:57:26 2019

@author: Laura

ICA Exercises
"""
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Exercise 7.1:  Generate samples of two independent components that follow a 
# Laplacian distribution (see Eq. 2.96). Mix them with three different random 
# mixing matrices. Plot the distributions of the independent components. Can 
# you see the matrix A in the plots? Do the same for ICs that are obtained by 
# taking absolute values of gaussian random variables.
# =============================================================================
""" Two independent component with Lapacian distribution """
# Laplacian is a supergaussian distribution
loc1 = 0   # mean
scale1 = 1 # lambda, greater lambda a high peak occur

loc2 = 0.5   # mean
scale2 = 0.5 # lambda, greater lambda a high peak occur

N = 1000  # size

p1 = np.random.laplace(loc1, scale1, size=N)
p2 = np.random.laplace(loc2, scale2, size=N)
"""
Draw samples from the Laplace or double exponential distribution with specified
location (or mean) and scale (decay). The Laplace distribution is similar to 
the Gaussian/normal distribution, but is sharper at the peak and has fatter 
tails. It represents the difference between two independent, identically 
distributed exponential random variables.

Parameters:	
loc : float or array_like of floats. The position, mu, of the distribution 
peak. Default is 0.

scale : float or array_like of floats. lambda, the exponential decay. 
Default is 1.

size : int or tuple of ints. Output shape. If the given shape is, e.g., 
(m, n, k), then m * n * k samples are drawn. If size is None (default), 
a single value is returned if loc and scale are both scalars. Otherwise, 
np.broadcast(loc, scale).size samples are drawn.
"""

""" Mixing Matrix A """
A0 = np.random.uniform(0,1,size=N)     # Uniform distribution with a=0 and b=1
A1 = np.random.randn(N)                # Normal distribution
A2 = np.random.binomial(1,0.5,size=N)  # Binomial distribution with n=1 and p=0.5

""" Mixing the components """
y01 = A0 * p1
y11 = A1 * p1
y21 = A2 * p1

y02 = A0 * p2
y12 = A1 * p2
y22 = A2 * p2

""" Plot the distributions of the independent components """
plt.figure(1)
plt.subplot(311)
plt.title("Laplacian Distribution with mean = 0 and lambda = 1")
count, bins, ignored = plt.hist(y01, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc1)/scale1)/(2.*scale1)
plt.plot(lin, pdf, 'k')

plt.subplot(312)
count, bins, ignored = plt.hist(y11, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc1)/scale1)/(2.*scale1)
plt.plot(lin, pdf, 'b')

plt.subplot(313)
count, bins, ignored = plt.hist(y21, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc1)/scale1)/(2.*scale1)
plt.plot(lin, pdf, 'r')
plt.show()

plt.figure(2)
plt.subplot(311)
plt.title("Laplacian Distribution with mean = 0.5 and lambda = 0.5")
count, bins, ignored = plt.hist(y02, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc2)/scale2)/(2.*scale2)
plt.plot(lin, pdf, 'k')

plt.subplot(312)
count, bins, ignored = plt.hist(y12, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc2)/scale2)/(2.*scale2)
plt.plot(lin, pdf, 'b')

plt.subplot(313)
count, bins, ignored = plt.hist(y22, 30, density=True)
lin = np.arange(-8., 8., .01)
pdf = np.exp(-abs(lin-loc2)/scale2)/(2.*scale2)
plt.plot(lin, pdf, 'r')
plt.show()
# =============================================================================
# Exercise 7.2: Generate samples of two independent gaussian random variables. 
# Mix them with a random mixing matrix. Compute a whitening matrix. Compute 
# the product of the whitening matrix and the mixing matrix. Show that this is 
# almost orthogonal. Why is it not exactly orthogonal?
# =============================================================================
p3 = np.random.normal(loc=0, scale=1, size=1000)
p4 = np.random.normal(loc=0, scale=1, size=1000)

A = np.random.uniform(0, 1, size=1000)
W = 
# =============================================================================
# Book Chapter 7
# =============================================================================
""" Two independent components with uniform distribution """
#p1 = 1/(2 * np.sqrt(3))  # if abs(x_i) <= np.sqrt(3)
#p2 = 0                   # otherwise
# The range of p1 and p2 make sure that we have mean zero and variance one

""" Mixing the two independent component """
#A = np.matrix([[5, 10],[10,2]]) #mixing matrix

#y1 = A * p1
#y2 = A * p2