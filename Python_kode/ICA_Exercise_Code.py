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
loc0 = 0   # mean
scale0 = 1 # lambda, greater lambda a high peak occur

loc1 = 0.5   # mean
scale1 = 0.5 # lambda, greater lambda a high peak occur

N = 1000  # size

p0 = np.random.laplace(loc0, scale0, size=N)
p1 = np.random.laplace(loc1, scale1, size=N)
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
y01 = A0 * p0
y11 = A1 * p0
y21 = A2 * p0

y02 = A0 * p1
y12 = A1 * p1
y22 = A2 * p1


""" Plot the distributions of the independent components """
lin = np.arange(-8., 8., .01)

pdf0 = np.exp(-abs(lin-loc0)/scale0)/(2.*scale0)
pdf1 = np.exp(-abs(lin-loc1)/scale1)/(2.*scale1)

plt.figure(1)
plt.subplot(311)
plt.title("Laplacian Distribution with mean = 0 and lambda = 1")
count, bins, ignored = plt.hist(y01, 30, density=True)
plt.plot(lin, pdf0, 'k')

plt.subplot(312)
count, bins, ignored = plt.hist(y11, 30, density=True)
plt.plot(lin, pdf0, 'b')

plt.subplot(313)
count, bins, ignored = plt.hist(y21, 30, density=True)
plt.plot(lin, pdf0, 'r')
plt.show()

plt.figure(2)
plt.subplot(311)
plt.title("Laplacian Distribution with mean = 0.5 and lambda = 0.5")
count, bins, ignored = plt.hist(y02, 30, density=True)
plt.plot(lin, pdf1, 'k')

plt.subplot(312)
count, bins, ignored = plt.hist(y12, 30, density=True)
plt.plot(lin, pdf1, 'b')

plt.subplot(313)
count, bins, ignored = plt.hist(y22, 30, density=True)
plt.plot(lin, pdf1, 'r')
plt.show()
# =============================================================================
# Exercise 7.2: Generate samples of two independent gaussian random variables. 
# Mix them with a random mixing matrix. Compute a whitening matrix. Compute 
# the product of the whitening matrix and the mixing matrix. Show that this is 
# almost orthogonal. Why is it not exactly orthogonal?
# =============================================================================
p2 = np.random.normal(loc=0, scale=1, size=1000) # (1000,)
p3 = np.random.normal(loc=0, scale=1, size=1000) # (1000,)

A3 = np.random.uniform(0, 1, size=1000)          # (1000,)

y3 = A3 * p2                                     # (1000,)
y4 = A3 * p3                                     # (1000,)

""" Whitening Matrix V """
# V = E D^{1/2} E^T with use of the eigenvalue decomposition (EVD)

C0 = np.cov(y3, y3.T)                            # (2,2)  
C1 = np.cov(y4, y4.T)                            # (2,2)

D0, E0 = np.linalg.eig(C0)   # D is the eigenvalues and E is the eigenvectors
D1, E1 = np.linalg.eig(C1)   # (2,2)

D0 = np.diag(D0)             # Matrix with eigenvalues in the diagonal
D1 = np.diag(D1)             # (2,2)

W0 = E0 * np.sqrt(D0) * E0.T # Whitening matrix
W1 = E1 * np.sqrt(D1) * E1.T # (2,2)

W0A3 = A3 * W0
W1A3 = A3 * W1
# =============================================================================
# Book Chapter 7, section 7.3
# =============================================================================
""" Two independent components with uniform distribution """
#p1 = 1/(2 * np.sqrt(3))  # if abs(x_i) <= np.sqrt(3)
#p2 = 0                   # otherwise
# The range of p1 and p2 make sure that we have mean zero and variance one

""" Mixing the two independent component """
#A = np.matrix([[5, 10],[10,2]]) #mixing matrix

#y1 = A * p1
#y2 = A * p2
