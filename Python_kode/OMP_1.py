# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:14:10 2019

@author: mathi
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal



# generate the data
#A = np.array([[0.8, -0.6, 0, 0], [0, 0.8, 0, 0], [np.sqrt(0.8), 0, 0, 0], [0, np.sqrt(0.8), 0, 0]])
#x = np.array([2.5, 2.5, 0, 0])  # it is x that we want to recover from y and A
#y = np.dot(A, x)
#
#supp, = x.nonzero()  # the support set of x
#
#n_sources, n_sensors = len(x), len(y)
#n_nonzero_coefs = len(supp)
x = np.array([ 0.08031541,  1.        , -0.91959799])
y = np.array([ 0.16071742,  1.12055972, -0.71872286])
A = np.matrix([[1. , 1. , 1. ],
       [0.5, 2. , 1. ],
       [1.5, 1. , 2. ]])
supp, = x.nonzero()
n_nonzero_coefs = len(supp)

# automatic generated data
#n_sources, n_sensors = 10, 10
#n_nonzero_coefs = 5
#
#y, A, x = make_sparse_coded_signal(n_samples=1,
#                                   n_components=n_sources,
#                                   n_features=n_sensors,
#                                   n_nonzero_coefs=n_nonzero_coefs,
#                                   random_state=0)

# plot original sparse signal  
plt.figure()
plt.subplot(3, 1, 1)
plt.title("Sparse signal")
plt.plot(x,'r')

# create reconstruction of x from signal y and A
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs).fit(A, y)  # create object from class
                                                     # fit model by training data, return self:object?  
x_rec = omp.coef_                                                 # coef of reconstructed x 
supp_r1, = x_rec.nonzero()                                        # supp of reconstructed x

# plot the reconstruction of x
#plt.subplot(3, 1, 3)
#plt.title("Recovered signal from A and y")
plt.plot(x_rec,'b.')

plt.show()

