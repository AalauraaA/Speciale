# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:33:08 2019

@author: trine
"""
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
import matplotlib.pyplot as plt
from scipy import signal
import data_generation

from sparselandtools.applications.denoising import KSVDImageDenoiser
from sparselandtools.applications.utils import example_image
from sparselandtools.pursuits import MatchingPursuit
from sparselandtools.dictionaries import DCTDictionary


# INITIALISING PARAMETERS
m = int(5)               # number of sensors
n = int(8)              # number of sources
non_zero = 4       # max number of non-zero coef. in rows of X
n_samples = 1      # number of sampels

Y, A_real, X_real = data_generation.random_sparse_data(m, n, non_zero, n_samples)


# sine signal

y = Y

# make noisy
np.random.seed(0)
y_noisy = y + np.random.randn(m)

# find sparse representation of y in a DCT-dictionary using MatchingPursuit
d = DCTDictionary(np.sqrt(m), np.sqrt(n))
a = MatchingPursuit(d, sparsity=3).fit(np.array([y_noisy]).T)
z = np.matmul(d.matrix, a)

## plot 
#plt.figure(figsize=(10, 3))
#plt.subplot(1, 3, 1)
#plt.title('original')
#plt.plot(x, y)
#plt.subplot(1, 3, 2)
#plt.title('noisy')
#plt.plot(x, y_noisy)
#plt.subplot(1, 3, 3)
#plt.title('denoised')
#plt.plot(x, z)
