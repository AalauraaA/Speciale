# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:57:26 2019

@author: Laura

ICA Exercises
"""
import numpy as np

# =============================================================================
# Gradient Algorithm with Kurtosis
# =============================================================================
def gradient_kurt(x, int_gam, int_w):
    gamma = int_gam
    w = int_w
    for i in range(len(x)):
        w = gamma * x[i] * (w[i].T * x[i])**3
        w[i+1] = w/np.linalg.norm(w)
    return w


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
