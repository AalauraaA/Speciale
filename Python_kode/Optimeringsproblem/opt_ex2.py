# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:16:50 2020

@author: trine

Minimisation Problem
    min || D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F

subject to
    di = vec(a_i*a_i^T), i = 1,...,n
    
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)

# hardcoded toy example with m = 3 n = 3

m = 3
n = 3
m_ = int(m*(m+1)/2.)

A = np.random.randint(1,5,size=(m,n)) # initial A 
a = np.reshape(A,(A.size)) # vectorization of initial A

U = np.random.randint(1,10,size=(m_,n))

# originale indexer for A 
#D_array = np.array([[A[0][0]**2,A[1][0]*A[0][0],A[1][0]**2,A[2][0]*A[0][0],A[2][0]*A[1][0],A[2][0]**2],
#              [A[0][1]**2,A[1][1]*A[0][1],A[1][1]**2,A[2][1]*A[0][1],A[2][1]*A[1][1],A[2][1]**2], 
#              [A[0][2]**2,A[1][2]*A[0][2],A[1][2]**2,A[2][2]*A[0][2],A[2][2]*A[1][2],A[2][2]**2]]).T


def D(a):
    return np.array([[a[0]**2,a[3]*a[0],a[3]**2,a[6]*a[0],a[6]*a[3],a[6]**2],
                     [a[1]**2,a[4]*a[1],a[4]**2,a[7]*a[1],a[7]*a[4],a[7]**2], 
                     [a[2]**2,a[5]*a[2],a[5]**2,a[8]*a[2],a[8]*a[5],a[8]**2]]).T

def D_term(a):
    return np.dot(np.dot(D(a),(np.linalg.inv(np.dot(D(a).T,D(a))))),D(a).T)

def U_term(U=U):
    return np.dot(np.dot(U,(np.linalg.inv(np.dot(U.T,U)))),U.T)

def cost1(a):
    return np.linalg.norm(D_term(a)-U_term())

# predefined optimization method, without defined the gradient og the cost. 
from scipy.optimize import minimize
res = minimize(cost1, a, method='nelder-mead',
            options={'xatol': 1e-8, 'disp': True})
a_new = res.x

A_new = np.reshape(a_new,(m,m)) 




