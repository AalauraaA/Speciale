# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:39:14 2020

@author: trine
"""

import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# test for least square minimization 

#Q = 2*cvx.matrix([ [2, .5], [.5, 1] ])
#p = cvx.matrix([1.0, 1.0])
#G = cvx.matrix([[-1.0,0.0],[0.0,-1.0]])
#h = cvx.matrix([0.0,0.0])
#A = cvx.matrix([1.0, 1.0], (1,2))
#b = cvx.matrix(1.0)
#sol = cvx.solvers.qp(Q, p, G, h, A, b)

# classic least square 
#np.random.seed(123)
#A = np.random.randint(0,10,size=(3,5))
#b = np.random.randint(0,10,size=(5,1))
#
#P = np.dot(A.T,A)
#q = np.dot(A,b)
#r = np.dot(b.T,b)

# least square of differens of two vectors. 
def test_cost(x):
    y = np.array([1,2,3,4,5,6])
    d = x
    lam = 1
    return np.linalg.norm(d-y)**2#+np.dot(lam,(d-x))



x0 = np.random.random(6) 
res = minimize(test_cost, x0, method='nelder-mead',
            options={'xatol': 1e-8, 'disp': True})
res.x
print(res.x)