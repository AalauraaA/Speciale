# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:43:46 2020

@author: trine
"""

import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


""" convex optimisation example from Qp-cvx.pdf """

#define parameters
# no equality constrint hence no A and b parameter 
# lists has to contain real numbers (doubles, 'd') instead of integers
# convert to type matrix for cvx 
P = cvx.matrix(np.array([[1.0,0.0],[0.0,0.0]]), tc='d')
q = cvx.matrix(np.array([3.0,4.0]), tc='d')
G = cvx.matrix(np.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]), tc='d')
h = cvx.matrix(np.array([0.0,0.0,-15.0,100.0,80.0]), tc='d')

sol = cvx.solvers.qp(P,q,G,h) #sol is a dictionary 

def f(x):
    return 3*x**2+0*x+10


fig = plt.figure()
xline = np.linspace(-10, 10, 100)
plt.plot(xline,f(xline))

P = cvx.matrix(np.array([3]))
q = cvx.matrix(np.array([3,0]))

sol1 = cvx.solvers.qp(c)


