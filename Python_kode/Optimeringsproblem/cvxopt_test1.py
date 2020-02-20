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

##define parameters
## no equality constrint hence no A and b parameter 
## lists has to contain real numbers (doubles, 'd') instead of integers
## convert to type matrix for cvx 
#P = cvx.matrix(np.array([[1.0,0.0],[0.0,0.0]]), tc='d')
#q = cvx.matrix(np.array([3.0,4.0]), tc='d')
#G = cvx.matrix(np.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]), tc='d')
#h = cvx.matrix(np.array([0.0,0.0,-15.0,100.0,80.0]), tc='d')
#
#sol = cvx.solvers.qp(P,q,G,h) #sol is a dictionary 
#
#
## new example - unconstraint optimization 
#
from scipy.optimize import minimize
#
##multiple varibles
#def rosen(x):
#     """The Rosenbrock function"""
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
#
#x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#res = minimize(rosen, x0, method='nelder-mead',
#                options={'xatol': 1e-8, 'disp': True})
#print(res.x)
#
## only one variable
#def f(x):
#    return 3*x**2+2*x+10
#
##fig = plt.figure()
##xline = np.linspace(-10, 10, 100)
##plt.plot(xline,f(xline))
#
#x0 = np.array([1.3])
#res = minimize(f, x0, method='nelder-mead',
#                options={'xatol': 1e-8, 'disp': True})
#
#print(res.x)


## optimization process 
m = 5
n = 8
u = np.random.rand(int(m*(m+1)/2.),n)
A = np.random.rand(m,n)
l = np.random.rand(n)
mu = 1 

limit = 10
iter_= 0


# definition of our function 
def D(X):
    D = np.zeros([int(len(X)*(len(X)+1)/2.),len(X.T)])
    for i in range(len(X.T)): # run over N
        d = np.outer(X.T[i],X.T[i])
        d_vec = d[np.tril_indices(len(X.T[i]))]
        D.T[i] = d_vec
    return D

def f(X,U=u):
    np.random.seed(123)
    D_prod = np.matmul(np.matmul(D(X), np.matmul(D(X).T,D(X))),D(X).T)
    U_prod = np.matmul(np.matmul(U, np.matmul(U.T,U)),U.T)
    return np.linalg.norm(D_prod - U_prod)**2

def h(x):
    d = np.outer(x,x)
    return d[np.tril_indices(len(x))] - d[np.tril_indices(len(x))]

# h will alway be zero right now ??? 

def Lagrange(X):
    sum1 = np.sum(np.sum([l[i]*h(X.T[i]) for i in range(len(X.T))]))
    sum2 = np.sum(np.sum([np.linalg.norm(h(X.T[i]))**2 for i in range(len(X.T))]))

    return f(X) + sum1 + (mu/2. * sum2)

cost = Lagrange(A)
#while cost > limit:
iter_ += 1
res = minimize(f, A, args=(u,), method='nelder-mead',
            options={'xatol': 1e-8, 'disp': True})
res.x


#### next 

from cvxopt import solvers, matrix, spdiag, sqrt, div

# rubust least square solver for unconstrint problem  

#def robls(A, b, rho):
#    m, n = A.size
#    def F(x=None, z=None):
#        if x is None: return 0, matrix(0.0, (n,1))
#        y = A*x-b
#        w = sqrt(rho + y**2)
#        f = sum(w)
#        Df = div(y, w).T * A
#        if z is None: return f, Df
#        H = A.T * spdiag(z[0]*rho*(w**-3)) * A
#        return f, Df, H
#    return solvers.cp(F)['x']

#
#    
#


