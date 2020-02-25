# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:30:47 2020

@author: Laura

Minimisation Problem
    min || D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F

subject to

    di = vec(Ai), i = 1,...,n
    Ai positive semi-definite
    
We want to minimise for Ai.

D = D(D^T D)^-1 * D^T
U = U(U^TU)^-1 * U^T

    || D - U ||^2_F
"""
""" CVXPY PAKKEN """
import cvxpy as cp
import numpy as np
np.random.seed(5)

# =============================================================================
# Least-Square
# =============================================================================
# Generate cost function 

#def D(X):
#    D = np.zeros([int(X.shape[0]*2),X.shape[1]])
#    for i in range(X.shape[1]): # run over N
##        d = np.outer(X.T[i],X.T[i])
##        d_vec = d[np.tril_indices(X.shape[0])]
#        d_vec = np.hstack((X.T[i],X.T[i]))
#        D.T[i] = d_vec
#    return D
#
#def D(X):
#    D = np.zeros([int(X.shape[0]*2),X.shape[1]])
#    for i in range(X.shape[1]): # run over N
##        d = np.outer(X.T[i],X.T[i])
##        d_vec = d[np.tril_indices(X.shape[0])]
#        d_vec = X.T[i]
#        D.T[i] = d_vec
#
#m = 3
#n = 4
#
#U = np.random.randint(1,10,size=(m,n))
#print(U)
#
#
## Define and solve the CVXPY problem.
#A = cp.Variable((m,n))
#cost = cp.sum_squares(D(A) - U)
#
##constraints = [D(A) == A]
##constraints += [
##    D[i] * A == b[i] for i in range(m)
##]
#
#prob = cp.Problem(cp.Minimize(cost))
#prob.solve()
#
## Print result.
#print("\nThe optimal value is", prob.value)
#print("The optimal x is")
#print(A.value)
#print("The norm of the residual is ", cp.norm(A - U, p=2).value)

## augmented lagrangian exsempel 

np.random.seed(1)

# Initialize data.
MAX_ITERS = 10
rho = 1.0
n = 20
m = 10
A = np.random.randn(m,n)
b = np.random.randn(m)

# Initialize problem.
x = cp.Variable(shape=n)
f = cp.norm(x, 1)

# Solve with CVXPY.
cp.Problem(cp.Minimize(f), [A*x == b]).solve()
print("Optimal value from CVXPY: {}".format(f.value))

# Solve with method of multipliers.
resid = A*x - b
y = cp.Parameter(shape=(m)); y.value = np.zeros(m)
aug_lagr = f + y.T*resid + (rho/2)*cp.sum_squares(resid)
for t in range(MAX_ITERS):
    cp.Problem(cp.Minimize(aug_lagr)).solve()
    y.value += rho*resid.value
    
