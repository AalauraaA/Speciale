# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:30:47 2020

@author: Laura

Minimisation Problem
    min || D(D^T D)^-1 * D^T - U(U^TU)^-1 * U^T||^2_F

subject to

    di = vec(Ai), i = 1,...,n
    Ai positive semo-definite
    
We want to minimise for Ai.

D = D(D^T D)^-1 * D^T
U = U(U^TU)^-1 * U^T

    || D - U ||^2_F
"""
""" CVXPY PAKKEN """
import cvxpy as cp
import numpy as np

# =============================================================================
# Random SDP - Eksempl
# =============================================================================
#np.random.seed(1)
#
#n = 3
#p = 3
#
#C = np.random.randn(n, n)
#A = []
#b = []
#for i in range(p):
#    A.append(np.random.randn(n, n))
#    b.append(np.random.randn())
#
## Define and solve the CVXPY problem.
## Create a symmetric matrix variable.
#X = cp.Variable((n,n), symmetric=True)
## The operator >> denotes matrix inequality.
#constraints = [X >> 0]
#constraints += [
#    cp.trace(A[i] * X) == b[i] for i in range(p)
#]
#prob = cp.Problem(cp.Minimize(cp.trace(C * X)),
#                  constraints)
#prob.solve(solver=cp.CVXOPT)
#
## Print result.
#print("The optimal value is", prob.value)
#print("A solution X is")
#print(X.value)

# =============================================================================
# Random SDP - Vores tilfælde
# =============================================================================
#np.random.seed(1)
#
#n = 3
#m = 3
#
#U = np.random.randn(n, n)
#D = []
#b = []
#for i in range(m):
#    D = np.random.randn(n, n)
#    b.append(np.random.randn())
#
#A = cp.Variable((n,n), symmetric=True)
#
#constraints = [A >> 0]
##constraints += [
##    D[i] * A == b[i] for i in range(m)
##]
#prob = cp.Problem(cp.Minimize(np.trace(U*A)),
#                  constraints)
#prob.solve(solver=cp.CVXOPT)
#
## Print result.
#print("The optimal value is", prob.value)
#print("A solution X is")
#print(A.value)
#

# =============================================================================
# Least-Square
# =============================================================================
# Generate data.
m = 2
n = 2
#np.random.seed(5)
d = np.random.randn(m,n)
print(d)
b = np.random.randn(m,n)
print(b)

# Define and solve the CVXPY problem.
A = cp.Variable((m,n))
cost = cp.sum_squares(d - b)

prob = cp.Problem(cp.Minimize(cost), [d == A])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(A.value)
print("The norm of the residual is ", cp.norm(d - b, p=2).value)








""" CVXOPT PAKKEN """
## =============================================================================
## Parameters and Variables
## =============================================================================
#from cvxopt import matrix, normal, spdiag, misc, lapack
#from ubsdp import ubsdp
#m = 5
#n = 5
#
#" Random initial Ai "
#A = normal(m**2,n)  # Normal distribution with mean = 0 and std = 1
#
#" Random positive definite matrix with maximum less than 1.0 "
#Z0 = normal(m,m) # Size 5x5
#Z0 = Z0 * Z0.T   # Size 5x5 
#
#" Zero vector "
#w = matrix(0.0, (m,1)) # Size 5x1
#
#" Random positive definite matrix "
#a = +Z0   # a is now Z0
#
## =============================================================================
## Eigenvalue Decomposition of real symmetric matrix of order n.
## =============================================================================
#lapack.syev(a, w, jobz = 'V')
#"""
#w is a real matrix with length m -- contains the eigenvalues in ascending order
#V mean that the eigenvectors are computed and stored in a
#"""
#
#wmax = max(w)  # Find the max value
#if wmax > 0.9:  
#    w = (0.9/wmax) * w
#
## =============================================================================
## Construct a Block-Diagonal Sparse Matrix
## =============================================================================
#Z0 = a * spdiag(w) * a.T # Z0 is now a block-diagonal sparse matrix
## spdiag(w) makes Z0 sparse
#
#" Construct the constrain di = -A(Z0)"
#d = matrix(0.0, (n,1))  # Zero matrix of size nx1
#misc.sgemv(A, Z0, d, dims = {'l': 0, 'q': [], 's': [m]}, trans = 'T', alpha = -1.0)
#
#
## Z1 = I - Z0
#Z1 = -Z0
#Z1[::m+1] += 1.0
#
#x0 = normal(n,1)
#
#X0 = normal(m,m)
#X0 = X0*X0.T
#
##S0 = normal(m,m)
##S0 = S0*S0.T
#
## B = A(x0) - X0 + S0
##B = matrix(A*x0 - X0[:] + S0[:], (m,m))
#D = matrix(A*x0 - X0[:], (m,m))
#
#X = ubsdp(d, A, D)
