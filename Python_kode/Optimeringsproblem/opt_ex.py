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

# =============================================================================
# Least-Square
# =============================================================================
# Generate data.
m = 6
n = 4

np.random.seed(5)
d = cp.Parameter(m,n)
d.value = np.random.randn

print(d)
b = np.random.randn(m,n)
print(b)

# Define and solve the CVXPY problem.
A = cp.Variable((m,n))
cost = cp.sum_squares(d - b)

#constraints = [A >> 0]
#constraints += [
#    D[i] * A == b[i] for i in range(m)
#]

prob = cp.Problem(cp.Minimize(cost), [d == A])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(A.value)
print("The norm of the residual is ", cp.norm(d - b, p=2).value)

