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
#    D = np.zeros([int(len(X)*(len(X)+1)/2.),len(X.T)])
#    for i in range(len(X.T)): # run over N
#        d = np.outer(X.T[i],X.T[i])
#        d_vec = d[np.tril_indices(len(X.T[i]))]
#        D.T[i] = d_vec
#    return D


#def D(X):
#    D = np.zeros([int(X.shape[0]),X.shape[1]])
#    for i in range(X.shape[1]): # run over N
#        d_vec = X.T[i]
#        D.T[i] = d_vec

m = 3
n = 3
m_ = int(m*(m+1)/2.)

U = np.random.randint(1,10,size=(m_,n))

#for m=3 og n=3
#D = np.array([[A[0][0]**2,A[1][0]*A[0][0],A[1][0]**2,A[2][0]*A[0][0],A[2][0]*A[1][0],A[2][0]**2],
#              [A[0][1]**2,A[1][1]*A[0][1],A[1][1]**2,A[2][1]*A[0][1],A[2][1]*A[1][1],A[2][1]**2], 
#              [A[0][2]**2,A[1][2]*A[0][2],A[1][2]**2,A[2][2]*A[0][2],A[2][2]*A[1][2],A[2][2]**2]]).T


print(U)

#Define and solve the CVXPY problem.
D = cp.Variable((m_,m_))
A = cp.Variable((m,n))
cost = cp.sum_squares(np.array([[cp.power(A[0,0],2),A[1,0]*A[0,0],cp.power(A[1,0],2),A[2,0]*A[0,0],A[2,0]*A[1,0],cp.power(A[2,0],2)],
                                [cp.power(A[0,1],2),A[1,1]*A[0,1],cp.power(A[1,1],2),A[2,1]*A[0,1],A[2,1]*A[1,1],cp.power(A[2,1],2)], 
                                [cp.power(A[0,2],2),A[1,2]*A[0,2],cp.power(A[1,2],2),A[2,2]*A[0,2],A[2,2]*A[1,2],cp.power(A[2,2],2)]]).T - U)

#constraints = [D[i] == np.outer(A.T[i],A.T[i])[np.tril_indices(m)] for i in range(m)
#
#constraints = [D.T[i] == np.array([A[0][i]**2,A[1][i]*A[0][i],A[1][i]**2,A[2][i]*A[0][i],A[2][i]*A[1][i],A[2][i]**2]) for i in range(n)]

prob = cp.Problem(cp.Minimize(cost))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(A.value)
print("The norm of the residual is ", cp.norm(D - U, p=2).value)

## augmented lagrangian exsempel 

#np.random.seed(1)
#
## Initialize data.
#MAX_ITERS = 10
#rho = 1.0
#n = 20
#m = 10
#A = np.random.randn(m,n)
#b = np.random.randn(m)
#
## Initialize problem.
#x = cp.Variable(shape=n)
#f = cp.norm(x, 1)
#
## Solve with CVXPY.
#cp.Problem(cp.Minimize(f), [A*x == b]).solve()
#print("Optimal value from CVXPY: {}".format(f.value))
#
## Solve with method of multipliers.
#resid = A*x - b
#y = cp.Parameter(shape=(m)); y.value = np.zeros(m)
#aug_lagr = f + y.T*resid + (rho/2)*cp.sum_squares(resid)
#for t in range(MAX_ITERS):
#    cp.Problem(cp.Minimize(aug_lagr)).solve()
#    y.value += rho*resid.value


#### nyt forsøg
#from scipy.optimize import minimize
#
#def cost(A, U=U):
#    return np.linalg.norm(np.array([[A[0][0]**2,A[1][0]*A[0][0],A[1][0]**2,A[2][0]*A[0][0],A[2][0]*A[1][0],A[2][0]**2],
#                    [A[0][1]**2,A[1][1]*A[0][1],A[1][1]**2,A[2][1]*A[0][1],A[2][1]*A[1][1],A[2][1]**2], 
#                    [A[0][2]**2,A[1][2]*A[0][2],A[1][2]**2,A[2][2]*A[0][2],A[2][2]*A[1][2],A[2][2]**2]]).T.flatten() - U.flatten())**2
#
#A0 = np.random.randint(1,10,size=(m,n))
#    
#res = minimize(cost, A0, method='nelder-mead',
#                options={'xatol': 1e-8, 'disp': True})
#print(res.x)