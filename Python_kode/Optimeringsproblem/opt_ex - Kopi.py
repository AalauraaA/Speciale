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
import matplotlib.pyplot as plt

np.random.seed(5)

# =============================================================================
# Define D
# =============================================================================
#def D_matrix(m, n):
#    """
#    Construct the D matrix when m = 3 and for a given A
#    """
#    A = np.random.randint(1,10, size=(m,m))
##    A = np.array([[2, 3, 8],
##       [1, 6, 1],
##       [1, 5, 5]])
#    D = np.zeros((m*2,m))
#    for j in range(n):
#        D = D.T
#        D[j] = np.array([A[0][j]**2, A[1][j]*A[0][j], A[1][j]**2, A[2][j]*A[0][j], A[2][0]*A[1][j], A[2][j]**2])
#        D = D.T
#    return D
#

# =============================================================================
# Define 
# =============================================================================
#def cost_func(theta, x, y):
#    """
#    Cost function is the MSE
#    """
#    m = len(y)
#    cost = (1/2 * m) * np.sum(np.square(x.dot(theta) - y))
#    return cost
#
#def derivative_cost(theta, x, y):
#    derivative = (x.T.dot((x.dot(theta) - y)))
#    return derivative
#
#def Gradient_Descent(x, y, theta, learning_rate, iterations):
#    m = len(y)
#    cost = np.zeros(iterations)
#    for it in range(iterations):
#        theta = theta - (1/m) * learning_rate * derivative_cost(theta, x, y)
#        cost[it] = cost_func(theta, x, y)
#    return theta, cost
#
#def Stoch_Gradient_Descent(x, y, theta, learning_rate, iterations):
#    m = len(y)
#    cost = np.zeros(iterations)  
#    for it in range(iterations):
#        cst = 0.0
#        for i in range(m):
#            rand = np.random.randint(0,m)
#            xi = x[rand, :].reshape(1, x.shape[1])
#            yi = y[rand, :].reshape(1, y.shape[1])
#            
#            theta = theta - (1/m) * learning_rate * derivative_cost(theta, xi, yi)
#            cst += cost_func(theta, xi, yi)
#            
#        cost[it] = cst
#    return theta, cost

# =============================================================================
# Perform GD and SGD
# =============================================================================
#lr = 0.01          # Learning Rate
#n_iter = 10        # Iterations
#
#m = 3              # Rows
#n = 3              # Columns
#D = D_matrix(m,n)  # D matrix --> find A from
#
#theta = np.random.randn(3,1) # Initial guess
#
##A = np.random.randint(1,10, size=(m*2,1))
##A_b = np.c_[np.ones((len(A),1)), A, A*2]   # Start point
#
#Start = np.c_[np.ones(6), D.T[0], D.T[1]]
#
#U = np.random.randint(1, 10, size=(m*2,m))
#
#theta1 ,cost1 = Gradient_Descent(Start, U, theta, lr, n_iter)
#theta2 ,cost2 = Stoch_Gradient_Descent(Start, U, theta, lr, n_iter)
#
#print('The value of D:')
#print(D)
#
#MSE1 = cost1[-1]
#print('MSE of GD:', MSE1)
#MSE2 = cost2[-1]
#print('MSE of SGD:', MSE2)
#
#print('Optimun GD:')
#print(theta1.reshape(m,m))
#print('Optimun SGD:')
#print(theta2.reshape(m,m))
#
#
#plt.figure(1)
#plt.plot(np.linspace(1,n_iter,n_iter), cost1, 'b.')
#
#plt.figure(2)
#plt.plot(np.linspace(1,n_iter,n_iter), cost2, 'b.')
#
#



# =============================================================================
# Least-Square
# =============================================================================
# Generate cost function 

def D(A):
    a = np.reshape(A, (A.size)) # vectorization of initial A
    A_tilde = np.zeros((m,m,m))
    a_tilde = np.zeros((int(m*(m+1)/2),m))
    d = np.zeros((int(m*(m+1)/2),m))
    for i in range(m):
        A_tilde[i] = np.outer(A.T[i],A.T[i])
        a_tilde.T[i] = A_tilde[i][np.tril_indices(len(A_tilde.T[i][0]))]
        
        d.T[i] = np.array([a[i]**2, a[i+3] * a[i], a[i+3]**2, a[i+6] * a[i], a[i+6]*a[i+3], a[i+6]**2])
    return A_tilde, a_tilde, d, a
#    return d

m = 3
n = 3 
l = m*(m+1)/2

A = np.random.randint(1,5,size=(m,n)) # initial A 



U = np.random.randint(1,10,size=(m*2,m))

# Define and solve the CVXPY problem.
#D = cp.Variable((int(l),m))
#A = cp.Variable((m,m))
#A = np.random.randint(1,10,size=(m,m))

A_tilde, a_tilde, d, a = D(A)

cost1 = cp.sum_squares(d - U)
cost2 = cp.sum_squares(a_tilde - U)

prob1 = cp.Problem(cp.Minimize(cost1))
prob1.solve()

prob2 = cp.Problem(cp.Minimize(cost2))
prob2.solve()



# Print result.
print("\nThe optimal value is", prob1.value)
print("\nThe optimal value is", prob2.value)
#print("The optimal A is")
#print(D2.value)
print("The norm of (d) the residual is ", cp.norm(d - U, p=2).value)
print("The norm of (a_tilde) the residual is ", cp.norm(a_tilde - U, p=2).value)

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
#    
