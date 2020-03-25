# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:31:16 2020

@author: Laura
URL: https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
"""

import numpy as np
import matplotlib.pyplot as plt

import data_generation


#x = 2 * np.random.rand(100,1)          # Gaussian Noise
#y = 4 + 3 * x + np.random.randn(100,1) # Linear Data with Gaussian noise
## y = b + m * x --> slope of a line
#
#plt.figure(1)
#plt.plot(x, y, 'b.')
#plt.xlabel('x', fontsize=18)
#plt.ylabel('y', rotation=0, fontsize=18)
#
##Y has a nice linear relationship with x -- only one independent variable = x
#
#""" Analytical Linear Regression """
#X_best = np.c_[np.ones((100,1)), x]
#theta_best = np.linalg.inv(X_best.T.dot(X_best)).dot(X_best.T).dot(y)
#print(theta_best)
#
##Not accurate due to the noise in data instead you can view it numerical
#
#""" Numerical Linear regression """
##Numerical, b and m can  found by J(theta) = theta0 + theta1 * x, where
##theta0 = b and theta1 = m

# =============================================================================
# Gradient Descent
# =============================================================================
"""
For the method of Gradient Descent you need following informations:
    - Learning rate (alpha)
    - Cost function (J(theta))
        J(theta) = 1/2 * m * sum_i=1^m (h(theta)^i - y^i)^2
    - Gradients (partial J(theta) / partial thetaj)
        partial J(theta) / partial thetaj = 1/m * sum_i=1^m (h(theta^i - y^i) * xj^i)
    - Number of observations
        m

For a linear regression you:
    - Start with an inital theta vector
    - Predict h(theta)
    - Derive cost function (often MSE) --> minimise cost
    - Next step(theta) is found from the partial  derivatives
"""

def cost_func(theta, x, y):
    """
    Cost function is the MSE
    """
    m = len(y)
    pred = x.dot(theta)
    cost = (1/2 * m) * np.sum(np.square(pred - y))
    return cost


def Gradient_Descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        pred = np.dot(x, theta)
        theta = theta - (1/m) * learning_rate * (x.T.dot((pred - y)))
        theta_history[it,:] = theta.T
        cost_history[it] = cost_func(theta, x, y)
    return theta, cost_history, theta_history

#
#lr1 = 0.01
#n_iter1 = 200
#
#theta1 = np.random.randn(2,1)
#
#X_b1 = np.c_[np.ones((len(x), 1)), x]
#theta1, cost_history1, theta_history1 = Gradient_Descent(X_b1, y, theta1, lr1, n_iter1)
#
#MSE1 = cost_history1[-1]
#
#plt.figure(2)
#plt.plot(np.linspace(1,200,200), cost_history1, 'b.')

"""
cost_history can be plotted to view how many iteration there is needed and it 
can be controlled with different learning rates.
Furthermore, the last value is the MSE
"""

# =============================================================================
# Stochastic Gradient Descent (SGD)
# =============================================================================
"""
In Gradient Descent we did gradients on each observation one by one.
In Stochastic Gradient Descent we choose random observations randomly.
"""
def Stoch_Gradient_Descent(x, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for it in range(iterations):
        cost = 0.0
        for i in range(m):
            rand = np.random.randint(0,m)
            xi = x[rand, :].reshape(1, x.shape[1])
            yi = y[rand].reshape(1,1)
            
            pred = np.dot(xi, theta)
            theta = theta - (1/m) * learning_rate * (xi.T.dot((pred - yi)))
            cost += cost_func(theta, xi, yi)
        cost_history[it] = cost
    return theta, cost_history

#
#lr2 =0.5
#n_iter2 = 50
#
#theta2 = np.random.randn(2,1)
#
#X_b2 = np.c_[np.ones((len(x),1)), x]
#theta2 ,cost_history2 = Stoch_Gradient_Descent(X_b2, y, theta2, lr2, n_iter2)
#
#MSE2 = cost_history2[-1]
#
#plt.figure(3)
#plt.plot(np.linspace(1,50,50), cost_history2, 'b.')
#

""" DATA GENERATION - AUTO-REGRESSIVE SIGNAL """

m = 8                         # number of sensors
n = 16                         # number of sources
k = 16                         # max number of non-zero coef. in rows of X
L = 1000                 # number of sampels
k_true = 16 

Y_real, A_real, X_real = data_generation.generate_AR_v2(n, m, L, k_true) 



