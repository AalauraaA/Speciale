# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:36:20 2019

@author: Mattek9b

Rossler Function to Making Differential Equations for 6 different cases
"""
import os
data_path = os.getcwd()

import numpy as np
import scipy.io

# =============================================================================
# Rossler Function
# =============================================================================
def rosslerpaper(t,y,conf,wc):
    """
    This function is used to generate different network configurations for
    the Roessler oscillator.
    
    Inputs: 
        t, y: parameters
        conf: different configuration in paper (values 1 to 6 
              corresponding to networks 1 to 6 in figure 1 of the paper)
        wc  : additional noise 
     
    Paper: Payam Shahsavari Baboukani, Ghasem Azemi, Boualem Boashash, Paul 
               Colditz, Amir Omidvarnia.
    
           A novel multivariate phase synchrony measure: Application to 
           multichannel newborn EEG analysis, Digital Signal Processing, 
           Volume 84, 2019, Pages 59-68, ISSN 1051-2004, 
           https://doi.org/10.1016/j.dsp.2018.08.019.
    """
    w  = [1.05,1.05,1.05,1.05,1.05,1.05]  #initial condition on W - a list of 6 elements
    dy = np.zeros((18,1))                 #differential equations array - array size 18  x 1
    
    # Paramters for oscillations
    u = 1.5   # Initial condition on U
    a = 0.35  # a
    b = 0.2   # b
    c = 10    # c
    
    " 6 different configuration of connection between node 1 to 6 "
    # See Al Khassaweneh's 2015 IEEE TSP (Fig 6) and Eq. 43 for more 
    # details.
    if conf == 1:
        e = np.array([
             [0, 0.5, 0, 0, 0, 0],
             [0.5, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0.5],
             [0, 0, 0, 0, 0.5, 0]])    # Configuration 1, connection 1-2 and 5-6
    
    elif conf == 2:
        e = np.array([
             [0, 0, 0, 0, 0, 0.5],
             [0, 0, 0.5, 0, 0, 0],
             [0, 0.5, 0, 0, 0, 0],
             [0, 0, 0, 0, 0.5, 0],
             [0, 0, 0, 0.5, 0, 0],
             [0.5, 0, 0, 0, 0, 0]]) # Configuration 2  connection 1-6, 2-3 and 4-5
    
    elif conf == 3:        
        e = np.array([
             [0, 0, 0, 0, 0.5, 0.5],
             [0, 0, 0.5, 0.5, 0, 0],
             [0, 0.5, 0, 0.5, 0, 0],
             [0, 0.5, 0.5, 0, 0, 0],
             [0.5, 0, 0, 0, 0, 0.5],
             [0.5, 0, 0, 0, 0.5, 0]])  # Configuration 3  connection 2-3-4 and 1-5-6
        
    elif conf == 4:
        e = np.array([
             [0, 0.5, 0, 0, 0.5, 0.5],
             [0.5, 0, 0.5, 0.5, 0, 0],
             [0, 0.5, 0, 0.5, 0, 0],
             [0, 0.5, 0.5, 0, 0, 0],
             [0.5, 0, 0, 0, 0, 0.5],
             [0.5, 0, 0, 0, 0.5, 0]]) # Configuration 4 connection 2-3-4, 1-5-6 and 1-2
        
    elif conf == 5:
        e = np.array([
             [0, 0.5, 0, 0.5, 0.5, 0.5],
             [0.5, 0, 0.5, 0.5, 0.5, 0],
             [0, 0.5, 0, 0.5, 0, 0],
             [0.5, 0.5, 0.5, 0, 0.5, 0],
             [0.5, 0.5, 0, 0.5, 0, 0.5],
             [0.5, 0, 0, 0, 0.5, 0]])  # Configuration 5 connection 2-3-4, 1-5-6, 1-2, 4-5, 2-5 and 1-4
    
    elif conf == 6:
        e = 0.5 * np.ones(6,6)         # Configuration 6 --> full connection

    " Rossler Oscilators - Differential Equations "
    dy[0] = -w[0] * y[1] - y[2] + (e[1][0] * (y[3] - y[0]) + e[2][0] * (y[6] - y[0]) + e[3][0] * (y[9] - y[0]) + e[4][0] * (y[12] - y[0]) + e[5][0] * (y[15] - y[0])) + u * wc[0]
    dy[1] = -w[0] * y[0] - a * y[1]
    dy[2] = b + (y[0] - c) * y[2]

    dy[3] = -w[1] * y[4] - y[5] + (e[0,1] * (y[0] - y[3]) + e[2,1] * (y[6] - y[3]) + e[3,1] * (y[9] - y[3]) + e[4,1] * (y[12] - y[3]) + e[5,1] * (y[15] - y[3])) + u * wc[1]
    dy[4] = -w[1] * y[3] - a * y[4]
    dy[5] = b + (y[3] - c) * y[5]

    dy[6] = -w[2] * y[7] - y[8] + (e[0,2] * (y[0] - y[6]) + e[1,2] * (y[3] - y[6]) + e[3,2] * (y[9] - y[6]) + e[4,2] * (y[12] - y[6]) + e[5,2] * (y[15] - y[6])) + u * wc[2]
    dy[7] = -w[2] * y[6] - a * y[7]
    dy[8] = b + (y[6] - c) * y[8]

    dy[9] = -w[3] * y[10] - y[11] + (e[0,3] * (y[0] - y[9]) + e[1,3] * (y[3] - y[9]) + e[2,3] * (y[6] - y[9]) + e[4,3] * (y[12] - y[9]) + e[5,3] * (y[15] - y[9])) + u * wc[3]
    dy[10] = -w[3] * y[9] - a * y[10]
    dy[11] = b + (y[9] - c) * y[11]

    dy[12] = -w[4] * y[13] - y[14] + (e[0,4] * (y[0] - y[12]) + e[1,4] * (y[3] - y[12]) + e[2,4] * (y[6] - y[12]) + e[3,4] * (y[9] - y[12]) + e[5,4] * (y[15] - y[12])) + u * wc[3]
    dy[13] = -w[4] * y[12] - a * y[13]
    dy[14] = b + (y[12] - c) * y[14]

    dy[15] = -w[5] * y[16] - y[17] + (e[0,5] * (y[0] - y[15]) + e[1,5] * (y[3] - y[15]) + e[2,5] * (y[6] - y[15]) + e[3,5] * (y[9] - y[15]) + e[4,5] * (y[12] - y[15])) + u * wc[5]
    dy[16] = -w[5] * y[15] - a * y[16]
    dy[17] = b + (y[15] - c) * y[17]
    return dy # Return the differential equation

# =============================================================================
# Generated Rossler
# =============================================================================
def Generate_Rossler():
    """   
    Inputs:
        conf: is number of configuration (see rosslerpaper function).
        N:    is number of samples used.
        
    Output:
    """
#    tspan = np.linspace(0, 50, N) # 50 second and sample rate is 60 Hz
#    initial = np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0]) #initial condition for solving Rossler oscillator equation (1 x 18)
#    c = np.random.normal(3,1,(6,1)) #additional noise which is selected from random normal distibution
#    a = np.nonzero(c < 0) # Find indices of nonzero element less than zero
#    if a != 0:
#        c = np.random.normal(3,1,(6,1))
#    
    #HELP
    #y = integrate.odeint(rosslerpaper(tspan,initial,conf,c), initial, tspan) # solving rossler model with s1 configuration (for more detail see rosslerpaper function)
    #y = integrate.ode(rosslerpaper(tspan, initial,conf,c)) # solving rossler model with s1 configuration (for more detail see rosslerpaper function)
    """
    Loaded the solution (mat_sol) to the Rossler Oscillation Equation (differential 
    equation of size 18 x 1) with configuration (conf = 4) and N = 2000 samples. 
    The solution was computed from a time span of 50 second with a sample rate 
    of 60 Hz.
    """
    mat_sol1 = scipy.io.loadmat('solution_oscillation1.mat')
    sol1 = mat_sol1['y']  
    mat_sol2 = scipy.io.loadmat('solution_oscillation2.mat')
    sol2 = mat_sol2['y']
    mat_sol3 = scipy.io.loadmat('solution_oscillation3.mat')
    sol3 = mat_sol3['y']    
    mat_sol4 = scipy.io.loadmat('solution_oscillation4.mat')
    sol4 = mat_sol4['y']
    mat_sol5 = scipy.io.loadmat('solution_oscillation5.mat')
    sol5 = mat_sol5['y']
    mat_sol6 = scipy.io.loadmat('solution_oscillation6.mat')
    sol6 = mat_sol6['y']
    """
    X has been chosen from the solution space (size 2000x18) with indices 3 
    step a part (1, 4, 7, 10, 13, 16 in MatLab indexes)
    
    X has size 1940 x 6 because we have remove the first 60 samples
    """
    x1 = sol1.T[0:16:3]  # Extracting 6 
    x1 = x1.T
    x1 = x1[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    x2 = sol2.T[0:16:3]  # Extracting 6 
    x2 = x2.T
    x2 = x2[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    x3 = sol3.T[0:16:3]  # Extracting 6 
    x3 = x3.T
    x3 = x3[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    x4 = sol4.T[0:16:3]  # Extracting 6 
    x4 = x4.T
    x4 = x4[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    x5 = sol5.T[0:16:3]  # Extracting 6 
    x5 = x5.T
    x5 = x5[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    x6 = sol6.T[0:16:3]  # Extracting 6 
    x6 = x6.T
    x6 = x6[59:-1]       # Remove the 60 first samples, X is now 1940 x 6
    return x1, x2, x3, x4, x5, x6