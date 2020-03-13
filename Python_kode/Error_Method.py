# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:45:22 2020

@author: Laura

Different Error Methods
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def MSE1(real,estimate):
    """
    Mean Squared Error (MSE)
    ----------------------------------------------------------------------
    Info:
        The difference between the estimated values from the observed values
        The MSE is always positive (because of the squared).
        A small value -- close to zero -- is a good estimation
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error = np.zeros(len(real))
    for i in range(len(real)):
        error[i] = np.average((real[i] - estimate[i])**2)
    return error
    
def MSE2(real,estimate):
    """
    Mean Squared Error (MSE)
    ----------------------------------------------------------------------
    Info:
        The difference between the estimated values from the observed values
        The MSE is always positive (because of the squared).
        A small value -- close to zero -- is a good estimation
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error1 = ((real - estimate)**2).mean()
    error2 = (np.square(real - estimate)).mean()
    return error1, error2
    
def MSE_all_errors(real,estimate):
    """
    Mean Squared Error (MSE) - m or n errors
    ----------------------------------------------------------------------
    Info:
        A small value -- close to zero -- is a good estimation.
        The inputs must be transponet to perform the action rowwise
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error = mean_squared_error(real.T, estimate.T, multioutput='raw_values')
    return error

def MSE_one_error(real,estimate):
    """
    Mean Squared Error (MSE) - One Error
    ----------------------------------------------------------------------
    Info:
        A small value -- close to zero -- is a good estimation.
        The inputs must be transponet to perform the action rowwise
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: MSE
    """
    error = mean_squared_error(real.T, estimate.T)
    return error


#y_true = np.array([[0.5, 1],[-1, 1],[7, -6]])
#y_pred = np.array([[0, 2],[-1, 2],[8, -5]])
#
#mse1 = MSE1(y_true,y_pred)
#mse2 = MSE2(y_true,y_pred)
#mse3 = MSE3(y_true,y_pred)




