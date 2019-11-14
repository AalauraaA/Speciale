## -*- coding: utf-8 -*-
#"""
#Created on Tue Sep 24 10:10:46 2019
#
#@author: Laura
#"""
#import numpy as np
#
#M = 20  # Sensors
#N = (M*(M+1))/2# Sources
#L = 1024 # Time samples
#n = 1
#k = N/5
#
#A = np.zeros(M,N) # Dictionary Matrix
#Y = np.zeros(M,L) # Observed Matrix
#X = np.zeros(N,L)
#
#gamma = np.zeros(N)
#G = np.diag(gamma)
#s = 0               # sigma**2
#Sigma = A * G * A.T + s * np.identity(N)
#
#""" Fixed Point Update """
#for i in range(len(N)):
#    gamma[i+1] = (gamma[i])/(np.sqrt(A[i].T * np.linalg.inv(Sigma[i])*A[i]))*(np.linalg.norm(Y.T * np.linalg.inv(Sigma[i]) * A[i], ord=2))/(np.sqrt(n))
#
#
#S = np.nonzero(gamma)     

"""
Algorithm Summary:
Given Y and a dictionary A 
1. Initialise gamma (gamma = 1)
2. Compute Sigma and Mu (posterior moments)
3. Update gamma with EM rule or fast fixed-point
4. Iterate step 2 and 3 until convergence to a fixed point gamma* 
5. Assuming a point estimate is desired for the unknown weights X_gen,
   choose X_m-sbl = Mu* = X_gen, with Mu* = E[X|Y ; gamma*]
6. Given that gamma* is sparse, the resultant estimator Mu* 
   will necessarily be row sparse.

Threshold = 10E-16
"""


from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

def gaus_post(n_params=100, n_sample=100, mean = 0, gamma = 1, n_size=100):
    params = np.linspace(-1, 1, n_params)
    
    sample = np.random.normal(0, gamma, n_size)
    likelihood = np.array([np.product(st.norm.pdf(sample,p)) for p in params])
    likelihood = likelihood / np.sum(likelihood)
    
    """
    Gaussian prior with mean = 0 and variance = gamma_i
    """
    prior_sample = np.random.normal(mean, gamma, n_size)
    prior = np.array([np.product(st.norm.pdf(prior_sample,p)) for p in params])
    prior = prior / np.sum(prior)
    
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
     
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(params, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(params, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(params, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()
     
    return prior, posterior


def MSBL_post(n_params=100, n_sample=100, mean = 0, n_size=100):
    params = np.linspace(-1, 1, n_params)
    #gamma = Gamma(A, n, Y)
    gamma = 1
    
    """
    Likelihood
    """
    sample = np.random.normal(0, gamma, n_size)
    likelihood = np.array([np.product(st.norm.pdf(sample,p)) for p in params])
    likelihood = likelihood / np.sum(likelihood)
    
    """
    Gaussian prior with mean = 0 and variance = gamma_i
    """
    prior_sample = np.random.normal(mean, gamma, n_size)
    prior = np.array([np.product(st.norm.pdf(prior_sample,p)) for p in params])
    prior = prior / np.sum(prior)
    
    """
    Posterior 
    """
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)
     
    """
    Plots
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(params, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(params, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(params, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()
     
    return prior, posterior

def Sigma(A):
    A = A
    Lambda = np.diag(gamma)
    I = np.identity(A.shape[0],A.shape[1])
    sigma = 0 
    return A * Lambda * A.T + sigma**2 * I

def gamma(A, n, Y):
    Sig = Sigma(A)
    a = A.shape[0]
    for i in range(10):
        gamma[i][0] = 0
        for k in range(10):
            gamma[i][k+1] = gamma[i][k]/np.sqrt(a[i].T *np.linalg.inv(Sig[k]) * a[i]) * np.linalg.norm(Y.T * np.linalg.inv(Sig[k]) * a[i], ord=2)/np.sqrt(n)