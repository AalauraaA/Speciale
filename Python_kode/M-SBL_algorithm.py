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

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

def gaus_post(n_params=100, n_sample=100, mean = 0, gamma = 1, n_size=100):
    params = np.linspace(-1, 1, n_params)
    
    sample = np.random.normal(0, gamma, n_size)
    likelihood = np.array([np.product(st.norm.pdf(sample,p)) for p in params])
    likelihood = likelihood / np.sum(likelihood)
    
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

prior, posterior = gaus_post()
