# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:48:09 2020

@author: trine
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

from main import Main_Algorithm
import simulated_data
import ICA
# #############################################################################
np.random.seed(1234)

# simulated deterministic data set 
M = 3
N = 3
k = 3
L = 1000
n_seg = 1
segment = 0

Y, A_real, X_real = simulated_data.mix_signals(L,M,version='test')

# Transpose to fit main algorithm with segments
Y = Y.T
A_real = A_real.T
X_real = X_real.T


#_Y[segment] += 0.2 * np.random.normal(size=_Y[segment].shape)  # Add noise
Y /= Y.std(axis=0)  # Standardize data

# Compute fast ICA
ica1 = FastICA(n_components=N)
X_ica = ica1.fit_transform(Y)  # Reconstruct signals
A_ica = ica1.mixing_  # Get estimated mixing matrix




# `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(Y, np.dot(X_ica, A_ica.T) + ica1.mean_)

# For comparison, compute PCA
pca1 = PCA(n_components=N)
H1 = pca1.fit_transform(Y)  # Reconstruct signals based on orthogonal components


# #############################################################################
# Generate sample data 
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)
    
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
#S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)


# #############################################################################
# Plot results

plt.figure(0)

models = [X, S, S_]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()

plt.figure(1)

models = [Y, X_real, X_ica]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()

plt.figure(2)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real[0])):
    if np.any(X_real.T[i]!=0) or np.any(X_ica.T[i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(X_real.T[i], 'g',label='Real X')
        plt.plot(X_ica.T[i],'r', label='ICA X')

plt.legend(loc='lower right')
plt.xlabel('sample')
#plt.tight_layout()
plt.show
#plt.savefig('figures/ICA_app1.png')

plt.figure(3)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(S[0])):
    if np.any(S.T[i]!=0) or np.any(S_.T[i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(S.T[i], 'g',label='Real X')
        plt.plot(S_.T[i],'r', label='ICA X')

plt.legend(loc='lower right')
plt.xlabel('sample')
#plt.tight_layout()
plt.show
plt.savefig('figures/ICA_app1.png')

