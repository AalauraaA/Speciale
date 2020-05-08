# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:25:42 2020

@author: trine
"""
## plot functions


def plot_seperate_sources_comparison(X_real,X_reconstruction,M,N,k,L,figsave,nr):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(nr)
    plt.title('M = {}, N = {}, k = {}, L = {}'.format(M,N,k,L))
    nr_plot=0
    for i in range(N):
        if np.any(X_real[i]!=0) or np.any(X_reconstruction[i]!=0):
            nr_plot += 1
            plt.subplot(k, 1, nr_plot)
           
            plt.plot(X_real[i], 'r',label='Recovered X')
            plt.plot(X_reconstruction[i],'g', label='ICA X')
    
    plt.legend()
    plt.xlabel('sample')
    plt.show()
    plt.savefig(figsave)
    

def plot_seperate_sources(X_reconstruction,M,N,k,L,figsave,nr):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(nr)
    plt.title('M = {}, N = {}, k = {}, L = {}'.format(M,N,k,L))
    nr_plot=0
    for i in range(N):
        if np.any(X_reconstruction[i]!=0):
            nr_plot += 1
            plt.subplot(k, 1, nr_plot)
           
            plt.plot(X_reconstruction[i],'g', label='Recovered X')
    
    plt.legend()
    plt.xlabel('sample')
    plt.show()
    plt.savefig(figsave)
    