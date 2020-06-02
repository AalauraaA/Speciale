# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:30:49 2020

@author: Mattek10b

This script perform the test on the synhetic data sets. You need to manuel
choose the kind of data set you want:
    - data = 'Mix' for deterministic data set
    - data = 'Ran' for stochastic data set
An you must choose the size of system and/or choice for mixing matrix A:
    - data = 'Mix'
        - branch = 'Cov-DL1' for system triggring Cov-DL1
        - branch = 'Cov-DL2' for system triggring Cov-DL2
        - branch = 'Fix' for system omitting Cov-DL, instead N(0,2)
        - branch = 'True1' for system with real A and M = N = k
        - branch = 'True2' for system with real A and Cov-DL2 system settings
        - branch = 'True3' for system with real A and Cov-DL1 system settings
        - branch = 'True4' for system with real A and N=k and M = 3
    
    - data = 'Ran'
        - branch = 'N8' for system with real A and N = k = 8 and M = 6
        - branch = 'N16' for system with real A and N = k = 16 and M = 6        

This script need the module:
    - Main_Algorithm
    - Data_Simulation
"""
from Main_Algorithm import Main_Algorithm
from M_SBL import M_SBL
import Data_Simulation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345) 

data = 'Ran'   # choose between 'Mix' or 'Ran'

# =============================================================================
# Simulated Data Set
# =============================================================================
if data == 'Mix':
    branch = 'True4'  # choose between 'Cov-DL1', 'Cov-DL2', 'Fix', 'True1', 'True2', 'True3' and 'True4'
    if branch == 'Cov-DL1':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 8        # Number of source
        
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=1, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
      
        " Recover A and X "
        A_result, A_init, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = False)
        
        " MSE "
        A_mse = Data_Simulation.MSE_one_error(A_real,A_result[0])
        A_mse0 = Data_Simulation.MSE_one_error(A_real,np.zeros(A_real.shape))

        print('\nMSE_A = {}'.format(np.round_(A_mse,decimals=3)))
        print('MSE_A_0 = {}'.format(np.round_(A_mse0,decimals=3)))
    
    if branch == 'Cov-DL2':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 5        # Number of source
            
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=None, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover A and X "
        A_result, A_init, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = False)

        " MSE "
        A_mse = Data_Simulation.MSE_one_error(A_real,A_result[0])
        A_mse0 = Data_Simulation.MSE_one_error(A_real,np.zeros(A_real.shape))

        print('\nMSE_A = {}'.format(np.round_(A_mse,decimals=3)))
        print('MSE_A_0 = {}'.format(np.round_(A_mse0,decimals=3)))
        
    if branch == 'Fix':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 5        # Number of source
            
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=None, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover A and X "
        A_result, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = True)

    if branch == 'True1':
        M = 4        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 4        # Number of source

        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=0, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)

        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))

    if branch == 'True2':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 5        # Number of source

        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=None, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)

        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))
        
    if branch == 'True3':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 8        # Number of source

        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=1, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)
        
        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))

    if branch == 'True4':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 4        # Number of source

        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=0, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)
        
        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))

# =============================================================================
# Stochastic Data Set
# =============================================================================
if data == 'Ran':
    branch = 'N16'
    if branch == 'N16':
        M = 6         # Number of sensors
        L = 1000      # Number of samples
        k = 16        # Number of active source
        N = 16        # Number of source
        
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.generate_AR(N, M, L, k)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
      
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)

        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))

    if branch == 'N8':
        M = 6         # Number of sensors
        L = 1000      # Number of samples
        k = 8        # Number of active source
        N = 8        # Number of source
        
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.generate_AR(N, M, L, k)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
      
        " Recover X "
        X_result = M_SBL(A_real, Y[0], M, N, k, iterations=1000, noise=False)
 
        " MSE "
        mse_array, mse_avg = Data_Simulation.MSE_segments(X_real[0],X_result)
        print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))
                

# =============================================================================
# Visulization of Deterministic Data Set
# =============================================================================
if data == 'Mix':
    if branch == 'Cov-DL1':
        plt.figure(0)
        models = [X_real[0], Y[0]]
        names = ['Source Signals, $\mathbf{X}$',
                 'Measurements, $\mathbf{Y}$',
                 ]
        colors = ['red', 'steelblue', 'orange', 'green', 'yellow', 'blue', 'cyan',
                  'purple']
        
        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(2, 1, ii)
            plt.title(name)
            for sig, color in zip(model, colors):
                plt.plot(sig, color=color)
        
        plt.tight_layout()
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show()
        plt.savefig('figures/simple_data.png')
        
        plt.figure(1)
        plt.title('Comparison of Mixing Matrix - COV-DL1')
        plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
        plt.plot(np.reshape(A_result[0], (A_result[0].size)),'o-r', label=r'Estimate $\hat{\mathbf{A}}$')
        plt.legend()
        plt.xlabel('index')
        plt.show()
        plt.savefig('figures/COV1_simple.png')
    
    if branch == 'Cov-DL2':
        plt.figure(0)
        models = [X_real[0], Y[0]]
        names = ['Source Signals, $\mathbf{X}$',
                 'Measurements, $\mathbf{Y}$',
                 ]
        colors = ['red', 'steelblue', 'orange', 'green', 'yellow', 'blue', 'cyan',
                  'purple']
        
        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(2, 1, ii)
            plt.title(name)
            for sig, color in zip(model, colors):
                plt.plot(sig, color=color)
        
        plt.tight_layout()
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show()
        plt.savefig('figures/simple_data.png')
    
        plt.figure(1)
        plt.title('Comparison of Mixing Matrix - COV-DL2')
        plt.plot(np.reshape(A_real, (A_real.size)), 'o-g',label=r'True $\mathbf{A}$')
        plt.plot(np.reshape(A_result[0], (A_result[0].size)),'o-r', label=r'Estimate $\hat{\mathbf{A}}$')
        plt.plot(np.reshape(A_init,(A_init.size)), 'o-b',label=r'init $\mathbf{A}$')
        plt.legend()
        plt.xlabel('index')
        plt.show()
        plt.savefig('figures/COV2_simple.png')
    
    
    if branch == 'True1':
        plt.figure(1)
        plt.title('Source matrix - M-SBL')
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
    
                plt.plot(X_real[0][i], 'g',label='Real X')
                plt.plot(X_result[i],'r', label='Recovered X')
        plt.legend(loc='lower right')
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show()
        plt.savefig('figures/M-SBL_simple0.png')
    
    if branch == 'True2':
        plt.figure(1)
        plt.title('Source matrix - M-SBL')
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
               
                plt.plot(X_real[0][i], 'g',label='Real X')
                plt.plot(X_result[i],'r',label='Recovered X')
        plt.legend()
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show
        plt.savefig('figures/M-SBL_simple1.png')
    
    if branch == 'True3':
        plt.figure(1)
        plt.title('Source matrix - M-SBL')
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
               
                plt.plot(X_real[0][i], 'g',label='Real X')
                plt.plot(X_result[i],'r',label='Recovered X')
        plt.legend()
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show
        plt.savefig('figures/M-SBL_simple2.png')
    
    if branch == 'True4':
        plt.figure(1)
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
               
                plt.plot(X_real[0][i], 'g',label='Real X')
                plt.plot(X_result[i],'r',label='Recovered X')
        plt.legend()
        plt.xlabel('sample')
        plt.tight_layout()
        plt.show
        plt.savefig('figures/M-SBL_simple3.png')
        
# =============================================================================
# Visualization of Stochastic Data sets
# =============================================================================
if data == 'Ran':
    if branch == 'N8':
        plt.figure(1)
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
               
                plt.plot(X_real[0][i][:100], 'g',label='Real X')
                plt.plot(X_result[i][:100],'r',label='Recovered X')
        plt.legend()
        plt.xlabel('sample')
        plt.show
        plt.savefig('figures/M-SBL_AR1.png')
    
    if branch == 'N16':
        plt.figure(1)
        nr_plot=0
        for i in range(len(X_real.T[0])):
            if np.any(X_real[0][i]!=0) or np.any(X_result[i]!=0):
                nr_plot += 1
                plt.subplot(k, 1, nr_plot)
               
                plt.plot(X_real[0][i][:100], 'g',label='Real X')
                plt.plot(X_result[i][:100],'r',label='Recovered X')
        plt.legend()
        plt.xlabel('sample')
        plt.show
        plt.savefig('figures/M-SBL_AR2.png')