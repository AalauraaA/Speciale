import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from main import Main_Algorithm
import simulated_data
import ICA
from sklearn.metrics import mean_squared_error


np.random.seed(1234)

#simulated deterministic data set
M = 4
N = 4
k = 4
L = 1000
n_seg = 1
segment = 0
 
Y, A_real, X_real = simulated_data.mix_signals(L,M,version=0)
Y = np.reshape(Y, (1,Y.shape[0],Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[:-2]
X_real = X_real.T

X_ica, A_ica = ICA.fast_ica_segments(Y, M)

X_mse = mean_squared_error(X_ica[segment].T, X_real[segment].T)
X_mselist = mean_squared_error(X_ica[segment].T, X_real[segment].T,multioutput='raw_values')
A_mse = mean_squared_error(A_ica[segment].T, A_real.T)

# make amplitude a like for plot

plt.figure(1)
models = [Y[segment], X_real[segment], X_ica[segment]]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue','green']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()
plt.savefig('figures/ICA_app1.png')

X_ica2 = np.zeros((X_ica.shape))
X_ica2[segment] = ICA.ica_fit(X_ica[segment],X_real[segment],N)

X_mse2 = mean_squared_error(X_ica2[segment].T, X_real[segment].T)
X_mselist2 = mean_squared_error(X_ica2[segment].T, X_real[segment].T,multioutput='raw_values')

print('MSE_X = {}',format(X_mse))
print('MSE_X2 = {}',format(X_mse2))

plt.figure(2)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_ica2[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(X_real[segment][i], 'g',label='$\mathbf{X}$ True')
        plt.plot(X_ica2[segment][i],'r', label='$\hat{\mathbf{X}}$ ICA')

plt.legend(loc='lower right')
plt.xlabel('sample')
plt.tight_layout()
plt.show
plt.savefig('figures/ICA_app2.png')

A_main, X_main = Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10) 

#
## #################################### Stachastic data
#from simulated_data import generate_AR

M = 4
N = 4
k = 4
L = 1000
n_seg = 1
segment = 0

Y, A_real, X_real = simulated_data.generate_AR(N, M, L, k)

Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

X_ica, A_ica = ICA.fast_ica_segments(Y, M)
A_main, X_main= Main_Algorithm(Y, M, L, n_seg, A_real, L_covseg=10)

X_mse = mean_squared_error(X_ica[segment].T, X_real[segment].T)
X_mselist = mean_squared_error(X_ica[segment].T, X_real[segment].T,multioutput='raw_values')
A_mse = mean_squared_error(A_ica[segment].T, A_real.T)

X_mse_m = mean_squared_error(X_main[segment].T, X_real[segment].T)
X_mselist_m = mean_squared_error(X_main[segment].T, X_real[segment].T,multioutput='raw_values')
A_mse_m = mean_squared_error(A_main[segment].T, A_real.T)

# fitting the ICA rows 

X_ica2 = np.zeros((X_ica.shape))
X_ica2[segment] = ICA.ica_fit(X_ica[segment],X_real[segment],N)

X_mse2 = mean_squared_error(X_ica2[segment].T, X_real[segment].T)
X_mselist2 = mean_squared_error(X_ica2[segment].T, X_real[segment].T,multioutput='raw_values')

print('MSE_X = {}'.format(X_mse))
print('MSE_X2 = {}'.format(X_mse2))

plt.figure(3)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_ica[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(X_real[segment][i][:100], 'g',label='$\mathbf{X}$ True')
        plt.plot(X_ica2[segment][i][:100],'r', label='$\hat{\mathbf{X}}$ ICA')
        #plt.plot(X_main[segment][i][:100], '--b',dashes = [5, 5, 5, 5], label='$\mathbf{X}$ Main')
        
plt.legend(loc='lower right')
plt.xlabel('sample')
plt.tight_layout()
plt.show
plt.savefig('figures/ICA_app3.png')

plt.figure(4)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    if np.any(X_real[segment][i]!=0) or np.any(X_ica[segment][i]!=0):
        
        nr_plot += 1
        plt.subplot(k, 1, nr_plot)
        #plt.xticks(" ")
       
        plt.plot(X_real[segment][i][:100], 'g',label='$\mathbf{X}$ True')
        plt.plot(X_ica2[segment][i][:100],'r', label='$\hat{\mathbf{X}}$ ICA')
        plt.plot(X_main[segment][i][:100], '--b',dashes = [5, 5, 5, 5], label='$\mathbf{X}$ Main')
        
plt.legend(loc='lower right')
plt.xlabel('sample')
plt.tight_layout()
plt.show
plt.savefig('figures/ICA_app4.png')