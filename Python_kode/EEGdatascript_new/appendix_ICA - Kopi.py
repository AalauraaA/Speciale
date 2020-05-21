import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from main import Main_Algorithm
import simulated_data
import ICA
from sklearn.metrics import mean_squared_error


np.random.seed(1234)


#from simulated_data import generate_AR

M = 6
N = 6
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

# fitting the ICA rows 
#

comb = np.array([0,2,1,3,4,5])
print(comb)
def cost(comb,X_ica,X_real):
    """
    the cost function shall return the mse() depending on the kombination of
    rows
    input data for one segment
    """
    X_ica_fit = np.zeros(X_ica.shape)
    for i in range(X_real.shape[0]):
        X_ica_fit[int(i)] = X_ica[int(comb[i])]

    return mean_squared_error(X_ica_fit, X_real)

bnds = ((0, N), (0, N), (0, N), (0, N), (0, N), (0, N))

# predefined optimization method, without defineing the gradient og the cost.
from scipy.optimize import minimize
res = minimize(cost, comb, args=(X_ica[segment],X_real[segment]), method='SLSQP', bounds=bnds,
                  options={'maxiter': 10000, 'disp': True})
comb_opt = res.x
print(comb_opt)
ms = cost(comb,X_ica[segment],X_real[segment])
comb_opt = comb


X_ica_fit = np.zeros(X_ica.shape)
for i in range(X_real[segment].shape[0]):
    X_ica_fit[segment][int(i)] = X_ica[segment][int(comb_opt[i])]


X_mse2 = mean_squared_error(X_ica_fit[segment].T, X_real[segment].T)
X_mselist2 = mean_squared_error(X_ica_fit[segment].T, X_real[segment].T,multioutput='raw_values')

print('MSE_X = {}'.format(X_mse))
print('MSE_X2 = {}'.format(X_mse2))

#for f in range(N):       
#    amp = np.max(X_real[segment][f])/np.max(X_ica_fit[segment][f])
#    X_ica_fit[segment][f] = X_ica_fit[segment][f]*amp

plt.figure(4)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    nr_plot += 1
    plt.subplot(N, 1, nr_plot)
    #plt.xticks(" ")
   
    plt.plot(X_real[segment][i][:100], 'g',label='$\mathbf{X}$ True')
    plt.plot(X_ica[segment][i][:100],'r', label='$\hat{\mathbf{X}}$ ICA')
        
plt.legend(loc='lower right')
plt.xlabel('sample')
#plt.tight_layout()
plt.show
#plt.savefig('figures/ICA_app4.png')

plt.figure(5)
plt.title('Source matrix - M-SBL')
nr_plot=0
for i in range(len(X_real.T[0])):
    nr_plot += 1
    plt.subplot(N, 1, nr_plot)
    #plt.xticks(" ")
   
    plt.plot(X_real[segment][i][:100], 'g',label='$\mathbf{X}$ True')
    plt.plot(X_ica_fit[segment][i][:100],'r', label='$\hat{\mathbf{X}}$ ICA_fit')
        
plt.legend(loc='lower right')
plt.xlabel('sample')
#plt.tight_layout()
plt.show
#plt.savefig('figures/ICA_app5.png')
