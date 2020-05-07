# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

@author: trine
"""

import numpy as np
from main import Main_Algorithm_EEG
import matplotlib.pyplot as plt
#from scipy import signal
from sklearn.decomposition import FastICA
import simulated_data
#import ICA
import data

np.random.seed(1234)

# =============================================================================
# Import EEG data file
# =============================================================================
data_name = 'S1_CClean.mat'
#data_name = 'S1_OClean.mat'
data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" Import Segmented Dataset "
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')

if data_name == 'S1_CClean.mat':
    " For S1_CClean.mat remove last sample of first segment "
    Y_ica[0] = Y_ica[0].T[:-1]
    Y_ica[0] = Y_ica[0].T

if data_name == 'S1_OClean.mat':
    for i in range(len(Y_ica)):
        if i <= 22:
            Y_ica[i] = Y_ica[i].T[:-1]
            Y_ica[i] = Y_ica[i].T
        else:
            continue
            
#X_ica, A_ica = ICA.fast_ica_segments(Y_ica, M_ica)
" Perform FastICA on Segmented Dataset "
n_seg = len(Y_ica)
N = Y_ica[0].shape[0]
L = Y_ica[0].shape[1]

X_ica = np.zeros((n_seg, N, L-2))
A_ica = np.zeros((n_seg, N, M_ica))
for i in range(len(Y_ica)):   
    X = Y_ica[i].T
    
    ica = FastICA(n_components=N)
    X_ICA = ica.fit_transform(X)  # Reconstruct signals
    A_ICA = ica.mixing_
    X_ica[i] = X_ICA[:L-2].T
    A_ica[i] = A_ICA              # Get estimated mixing matrix

" Copy the X_ica to an Array "
X_ica_new = np.array(X_ica, copy = True)
X_ica_array = []
for i in range(len(Y_ica)):
    X_ica_array.append(X_ica_new[i])

" Replacing small values with zero and creating X_ica of size k x samples for each segment "
X_ica_nonzero = []
tol = 10E-5
for seg in range(len(X_ica_array)): 
    # Looking at one segment at time
    temp = []                       # temporary variable to store the nonzero array for one segment
    for i in range(len(X_ica_array[seg])): 
        # Looking at on row of one segment at the time   
        if np.average(X_ica_array[seg][i]) < tol and np.average(X_ica_array[seg][i]) > -tol:  # if smaller than 
            X_ica_array[seg][i] = 0   # replace by zero
        else:
            temp.append(X_ica_array[seg][i])
            
    X_ica_nonzero.append(temp)


" Finding the number of active sources (k) for each segment "
k = np.zeros(len(X_ica_nonzero))
for seg in range(len(X_ica_nonzero)):
    # count the number of nonzeros rows in one segment
    k[seg] = len(X_ica_nonzero[seg])

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
#request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
request = 'none'

Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

if data_name == 'S1_CClean.mat':
    " For S1_CClean.mat remove last sample of first segment "
    Y[0] = Y[0].T[:-1]
    Y[0]=Y[0].T

if data_name == 'S1_OClean.mat':
    for i in range(len(Y)):
        if i <= 22:
            Y[i] = Y[i].T[:-1]
            Y[i] = Y[i].T
        else:
            continue

" Perform Main Algorithm on Segmented Dataset "
X_result = []    # Original recovered source matrix X
X_result2 = []   # Original recovered source matrix X used to fitting X_ica

mse = []         # Original MSE for each rows of the original recovered source matrix X and X_ica
mse2 = []        # Fitted MSE for each rows of the original recovered source matrix X and fitted X_ica

average_mse = np.zeros(len(Y))  # Original average MSE for each rows of the original recovered source matrix X and X_ica
average_mse2 = np.zeros(len(Y)) # Fitted average MSE for each rows of the original recovered source matrix X and fitted X_ica

for i in range(k.shape[0]):
    " Making the right size of X for all segments "
    X_result.append(np.zeros([len(Y), int(k[i])]))
    X_result2.append(np.zeros([len(Y), int(k[i])]))
    
    " Making the mse for all sources in all segments "
    mse.append(np.zeros([len(Y), int(k[i])]))
    mse2.append(np.zeros([len(Y), int(k[i])]))


average_mse = np.zeros(len(Y))
average_mse2 = np.zeros(len(Y))

" Original Recovered Source Matrix X, MSE and Average MSE with X_ica "
for seg in range(len(Y)): 
    # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[seg])))
    X_result[seg] = Main_Algorithm_EEG(Y[seg], A, M, int(k[seg]), L)
    mse[seg], average_mse[seg] = simulated_data.MSE_segments(X_result[seg], X_ica_nonzero[seg])

#for seg in range(len(X_ica_nonzero)):
#    for f in range(len(X_ica_nonzero[seg])):       
#        amp = np.max(X_result2[seg][f])/np.max(X_ica_nonzero[seg][f])
#        X_ica_nonzero[seg][f] = X_ica_nonzero[seg][f]*amp

" Original Recovered Source Matrix X, MSE and Average MSE with fitted amplitude X_ica "
for seg in range(len(Y)): 
    # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[seg])))
    X_result2[seg] = Main_Algorithm_EEG(Y[seg], A, M, int(k[seg]), L)
    for f in range(len(X_ica_nonzero[seg])):
        amp = np.max(X_result2[seg][f])/np.max(X_ica_nonzero[seg][f])
        X_ica_nonzero[seg][f] = X_ica_nonzero[seg][f]*amp
    mse2[seg], average_mse2[seg] = simulated_data.MSE_segments(X_result2[seg], X_ica_nonzero[seg])

#" Original Recovered Source Matrix X, MSE and Average MSE with fitted amplitude and location of X_ica "
#for seg in range(len(X_ica_nonzero)):
#    X_result1 = np.array(X_result, copy=True)  # size 144 x k x 513
#    X_result1 = np.reshape(X_result1[seg], (X_result[seg].shape))  # size k x 513
#    temp = [] # temporary variable to store the nonzero array for one segment
#    for i in range(X_result1.shape[0]):
#        _list = np.zeros(X_result1.shape[0])  # size k
#        for j in range(X_result1.shape[0]): # loop through k rows
#            _list[j] = simulated_data.MSE_one_error(X_ica_nonzero[seg][i], X_result1[j])
#        index = int(np.argmin(_list))
#        temp.append(X_result1[index])
#        X_result1 =  np.delete(X_result1,index,axis=0)
#    X_result2.append(temp)
#
#for seg in range(len(Y)): 
#    # Looking at one time segment
#    A = np.random.normal(0,2,(M,int(k[seg])))
#    X_result2[seg] = Main_Algorithm_EEG(Y[seg], A, M, int(k[seg]), L)
#    for f in range(len(X_ica_nonzero[seg])):
#        amp = np.max(X_result2[seg][f])/np.max(X_ica_nonzero[seg][f])
#        X_ica_nonzero[seg][f] = X_ica_nonzero[seg][f]*amp
#    mse2[seg], average_mse2[seg] = simulated_data.MSE_segments(X_result2[seg], X_ica_nonzero[seg])
    

" Plots of second (seg = 54) segment "
seg = 54
figsave = "figures/EEG_non_removed_timeseg54" + str(data_name) + '_' + str(seg) + ".png"

plt.figure(1)
index = [0, 5, 10, int(k[seg])-1]
plt.subplot(4, 1, 1)
plt.plot(X_result[seg][index[0]], 'g', label='Main Alg. - Source 0')
plt.plot(X_ica_nonzero[seg][index[0]], 'r', label='ICA - Source 0')
plt.title('Recovered Source Matrix X for Time Segment = 55')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(X_result[seg][index[1]], 'g', label='Main Alg. - Source 5')
plt.plot(X_ica_nonzero[seg][index[1]], 'r', label='ICA - Source 5')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(X_result[seg][index[2]], 'g', label='Main Alg. - Source 10')
plt.plot(X_ica_nonzero[seg][index[2]], 'r', label='ICA - Source 10')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(X_result[seg][index[3]], 'g', label='Main Alg. - Source 13')
plt.plot( X_ica_nonzero[seg][index[3]], 'r', label='ICA - Source 13')
plt.legend()
plt.xlabel('Sample')
plt.show()
plt.savefig(figsave)
    
plt.figure(2)
plt.plot(average_mse, '-ro', label = 'Average MSE')
#plt.plot(average_mse2, '-bo', label = 'Average MSE + amp')
plt.title('Average MSE Values of All Time Segments')
plt.xlabel('Time Segment')
plt.ylabel('Average MSE')
plt.legend()
plt.savefig('figures/average_mse_non_removed_ica.png')

plt.figure(3)
plt.plot(average_mse, '-ro', label = 'Average MSE')
#plt.plot(average_mse2, '-bo', label = 'Average MSE + amp')
plt.title('Average MSE Values of All Time Segments - zoom')
plt.legend()
plt.axis([-1,145, -10,50])
plt.savefig('figures/average_mse_non_removed_ica_zoom.png')

plt.figure(4)
plt.plot(mse[seg], '-ro', label = 'MSE')
#plt.plot(mse2[seg], '-bo', label = 'MSE')
plt.title('MSE Values of One Time Segment = 55')
plt.xlabel('Sources')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/mse_non_removed_ica_timeseg54.png')

# =============================================================================
# Main Algorithm with A_ica
# =============================================================================
# remove sensors and the same sources from dataset - every third
#request='remove 1/3'
#Y, M, L, n_seg = data._import(data_file, segment_time, request=request)
#Y[0] = Y[0].T[:-1]
#Y[0] = Y[0].T
#
#A_ica_array = np.array(A_ica, copy=True)
#A_ica_array = data._reduction(A_ica_array, request)
##
##for i in range(len(X_ica_nonzero)):
##    for j in range(len(X_ica_nonzero[i])):
##        X_ica_nonzero[i][j] = X_ica_nonzero[i][j][:-1]
#
#X_result = []
#A = []
#mse = []
#for i in range(len(k)):
#    " Making the right size of X for all segments "
#    X_result.append(np.zeros([len(Y), int(k[i])]))
#    
#    " Making the mixing matrix A from ICA for all segments "
#    A.append(A_ica_array[:,:int(k[i])])
#    
#    " Making the mse for all sources in all segments "
#    mse.append(np.zeros([len(Y), int(k[i])]))
#
#average_mse = np.zeros(len(Y)) # Average MSE for each segments (size 28 x 1)
#
#for i in range(len(Y)):
#    # Looking at one time segment
#    X_result[i] = Main_Algorithm_EEG(Y[i], A[i], M, int(k[i]), L)
#    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_nonzero[i])
#
##print('MSE = {}'.format(average_mse))
#
#" Plot the all the sources of time segment 1 "
#plot_seperate_sources_comparison(X_result[1], X_ica_nonzero[1], M, int(k[1]), int(k[1]), L)