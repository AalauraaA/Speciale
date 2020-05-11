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
import X_ICA
import X_MAIN

np.random.seed(1234)

# =============================================================================
# Import EEG data file
# =============================================================================
data_name = 'S1_CClean.mat'
#data_name = 'S4_OClean.mat'
data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" Import Segmented Dataset "
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')

X_ica_nonzero, k = X_ICA.X_ica(data_name, Y_ica, M_ica)

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
#request = 'none'

Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

X_result, Y = X_MAIN.X_main(data_name, Y, M, k)

def scale(Y, X_ica, X_main):
    for seg in range(len(Y)): 
    # Looking at one time segment
        for f in range(len(X_ica[seg])):
            amp = np.max(X_main[seg][f])/np.max(X_ica[seg][f])
            X_ica[seg][f] = X_ica[seg][f]*amp
    return X_ica

X_copy = scale(Y, X_ica_nonzero, X_result)

    
" Calculate the MSE and Average MSE with X_ica and fitted amplitude X_ica "
mse = []         # Original MSE for each rows of the original recovered source matrix X and X_ica
#mse2 = []        # Fitted MSE for each rows of the original recovered source matrix X and fitted X_ica

average_mse = np.zeros(len(Y))  # Original average MSE for each rows of the original recovered source matrix X and X_ica
#average_mse2 = np.zeros(len(Y)) # Fitted average MSE for each rows of the original recovered source matrix X and fitted X_ica

for seg in range(k.shape[0]):   
    " Making the mse for all sources in all segments "
    mse.append(np.zeros([len(Y), int(k[seg])]))
#    mse2.append(np.zeros([len(Y), int(k[seg])]))

for seg in range(len(Y)): 
    # Looking at one time segment
    mse[seg], average_mse[seg] = simulated_data.MSE_segments(X_result[seg], X_ica_nonzero[seg])
#    mse2[seg], average_mse2[seg] = simulated_data.MSE_segments(X_result2[seg], X_ica_nonzero[seg])

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
    

" Plots of (seg = 55) segment "
segment = 55
figsave1 = "figures/EEG_second_removed_timeseg55" + str(data_name) + '_' + ".png"
figsave2 = "figures/EEG_second_removed_scaled_timeseg55" + str(data_name) + '_' + ".png"
index = [0, 5, 10, int(k[segment])-1]

plt.figure(1)
plt.subplot(4, 1, 1)
plt.plot(X_result[segment][index[0]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[0]], 'r', label='ICA')
plt.title('Recovered Source Matrix X for Time Segment = 56')
plt.xlabel('Source 1')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(4, 1, 2)
plt.plot(X_result[segment][index[1]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[1]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 6')
plt.subplot(4, 1, 3)
plt.plot(X_result[segment][index[2]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[2]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 11')
plt.subplot(4, 1, 4)
plt.plot(X_result[segment][index[3]], 'g', label='Main Alg.')
plt.plot( X_ica_nonzero[segment][index[3]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 18')
plt.tight_layout() 
plt.show()
plt.savefig(figsave1)
    
plt.figure(2)
plt.plot(average_mse, '-ro', label = 'Average MSE')
#plt.hlines(5, 0, 144, label='Tolerance = 5') # horizontial line
#plt.plot(average_mse2, '-bo', label = 'Average MSE + amp')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for all time segments')
plt.xlabel('Time Segment')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/average_mse_second_removed_ica.png')

plt.figure(3)
plt.plot(average_mse, '-ro', label = 'Average MSE')
plt.hlines(5, 0, 144, label='Tolerance = 5') # horizontal line
#plt.plot(average_mse2, '-bo', label = 'Average MSE + amp')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for all time segments')
plt.xlabel('Time Segment')
plt.ylabel('MSE')
plt.legend()
plt.axis([-1,145, -10,50])
plt.savefig('figures/average_mse_second_removed_ica_zoom.png')

plt.figure(4)
plt.plot(mse[segment], '-ro', label = 'MSE per row')
#plt.plot(mse2[seg], '-bo', label = 'MSE')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for time segment = 56')
plt.xlabel('Sources')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/mse_second_removed_ica_timeseg55.png')
          
plt.figure(5)
plt.subplot(4, 1, 1)
plt.plot(X_result[segment][index[0]], 'g', label='Main Alg.')
plt.plot(X_copy[segment][index[0]], 'r', label='Scale ICA')
plt.title('Recovered Source Matrix X for Time Segment = 56')
plt.xlabel('Source 1')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(4, 1, 2)
plt.plot(X_result[segment][index[1]], 'g', label='Main Alg.')
plt.plot(X_copy[segment][index[1]], 'r', label='Scale ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 6')
plt.subplot(4, 1, 3)
plt.plot(X_result[segment][index[2]], 'g', label='Main Alg.')
plt.plot(X_copy[segment][index[2]], 'r', label='Scale ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 11')
plt.subplot(4, 1, 4)
plt.plot(X_result[segment][index[3]], 'g', label='Main Alg.')
plt.plot(X_copy[segment][index[3]], 'r', label='Scale ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source 18')
plt.tight_layout() 
plt.show()
plt.savefig(figsave2)

# =============================================================================
# Calculating Average of the Average MSE 
# =============================================================================
#Find gennemsnittet og find dem der ligger over og under tol = 5
one_average = np.average(average_mse)
print('The average mse of all average time segments: ', one_average)

tol = 5
under = 0
on = 0
over = 0
for seg in range(len(average_mse)):
    if average_mse[seg] < tol:
        under += 1
    if average_mse[seg] > tol:
        over += 1
    if average_mse[seg] == tol:
        on += 1

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