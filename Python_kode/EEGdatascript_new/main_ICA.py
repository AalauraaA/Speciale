# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

@author: trine
"""
import numpy as np
from main import Main_Algorithm_EEG
import simulated_data
import ICA
import data
from plot_functions import plot_seperate_sources_comparison

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
" Perform ICA on full dataset "
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
            
X_ica, A_ica = ICA.ica_segments(Y_ica, 1000)

" Remove the last column from X_ica to match size of X "
X_ica_new = np.array(X_ica, copy = True)
X_ica_array = []
for i in range(len(Y_ica)):
    X_ica_array.append(X_ica_new[i])
#    if i == 0:     
#        X_ica_array.append(X_ica_new[i,:,:-1])
#    else:
#       X_ica_array.append(X_ica_new[i]) 

" Replacing small values with zero and creating X_ica of size k x samples for each segment "
X_ica_nonzero = []
tol = 10E-3
for i in range(len(X_ica_array)): # Looking at one segment at time
    temp = [] # temporary variable to store the nonzero array for one segment
    for j in range(len(X_ica_array[i])): # Looking at on row of one segment at the time
        
        if np.average(X_ica_array[i][j]) < tol and np.average(X_ica_array[i][j]) > -tol:  # if smaller than 
            X_ica_array[i][j] = 0   # replace by zero
        else:
            temp.append(X_ica_array[i][j])
    X_ica_nonzero.append(temp)

" Finding the number of active sources (k) for each segment "
k = np.zeros(len(X_ica_nonzero))
for i in range(len(X_ica_nonzero)):
    # count the number of nonzeros rows in one segment
    k[i] = len(X_ica_nonzero[i])

# =============================================================================
# Main Algorithm with random A
# =============================================================================
request='remove 1/3' # remove sensors and the same sources from dataset - every third
Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

if data_name == 'S1_CClean.mat':
    " For S1_CClean.mat remove last sample of first segment "
    Y[0] = Y[0].T[:-1]
    Y[0] = Y[0].T

if data_name == 'S1_OClean.mat':
    for i in range(len(Y)):
        if i <= 22:
            Y[i] = Y[i].T[:-1]
            Y[i] = Y[i].T
        else:
            continue

X_result = []
mse = []
mse2 = []
for i in range(k.shape[0]):
    " Making the right size of X for all segments "
    X_result.append(np.zeros([len(Y), int(k[i])]))
    
    " Making the mse for all sources in all segments "
    mse.append(np.zeros([len(Y), int(k[i])]))
    mse2.append(np.zeros([len(Y), int(k[i])]))
  
average_mse = np.zeros(len(Y))
average_mse2 = np.zeros(len(Y))

for i in range(len(Y)): # Looking at one time segment
    A = np.random.normal(0,2,(M,int(k[i])))
    X_result[i] = Main_Algorithm_EEG(Y[i], A, M, int(k[i]), L)
    mse[i], average_mse[i] = simulated_data.MSE_segments(X_result[i], X_ica_nonzero[i])

""" Moving the Sources in X_result to match X_ica """   
X_result2 = []
for i in range(len(X_ica_nonzero)):
    segment=i
    X_result1 = np.array(X_result, copy=True)  # size 144 x k x 513
    X_result1 = np.reshape(X_result1[segment], (X_result[segment].shape))  # size k x 513
    temp = [] # temporary variable to store the nonzero array for one segment
    for j in range(X_result1.shape[0]):
        _list = np.zeros(X_result1.shape[0])  # size k
        for p in range(X_result1.shape[0]): # loop through k rows
            print('p', p)
            _list[p] = simulated_data.MSE_one_error(X_ica_nonzero[segment][j], X_result1[p])
        
        index = int(np.argmin(_list))
        temp.append(X_result1[index])
        X_result1 =  np.delete(X_result1,index,axis=0)
    X_result2.append(temp)
    
for i in range(len(Y)): # Looking at one time segment
    mse2[i], average_mse2[i] = simulated_data.MSE_segments(X_result2[i], X_ica_nonzero[i])
 

" Plots of second (i = 1) segment "
import matplotlib.pyplot as plt
i = 1
figsave = "figures/EEG_third_removed_ori" + str(data_name) + '_' + str(i) + ".png"
plot_seperate_sources_comparison(X_result[i],X_ica_nonzero[i],M,int(k[i]),int(k[i]),L,figsave,1)

i = 1
figsave = "figures/EEG_third_removed_new" + str(data_name) + '_' + str(i) + ".png"
plot_seperate_sources_comparison(X_result2[i],X_ica_nonzero[i],M,int(k[i]),int(k[i]),L,figsave,2)

plt.figure(3)
plt.plot(average_mse, '-bo', label = 'Original Sources Locations')
plt.plot(average_mse2, '-ro', label = 'New Sources Locations')
plt.title('Average MSE Values of Original and New Source Locations')
plt.legend()
plt.savefig('figures/average_mse_ori_new_third_removed_timeseg_1.png')

plt.figure(4)
plt.plot(average_mse, '-bo', label = 'Original Sources Locations')
plt.plot(average_mse2, '-ro', label = 'New Sources Locations')
plt.title('Average MSE Values of Original and New Source Locations - Zoom')
plt.legend()
plt.axis([-1,145, -2,10])
plt.savefig('figures/average_mse_ori_new_third_removed_timeseg_1_zoom.png')

plt.figure(5)
plt.plot(mse[1], '-bo', label = 'Original Sources Locations')
plt.plot(mse2[1], '-ro', label = 'New Sources Locations')
plt.title('MSE Values of Original and New Source Locations')
plt.legend()
plt.savefig('figures/mse_ori_new_third_removed_timeseg_1.png')


# =============================================================================
# Main Algorithm with A_ica
# =============================================================================
# remove sensors and the same sources from dataset - every third
#request='remove 1/3'
#Y, M, L, n_seg = data._import(data_file, segment_time, request=request)
#Y[0] = Y[0].T[:-1]
#Y[0] = Y[0].T

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
