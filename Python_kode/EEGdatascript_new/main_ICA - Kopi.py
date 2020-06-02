# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:36 2020

@author: trine
"""

import numpy as np
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

data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" Import Segmented Dataset "
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')
Y_ica = data.fit_Y(Y_ica, data_name)

X_ica_nonzero, k = X_ICA.X_ica(data_name, Y_ica, M_ica)

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
#request = 'none'

Y, M, L, n_seg = data._import(data_file, segment_time, request=request)
Y = data.fit_Y(Y, data_name)

X_result, Y = X_MAIN.X_main(data_name, Y, M, k)

    
" Calculate the MSE "
mse_rows = []         # Original MSE for each rows of the original recovered source matrix X and X_ica
mse_segment = np.zeros(len(Y))  # Original average MSE for each rows of the original recovered source matrix X and X_ica

for seg in range(n_seg):   
    " Making the mse for all sources in all segments "
    mse_rows.append(np.zeros(int(k[seg])))

for seg in range(n_seg): 
    # Looking at one time segment
    mse_rows[seg], mse_segment[seg] = simulated_data.MSE_segments(X_result[seg], np.array(X_ica_nonzero[seg]))

## remove mse values over 1000, as they are considered outlies
_temp = []
for p in range(n_seg):
    if mse_segment[p]>1000:
        _temp.append(p)
mse_segment = np.delete(mse_segment,_temp)

mse_average = np.average(mse_segment)

# choose segment to plot
segment = 5

plt.figure(2)
plt.plot(mse_segment, '-ro', label = 'MSE')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for all time segments')
plt.xlabel('Time Segment')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/average_mse_none_removed_ica.png')

plt.figure(3)
plt.plot(mse_segment, '-ro', label = 'MSE')
plt.hlines(5, 0, 144, label='Tolerance = 5') # horizontal line
#plt.plot(average_mse2, '-bo', label = 'Average MSE + amp')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for all time segments')
plt.xlabel('Time Segment')
plt.ylabel('MSE')
plt.legend()
plt.axis([-1,145, -10,50])
plt.savefig('figures/average_mse_none_removed_ica_zoom.png')


plt.figure(4)
plt.plot(mse_rows[segment], '-ro', label = 'MSE per row')
plt.title(r'MSE($\^\mathbf{X}_{main}$,$\^\mathbf{X}_{ICA}$) for time segment = 5')
plt.xlabel('Sources')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figures/mse_none_removed_ica_timeseg5.png')

# =============================================================================
# Calculating Average of the Average MSE 
# =============================================================================
#Find gennemsnittet og find dem der ligger over og under tol = 5
print('{} values was removed as outliers'.format(n_seg-(len(mse_segment))))
print('The average mse over all segments: ', mse_average)

tol = 5
under = 0
on = 0
over = 0
for seg in range(len(mse_segment)):
    if mse_segment[seg] < tol:
        under += 1
    if mse_segment[seg] > tol:
        over += 1
    if mse_segment[seg] == tol:
        on += 1

print('over tol: ', over)
print('under tol: ', under)
print('percentage under: ', (under/n_seg*100))
print('n_seg = {} and Ls = {}'.format(n_seg, L))

#save source estimate to .mat
import scipy.io as sio
sio.savemat('main_sources_M'+ str(M) +'_'+ str(data_name) + '.mat', {'sources':X_result})




# ############################################################################
" Plots of segment "

figsave1 = "figures/EEG_none_removed_timeseg5" + str(data_name) + '_' + ".png"
figsave2 = "figures/EEG_second_removed_scaled_timeseg5" + str(data_name) + '_' + ".png"
index = [5, 10, 15, int(k[segment])-1]

plt.figure(5)
plt.subplot(4, 1, 1)
plt.plot(X_result[segment][index[0]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[0]], 'r', label='ICA')
plt.title('Recovered Source Matrix X for Time Segment = {}'.format(segment))
plt.xlabel('Source {}'.format(index[0]))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(4, 1, 2)
plt.plot(X_result[segment][index[1]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[1]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[1]))
plt.subplot(4, 1, 3)
plt.plot(X_result[segment][index[2]], 'g', label='Main Alg.')
plt.plot(X_ica_nonzero[segment][index[2]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[2]))
plt.subplot(4, 1, 4)
plt.plot(X_result[segment][index[3]], 'g', label='Main Alg.')
plt.plot( X_ica_nonzero[segment][index[3]], 'r', label='ICA')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[3]))
plt.tight_layout() 
plt.show()
plt.savefig(figsave1)

def scale(n_seg, k, X_ica, X_main):
        
    for seg in range(n_seg): 
    # Looking at one time segment
        for f in range(len(X_ica[seg])):
            amp = np.max(X_main[seg][f])/np.max(X_ica[seg][f])
            X_ica[seg][f] = X_ica[seg][f]*amp
    return X_ica

X_ica_scale = scale(n_seg, k, X_ica_nonzero, X_result)

plt.figure(6)
plt.subplot(4, 1, 1)
plt.plot(X_result[segment][index[0]], 'g', label='Main Alg.')
plt.plot(X_ica_scale[segment][index[0]], 'r', label='ICA scaled')
plt.title('Recovered Source Matrix X for Time Segment  = {}'.format(segment))
plt.xlabel('Source {}'.format(index[0]))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplot(4, 1, 2)
plt.plot(X_result[segment][index[1]], 'g', label='Main Alg.')
plt.plot(X_ica_scale[segment][index[1]], 'r', label='ICA scaled')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[1]))
plt.subplot(4, 1, 3)
plt.plot(X_result[segment][index[2]], 'g', label='Main Alg.')
plt.plot(X_ica_scale[segment][index[2]], 'r', label='ICA scaled')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[2]))
plt.subplot(4, 1, 4)
plt.plot(X_result[segment][index[3]], 'g', label='Main Alg.')
plt.plot( X_ica_scale[segment][index[3]], 'r', label='ICA scaled')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Source {}'.format(index[3]))
plt.tight_layout() 
plt.show()
plt.savefig(figsave2)
    