# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:30:49 2020

@author: Laura
"""
import X_ICA
import X_MAIN
import data
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz


np.random.seed(1234)

# =============================================================================
# Import EEG data file
# =============================================================================
data_name_C = 'S1_CClean.mat'                # Closed eyes - Subject 1
data_name_O = 'S1_OClean.mat'                # Open eyes - Subject 1
data_file_C = 'data/' + data_name_C          # file path
data_file_O = 'data/' + data_name_O          # file path

segment_time = 1                             # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" ICA - Closed Eyes Dataset "
Y_ica_C, M_ica_C, L_ica_C, n_seg_ica_C = data._import(data_file_C, segment_time, request='none')
X_ica_C, k_C = X_ICA.X_ica(data_name_C, Y_ica_C, M_ica_C)

" ICA - Open Eyes Dataset "
Y_ica_O, M_ica_O, L_ica_O, n_seg_ica_O = data._import(data_file_O, segment_time, request='none')
X_ica_O, k_O = X_ICA.X_ica(data_name_O, Y_ica_O, M_ica_O)

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
#request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
request = 'none'

" Main - Closed Eyes Dataset "
Y_C, M_C, L_C, n_seg_C = data._import(data_file_C, segment_time, request=request)
X_C, Y_C = X_MAIN.X_main(data_name_C, Y_C, M_C, k_C)

" Main - Open Eyes Dataset "
Y_O, M_O, L_O, n_seg_O = data._import(data_file_O, segment_time, request=request)
X_O, Y_O = X_MAIN.X_main(data_name_O, Y_O, M_O, k_O)
# =============================================================================
# DFT
# =============================================================================
seg = 15
row = 10

Y_C_signal = Y_C[seg][row]                  # one measurement signal
Y_O_signal = Y_O[seg][row]                  # one measurement signal
X_C_signal = X_C[seg][row]                  # one recovered source signal
X_O_signal = X_O[seg][row]                  # one recovered source signal

X_C_matrix = X_C[seg]
X_O_matrix = X_O[seg]
Y_C_matrix = Y_C[seg]
Y_O_matrix = Y_O[seg]

X_C_time = np.linspace(0,1,len(X_C_signal)) # time signal (0ne second) for source signal
X_O_time = np.linspace(0,1,len(X_O_signal)) # time signal (0ne second) for measurment signal
Y_C_time = np.linspace(0,1,len(Y_C_signal)) # time signal (0ne second) for source signal
Y_O_time = np.linspace(0,1,len(Y_O_signal)) # time signal (0ne second) for measurment signal

def DFT(signal):
    fft = np.fft.rfft(signal)    # FFT of signal
    fft_power = np.abs(fft)      # |FFT|
    return fft, fft_power
     
X_C_fft, X_C_power = DFT(X_C_signal)   # FFT of source signal
X_O_fft, X_O_power = DFT(X_O_signal)   # FFT of source signal

Y_C_fft, Y_C_power = DFT(Y_C_signal)   # FFT of measurement signal
Y_O_fft, Y_O_power = DFT(Y_O_signal)   # FFT of measurement signal

def DFT_matrix(matrix):
    fft = np.fft.rfft2(matrix)    # FFT of matrix
    fft_power = np.abs(fft)       # |FFT|
    return fft, fft_power

X_C_fft_matrix, X_C_power_matrix = DFT_matrix(X_C_matrix)   # FFT of source matrix
X_O_fft_matrix, X_O_power_matrix = DFT_matrix(X_O_matrix)   # FFT of source matrix

Y_C_fft_matrix, Y_C_power_matrix = DFT_matrix(Y_C_matrix)   # FFT of source matrix
Y_O_fft_matrix, Y_O_power_matrix = DFT_matrix(Y_O_matrix)   # FFT of source matrix

# =============================================================================
# Butterworth Bandpass filter
# =============================================================================
lowcut = 8     # low cut off frequency (Hz)
highcut = 13   # high cut off frequency (Hz)
fs = 512       # sample frequency
order = 5      # ordre of Butterworth filter

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs      # nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass') # coefficients of transfer function
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)   # filter the signal with FIR filter
    return y

" Frequency Response "
b, a = butter_bandpass(lowcut, highcut, fs, order) # coefficients of transfer function
w, h = freqz(b, a, worN=2000)                      # frequency and frequency response
hz = (fs * 0.5 / np.pi) * w                        # frequncies

# =============================================================================
# Filtering
# =============================================================================
" X signal and Y signal Filtering "
def filtering(signal, lowcut, highcut, fs, order=5):
    filt = butter_bandpass_filter(signal, lowcut, highcut, fs, order)
    fft_filt = np.fft.rfft(filt)
    fft_power_filt = np.abs(fft_filt)
    return filt, fft_filt, fft_power_filt

X_C_filter, X_C_fft_filter, X_C_power_filter = filtering(X_C_signal, lowcut, highcut, fs, order=5)
X_O_filter, X_O_fft_filter, X_O_power_filter = filtering(X_O_signal, lowcut, highcut, fs, order=5)

Y_C_filter, Y_C_fft_filter, Y_C_power_filter = filtering(Y_C_signal, lowcut, highcut, fs, order=5)
Y_O_filter, Y_O_fft_filter, Y_O_power_filter = filtering(Y_O_signal, lowcut, highcut, fs, order=5)

" X matrix and Y matrix Filtering "
def filtering_matrix(matrix, lowcut, highcut, fs, order=5):
    filter_matrix = []
    for i in range(len(matrix)):
        filter_matrix.append(butter_bandpass_filter(matrix[i], lowcut, highcut, fs, order))
    fft_filter_matrix = np.fft.rfft2(filter_matrix)
    fft_power_filter_matrix = np.abs(fft_filter_matrix)
    return filter_matrix, fft_filter_matrix, fft_power_filter_matrix

X_C_filter_matrix, X_C_fft_filter_matrix, X_C_power_filter_matrix = filtering_matrix(X_C_matrix, lowcut, highcut, fs, order=5)
X_O_filter_matrix, X_O_fft_filter_matrix, X_O_power_filter_matrix = filtering_matrix(X_O_matrix, lowcut, highcut, fs, order=5)

Y_C_filter_matrix, Y_C_fft_filter_matrix, Y_C_power_filter_matrix = filtering_matrix(Y_C_matrix, lowcut, highcut, fs, order=5)
Y_O_filter_matrix, Y_O_fft_filter_matrix, Y_O_power_filter_matrix = filtering_matrix(Y_O_matrix, lowcut, highcut, fs, order=5)

# =============================================================================
# Average Differences
# =============================================================================
print('Average difference between Y Closed and Y Open: ', abs(np.average(Y_C_filter[:-1]/Y_O_filter)))
print('Average difference between X Closed and X Open: ', abs(np.average(X_C_filter[:-1]/X_O_filter)))

#for i in range(len(Y_C_filter_matrix)):    
#
#print('Average difference between Y Closed and Y Open: ', abs((sum(Y_C_filter_matrix))/sum(Y_O_filter_matrix)))
#print('Average difference between X Closed and X Open: ', abs((sum(X_C_filter_matrix))/sum(X_O_filter_matrix)))
#


# =============================================================================
# Plots
# =============================================================================
" Source signal X Plots "
plt.figure(1)
plt.subplot(511)
plt.plot(X_C_time, X_C_signal, label='Source 10')
plt.xlabel('Time')
plt.title('Source Signal from S1_CClean')
plt.legend()

plt.subplot(512)
plt.stem(X_C_power, label = 'Source 10' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title('FFT of Source Signal from S1_CClean')
plt.axis([-1,70,0,150])
plt.legend()

plt.subplot(513)
plt.plot(hz[:150], abs(h[:150]), label="Frequency Response of Order = 5")
plt.axvline(x=8)
plt.axvline(x=13)
plt.title('Butterworth Bandpass')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.legend()

plt.subplot(514)
plt.plot(X_C_time, X_C_filter, label='Source = 10')
plt.title('Filtered Source Signal from S1_CClean')
plt.xlabel('Time')
plt.legend()

plt.subplot(515)
plt.stem(X_C_power_filter,label='Source = 10' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title('FFT Filtered Source Signal from S1_CClean')
plt.axis([-1,70,0,150])
plt.legend()
plt.show()
plt.savefig('figures/DFT_plot_X_timeseg15_source10.png')


" Measurement signal Y and Source signal X Plots "
plt.figure(2)
plt.subplot(411)
plt.plot(Y_C_time, Y_C_filter, 'b', label='Measurement 10')
plt.xlabel('Time')
plt.title('Filtered Measurement Signal from S1_CClean')
plt.legend()

plt.subplot(412)
plt.plot(Y_O_time, Y_O_filter, 'b', label='Measurement 10')
plt.xlabel('Time')
plt.title('Filtered Measurement Signal from S1_OClean')
plt.legend()

plt.subplot(413)
plt.plot(X_C_time, X_C_filter, 'r', label='Source 10')
plt.xlabel('Time')
plt.title('Filtered Source Signal from S1_CClean')
plt.legend()

plt.subplot(414)
plt.plot(X_O_time, X_O_filter, 'r', label='Source 10')
plt.xlabel('Time')
plt.title('Filtered Source Signal from S1_OClean')
plt.legend()
plt.show()
plt.savefig('figures/DFT_plot_X_and_Y_signal_timeseg15_source10.png')


" Measurement Matrix Y and Source Matrix X Plots "
plt.figure(3)
plt.subplot(411)
plt.plot(Y_C_time, sum(Y_C_filter_matrix), label='Time Segement 15')
plt.xlabel('Time')
plt.title('Filtered Measurement Matrix from S1_CClean')
plt.legend()

plt.subplot(412)
plt.plot(Y_O_time, sum(Y_O_filter_matrix), label='Time Segment 15')
plt.xlabel('Time')
plt.title('Filtered Measurement Matrix from S1_OClean')
plt.legend()

plt.subplot(413)
plt.plot(X_C_time, sum(X_C_filter_matrix), label='Time Segment 15')
plt.xlabel('Time')
plt.title('Filtered Source Matrix from S1_CClean')
plt.legend()

plt.subplot(414)
plt.plot(X_O_time, sum(X_O_filter_matrix), label='Time Segment 15')
plt.xlabel('Time')
plt.title('Filtered Source Matrix from S1_OClean')
plt.legend()
plt.show()
plt.savefig('figures/DFT_plot_X_and_Y_matrix_timeseg15.png')
