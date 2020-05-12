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

X_C_time = np.linspace(0,1,len(X_C_signal)) # time signal (0ne second) for source signal
X_O_time = np.linspace(0,1,len(X_O_signal)) # time signal (0ne second) for measurment signal


def DFT(signal):

Y_C_time = np.linspace(0,1,len(Y_C_signal)) # time signal (0ne second) for source signal
Y_O_time = np.linspace(0,1,len(Y_O_signal)) # time signal (0ne second) for measurment signal

X_C_stepsize = X_C_time[1]-X_C_time[0]
X_O_stepsize = X_O_time[1]-X_O_time[0]
Y_C_stepsize = Y_C_time[1]-Y_C_time[0]
Y_O_stepsize = Y_O_time[1]-Y_O_time[0]


X_C_fft = np.fft.rfft(X_C_signal)   # FFT of source signal
X_O_fft = np.fft.rfft(X_O_signal)   # FFT of source signal

X_C_power = np.abs(X_C_fft)         # |FFT|
X_O_power = np.abs(X_O_fft)         # |FFT|
#X_sample_f = np.fft.fftfreq(int(X_signal.size/2)+1, d=X_stepsize) # frequencies for source FFT

Y_C_fft = np.fft.rfft(Y_C_signal)   # FFT of source signal
Y_O_fft = np.fft.rfft(Y_O_signal)   # FFT of source signal

Y_C_power = np.abs(Y_C_fft)         # |FFT|
Y_O_power = np.abs(Y_O_fft)         # |FFT|
#Y_sample_f = np.fft.fftfreq(int(Y_signal.size/2)+1, d=Y_stepsize) # frequencies for measurement FFT

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

" X_signal Filtering "
X_C_filter = butter_bandpass_filter(X_C_signal, lowcut, highcut, fs, order)
X_C_fft_filter = np.fft.rfft(X_C_filter)
X_C_power_filter = np.abs(X_C_fft_filter)
X_O_filter = butter_bandpass_filter(X_O_signal, lowcut, highcut, fs, order)
X_O_fft_filter = np.fft.rfft(X_O_filter)
X_O_power_filter = np.abs(X_O_fft_filter)

#X_sample_f2 = np.fft.fftfreq(int(X_filter.size/2)+1, d=X_stepsize)

" Y_signal Filtering "
Y_C_filter = butter_bandpass_filter(Y_C_signal, lowcut, highcut, fs, order)
Y_C_fft_filter = np.fft.rfft(Y_C_filter)
Y_C_power_filter = np.abs(Y_C_fft_filter)
Y_O_filter = butter_bandpass_filter(Y_O_signal, lowcut, highcut, fs, order)
Y_O_fft_filter = np.fft.rfft(Y_O_filter)
Y_O_power_filter = np.abs(Y_O_fft_filter)
#Y_sample_f2 = np.fft.fftfreq(int(Y_filter.size/2)+1, d=Y_stepsize)


" Source Matrix X Plots "
plt.figure(1)
plt.subplot(511)
plt.plot(X_C_time, X_C_signal, label='Time Signal X - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(512)
plt.stem(X_C_power, label = 'FFT Signal X - Closed Eyes' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,200])
plt.legend()

plt.subplot(513)
plt.plot(hz[:150], abs(h[:150]), label="Frequency Response of Order = 5")
plt.axvline(x=8)
plt.axvline(x=13)
#plt.title('Butterworth Bandpass')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend()

plt.subplot(514)
plt.plot(X_C_time, X_C_filter, label='Time Filtered Signal X - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(515)
plt.stem(X_C_power_filter,label='FFT Filtered Signal X - Closed Eyes' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,200])
plt.legend()

" Measurement Matrix Y and Source Matrix Plots "
plt.figure(2)
plt.subplot(411)
plt.plot(Y_C_time, Y_C_filter, 'b', label='Time Filtered Signal Y - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(412)
plt.plot(Y_O_time, Y_O_filter, 'b', label='Time Filtered Signal Y - Open Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(413)
plt.plot(X_C_time, X_C_filter, 'r', label='Time Filtered Signal X - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(414)
plt.plot(X_O_time, X_O_filter, 'r', label='Time Filtered Signal X - Open Eyes')
plt.xlabel('Time')
plt.legend()

print('Average difference between Y Closed and Y Open: ', abs(np.average(Y_C_filter[:-1]/Y_O_filter)))
print('Average difference between X Closed and X Open: ', abs(np.average(X_C_filter[:-1]/X_O_filter)))

# =============================================================================
# DFT Matrix
# =============================================================================
" X_matrix Filtering "
X_C_matrix = X_C[seg]
X_C_fft_matrix = np.fft.rfft2(X_C_matrix)   # FFT of source matrix
X_C_power_matrix = np.abs(X_C_fft_matrix)   # |FFT|

X_O_matrix = X_O[seg]
X_O_fft_matrix = np.fft.rfft2(X_O_matrix)   # FFT of source matrix
X_O_power_matrix = np.abs(X_O_fft_matrix)   # |FFT|

X_C_filter_matrix = []
X_O_filter_matrix = []
for i in range(len(X_C_matrix)):
    X_C_filter_matrix.append(butter_bandpass_filter(X_C_matrix[i], lowcut, highcut, fs, order))

for i in range(len(X_O_matrix)):
    X_O_filter_matrix.append(butter_bandpass_filter(X_O_matrix[i], lowcut, highcut, fs, order))

X_C_fft_filter_matrix = np.fft.rfft(X_C_filter_matrix)
X_C_power_filter_matrix = np.abs(X_C_fft_filter_matrix)
X_O_fft_filter_matrix = np.fft.rfft(X_O_filter_matrix)
X_O_power_filter_matrix = np.abs(X_O_fft_filter_matrix)

" Y_matrix Filtering "
Y_C_matrix = Y_C[seg]
Y_C_fft_matrix = np.fft.rfft2(Y_C_matrix)   # FFT of source matrix
Y_C_power_matrix = np.abs(Y_C_fft_matrix)   # |FFT|

Y_O_matrix = Y_O[seg]
Y_O_fft_matrix = np.fft.rfft2(Y_O_matrix)   # FFT of source matrix
Y_O_power_matrix = np.abs(Y_O_fft_matrix)   # |FFT|

Y_C_filter_matrix =[]
Y_O_filter_matrix = []
for i in range(len(Y_C_matrix)):
    Y_C_filter_matrix.append(butter_bandpass_filter(Y_C_matrix[i], lowcut, highcut, fs, order))

for i in range(len(Y_O_matrix)):
    Y_O_filter_matrix.append(butter_bandpass_filter(Y_O_matrix[i], lowcut, highcut, fs, order))

Y_C_fft_filter_matrix = np.fft.rfft(Y_C_filter_matrix)
Y_C_power_filter_matrix = np.abs(Y_C_fft_filter_matrix)
Y_O_fft_filter_matrix = np.fft.rfft(Y_O_filter_matrix)
Y_O_power_filter_matrix = np.abs(Y_O_fft_filter_matrix)

plt.figure(3)
plt.subplot(411)
plt.plot(Y_C_time, sum(Y_C_filter_matrix), label='Time Filtered Matrix Y - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(412)
plt.plot(Y_O_time, sum(Y_O_filter_matrix), label='Time Filtered Matrix Y - Open Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(413)
plt.plot(X_C_time, sum(X_C_filter_matrix), label='Time Filtered Matrix X - Closed Eyes')
plt.xlabel('Time')
plt.legend()

plt.subplot(414)
plt.plot(X_O_time, sum(X_O_filter_matrix), label='Time Filtered Matrix X - Open Eyes')
plt.xlabel('Time')
plt.legend()

for i in range(len(Y_C_filter_matrix)):
    
    

print('Average difference between Y Closed and Y Open: ', abs((sum(Y_C_filter_matrix))/sum(Y_O_filter_matrix)))
print('Average difference between X Closed and X Open: ', abs((sum(X_C_filter_matrix))/sum(X_O_filter_matrix)))

