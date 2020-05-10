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
data_name = 'S1_CClean.mat'                # Closed eyes - Subject 1
#data_name = 'S1_OClean.mat'               # Open eyes - Subject 1
data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" Import Segmented Dataset "
Y_ica, M_ica, L_ica, n_seg_ica = data._import(data_file, segment_time, request='none')

X_ica, k = X_ICA.X_ica(data_name, Y_ica, M_ica)

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
#request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
request = 'none'

Y, M, L, n_seg = data._import(data_file, segment_time, request=request)

X_result, Y = X_MAIN.X_main(data_name, Y, M, k)

# =============================================================================
# DFT
# =============================================================================
seg = 15
row = 10

Y_signal = Y[seg][row]                  # one measurement signal
X_signal = X_result[seg][row]           # one recovered source signal

X_time = np.linspace(0,1,len(X_signal)) # time signal (0ne second) for source signal
Y_time = np.linspace(0,1,len(Y_signal)) # time signal (0ne second) for measurment signal

X_stepsize = X_time[1]-X_time[0]
Y_stepsize = Y_time[1]-Y_time[0]

X_fft = np.fft.rfft(X_signal)   # FFT of source signal
X_power = np.abs(X_fft)         # |FFT|
X_sample_f = np.fft.fftfreq(int(X_signal.size/2)+1, d=X_stepsize) # frequencies for source FFT


Y_fft = np.fft.rfft(Y_signal)  # FFT of measurement signal
Y_power = np.abs(Y_fft)        # |FFT|
Y_sample_f = np.fft.fftfreq(int(Y_signal.size/2)+1, d=Y_stepsize) # frequencies for measurement FFT

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
    b, a = butter(order, [low, high], btype='band') # coefficients of transfer function
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
X_filter = butter_bandpass_filter(X_signal, lowcut, highcut, fs, order)
X_fft_filter = np.fft.rfft(X_filter)
X_power_filter = np.abs(X_fft_filter)
X_sample_f2 = np.fft.fftfreq(int(X_filter.size/2)+1, d=X_stepsize)

" Y_signal Filtering "
Y_filter = butter_bandpass_filter(Y_signal, lowcut, highcut, fs, order)
Y_fft_filter = np.fft.rfft(Y_filter)
Y_power_filter = np.abs(Y_fft_filter)
Y_sample_f2 = np.fft.fftfreq(int(Y_filter.size/2)+1, d=Y_stepsize)

# =============================================================================
# Plots
# =============================================================================
" Source Matrix X Plots "
plt.figure(1)
plt.subplot(511)
plt.plot(X_time, X_signal, label='Original signal X')
plt.xlabel('Time')
plt.legend()

plt.subplot(512)
plt.stem(X_sample_f, X_power,label='fft signal X' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,200])
plt.legend()

plt.subplot(513)
plt.plot(hz[:150], abs(h[:150]), label="Frequency response of order = 5")
plt.axvline(x=8)
plt.axvline(x=13)
#plt.title('Butterworth Bandpass')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend()

plt.subplot(514)
plt.plot(X_time, X_filter, label='Original filtret signal X')
plt.xlabel('Time')
plt.legend()

plt.subplot(515)
plt.stem(X_sample_f2, X_power_filter,label='fft filtret signal X' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,200])
plt.legend()


" Measurement Matrix Y Plots "
plt.figure(2)
plt.subplot(511)
plt.plot(Y_time, Y_signal, label='Original signal Y')
plt.xlabel('Time')
plt.legend()

plt.subplot(512)
plt.stem(Y_sample_f, Y_power,label='fft signal Y' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,500])
plt.legend()

plt.subplot(513)
plt.plot(hz[:150], abs(h[:150]), label="Frequency response of order = 5")
plt.axvline(x=8)
plt.axvline(x=13)
#plt.title('Butterworth Bandpass')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend()

plt.subplot(514)
plt.plot(Y_time, Y_filter, label='Original filtret signal Y')
plt.xlabel('Time')
plt.legend()

plt.subplot(515)
plt.stem(Y_sample_f2, Y_power_filter,label='fft filtret signal Y' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,70,0,300])
plt.legend()
