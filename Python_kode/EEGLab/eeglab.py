# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:01:30 2019

@author: Mattek9b

Information about MNE package: 
    https://mne.tools/0.16/documentation.html
    https://cbrnr.github.io/2017/10/23/loading-eeg-data/
"""
# =============================================================================
# Import Important Packages
# =============================================================================
import mne
import os
from matplotlib import pyplot as plt

# =============================================================================
# Import EEG Data from EEGLab
# =============================================================================
data_path = os.getcwd()                    # Get the current working directory
fname = data_path + "/eeglab_data.set"
montage = data_path + "/eeglab_chan32.locs"

# =============================================================================
# Creating EEG Data
# =============================================================================
event_id = {"rt": 1, "square": 2}
raw = mne.io.read_raw_eeglab(fname)         # Read the EEGLab data from MatLab

" All the sensors/channels name "
mapping = {
    'EEG 000': 'Fpz', 'EEG 001': 'EOG1', 'EEG 002': 'F3', 'EEG 003': 'Fz',
    'EEG 004': 'F4', 'EEG 005': 'EOG2', 'EEG 006': 'FC5', 'EEG 007': 'FC1',
    'EEG 008': 'FC2', 'EEG 009': 'FC6', 'EEG 010': 'T7', 'EEG 011': 'C3',
    'EEG 012': 'C4', 'EEG 013': 'Cz', 'EEG 014': 'T8', 'EEG 015': 'CP5',
    'EEG 016': 'CP1', 'EEG 017': 'CP2', 'EEG 018': 'CP6', 'EEG 019': 'P7',
    'EEG 020': 'P3', 'EEG 021': 'Pz', 'EEG 022': 'P4', 'EEG 023': 'P8',
    'EEG 024': 'PO7', 'EEG 025': 'PO3', 'EEG 026': 'POz', 'EEG 027': 'PO4',
    'EEG 028': 'PO8', 'EEG 029': 'O1', 'EEG 030': 'Oz', 'EEG 031': 'O2'
}
raw.rename_channels(mapping)
raw.set_channel_types({"EOG1": 'eog', "EOG2": 'eog'})
raw.set_montage('standard_1020')               # Standard placement of channels

events = mne.events_from_annotations(raw, event_id)[0]

"""
EEGLab Data set with the following specifications
    - 32 channels pr frame
    - 30504 frames pr epoch
    - 1 epoch
    - 154 events (square and rt)
    - 128 Hz sample rate
    - 0 - 238,305 sec epoch
"""
# =============================================================================
# Dividing the EEG Data into Epochs
# =============================================================================
epochs = mne.Epochs(raw, events=events, tmin = -1, tmax= 2,
                    event_id={"square": 2}, preload=True)

"""
The new specifications of EEGLab
    - 32 channels pr frame
    - 384 frames pr epoch
    - 80 epoch
    - 154 events (square and rt)
    - 128 Hz sample rate
    - -1 - 2 (3) sec epoch
"""
# =============================================================================
# Plot of the epochs - Our observed data Y
# =============================================================================
"""
Epochs objects are a way of representing continuous data as a collection of
time-locked trials, stored in an array of shape (n_events, n_channels, 
n_times). They are useful for many statistical methods in neuroscience, 
and make it easy to quickly overview what occurs during a trial.
"""
# Plot all channel in one epoch
# epochs.plot(n_epochs=1)

" Print the first epoch with 32 channel and 385 frames/samples "
epc = []
for ep in epochs[0]: # Highest value is 79 corresponding to epoch number 80
    epc = ep

" Plots "
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(epc[0])
plt.title("First Channel of Epoch 1")

plt.subplot(2, 1, 2)
for p in epc:
    plt.plot(p)
plt.title("All Channels of Epoch 1")

# Epochs own plotting function
epochs.plot(n_epochs=1, n_channels=1)
plt.title("Epoch Plot - epoch 1 and channel 1")

