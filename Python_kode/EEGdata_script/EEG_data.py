# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:55:56 2020

@author: trine
"""
import numpy as np
import scipy.io

# suggested input to data_import function
data_file = 'data/S1_CClean.mat'            # file path
segment_time = 1                            # length of segment in seconds

def _import(data_file, segment_time, request='none', fs=512):
    """
    Import datafile and perform segmentation

    Input:      data_file       -> string with file path
                segment_time    -> float, lengt of segment i seconds

    Output:     Y       -> array (m x L)
                Ys      -> list of arrays, segments in axis 0 array (m, Ls)
                m       -> int, number of sensors
                Ls      -> int, length of one segment
    """
    # import data
    mat = scipy.io.loadmat(data_file)
    Y = np.array([mat['EEG']['data'][0][0][i] for i in
                  range(len(mat['EEG']['data'][0][0]))])
    if request != 'none':
        Y = _reduction(Y, request)
    m, L = Y.shape
    # no segmentation
    if segment_time == 0:
        n_seg = 1
        Y = np.reshape(Y,(1,Y.shape[0],Y.shape[1]))
        return Y, m, L, n_seg
 
    # segmentation
    Ls = int(fs * segment_time)             # number of samples in one segment
    if Ls > L:
        raise SystemExit('segment_time is to high')
    n_seg = int(L/Ls)                       # total number of segments
    Y = Y[:n_seg*Ls]                        # remove last segement if too small
    Ys = np.array_split(Y, n_seg, axis=1)   # Matrixs with segments in axis=0

    m, Ls = Ys[0].shape

    return Ys, m, Ls, n_seg

def _reduction(Y, request='remove 1/2'):
    """
    Remove a number of sensors (reducing m) corresponding to pre defined
    request. Function used inside data_import()
    input:      string  'remove 1/2' -> remove every second sensor
                        'remove 1/3' -> remove every third sensor
                        'remove 2'   -> remove sensor of index 4 and 8

    output:     array (m_new x L), reduced data set Y
    """
    if request == 'remove 1/2':
        Y_new = np.delete(Y, np.arange(0, Y.shape[0], 2), axis=0)
        return Y_new

    if request == 'remove 1/3':
        Y_new = np.delete(Y, np.arange(0, Y.shape[0], 3), axis=0)
        return Y_new

    if request == 'remove 2':
        Y_new = np.delete(Y, [4, 8], axis=0)
        return Y_new

    else:
        raise SystemExit('data removeal request is not possible')