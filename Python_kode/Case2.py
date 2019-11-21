# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:33:33 2019

@author: Laura

Arbejdsblad - Case 2 - mixed Signals
m = 3
n = 8
L = n_samples = 20
k = non-zero = 4 
Y = signal - m x L
A = Kendt mixing matrix - m x n
X = Ukendt signal - n x L 

Full recovery because k < m

Uden segmenentering og 1000 iterationer
"""
import numpy as np
import matplotlib.pyplot as plt
from Cov_DL import data_generation
from Cov_DL import MSBL

np.random.seed(1)

# =============================================================================
# Import data
# =============================================================================
m = 3               # number of sensors
n = 8                # number of sources
non_zero = 4         # max number of non-zero coef. in rows of X
n_samples = 20       # number of sampels
duration = 8
iterations = 1000

"Mixed Signals Generation - Sinus, sign, saw tooth and zeros"
Y_mix, A_mix, X_mix = data_generation.mix_signals(n_samples, duration, m)

# =============================================================================
# M-SBL
# =============================================================================
X_Rec_mix = MSBL.M_SBL(A_mix, Y_mix, m, n, n_samples, non_zero, iterations)

# =============================================================================
# Plot
# =============================================================================
plt.figure(1)
plt.title('Plot of row 5 of X - Mixed Signal Data')
plt.plot(X_mix[5], 'r',label='Real X')
plt.plot(X_Rec_mix[5],'g', label='Computed X')
plt.legend()
plt.show

plt.figure(2)
plt.title('Plot of column 5 of X - Mixed Signal Data')
plt.plot(X_mix.T[5], 'r',label='Real X')
plt.plot(X_Rec_mix.T[5],'g', label='Computed X')
plt.legend()
plt.show
