# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:43:29 2019

@author: Laura
"""
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Illustration of ICA
# =============================================================================
""" Two independent components with uniform distribution """
lin = np.linspace(-1-np.sqrt(3), 1+np.sqrt(3),2)  #1000 samples
s = np.random.randn(len(lin),1)                        #1000 samples


ps1 = np.random.uniform(s)
ps2 = np.random.uniform(s)


" Joint Density "
ps_joint = np.zeros(len(lin))
for i in range(len(lin)):
    ps_joint[i] = ps1[i] * ps2[i]
    
#plt.plot(ps_joint,'o')

""" Mixing the two independent component """
A = np.matrix([[2, 3],[2,1]]) #mixing matrix

x1 = A * ps1
x2 = A * ps2

" The joint distribution of the independent components s1 and s2 with uniform distributions. Horizontal axis: s1, vertical axis: s2. "
for i in range(1000):
    lin = np.linspace(-1-np.sqrt(3), 1+np.sqrt(3),2)  #1000 samples
    s = np.random.randn(len(lin),1)  #1000 samples
    for j in range(len(lin)):
        if s[j] <= np.sqrt(3):                      
            ps1 = np.random.uniform(s)
            ps2 = np.random.uniform(s)
        else:
            ps1 = 0
            ps2 = 0
    #plt.plot(ps1,ps2,'o')
    
    A = np.matrix([[2, 3],[2,1]])
    x1 = A * ps1
    x2 = A * ps2
    plt.plot(x1,x2,'o')
