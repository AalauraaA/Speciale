# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:28:33 2019

@author: trine
"""

import matplotlib.pyplot as plt
import numpy as np


# antal mulige sources som M stiger 
M = range(10)
M2 = [(M[i]*(M[i]+1))/2 for i in M]
plt.plot(M)
plt.plot(M2)

