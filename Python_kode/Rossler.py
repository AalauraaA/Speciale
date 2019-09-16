# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:36:20 2019

@author: Laura

Topic: Generating Synthetic EEG Data
"""
import numpy as np
# =============================================================================
# Generated Rossler
# =============================================================================
def Generate_Rossler(N,conf):
    """
    It is recommended to run this code several times and average the 
    results to show that results are not dependent on additional random 
    noise. 
    conf is number of configuration (see rosslerpaper function).
    N is number of samples used.
    """
    tspan = np.linspace(0, 50, N) #50 second and sample rate is 60 Hz
    Initial = np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0]) #initial condition for solving Rossler oscillator equation
    c = np.randn('norm',3,1,6,1) #additional noise which is selected from random normal distibution
    a=find(c<0)
    if ~isempty(a)
        c=random('norm',3,1,6,1)

    [t,y] = ode45(@(t,y) rosslerpaper(t,y,conf,c),tspan,Initial) # solving rossler model with s1 configuration (for more detail see rosslerpaper function)
    x=y(:,[1,4,7,10,13,16])  # extracting X segments
    x=x(61:end,:)


# =============================================================================
# Rossler Paper
# =============================================================================
def rosslerpaper(t,y,conf,wc):
    """
    This function is used to generate different network configurations for
    the Roessler oscillator. 
    Inputs: t and y are parameters
        conf: different configuration in this paper (values 1 to 6 
              corresponding to networks 1 to 6 in figure 1 of the paper)
        wc  : additional noise 
    
    Reference: 
        Payam Shahsavari Baboukani, Ghasem Azemi, Boualem Boashash, Paul 
        Colditz, Amir Omidvarnia.
    
        A novel multivariate phase synchrony measure: Application to 
        multichannel newborn EEG analysis, Digital Signal Processing, 
        Volume 84, 2019, Pages 59-68, ISSN 1051-2004, 
        https://doi.org/10.1016/j.dsp.2018.08.019.
    """
    w = [1.05,1.05,1.05,1.05,1.05,1.05] # initial condition on W
    dy = np.zeros(18,1)
    u = 1.5   # Initial condition on U
    a = 0.35  # a
    b = 0.2   # b
    c=10      # c

    # See Al Khassaweneh's 2015 IEEE TSP (Fig 6) and Eq. 43 for more 
    # details.
    switch conf
        case 1
            e = [0 0.5 0 0 0 0
                 0.5 0 0 0 0 0
                 0 0 0 0 0 0
                 0 0 0 0 0 0
                 0 0 0 0 0 0.5
                 0 0 0 0 0.5 0]    # Configuration 1 (1-2,5-6)
        case 2
            e = [0 0 0 0 0 0.5
                 0 0 0.5 0 0 0
                 0 0.5 0 0 0 0
                 0 0 0 0 0.5 0
                0 0 0 0.5 0 0
                0.5 0 0 0 0 0]     # Configuration 2 (1-6, 2-3, 4-5)
        case 3
            e = [0 0 0 0 0.5 0.5
                 0 0 0.5 0.5 0 0
                 0 0.5 0 0.5 0 0
                 0 0.5 0.5 0 0 0
                 0.5 0 0 0 0 0.5
                 0.5 0 0 0 0.5 0]  # Configuration 3 (2-3-4, 1-5-6)
        case 4
            e = [0 0.5 0 0 0.5 0.5
                 0.5 0 0.5 0.5 0 0
                 0 0.5 0 0.5 0 0
                 0 0.5 0.5 0 0 0
                 0.5 0 0 0 0 0.5
                 0.5 0 0 0 0.5 0]  # Configuration 4 (2-3-4, 1-5-6, 1-2)
        
        case 5
            e = [0 0.5 0 0.5 0.5 0.5
                 0.5 0 0.5 0.5 0.5 0
                 0 0.5 0 0.5 0 0
                 0.5 0.5 0.5 0 0.5 0
                 0.5 0.5 0 0.5 0 0.5
                 0.5 0 0 0 0.5 0]    # Configuration 5 (2_3_4, 1-5-6, 1-2, 4-5, 2-5, 1-4)
        case 6
            e = 0.5*ones(6,6);       # Configuration 6 --> full

    # Roessler oscilators
    dy[1] = -w[1] * y[2] - y[3] + (e[2][1] * (y[4] - y[1]) + e[3][1] * (y[7] - y[1]) + e[4][1] * (y[10] - y[1]) + e[5][1] * (y[13] - y[1]) + e[6][1] * (y[16] - y[1])) + u * wc[1]
    dy[2] = -w[1] * y[1] - a * y[2]
    dy[3] = b + (y[1] - c) * y[3]

    dy[4] = -w(2)*y(5)-y(6)+(e(1,2)*(y(1)-y(4))+e(3,2)*(y(7)-y(4))+e(4,2)*(y(10)-y(4))+e(5,2)*(y(13)-y(4))+e(6,2)*(y(16)-y(4)))+u*wc(2);
    dy[5] = -w[2] * y[4] - a * y[5]
    dy[6] = b + (y[4] - c) * y[6]

    dy[7] = -w(3)*y(8)-y(9)+(e(1,3)*(y(1)-y(7))+e(2,3)*(y(4)-y(7))+e(4,3)*(y(10)-y(7))+e(5,3)*(y(13)-y(7))+e(6,3)*(y(16)-y(7)))+u*wc(3);
    dy[8] = -w[3] * y[7] - a * y[8]
    dy[9] = b + (y[7] - c) * y[9]

    dy[10] = -w(4)*y(11)-y(12)+(e(1,4)*(y(1)-y(10))+e(2,4)*(y(4)-y(10))+e(3,4)*(y(7)-y(10))+e(5,4)*(y(13)-y(10))+e(6,4)*(y(16)-y(10)))+u*wc(4);
    dy[11] = -w(4)*y(10) -a*y(11);
    dy[12] = b+(y(10)-c)*y(12);

    dy[13] = -w(5)*y(14)-y(15)+(e(1,5)*(y(1)-y(13))+e(2,5)*(y(4)-y(13))+e(3,5)*(y(7)-y(13))+e(4,5)*(y(10)-y(13))+e(6,5)*(y(16)-y(13)))+u*wc(5);
    dy[14] = -w(5)*y(13) -a*y(14);
    dy[15] = b+(y(13)-c)*y(15);

    dy[16] = -w(6)*y(17)-y(18)+(e(1,6)*(y(1)-y(16))+e(2,6)*(y(4)-y(16))+e(3,6)*(y(7)-y(16))+e(4,6)*(y(10)-y(16))+e(5,6)*(y(13)-y(16)))+u*wc(6);
    dy[17] = -w(6)*y(16) -a*y(17);
    dy[18] = b+(y(16)-c)*y(18);
