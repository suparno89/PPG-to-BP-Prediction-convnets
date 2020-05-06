#!/usr/bin/env python
# coding: utf-8

#wavelet guide
#http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
#wavelet browser:
#http://wavelets.pybytes.com/

import json
import os
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.signal as Signal
plt.close("all")


def plot_wavelet(time, signals, scales, 
                 waveletname, 
                 title = 'Wavelet Transform of signal', 
                 ylabel = 'Frequencies (Hz)', 
                 xlabel = 'Time (sec)'):
    
    for v in signals:

        fig, (ax) = plt.subplots(nrows=1,figsize=(10, 8),sharex=True)
        signal1 = v['ppg']
        print(len(signal1))
        
        signal = Signal.detrend(signal1, type='constant')
        ax = plt.axes()
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2

        #display log2 of power
        #levels = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        #contourlevels = np.log2(levels)
        #im = ax.contourf(time, np.log2(frequencies[1:]), np.log2(power[:][1:]), contourlevels, extend='both')
        
        #display power directly
        lev_exp = np.arange(-5, np.ceil(np.log10(power.max())+1))
        levs = np.power(10, lev_exp)
        ##for cardioveg

        print(time.shape)
        print(frequencies.shape)
        print(power[:,1:].shape)
        im = ax.contourf(time, np.log2(frequencies[:]), power[:,1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")
    
        #im = ax.contourf(time, np.log2(frequencies[1:]), power[:][1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")
        ax.set_title(title, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        
        yticks = 2**np.arange(-1, np.floor(np.log2(frequencies.max())))
        print(yticks)
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ylim = ax.get_ylim()
        print(ylim)
        #ax.set_ylim(ylim[0], -1)
        ax.set_ylim(ylim[0], 1)
        #ax.set_ylim(0, ylim[1])
        
        cbar_ax = fig.add_axes([0.9, 0.5, 0.03, 0.25])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")


        #ax2.plot(time, signal1[1:])
        #ax2.plot(time,np.max(power,0))

        plt.show()

# In[12]:


def plot_signal(time, signals):

    for v in signals: 
        #signal1 = Signal.detrend(v['ppg'], type='constant')
        plt.figure(figsize=(10, 8))
        ax = plt.axes()
        ax.set_title("Normalized Signal", fontsize = 20)
        ax.set_ylabel("Amplitude", fontsize=18)
        ax.set_xlabel("Time (sec)", fontsize=18)
        plt.plot(time, v['ppg'][1:])
        plt.show()


##readppg data, change the path according to your own directories
with open('../intermediate_data/ppg_snippets_kaggle.json', 'r') as JSON:
    json_dict = json.load(JSON)



#scalograms
T = 30 #sec
Fs=1000 #Hz
dt=1/Fs #sec
time = np.arange(0, T, dt)
#signal = signal.detrend(json_dict[1]['ppg'], type='constant')


wavelet =  "cmor3-60" #"cmor3-60" # "morl"# "cmor" "gaus1"

scales = np.arange(1,512,2)
#print(pywt.scale2frequency(wavelet, scales)/dt) 

plot_wavelet(time, json_dict[115:116], scales, waveletname=wavelet)
#plot_signal(time, json_dict[90:95])



#todo
#remove mean - ok
#apply ramp function

# In[ ]:
print("done")
exit()


