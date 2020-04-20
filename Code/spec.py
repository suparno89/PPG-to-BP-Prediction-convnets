#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import os
#import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
plt.close("all")

# In[12]:


##readppg data, change the path according to your own directories
with open('../intermediate_data/ppg_snippets_kaggle.json', 'r') as JSON:
    json_dict = json.load(JSON)


#############
#Constants
#############
Fs=1000; #sampling frequency

#############
#independent variables
#############
detrends = ['mean'];   #none, mean, linear

#number of frequencies in spectrogram #Nfft=pow(2,pot);
#pots = np.linspace(8,14,dtype="int"); #potence of 2.
pots = [10, 12]

#meas= np.arange(1,np.size((json_dict[:])),10,dtype="int")#
meas= [115]  #set which measurements to analyse


for detrend in detrends:
    for pot in pots:
        Nfft = int
        Nfft=pow(int(2),pot);
        #meas_i = 115;   #index of investigated measurement
        for meas_i in meas:
            print(Nfft, meas_i)
            #############
            ##spectograms
            #############
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,9));
            
            Pxx, freqs, bins, im = ax1.specgram(json_dict[meas_i]['ppg'], Fs=Fs, NFFT=Nfft,noverlap=Nfft/2,detrend=detrend,window=np.hanning(Nfft))#,pad_to=Nfft, 
            # The `specgram` method returns 4 objects. They are:
            # - Pxx: the periodogram
            # - freqs: the frequency vector
            # - bins: the centers of the time bins
            # - im: the matplotlib.image.AxesImage instance representing the data in the plot
            
            ax2.loglog(freqs,np.transpose([np.min(Pxx,1),np.mean(Pxx,1),np.max(Pxx,1),]))
            
            print("# = ", meas_i,"Nfft = ", Nfft, ", Nt = ", np.size(Pxx[1,:]) )
            #set 
            fig.suptitle('meas #'+str(meas_i)+', Nfft = '+str(Nfft)+', Nt = '+ str(np.size(Pxx[1,:]))+'__detrend_'+ detrend)
            ax1.set_title("Spectrogram")
            ax1.set_yscale("log");
            ax1.set_xlabel("t in s")
            ax1.set_ylabel("f in Hz");
            ax1.set_ylim(freqs[1], freqs[-1])
            ax2.set_title("double log magnitude plot")
            ax2.set_xlabel("frequency [Hz]")
            ax2.set_ylabel("Magnitude [unit?]");
            ax2.legend(["min","mean of "+str(np.size(Pxx[1,:])),"max"])
            plt.show()
            
            #plt.savefig('fig_out/spectrogram__meas_'+str(meas_i)+'__Nfft_'+str(Nfft)+'__detrend_'+ detrend)
            plt.close()
# In[14]:


##scalograms
##takes some time execute based on the widths
#fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,10))
#ax1.plot(json_dict[115]['ppg'])
#widths = np.arange(950, 1050)
#cwtmatr, freqs = pywt.cwt(json_dict[115]['ppg'], widths, 'morl')
#plt.imshow(cwtmatr, cmap='seismic', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


# In[ ]:

exit()

