
import json
import os
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob
import scipy.signal as signal
import pandas as pd
from img2vec_pytorch import Img2Vec
from PIL import Image
import gc


plt.close("all")

##global variables
PATH_SCALOGRAMS = '../scalograms_cardioveg'
PATH_SPECTOGRAMS = '../spectograms_cardioveg'
FILE_JSON_DICT_IN = '../intermediate_data/all_features_30sec_bfffill_all.json'
FILE_REPRESENTATION_OUT = '../intermediate_data/cardioveg_bfffill_all_scalograms_alexnet.pkl'
##choose one of the following models('resnet-18', 'alexnet') 
MODEL_TO_LEARN = 'alexnet'
DATA = 'cardioveg'
FREQUENCY = 64
TIME = 30


def delete_contents_in_folder(path):
    ##provided a path this function asks the user if it can delete the contents, and proceeds accordingly
   
    delete_flag = input("The folder already contains files. Should the contents be deleted? (Y/N)")
    if (str(delete_flag).lower() == 'y'):
        for files in glob.glob(path+'/*'):
            os.remove(files)
        return True
    else:
        print("can't proceed")
        return False

def empty_folder(path):

    ##this function checks if a folder is empty or not. 
    if(len(os.listdir(path) ) > 0):
        print("The folder contains " + str(len(os.listdir(path))) + " files")
        return False
    else:
        print("the folder is empty")
        return True
    

              

def get_spectograms(json_dict, path): 
    
    ##given the ppg signal this function writes the spectograms out to a provided path
    # if (empty_folder(path) == False):
    #     if(delete_contents_in_folder(path) == False):
    #       return

    for val in json_dict:

        ppg = val['ppg']
        patientid = val['patientid']
        sbp = val['sbp']
        dbp=val['dbp']
            
        pot = 12
        Fs=64
        Nfft=pow(int(2),pot)
        detrend = 'mean'
        ax = plt.axes()
        Pxx, freqs, bins, im = ax.specgram(ppg, Fs=Fs, NFFT=Nfft,noverlap=Nfft/2, window=np.hanning(Nfft), detrend=detrend, cmap='seismic')
        ax.set_yscale('log')
        ax.set_ylim(freqs[1], freqs[-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(os.path.join(PATH_SPECTOGRAMS, '_{}_{}_{}_spectogram.jpg'.format(patientid, sbp, dbp)), bbox_inches='tight', pad_inches=0)
        del ax
        gc.collect()

def get_scalograms(json_dict, path):

    #given the ppg signal this function writes the scalograms out to a provided path
    if (empty_folder(path) == False):
        if(delete_contents_in_folder(path) == False):
            return

    i = 0
    for val in json_dict:
        i += 1

        ppg = val['ppg']
        patientid = val['patientid']
        sbp = val['sbp']
        dbp=val['dbp']

        ##params for wavelet function
        T = 30.0 #sec
        Fs=64 #Hz
        dt=1/Fs #sec
        time = np.arange(0, T, dt)
        signal_detrend = signal.detrend(ppg, type='constant')
        wavelet =  "morl"

        scales = np.arange(1,512 ,2)
        
        ##plot
        ax = plt.axes()
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal_detrend, scales, wavelet, dt)
        power = (abs(coefficients)) ** 2
        lev_exp = np.arange(-5, np.ceil(np.log10(power.max())+1))
        levs = np.power(10, lev_exp)

        if DATA == 'cardioveg':
             ##for cardioveg
            im = ax.contourf(time, np.log2(frequencies[:]), power[:,1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")
        else:
            im = ax.contourf(time, np.log2(frequencies[1:]), power[:][1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")

        yticks = 2**np.arange(1, np.floor(np.log2(frequencies.max())))
        ax.set_yticks(np.log2(yticks))
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], 0)

        #plt.show()
        plt.savefig(os.path.join(path, '{}_{}_{}_{}_scalogram.jpg'.format(i, patientid, sbp, dbp)), bbox_inches='tight', pad_inches=0)


def plot_wavelet(signals, path, data, frequency, time, wavelet_func = "morl"):

    if (empty_folder(path) == False):
        if(delete_contents_in_folder(path) == False):
            return
    
    T = time #sec
    Fs=frequency #Hz
    dt=1/Fs #sec
    time = np.arange(0, T, dt)
    wavelet =  wavelet_func # "morl"# "cmor" "gaus1"
    scales = np.arange(1,512,2)

    i = 0
    for val in signals:

        i += 1
        ppg = val['ppg']
        patientid = val['patientid']
        sbp = val['sbp']
        dbp=val['dbp']
        
        signal_detrend = signal.detrend(ppg, type='constant')
        #ax = plt.axes()
        fig = plt.figure()
        ax = fig.add_subplot()
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal_detrend, scales, wavelet, dt)
        power = (abs(coefficients)) ** 2
        lev_exp = np.arange(-5, np.ceil(np.log10(power.max())+1))
        levs = np.power(10, lev_exp)
        ##for cardioveg
        try: 
            if data == 'cardioveg':
                ##for cardioveg
                im = ax.contourf(time, np.log2(frequencies[:]), power[:,1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")
            else:
                im = ax.contourf(time, np.log2(frequencies[1:]), power[:][1:], levs, norm=mpl.colors.LogNorm(), extend='both',cmap="RdBu_r")
        except TypeError:
            pass

        yticks = 2**np.arange(-2, np.floor(np.log2(frequencies.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], -2)
        fig.savefig(os.path.join(path, '{}_{}_{}_{}_scalogram.jpg'.format(i, patientid, sbp, dbp)), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)



def get_representations(path, model = 'resnet-18'):

    if(empty_folder(path) == True):
        print("exiting the program as there was no files in the folder found")
        return
    
    list_dicts = []
    img2vec = Img2Vec(cuda=False, model = model)
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        im=Image.open(filename)
        # Get a vector from img2vec, returned as a torch FloatTensor
        vec = img2vec.get_vec(im, tensor=True)
        dict_temp = {}
        np_vec = vec.numpy().flatten()
        dict_temp['representation'] = np_vec
        dict_temp['patientid'] = filename.strip().split('_')[2]
        dict_temp['sbp'] = filename.strip().split('_')[3]
        dict_temp['dbp'] = filename.strip().split('_')[4]
        list_dicts.append(dict_temp)

    df_representation = pd.DataFrame(list_dicts)
    return df_representation


if __name__ == "__main__":
    with open(FILE_JSON_DICT_IN, 'r') as JSON:
       json_dict = json.load(JSON)
    print('number of subjects in file'  + str(len(json_dict)))


    #get_spectograms(json_dict, PATH_SPECTOGRAMS)
    #get_scalograms(json_dict[1:10], PATH_SCALOGRAMS)
    #plot_wavelet(json_dict, PATH_SCALOGRAMS, DATA, FREQUENCY, TIME)
    df_representations = get_representations(PATH_SCALOGRAMS, MODEL_TO_LEARN)
    print(df_representations.head())
    print(df_representations.shape)
    print(np.stack(df_representations['representation']).any())
    #df_representations.to_pickle(FILE_REPRESENTATION_OUT)
    
