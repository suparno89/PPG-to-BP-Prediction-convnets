
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
PATH_SCALOGRAMS = '../scalograms_cardioveg' ##output path where scalograms will be saved ../scalograms_cardioveg
PATH_SPECTOGRAMS = '../spectograms_cardioveg' ##output path where spectograms will be saved ../spectograms_cardioveg
FILE_JSON_DICT_IN = '../intermediate_data/all_features_30sec_bfill_fullday.json' ##path of input file with ppg and bp json
FILE_REPRESENTATION_OUT = '../intermediate_data/cardioveg_fullday_scalogram_resnet.pkl' ##path of output file with representation
MODEL_TO_LEARN = 'resnet-18' ##choose one of the following models('resnet-18', 'alexnet') 
DATA = 'cardioveg' ##kaggle or cardioveg, needed for the scalograms
FREQUENCY = 64 ##64 for cardioveg, 1000 for kaggle 
TIME = 30 ##in secs


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
    

              

def get_spectograms(json_dict, path, frequency): 
    
    ##given the ppg signal this function writes the spectograms out to a provided path
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
            
        pot = 6 ##use 12 for kaggle, 6 for cardioveg 
        Fs=frequency
        Nfft=pow(int(2),pot)
        detrend = 'mean'
        #ax = plt.axes()
        fig = plt.figure()
        ax = fig.add_subplot()
        Pxx, freqs, bins, im = ax.specgram(ppg, Fs=Fs, NFFT=Nfft,noverlap=Nfft/2, window=np.hanning(Nfft), detrend=detrend, cmap='seismic')
        ax.set_yscale('log')
        ax.set_ylim(freqs[1], freqs[-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(os.path.join(PATH_SPECTOGRAMS, '{}_{}_{}_{}_spectogram.jpg'.format(i, patientid, sbp, dbp)), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        gc.collect()

def get_scalograms(signals, path, data, frequency, time, wavelet_func = "cmor3-60"):

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
            print("length of the ppg signals are not similar for index {}".format(i-1))
            pass

        yticks = 2**np.arange(-2, np.floor(np.log2(frequencies.max()))) ##-2 forcardioveg
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], 1) ##can set the last parameter to -2 for cardioveg
        fig.savefig(os.path.join(path, '{}_{}_{}_{}_scalogram.jpg'.format(i, patientid, sbp, dbp)), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


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


    #get_spectograms(json_dict, PATH_SPECTOGRAMS, FREQUENCY)
    get_scalograms(json_dict, PATH_SCALOGRAMS, DATA, FREQUENCY, TIME)
    df_representations = get_representations(PATH_SCALOGRAMS, MODEL_TO_LEARN)
    print(df_representations.head())
    print(df_representations.shape)
    df_representations.to_pickle(FILE_REPRESENTATION_OUT)
    
