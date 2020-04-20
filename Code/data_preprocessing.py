#!/usr/bin/env python
# coding: utf-8

##BP Points estimation taken from: https://www.kaggle.com/bguberfain/reading-and-showing-data
##Author: Suparno Datta



import numpy as np
import pandas as pd
import json
from os.path import join
import matplotlib.pyplot as plt
import scipy.signal as sig
import os



get_ipython().run_line_magic('run', 'Helper_Functions.py')

##User Defined Variables
input_path = join('..', 'Data', 'Cuff-less Non-invasive Blood Pressure Estimation Data Set') 
time_window_secs = 30


##Functions



##rolling window trick numpy
# http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


##find mimimums in a window 
def find_mins(a, num_mins, window):
    found_mins = []
    amax = a.max()
    
    hwindow = window // 2
    
    a = np.array(a)

    for i in range(num_mins):
        found_min = np.argmin(a)
        found_mins.append(found_min)
        a[found_min-hwindow:found_min+hwindow] = amax
        
    del a
        
    return sorted(found_mins)

def find_bp_from_fsr(data_FSR):
    
    first_point = data_FSR[0] 
    
    if first_point == 0:
        index = (data_FSR != 0).argmax()
        data_FSR = data_FSR[index:]
    
    plt.figure(figsize=(10, 6))
    max_diff = 50
    data_FSR_clear = np.array(data_FSR, dtype=np.float)
    data_FSR_outliers = np.abs(data_FSR[1:] - data_FSR[:-1]) > max_diff
    data_FSR_outliers = np.append(data_FSR_outliers, False)
    data_FSR_clear[data_FSR_outliers] = np.nan    
    
    mean_window = 10
    
    data_FSR_roll_mean = np.nanmean(rolling_window(data_FSR_clear, mean_window), axis=-1)
    
    data_FSR_clear[np.isnan(data_FSR_clear)] = data_FSR_roll_mean[np.isnan(data_FSR_clear)[:1-mean_window]]
        
    assert np.isnan(data_FSR_clear).sum() == 0
    
    data_FSR_smooth = sig.savgol_filter(data_FSR_clear, 51, 0)

    
    diff_n = 1000
    roll_window = 21
    data_FSR_diff = data_FSR_smooth[diff_n:] - data_FSR_smooth[:-diff_n]
    data_FSR_diff_roll = rolling_window(data_FSR_diff, roll_window).mean(axis=-1)

    
    num_mins = len(data['data_BP'])
    min_window = 15000
    
    data_FSR_mins = find_mins(data_FSR_diff_roll, num_mins, min_window)
    
    
    plt.figure(figsize=(14, 6))
    plt.plot(data_FSR_smooth, label='Smoothed FSR')
    
    data_FSR_max, data_FSR_min = data_FSR_smooth.max(), data_FSR_smooth.min()
    for m in data_FSR_mins:
        plt.vlines(m + diff_n/2, data_FSR_min, data_FSR_max, color='red')
    plt.legend()
    plt.title('BP measures points');
    
    if first_point == 0:
        data_FSR_mins = (data_FSR_mins + index).tolist()
        
    return data_FSR_mins


def clear_ppg(data_PPG, verbose = False):
    
    max_diff = 30    
    
    data_PPG_clear = np.array(data_PPG, dtype=np.float)
    data_PPG_outliers = np.abs(data_PPG[1:] - data_PPG[:-1]) > max_diff
    data_PPG_outliers = np.append(data_PPG_outliers, False)
    data_PPG_clear[data_PPG_outliers] = np.nan
    
    
      
    
    data_PPG_clear = pd.Series(data_PPG_clear).interpolate().values
    
    if verbose:
        
        fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(10,8))
        ax0.plot(data_PPG[10000:20000])
        ax1.plot(data_PPG_clear[10000:20000])
        ax2.plot(data_PPG_clear[10000:20000])
        plt.tight_layout()
        plt.plot()
    return data_PPG_clear

##end of functions

##initializing empty list to store the dicts of BP values and corresponding ppg segments
list_dicts = []
list_dfs = []
total_bp_values = 0 
filenumber = 0


for filename in os.listdir(input_path):
    
    filenumber = filenumber + 1

    with open(join(input_path, filename), 'r') as f:
        data = json.load(f)

    print("number of BP measurements in this file is : " + str(len(data['data_BP'])))
    total_bp_values += len(data['data_BP'])


    data_FSR = -np.array(data['data_FSR'])
    data_FSR_mins= find_bp_from_fsr(data_FSR)
    
    data_PPG = -np.array(data['data_PPG'])
    data_PPG_clear = clear_ppg(data_PPG, verbose = False) 
    
    
    data_PPG_clear_normalized = normalize(data_PPG_clear)
    
    bp = pd.DataFrame()
    time_window_hz = time_window_secs * 1000

    ##start of looping through the BP values
    for index, mins in enumerate(data_FSR_mins):
        
        data_PPG_subsection = data_PPG_clear_normalized[(mins - time_window_hz): mins]
        
        if(len(data_PPG_subsection) == 0):
            print("NO PPG FOUND IN" + filename)
       # print(type(extract_features_for_window(pd.Series(data_PPG_subsection), verbose = True)))
        data_PPG_subsection_df = pd.Series(data_PPG_subsection)
        data_PPG_subsection_df.index = pd.to_datetime(data_PPG_subsection_df.index, unit='ms')
        window_features = extract_features_for_window(data_PPG_subsection_df, verbose=False)
        sbp = data['data_BP'][index]['SBP']
        dbp = data['data_BP'][index]['DBP']
        bp.loc[index, 'SBP'] = sbp
        bp.loc[index, 'DBP'] = dbp
        bp.loc[index, 'patientid'] = filenumber
        
        bp_dict = {'patientid': filenumber,
                   'sbp': sbp, 
                   'dbp': dbp,
                   'ppg': data_PPG_subsection.tolist()}
        list_dicts.append(bp_dict)
        for col in window_features.columns:
            if col.find('ts') == -1:
                bp.loc[index, col+'_mean'] = window_features[col].mean()
                bp.loc[index, col+'_var'] = window_features[col].var()
        ##end of looping through bp values  
            
    list_dfs.append(bp)
    print("total BP values found till now : "+ str(total_bp_values))      
    #end of looping through files   
    
    
with open('../intermediate_data/ppg_snippets.json', 'w') as fout:
    json.dump(list_dicts , fout)
    
df_bp_complete = pd.concat(list_dfs, ignore_index = True)    
#df_bp_complete.to_csv('../intermediate_data/bp_features.csv', index=False)




