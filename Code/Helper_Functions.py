import pytz
import numpy as np
import json
import os
import pandas as pd
import math
import re
import sys
import datetime

import pyedflib
import scipy.signal as sig
from scipy.fftpack import fft
import scipy.stats as stats
import matplotlib.pyplot as plt
import biosppy

import import_ipynb
import glob

def find_clean_cycles_with_template(signal, verbose=False):
    initial_cycle_starts = find_cycle_starts(signal)
    if len(initial_cycle_starts) <= 1:
        return []
    template_length = math.floor(np.median(np.diff(initial_cycle_starts)))
    cycle_starts = initial_cycle_starts[:-1]
    while cycle_starts[-1] + template_length > len(signal):
        cycle_starts = cycle_starts[:-1]
    template = []
    for i in range(template_length):
        template.append(np.mean(signal[cycle_starts + i]))
    
    corr_coef = []
    for cycle_start in cycle_starts:
        corr_coef.append(np.corrcoef(template, signal[cycle_start:cycle_start+template_length])[0,1])

    valid_indices = np.argwhere(np.array(corr_coef) >= 0.8)
    if (len(valid_indices) > len(cycle_starts) / 2) and len(valid_indices) > 1:
        cycle_starts = cycle_starts[np.squeeze(valid_indices)]
        template2 = []
        for i in range(template_length):
            template2.append(np.mean(signal[cycle_starts + i]))
        template = template2
        
    if verbose:
        print('Cycle Template')
        plt.plot(template)
        plt.show()
    
    
    # Check correlation of cycles with template
    # SQI1: Pearson Correlation
    sqi1_corr = []
    for cycle_start in cycle_starts:
        corr, _ = stats.pearsonr(template, signal[cycle_start:cycle_start+template_length])
        sqi1_corr.append(corr)
        
    # SQI2: Pearson Correlation with 
    sqi2_corr = []
    for cycle_start in cycle_starts:
        cycle_end = initial_cycle_starts[np.squeeze(np.argwhere(initial_cycle_starts==cycle_start)) + 1] 
        corr, _ = stats.pearsonr(template, sig.resample(signal[cycle_start:cycle_end], template_length))
        sqi2_corr.append(corr)
        
    # SQI3: Pearson Correlation
    #sq2_corr = []
    #for cycle_start in cycle_starts:
    #    cycle_end = initial_cycle_starts[initial_cycle_starts.index(cycle_start) + 1] 
    #    corr, _ = stats.pearsonr(template, df['bvp'].iloc[cycle_start:cycle_end])
    #    sq2_corr.append(corr)
    corrs = np.array([sqi1_corr, sqi2_corr]).transpose()
    cycle_starts = cycle_starts[np.all(corrs >= 0.8, axis=1)]
    
    if verbose:
        print('Detected Valid Cycles')
        plt.figure()
        for cycle_start in cycle_starts:
            plt.plot(signal[cycle_start:cycle_start+template_length].to_numpy())
        plt.show()
        
    cycles = []
    for cycle_start in cycle_starts:
        cycle_end = initial_cycle_starts[np.squeeze(np.argwhere(initial_cycle_starts==cycle_start)) + 1]
        if (cycle_end - cycle_start) > template_length*1.2:
            cycle_end = cycle_start + template_length
        cycles.append((cycle_start, cycle_end))

    return cycles

# finds the local minima that correspond to the starts of a cardiac cycle
def find_cycle_starts(df, sample_rate=500):
    minima = sig.find_peaks(-df.values, distance=0.7*sample_rate)[0] # Todo: Check other parameters
    #diffs = np.diff(minima)
    #minima = np.delete(minima, np.argwhere(diffs < 0.75*np.mean(diffs)))
    return minima

# returns the x values for those samples in the signal, that are closest to some given y value
def find_xs_for_y(ys, y_val, sys_peak):
    diffs = abs(ys - y_val)
    x1 = diffs[:sys_peak].idxmin()
    x2 = diffs[sys_peak:].idxmin()
    return x1, x2

# takes a dataframe of calculated features and removes the outliers occuding due to inaccuracies in the signal
def clean_window_features_of_outliers(df):
    quant = df.quantile(0.8)
    for col in df.columns:
        if col.find('ts') == -1:
            df = df[df[col] < quant[col]*2]
    return df

# finds sections with high acceleration magnitude and removes them
def remove_motion_sections(df, limit=100, min_size=5, padding=15, std_mult=0.25):
    # Todo check frequency domain for high frequencies
    acc_mag_mean = df['acc_mag'].mean()
    acc_mag_std =  df['acc_mag'].std()
    # Comparison with overall mean and std
    thresh_indices = np.squeeze(np.argwhere((df['acc_mag'] > acc_mag_mean + std_mult * acc_mag_std) | 
                                            (df['acc_mag'] < acc_mag_mean - std_mult * acc_mag_std)))
        
    section_indices = []
    section_start = thresh_indices[0]
    for i in range(1, len(thresh_indices) - 1):
        if thresh_indices[i] - thresh_indices[i-1] > limit:
            if thresh_indices[i-1] >= section_start + min_size:
                section_indices.append((section_start - padding, thresh_indices[i-1] + padding))
            section_start = thresh_indices[i]
    if thresh_indices[-1] != section_start:
        section_indices.append((section_start, thresh_indices[-1]))
    
    # Check local variance for window
    #section_indices = []
    #window_vars = []
    #print(df.count()['bvp'])
    #for i in range((df.count()['bvp'] - limit) // padding + 1):
    #    window = df['acc_mag'].iloc[i*padding:i*padding+limit]
    #    var = window.var()
    #    window_vars.append(var)
    #ind = np.squeeze(np.argwhere(window_vars > np.float64(1)))
    #print(ind)
    #section_indices = []
    #section_start = ind[0]
    #for i in range(1, len(ind) - 1):
    #    if ind[i] - ind[i-1] > 1:
    #        section_indices.append((section_start*padding, ind[i-1]*padding+limit))
    #        section_start = ind[i]
    #if ind[-1] != section_start:
    #    section_indices.append((section_start*padding, ind[-1]*padding+limit))
    #print(section_indices)
        
    section_indices.reverse()
    for (start, end) in section_indices:
        df = df.drop(index=df.iloc[start:end].index)
    return df

def extract_features_for_cycle(window_df, signal, verbose=False):
    cur_index = window_df.index.max() + 1
    if np.isnan(cur_index):
        cur_index = 0
    signal = signal.resample('ms').nearest(limit=1).interpolate(method='time')
    signal = signal - signal.min()
    max_amplitude = signal.max()
    
    peaks = sig.find_peaks(signal.values)[0]
    sys_peak_ts = signal.index[peaks[0]]
    
    if verbose:
        plt.figure()
        plt.xlim((signal.index.min(), signal.index.max()))
        plt.scatter(signal.index[peaks], signal[peaks])
        plt.plot(signal.index, signal.values)
    # Features
    window_df = window_df.append(pd.DataFrame({'start_ts': signal.index.min(),
                                               'sys_peak_ts': sys_peak_ts,
                                               'T_S': (sys_peak_ts - signal.index.min()).total_seconds(),
                                               'T_D': (signal.index.max() - sys_peak_ts).total_seconds()
                                              }, index=[cur_index]), sort=False)
    for p in [10, 25, 33, 50, 66, 75]:
        p_ampl = p / 100 * max_amplitude
        x1, x2 = find_xs_for_y(signal, p_ampl, peaks[0])
        if verbose:
            plt.scatter([x1, x2], signal[[x1, x2]])
        window_df.loc[cur_index, 'DW_'+str(p)] = (x2 - sys_peak_ts).total_seconds()
        window_df.loc[cur_index, 'DW_SW_sum_'+str(p)] = (x2 - x1).total_seconds()
        window_df.loc[cur_index, 'DW_SW_ratio_'+str(p)] = (x2 - sys_peak_ts) / (sys_peak_ts - x1)
    if verbose:
        plt.show()
    return window_df
    
def extract_features_for_window(signal, verbose=False):
    cycles = find_clean_cycles_with_template(signal, verbose=verbose)
    print("the number of cycles is found: " + str(len(cycles)))
    if len(cycles) == 0:
        return pd.DataFrame()
    #plt.figure(figsize=(16, 6))
    #plt.plot(df['bvp'].to_numpy())
    #plt.scatter(cycle_starts, df['bvp'].iloc[cycle_starts])
    #plt.show()
    
    window_features = pd.DataFrame()
    cur_index = 0
    for i in range(len(cycles)):
        window_features = extract_features_for_cycle(window_features, signal.iloc[cycles[i][0]:cycles[i][1]], verbose=verbose)
        if i > 0:
            window_features.loc[cur_index-1, 'CP'] = (window_features.loc[cur_index, 'sys_peak_ts'] - window_features.loc[cur_index-1, 'sys_peak_ts']).total_seconds()
        cur_index = cur_index + 1
    if verbose:
        print('Cycle Features within Window:')
        print(window_features)
    window_features = clean_window_features_of_outliers(window_features)
    return window_features

def normalize(signal):
    # No smoothing neccessary due to relatively low sampling frequency
    signal = (signal - signal.min()) / (signal.max() - signal.min())
    return signal