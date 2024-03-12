#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:48:18 2024

@author: danielvalmassei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylandau
from scipy.optimize import curve_fit
import warnings

def gauss(x, A, x0, sigma): 
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussFit(x, y, A, x0, sigma):
    coeff, pcov = curve_fit(gauss, x, y,
                            absolute_sigma=True,
                            p0=(A, x0, sigma))
        
    return coeff, pcov

def langauFit(x, y, mpv, sigma, eta, A):
    coeff, pcov = curve_fit(pylandau.langau, x, y,
                        absolute_sigma=True,
                        p0=(mpv, sigma, eta, A),                    
                        bounds=(-2.0,1000.0))
    
    return coeff, pcov

def read_signals(data_folder, runs, channels):
    print('Loading data. This could take a couple minutes. Please wait...')

    dfs = []
    dfs.append(pd.read_csv(data_folder + f'/{runs[0]}/TR_0_0.txt', header=None))
    for ch in channels:
        dfs.append(pd.read_csv(data_folder + f'/{runs[0]}/wave_{ch}.txt', header=None))
    
    if len(runs) > 1:
        for i in range(len(runs) - 1):
            dfs[0] = pd.concat([dfs[0], pd.read_csv(data_folder + f'/{runs[i+1]}/TR_0_0.txt', header= None)])
            for ch in channels:
                dfs[ch+1] = pd.concat([dfs[ch+1],pd.read_csv(data_folder + f'/{runs[i+1]}/wave_{ch}.txt',header = None)])    
            
    print('Data loaded!')

    signals = np.array(dfs)/50/4096
    return signals

def calculate_offsets(signals, channels):
    offsets = [np.mean(signals[ch][-100:]) for ch in range(len(channels) + 1)]
    return offsets 

def find_events(trigger, threshold, offset):
    print('Finding events...')
    events = []
    trigger_state = False
    for i in range(len(trigger)):
        if trigger[i] < offset - threshold and not trigger_state:
            events.append(i)
            trigger_state = True
        elif trigger[i] > offset - threshold and trigger_state:
            trigger_state = False
    return events

def integrate_signals(channels, events, signalThreshold, signals, offsets, atten):
    charge = [np.zeros(len(events)) for _ in range(len(channels))]

    for i in range(len(events)):
        start_index = events[i]
        for ch in channels[:-1]:
            if (max(offsets[ch+1] - signals[ch+1][start_index:start_index + 1000])> signalThreshold):
                charge[ch][i] = sum(offsets[ch+1] - signals[ch+1][start_index:start_index + 1000]) / (5 * 10E9) * 1E12 / atten
                
        charge[-1][i] = sum(offsets[-1] - signals[-1][start_index:start_index + 1000]) / (5 * 10E9) * 1E12 / atten
        
    print('Finished processing signals!')
    return charge

def calculate_stats(channels,charge):
    mean = []
    rms = []
    std = []
    
    for i in range(len(channels)):
        mean.append(np.mean(charge[i]))
        rms.append(np.mean(charge[i]**2))
        std.append(np.std(charge[i]))
    
    return np.array(mean), np.array(rms), np.array(std)

def histogram_charges(charge, channels, histEndpoint, nBins):
    hists = []
    bins = []
    for i in range(len(channels)):
        hist0, bins0 = np.histogram(charge[i][charge[i] != 0], bins=nBins, range=(-2, histEndpoint))
        hists.append(hist0)
        bins.append((bins0[:-1] + bins0[1:]) /2) #calculate bin centers
    
    return hists, bins

def fit(hists,bins,channels,charge,means):
    coeffs = []
    pcovs = []
    resLangau = []
    ss_resLangau = []
    ss_tot = []
    failed_fit_channels = []

    for i in range(len(channels)):
        try:
            coeff,pcov = langauFit(bins[i], hists[i], means[i]/2, np.std(charge[i])/2, 0.1, np.max(hists[i]))
            coeffs.append(coeff)
            pcovs.append(pcov)
            resLangau.append(pylandau.langau(bins[i], *coeffs[i]) - hists[i])
            ss_resLangau.append(np.sum(np.array(resLangau[i]) **2))
            ss_tot.append(np.sum((hists[i] - np.mean(hists[i]))**2))
            
        except Exception as e:
            print(f"Failed to fit Langau for Channel {channels[i]}. Error: {e}")
            coeffs.append(0)
            pcovs.append(0)
            resLangau.append(1)
            ss_resLangau.append(1)
            ss_tot.append(1)
            failed_fit_channels.append(channels[i])
            
    r_squared = 1 - (np.array(ss_resLangau) / np.array(ss_tot))
    return coeffs, resLangau, r_squared, failed_fit_channels

def pedestal_fit(hists,bins,noise_channel,charge,mean):
    resLangau = []
    ss_resLangau = []
    ss_tot = []
    failed_fit_channels = []
  
    coeff,pcov = gaussFit(bins, hists, np.max(hists), mean, np.std(charge))
            
    r_squared = 1 - (np.array(ss_resLangau) / np.array(ss_tot))
    return coeff, resLangau, r_squared, failed_fit_channels

def main(data_folder, runs, data_channels, noise_channel, atten, triggerThreshold, signalThreshold, nBins, histEndpoint):
    
    all_channels = data_channels + noise_channel
    signals = read_signals(data_folder, runs, all_channels)

    plt.figure(figsize=(10, 5))
    for signal in signals:
        plt.plot(signal, linewidth=1)

    plt.ylabel('current [A]')
    plt.xlabel('Sample # @ 5GS/s')
    plt.show()
    
    trigger = signals[0]
    offsets = calculate_offsets(signals, all_channels)
    
    triggerThreshold /= 4096 * 50  # convert digitizer bins to current
    events = find_events(trigger,triggerThreshold,offsets[0])
    print(f'Number of Events: {len(events)}')
    
    signalThreshold /= 4096 * 50
    charge = np.array(integrate_signals(all_channels, events, signalThreshold,signals, offsets, atten))

    hists, bins = histogram_charges(charge,data_channels,histEndpoint, nBins)
    pedhist, pedbins = np.histogram(charge[-1], bins=nBins, range=(-2, 2))
    pedbins = (pedbins[:-1] + pedbins[1:]) /2
    
    mean, rms, std = calculate_stats(all_channels, charge)

    print(f'mean: {mean}')
    print(f'rms: {rms}')
    print(f'std: {std}')
    print(f'rms/mean: {rms/mean}')
    print(f'std/mean: {std/mean}')
    
    coeffs, resLangau, r_squared, failed_fit_channels = fit(hists,bins,data_channels,charge,mean)
    ped_coeffs,_,_,_ = pedestal_fit(pedhist, pedbins, noise_channel, charge[-1], mean[-1])
    
    for i in range(len(hists)):    
        plt.step(bins[i], hists[i])
        plt.plot(bins[i], pylandau.langau(bins[i], *coeffs[i]))
        
    plt.step(pedbins, pedhist)
    plt.plot(pedbins, gauss(pedbins, *ped_coeffs))
        
    plt.show()
    print(f'r_squared: {r_squared}')   

    fig, axis = plt.subplots(2, 1, height_ratios=[4, 1], sharex=True, figsize=(12, 8))
    
    for i in range(len(hists)):
        if i in failed_fit_channels:
            axis[0].step(bins[i], hists[i], linewidth=1, label=f'ch. {data_channels[i]}, res = {std[i]/mean[i]:.4f}')
        else:
            axis[0].step(bins[i], hists[i], linewidth=1, label=f'ch. {data_channels[i]}, std/mean = {std[i]/mean[i]:.4f}')
            axis[0].plot(bins[i], pylandau.langau(bins[i], *coeffs[i]), label = f'ch. {data_channels[i]} fit, $r^2 = {r_squared[i]:.4f}$')
            axis[1].scatter(bins[i], resLangau[i], s=3, label=f'ch. {data_channels[i]}')

    if len(noise_channel) > 0:
        axis[0].step(pedbins,pedhist/2,linewidth=1,label=f'ch. {noise_channel[0]} (noise)') # divide by 2 so we can still see data histogram
    axis[0].legend(fontsize='medium', frameon=True)
    axis[0].set_ylabel('# events', fontsize='medium', labelpad=2.0)
    axis[0].set_title(runs)

    axis[1].legend(fontsize='small', frameon=True)
    axis[1].set_ylabel('residuals', fontsize='medium', labelpad=2.0)
    axis[1].set_xlabel('pC', fontsize='medium', labelpad=2.0)

    plt.show()
    
    peak_charge = [coeffs[i][0] - mean[-1] for i in range(len(data_channels))]
    print(f'Peak Charge: {peak_charge}')

if __name__ =='__main__':
    # Example usage:
    data_folder = '../data/'
    runs = ['sam_012224_0','sam_012324_0','sam_012524_0','sam_012624_0']  # Example list of runs 'sam_012224_0','sam_012324_0','sam_012524_0','sam_012624_0'
    data_channels = [0]
    noise_channel = [1]
    atten = 1.0
    triggerThreshold = 500
    signalThreshold = 200
    nBins = 128
    histEndpoint = 20
    warnings.simplefilter("ignore")
    
    main(data_folder, runs, data_channels, noise_channel, atten, triggerThreshold, signalThreshold, nBins, histEndpoint)