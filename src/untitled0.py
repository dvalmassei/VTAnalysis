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
import csv

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
    offsets = [np.mean(signals[ch][:300]) for ch in range(len(channels)+1)]
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
                charge[ch][i] = sum(offsets[ch+1] - signals[ch+1][start_index:start_index + 1000]) / (5*10E9) * 1E12 / atten
                
        charge[-1][i] = sum(offsets[-1] - signals[-1][start_index:start_index + 1000]) / (5*10E9) * 1E12 / atten
        
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
        hist0, bins0 = np.histogram(charge[i][charge[i] != 0], bins=nBins, range=(-5, histEndpoint))
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
    
    plt.style.use('seaborn-v0_8')
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
    
    histzip = zip(bins[0],hists[0])#,hists[1])
    with open('test.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("charge (pC)","ch.0","ch.1"))
      wr.writerows(histzip)
    myfile.close()
    
    '''
    histzip = zip(bins[-1],hists[-1])
    with open('pedHist.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("charge (pC)","ch.2"))
      wr.writerows(histzip)
    myfile.close()
    '''
    mean, rms, std = calculate_stats(all_channels, charge)
    #coeffs, resLangau, r_squared, failed_fit_channels = fit(hists,bins,data_channels,charge,mean)
    #ped_coeffs,_,_,_ = pedestal_fit(pedhist, pedbins, noise_channel, charge[-1], mean[-1])

    print(f'Number of Events above signal threshold: {sum(hists[0])}')
    print(f'mean: {mean}')
    print(f'rms: {rms}')
    print(f'std: {std}')
    print(f'rms/mean: {rms/mean}')
    print(f'std/mean: {std/mean}')
    #print(f'r_squared: {r_squared}') 
    #print(f'coeffs: {coeffs}') 
    '''
    for i in range(len(hists)):    
        plt.step(bins[i], hists[i])
        plt.plot(bins[i], pylandau.langau(bins[i], *coeffs[i]))
        
    plt.step(pedbins, pedhist)
    plt.plot(pedbins, gauss(pedbins, *ped_coeffs))
    plt.show()  
'''
    fig, axis = plt.subplots(figsize=(12, 8))
    
    for i in range(len(hists)):
        axis.step(bins[i], hists[i], linewidth=1, label=f'ch. {data_channels[i]}, res = {std[i]/mean[i]:.4f}')
    '''
        else:
            axis[0].step(bins[i], hists[i], linewidth=1, label=f'ch. {data_channels[i]}, std/mean = {std[i]/mean[i]:.4f}')
            axis[0].plot(bins[i], pylandau.langau(bins[i], *coeffs[i]), label = f'ch. {data_channels[i]} fit, $r^2 = {r_squared[i]:.4f}$')
            axis[1].scatter(bins[i], resLangau[i], s=3, label=f'ch. {data_channels[i]}')
    '''
    
    if len(noise_channel) > 0:
        axis.step(pedbins,pedhist/2,linewidth=1,label=f'ch. {noise_channel[0]} (noise)') # divide by 2 so we can still see data histogram
    axis.legend(fontsize='medium', frameon=True)
    axis.set_ylabel('# events', fontsize='medium', labelpad=2.0)
    axis.set_title(runs)
    #axis.set_yscale('log')

    plt.show()
    
    #peak_charge = [coeffs[i][0] - mean[-1] for i in range(len(data_channels))]
    #print(f'Peak Charge: {peak_charge}')
    
    '''
    plt.hist(sum(charge[:-1]),bins=100,range=(0,20))
    plt.show()
    '''
    chargezip = zip(charge[0])#,charge[1])
    with open('charge.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(('ch.0','ch.1'))
      wr.writerows(chargezip)
    myfile.close()
    

if __name__ =='__main__':
    # Example usage:
    data_folder = '../../data'
    runs = ['sam_ds_05072024_0']  # Example list of runs 'sam_012224_0','sam_012324_0','sam_012524_0','sam_012624_0'
    data_channels = [0,1]
    noise_channel = []
    atten = 1.0
    triggerThreshold = 500
    signalThreshold = 0
    nBins = 256
    histEndpoint = 20
    warnings.simplefilter("ignore")
    
    main(data_folder, runs, data_channels, noise_channel, atten, triggerThreshold, signalThreshold, nBins, histEndpoint)
