#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:49:10 2024

@author: danielvalmassei
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pylandau



def gauss(x, A, x0, sigma): 
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussFit(x, y, A, x0, sigma):
    coeff, pcov = curve_fit(gauss, x, y,
                            absolute_sigma=True,
                            p0=(A, x0, sigma))
        
    return coeff, pcov

def doubleGauss(x, A0, x0, sigma0, A1, x1, sigma1):
    return A0 * np.exp(-(x - x0) ** 2 / (2 * sigma0 ** 2)) + A1 * np.exp(-(x - x1) ** 2 / (2 * sigma1 ** 2))

def doubleGaussfit(x,y, A0, x0, sigma0, A1, x1, sigma1):
    coeff, pcov = curve_fit(doubleGauss, x, y, absolute_sigma=True,
                            p0=(A0, x0, sigma0, A1, x1, sigma1))
    
    return coeff, pcov

def langauFit(x, y, mpv, sigma, eta, A):
    coeff, pcov = curve_fit(pylandau.langau, x, y,
                        absolute_sigma=True,
                        p0=(mpv, sigma, eta, A),                    
                        bounds=(-2.0,1000.0))
    
    return coeff, pcov

def main():
    ########################### Fit Parameters ################################
    
    ch0SPE = 20 #3.88992 #.534 #1.54/2 # Set to 1 when doing SPE calibrations
    ch1SPE = 20
    
    pedA = 2000
    pedMean = 0
    pedSigma = 1
    
    ch0A = 4
    ch0Mean = 20
    ch0Sigma = 5
    
    ch1A = 4
    ch1Mean = 20
    ch1Sigma = 50
    
    sumPedA = 2000
    sumPedMean = 0
    sumPedSigma = 1
    
    sumA = 20
    sumMean = 100
    sumSigma = 10
    
    ########################################################################
    
    plt.style.use('seaborn-v0_8')

    hist = pd.read_csv('chargeHistogram.csv')
    charge = pd.read_csv('charge.csv')
    
    coeff0,coeff0cov = doubleGaussfit(hist['charge (pC)']/ch0SPE,hist['ch.0'],pedA, pedMean,pedSigma,ch0A,ch0Mean, ch0Sigma)
    coeff1,coeff1cov = doubleGaussfit(hist['charge (pC)']/ch1SPE,hist['ch.1'],pedA, pedMean,pedSigma,ch1A,ch1Mean, ch1Sigma)
    #coeff1,_ = gaussFit(hist['charge (pC)'][60:200]/ch0SPE,hist['ch.0'][60:200],ch0A,ch0Mean, ch0Sigma)
    #print(coeff1)
    
    plt.step(hist['charge (pC)']/ch0SPE,hist['ch.0'],label=f'ch.0: A={coeff0[0]:.2f}, x0={coeff0[1]:.2f}, sigma={coeff0[2]:.2f}')
    plt.step(hist['charge (pC)']/ch1SPE,hist['ch.1'],label=f'ch.1: A={coeff1[0]:.2f}, x0={coeff1[1]:.2f}, sigma={coeff1[2]:.2f}')
    
    plt.plot(hist['charge (pC)']/ch0SPE,doubleGauss(hist['charge (pC)']/ch0SPE,*coeff0),label='ch.0 fit')
    plt.plot(hist['charge (pC)']/ch1SPE,doubleGauss(hist['charge (pC)']/ch1SPE,*coeff1),label='ch.1 fit')
    plt.plot(hist['charge (pC)']/ch0SPE,gauss(hist['charge (pC)']/ch0SPE,coeff0[0],coeff0[1],coeff0[2]),label=f'pedestal 0: A={coeff0[0]:.2f}, x0={coeff0[1]:.2f}, sigma={coeff0[2]:.2f}')
    plt.plot(hist['charge (pC)']/ch0SPE,gauss(hist['charge (pC)']/ch0SPE,coeff0[3],coeff0[4],coeff0[5]),label=f'signal 0: A={coeff0[3]:.2f}, x0={coeff0[4]:.2f}, sigma={coeff0[5]:.2f}')
    plt.plot(hist['charge (pC)']/ch1SPE,gauss(hist['charge (pC)']/ch1SPE,coeff1[0],coeff1[1],coeff1[2]),label=f'pedestal 1: A={coeff1[0]:.2f}, x0={coeff1[1]:.2f}, sigma={coeff1[2]:.2f}')
    plt.plot(hist['charge (pC)']/ch1SPE,gauss(hist['charge (pC)']/ch1SPE,coeff1[3],coeff1[4],coeff1[5]),label=f'signal 1: A={coeff1[3]:.2f}, x0={coeff1[4]:.2f}, sigma={coeff1[5]:.2f}')
    plt.xlabel('PEs')
    
    #plt.yscale('log')
    plt.title('HV SCAN 1050V PMT 9305 1304-B828')
    plt.legend()
    plt.xlim((-1,30))
    plt.ylim((0.1,100))
    plt.yscale('log')
    plt.show()
    perr = np.sqrt(np.diag(coeff0cov))
    print(perr)
    
    
    summed_pe = charge['ch.0']/ch0SPE + charge['ch.1']/ch1SPE
    nevents,bins = np.histogram(summed_pe,bins=512,range=(-5,600))
    bins = (bins[:-1] + bins[1:]) /2
    mean = np.sum(bins * nevents)/np.sum(nevents)
    stddev = np.sqrt(np.sum( nevents*(bins - mean)**2/np.sum(nevents)))
    rms = np.sqrt(np.sum(bins**2 * nevents)/ np.sum(nevents))
    
    # Example usage:
    
    
    
    coeff,_ = doubleGaussfit(bins,nevents,sumPedA, sumPedMean, sumPedSigma, sumA, sumMean, sumSigma)
    #print(coeff)
    
    plt.step(bins,nevents,label=f'mean={mean:.2f}, std dev={stddev:.2f}, rms={rms:.2f}')
    plt.plot(bins,doubleGauss(bins,*coeff))
    plt.plot(bins,gauss(bins,coeff[0],coeff[1],coeff[2]),label=f'gaussian: A={coeff[0]:.2f}, x0={coeff[1]:.2f}, sigma={coeff[2]:.2f}')
    plt.plot(bins,gauss(bins,coeff[3],coeff[4],coeff[5]),label=f'gaussian: A={coeff[3]:.2f}, x0={coeff[4]:.2f}, sigma={coeff[5]:.2f}')
    plt.title('Summed Charge')
    plt.xlabel('PEs')
    plt.yscale('log')
    plt.ylim((1,1000))
    plt.xlim(-1,60)
    plt.legend(loc='upper right')
    plt.show()   
    
    
    plt.step(bins,nevents)
    plt.plot(bins,doubleGauss(bins,*coeff))
    plt.plot(bins,gauss(bins,coeff[0],coeff[1],coeff[2]),label=f'gaussian: A={coeff[0]:.2f}, x0={coeff[1]:.2f}, sigma={coeff[2]:.2f}')
    plt.plot(bins,gauss(bins,coeff[3],coeff[4],coeff[5]),label=f'gaussian: A={coeff[3]:.2f}, x0={coeff[4]:.2f}, sigma={coeff[5]:.2f}')
    plt.title('Summed PEs')
    plt.xlabel('PEs')
    plt.yscale('log')
    plt.xlim(-1,60)
    plt.ylim(0.9,1030)
    plt.legend(loc='upper right')
    plt.show()  
    
    


    
if __name__ == '__main__':
    main()