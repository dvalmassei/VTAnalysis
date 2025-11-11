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
    plt.style.use('seaborn-v0_8')

    hist = pd.read_csv('test.csv')
    charge = pd.read_csv('charge.csv')
    
    coeff0,_ = doubleGaussfit(hist['charge (pC)']/0.569,hist['ch.0'],4000,-1,0.5,15,8,4)
    coeff1,_ = doubleGaussfit(hist['charge (pC)']/0.534,hist['ch.1'],4000,-1,0.5,15,8,4)

    
    plt.step(hist['charge (pC)']/0.569,hist['ch.0'],label=f'ch.0: A={coeff0[0]:.2f}, x0={coeff0[1]:.2f}, sigma={coeff0[2]:.2f}')
    plt.step(hist['charge (pC)']/0.534,hist['ch.1'],label=f'ch.1: A={coeff1[0]:.2f}, x0={coeff1[1]:.2f}, sigma={coeff1[2]:.2f}')
    plt.plot(hist['charge (pC)']/0.569,doubleGauss(hist['charge (pC)']/0.569,*coeff0),label=f'gaussian: A={coeff0[3]:.2f}, x0={coeff0[4]:.2f}, sigma={coeff0[5]:.2f}')
    plt.plot(hist['charge (pC)']/0.534,doubleGauss(hist['charge (pC)']/0.534,*coeff1),label=f'gaussian: A={coeff1[3]:.2f}, x0={coeff1[4]:.2f}, sigma={coeff1[5]:.2f}')
    plt.xlabel('PEs')
    plt.yscale('log')
    plt.legend()
    plt.xlim((-2,40))
    plt.ylim((1,1000))
    plt.show()    
    
    
    summed_pe = charge['ch.0']/0.569 + charge['ch.1']/0.534
    nevents,bins = np.histogram(summed_pe,bins=56,range=(-7,50))
    bins = (bins[:-1] + bins[1:]) /2
    mean = np.sum(bins * nevents)/np.sum(nevents)
    stddev = np.sqrt(np.sum( nevents*(bins - mean)**2/np.sum(nevents)))
    rms = np.sqrt(np.sum(bins**2 * nevents)/ np.sum(nevents))
    
    
    coeff,_ = doubleGaussfit(bins,nevents,2600,-1,1,20,20,10)
    print(coeff)
    
    
    plt.step(bins,nevents,label=f'mean={mean:.2f}, std dev={stddev:.2f}, rms={rms:.2f}')
    plt.plot(bins,doubleGauss(bins,*coeff))
    plt.plot(bins,gauss(bins,coeff[0],coeff[1],coeff[2]),label=f'gaussian: A={coeff[0]:.2f}, x0={coeff[1]:.2f}, sigma={coeff[2]:.2f}')
    plt.plot(bins,gauss(bins,coeff[3],coeff[4],coeff[5]),label=f'gaussian: A={coeff[3]:.2f}, x0={coeff[4]:.2f}, sigma={coeff[5]:.2f}')
    plt.title('Summed PEs')
    plt.xlabel('PEs')
    plt.yscale('log')
    plt.ylim(1,3000)
    plt.legend()
    plt.show()   
    
    
    plt.step(bins,nevents)
    plt.plot(bins,doubleGauss(bins,*coeff))
    plt.plot(bins,gauss(bins,coeff[0],coeff[1],coeff[2]),label=f'gaussian: A={coeff[0]:.2f}, x0={coeff[1]:.2f}, sigma={coeff[2]:.2f}')
    plt.plot(bins,gauss(bins,coeff[3],coeff[4],coeff[5]),label=f'gaussian: A={coeff[3]:.2f}, x0={coeff[4]:.2f}, sigma={coeff[5]:.2f}')
    plt.title('Summed PEs')
    plt.xlabel('PEs')
    #plt.yscale('log')
    plt.xlim(-0,50)
    plt.ylim(-1,15)
    plt.legend()
    plt.show()  
    
    


    
if __name__ == '__main__':
    main()