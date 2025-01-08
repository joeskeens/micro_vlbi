#!/usr/bin/python
"""
Contains utilities used by other public release python scripts 

==============================================================================
  This file is part of software and data needed to reproduce the figures and
  data in 
    A novel interferometer utilizing a radio telescope and a GNSS antenna
  submitted to Radio Science in 2023.  It has been prepared under the 
  NASA Open-Source Science initiative.

  This is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 3.0 of the License, or
  any later version.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with GPSTk; if not, write to the Free Software Foundation,
  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110, USA

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin, , under NASA Grants xxx-xxx-xxx and 
  xxx-xxx-xxx.

  Copyright 2023, The Board of Regents of The University of Texas System
==============================================================================
"""

from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
import numpy as np
import scipy.constants as const
import os

def save_data(filename, arr_list, col_names):
    """ Save plot data to a csv file
    Arguments:
        filename: The name of the csv file to be saved
        arr_list: The data to be saved, given as a list of arrays
        col_names: The names of the data arrays in arr_list
    Returns: 
        None
    """
    # determine the maximum length of the arrays
    max_len = max(len(arr) for arr in arr_list)

    # create a new array with the maximum length, fill with NaN
    new_arr = np.full((max_len, len(arr_list)), np.nan)

    # copy the arrays to the new array
    for i, arr in enumerate(arr_list):
        new_arr[:len(arr), i] = arr

    # save the new array to a csv file
    np.savetxt('gain_fig.csv', new_arr, delimiter=',', fmt='%.2f', header=','.join(col_names))
    
def gen_gain_pattern(antenna_file):
    """ Generate 2D interpolater function from antenna csv file
    Arguments:
        antenna_file: The name of the antenna gain csv file to be opened

    Returns: 
    
    
    """
    if os.path.exists(antenna_file):
        topcon_data = np.loadtxt(open(antenna_file,'rb'),delimiter=',',skiprows=1,usecols=(2,3,4))
    else:
        raise Exception('File ' + antenna_file + ' does not exist')
        
    elev_angle = topcon_data[:,0]
    freq = topcon_data[:,1]
    gain = topcon_data[:,2]
    interp_fcn = interp2d(elev_angle, freq, gain)
    
    return interp_fcn 

GHZ_TO_HZ = 1e9
class VLBA_properties():
    """ properties for the Fort Davis VLBA dish-- generate
    interpolation functions for DPFU and Tsys
    """
    def __init__(self):
            freq_arr = np.array([1.438,1.658])*GHZ_TO_HZ # Hz, measurement Freqs, vlba_gains.key
            FD_DPFU = np.array([0.111,0.105]) # vlba_gains.key
            FD_Tsys = np.array([29,28]) # vlba_gains.key, zenith system temperature
            self.DPFU_interp = interp1d(freq_arr, FD_DPFU, fill_value="extrapolate")
            self.Tsys_interp = interp1d(freq_arr, FD_Tsys, fill_value="extrapolate")

class antenna_properties():
    """ Properties for the GNSS antenna. Allows for calculating the effective antenna area
    for a given wavelength/gain 
    """
    def __init__(self, lam): 
            self.lam = lam # wavelength
    def area(self, gain):
            return gain * self.lam**2 / (4 * const.pi)
