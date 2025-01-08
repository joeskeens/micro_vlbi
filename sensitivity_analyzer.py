#!/usr/bin/python
"""
This script takes input flux density, observed SNR, and elevation to compute
the observed system temperature for a dish-patch antenna interferometer.

Usage:

  python sensitivity_analyzer.py --bandwidth $BANDWIDTH --start_freq $START_FREQ --input_name $FILEIN --output_name $FILEOUT --antenna_file $ANTENNA_CSV

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

import numpy as np
import scipy.constants as const
from scipy.interpolate import interp1d, interp2d
import utilities
import argparse
JY_TO_SI=1e26

def parser_fcn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type = str, help='name of input file and path (e.g. /here/name_here.txt)',default='source_data_sensitivity.txt')
    parser.add_argument('--output_name', type = str, help='name of output file (e.g. name_here.txt)',default=None)
    parser.add_argument('--antenna_file', type = str, help='name and path of antenna gain file (csv)',default='antenna_gain.csv')
    parser.add_argument('--bandwidth', type = float, help='bandwidth of data (Hz)', default=120e6)
    parser.add_argument('--start_freq', type = float, help='lowest frequency in band (Hz)', default=1376e6)
    parsed_args = parser.parse_args()
    return parsed_args
    
if __name__=='__main__':

    parsed_args = parser_fcn()
    
    dv = parsed_args.bandwidth # bandwidth
    avg_freq = parsed_args.start_freq + dv/2 # middle of band
    lam = const.c/(avg_freq)
    
    #antenna/dish properties
    dish_prop = utilities.VLBA_properties()
    antenna_prop = utilities.antenna_properties(lam)
    interp_fcn = utilities.gen_gain_pattern(parsed_args.antenna_file)
    
    eta_Q = 0.8815 # digitization efficiency (2-bit), Schwab et al. 1986 
 
    #array initializations
    source_dat = np.loadtxt(parsed_args.input_name, skiprows=1, usecols=[1,2,3,4,5], delimiter = ",") # load source data
    source_output = np.zeros((len(source_dat[:,0]),2))
    T_sys_dish = dish_prop.Tsys_interp(avg_freq)
    gain_zenith = interp_fcn(90, avg_freq) 
    gain_zenith = 10**(gain_zenith/10) 
    area_zenith = antenna_prop.area(gain_zenith)
    dish_area = 2*const.k*10**26*dish_prop.DPFU_interp(avg_freq)

    # run data
    for idx in range(len(source_dat[:,0])):
        int_time = source_dat[idx,0] 
        flux_dens = source_dat[idx,1] * 10**-26 # W/(m^2-Hz)
        SNR = source_dat[idx,2]
        elev = source_dat[idx,3]
        bw = source_dat[idx,4] * 10**6
        gain_ant = interp_fcn(elev,avg_freq)
        gain_ant = 10**(gain_ant/10) # convert to linear units
        T_sys = (eta_Q*flux_dens/SNR)**2 * dish_prop.DPFU_interp(avg_freq)*JY_TO_SI/(const.k*T_sys_dish)\
                *antenna_prop.area(gain_ant)*bw*int_time
        SEFD = 2*const.k*T_sys/area_zenith * JY_TO_SI/1e6 # MJy
        source_output[idx,0] = T_sys
        source_output[idx,1] = SEFD
        
    source_desigs = np.loadtxt(parsed_args.input_name, "str", skiprows=1, delimiter = ",", usecols = 0)

    if parsed_args.output_name is not None:
        output = np.concatenate((source_desigs[:,np.newaxis], source_output), axis=1)
        np.savetxt(parsed_args.output_name, output, fmt="%s")
    else: # print output
        for idx in range(len(source_dat[0,:])):
            print('For source '+source_desigs[idx]+' estimated system temperature is '\
                    +str(np.round_(source_output[idx],decimals=3))+' K\n')
