#!/usr/bin/python
"""
Find the number of visible sources in the NVSS catalog for 
given interferometer characteristics

Usage:

  python find_nsource.py

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

import utilities
import numpy as np
import scipy.constants as const
from scipy.spatial import KDTree
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_flux(filename):
    """
    Open the NVSS catalog and find locations and flux densities 

    Arguments:
        filename: filename and location of TXT catalog

    Returns:
        RA_deg_arr: array of source right ascensions (deg)
        DEC_deg_arr: array of source declinations (deg)
        flux_arr: array of flux densities (Jy)
    """ 

    with open(filename) as fn:
        lines = fn.readlines()
        names = []
        RA_deg_arr = []
        DEC_deg_arr = []
        flux_arr = []
        for line in lines:
            words = line.split()
            if len(words) > 4: 
               RA_deg = (float(words[0])+float(words[1])/60+float(words[2])/3600)/24*360  # IVS name
               RA_deg_arr.append(RA_deg)
               if words[3] == '-00' or float(words[3])<0:
                   neg = -1
               else:
                   neg = 1
               DEC_deg =  neg*(np.abs(float(words[3]))+float(words[4])/60+float(words[5])/3600)  
               DEC_deg_arr.append(DEC_deg)
               flux_arr.append(words[6])
    RA_deg_arr = np.array(RA_deg_arr, dtype=float)
    DEC_deg_arr = np.array(DEC_deg_arr, dtype=float)
    flux_arr = np.array(flux_arr, dtype=float)
    # when a position is not recorded, NVSS reports a >3000 Jy flux dens.
    # get rid of this here
    idxs_good = flux_arr<3000
    flux_arr = flux_arr[flux_arr<3000] 
    print('Max flux in set: ' + str(np.amax(flux_arr)) + ' Jy')
    print('\nNumber of sources considered: ' + str(len(flux_arr)))
    return RA_deg_arr[idxs_good], DEC_deg_arr[idxs_good], flux_arr

def find_nsource_rad(rad, Sf_lim, RA_deg_arr, DEC_deg_arr, flux_arr):
    """
    Use a KDTree to combine the flux density of sources according to the radius given. 
    Sources are combined by finding the highest flux density neighbor
    and using this as the center of a ball to absorb surrounding sources.
    Determine which of these combined sources are observable at SNR_lim SNR for a 5 minute 
    coherent accumulation of bandwidth dv

    Arguments:
        rad: radius in arcmin to sum
        Sf_lim: limiting flux density (Jy)
        RA_deg_arr: array of source right ascensions (deg)
        DEC_deg_arr: array of source declinations (deg)
        flux_arr: array of flux densities (Jy)

    Returns:
        None
    """ 

    rad = rad/60 # convert to degrees from arcmin
    sorted_idxs = np.flip(np.argsort(flux_arr))
    flux_arr = flux_arr[sorted_idxs]
    RA_deg_arr = RA_deg_arr[sorted_idxs]
    DEC_deg_arr = DEC_deg_arr[sorted_idxs]

    source_pts = np.vstack((RA_deg_arr,DEC_deg_arr)).T
    sources_tree = KDTree(source_pts)
    source_nums = np.arange(len(RA_deg_arr))
       
    sources_used = []
    max_sources = []
    fluxes_used = []
    for source in source_nums:
        if source in sources_used:
            continue
        idxs_ball = sources_tree.query_ball_point(source_pts[source,:], rad, workers=-1) 
        
        fluxes = flux_arr[idxs_ball] 
        fluxes_used.append(np.sum(fluxes))
        max_sources.append(source)
        for ball_idx in idxs_ball: sources_used.append(ball_idx)
    
    n_above_threshold = np.sum(fluxes_used > Sf_lim) 
    print('Number of source above ' + str(Sf_lim) + ' Jy: ' + str(n_above_threshold))
    
    return 


def parser_fcn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--antenna_file', type = str, help='name and path of antenna gain file (csv)',default='antenna_gain.csv')
    parser.add_argument('--catalog_file', type = str, help='name and path of ICRF catalog file',default='ICRF3_Lband_FD.txt')
    parser.add_argument('--bandwidth', type = float, help='bandwidth of data (Hz)', default=200e6)
    parser.add_argument('--start_freq', type = float, help='lowest frequency in band (Hz)', default=1376e6)
    parser.add_argument('--rad', type = float, help='radius to sum (arcmin)', default=10)
    parser.add_argument('--SNR_lim', type = float, help='SNR for useful source detect in plot', default=10)
    parser.add_argument('--t_obs', type = float, help='integration time (min)', default=5)
    parser.add_argument('--Tsys_antenna', type = float, help='System temperature for GNSS antenna', default=200)
    parsed_args = parser.parse_args()
    return parsed_args
    
if __name__=='__main__':

    parsed_args = parser_fcn()
   
    cat_file = parsed_args.catalog_file
    SNR_lim = parsed_args.SNR_lim
    Tsys_ant = parsed_args.Tsys_antenna
    t_obs = parsed_args.t_obs*60
    rad = parsed_args.rad

    #system properties
    dv = parsed_args.bandwidth # bandwidth
    eta_Q = 0.8825 # digitization efficiency (1-2 bit)
    avg_freq = parsed_args.start_freq + dv/2 # middle of band 
    lam = const.c/(avg_freq)
    
    dish_prop = utilities.VLBA_properties()
    antenna_prop = utilities.antenna_properties(lam)
    interp_fcn = utilities.gen_gain_pattern(parsed_args.antenna_file)

    Tsys_dish = dish_prop.Tsys_interp(avg_freq)
    dish_area = 2*const.k*10**26*dish_prop.DPFU_interp(avg_freq)
    elev_angle = 45 # deg 

    gain_interp = interp_fcn(elev_angle, avg_freq)
    gain_interp = 10**(gain_interp/10)
    

    T_sys = np.sqrt(Tsys_dish * Tsys_ant)
    A_eff = np.sqrt(antenna_prop.area(gain_interp) * dish_area) # effective combined area
    
    Sf_min = np.sqrt(2) * const.k * SNR_lim * T_sys/ (eta_Q * A_eff * np.sqrt(dv * t_obs)) * 10**26
    
    RA_deg_arr, DEC_deg_arr, flux_arr = parse_flux(cat_file)
    num_sources = len(flux_arr)
    find_nsource_rad(rad, Sf_min, RA_deg_arr, DEC_deg_arr, flux_arr) 
