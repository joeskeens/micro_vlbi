#!/usr/bin/python
"""
Correct the phase of complex visibilities in a FITS file with data 
from a PPP clock solution

Usage:

  python clock_correct_ppp.py --ant1 $ANTENNA1 --ant2 $ANTENNA2 \
  --pppclockfile $PPPCLOCKFILE --filename $FILEINPATH/$FILEIN --fileout $FILEOUTPATH/$FILEOUT 

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
import os
import argparse as ag
import numpy as np
import fits_parser as fp
import scipy.signal as sig
import scipy.constants as const
# Astropy provides the reader for the FITS file
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def add_args_to_parser(parser_in):
    """Add arguments to parser."""

    # input properties
    parser.add_argument('--filename',type=str,help="Filename of input FITS file")
    parser.add_argument('--ant1',type=str,default="DBR205",help="Antenna 1 in relevant cross-correlation")
    parser.add_argument('--ant2',type=str,default="FD",help="Antenna 2 in relevant cross-correlation")
    parser.add_argument('--pppclockfile', type=str,help="ppp clock file (full path, .txt).")
    # save properties
    parser.add_argument('--fileout',type=str,default="converted",help="Filename to save FITS files as;")

class ClockCorrect(object):
    """ Handles clock correction  """
    def __init__(self, filename, fileout, clockfile, ant1, ant2):
        
        if not os.path.exists(filename):
            raise ValueError('Filepath does not exist')

        self.filename = filename
        self.fileout = fileout
        self.clockfile = clockfile
        self.fits_parser = fp.FITSParser(filename)
        self.ant1 = ant1.upper()
        self.ant2 = ant2.upper()
    
    def openClockFile(self):    
        self.interp_fcn = buildClockModel(self.clockfile)

    def processClockFile(self):
        """ Create and write mdh attributes, call process_clock  """
        self.freq_table = self.fits_parser.freq_table.freq_array(0)
        self.time_start = self.fits_parser._time_sec_of_day[0]
        self.freq_table = self.freq_table + self.fits_parser._freq # convert to sky freq
        self.cvis_new = {}

        self.sources = self.fits_parser.sources()
        if self.plot_source > 0 and self.plot_source not in self.sources:
            raise Exception('Source number not in experiment, valid source numbers: '+str(tuple(self.sources)))
        baselines = [list(self.fits_parser.baselines(source)) for source in self.sources]
        baselines_flattened = [item for sublist in baselines for item in sublist]
        first_call = True
        for baseline in baselines_flattened:
            if not baseline.is_zero() and self.ant1 in baseline.baseline_pair and self.ant2 in baseline.baseline_pair:
                complex_vis_orig = baseline.vis_data.complex_vis(0) 
                time_pts = baseline.vis_data.time_points + self.time_start
                if len(time_pts)>0:
                    complex_vis_correct = process_clock(complex_vis_orig, time_pts, self.freq_table, self.interp_fcn)
                    first_call = False
                    self.cvis_new[baseline.source] = complex_vis_correct
                else: self.cvis_new[baseline.source] = complex_vis_orig
                self.baseline_ccorr = baseline.baseline
    
    def saveClockFile(self):
      """
      Save the clock data to the destination FITS file
      """
        fits_orig = fits.open(filename)
        header_orig = fits_orig[0].header
        data_orig = fits_orig[0].data
        extname = 'EXTNAME'
        for idx, head in enumerate(fits_orig): 
            if extname in list(head.header.keys()): 
                if head.header['EXTNAME'] == 'UV_DATA':
                    for sdx, source in enumerate(self.sources,1):
                        cvis_source = self.cvis_new[source]
                        cvis_real = np.real(cvis_source)
                        cvis_imag = np.imag(cvis_source)
                        cvis_FITS = np.zeros((cvis_source.shape[0],2*cvis_source.shape[1]))
                        cvis_FITS[:,0::2] = cvis_real
                        cvis_FITS[:,1::2] = cvis_imag
                        source_indices = np.bitwise_and(head.data['BASELINE']==self.baseline_ccorr, \
                                head.data['SOURCE']==sdx)
                        fits_orig[idx].data['FLUX'][source_indices,:] = cvis_FITS
        fits_orig.writeto(fileout, overwrite=True)

def buildClockModel(clock_file):
    """ Create the clock model with scipy interp1d
    """
    ppp_data = np.loadtxt(clock_file)
    time_sec_of_day = ppp_data[:,0]
    cb_data = ppp_data[:,1]
    interp_fcn = interp1d(time_sec_of_day, cb_data, fill_value='extrapolate')

    return interp_fcn

def process_clock(complex_vis, times, freqs, interp_fcn):
    """ correct FITS data for given clock model 
    """
    clock_bias_m = interp_fcn(times)

    for idx, freq in enumerate(freqs):

        clock_offset_radians = (clock_bias_m-clock_bias_m[0])*2*np.pi*freq/const.speed_of_light

        complex_vis[:,idx]= complex_vis[:,idx]*np.exp(-1j*clock_offset_radians.astype(float))
    
    return complex_vis

if __name__ == '__main__':
    parser = ag.ArgumentParser()
    add_args_to_parser(parser)    
    args = parser.parse_args()
    filename = args.filename
    fileout = args.fileout
    clockfile = args.pppclockfile
    ant1 = args.ant1
    ant2 = args.ant2
    clock_corr = ClockCorrect(filename, fileout, clockfile, ant1, ant2)
    clock_corr.openClockFile()
    clock_corr.processClockFile()
    clock_corr.saveClockFile()

