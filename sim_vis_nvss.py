#!/usr/bin/python
"""
Simulate radioastronomical 'visibilities' for a specified observation scenario using the NVSS catalog.
                      Generate several figures showing the components of the source on the sky and the flux density in different scenarios

Usage:

  python sim_vis_nvss.py --rxname1 FD_VLBA --rxname2 $DEVICE --rxpos1 "$RXPOS1" --rxpos2 "$RXPOS2" --time $TIME --catalog $CATALOG --centerFreqHz 1440e6 --bandwidth 72e6 --genfringe --scanLen $SCANLEN 
  --ra $RIGHT_ASC --dec $DEC --Npt $NPT --source $SOURCE --searchRad 50

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

from astropy.io import fits
from astropy import wcs, coordinates, time, units, constants
from astropy.utils import iers
import locations as loc
from os import path
import re
import numpy as np
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp2d
from scipy.spatial import KDTree
from collections import namedtuple
from math import pow
from math import log as logn
import utilities
import scipy.constants as const
import logging
import argparse
color_map = 'plasma'
# Note: The following file is included as an example.  It should be updated occassionally.
leap_sec_path = path.join('.','Leap_Second.dat')

# The IERS A table
iers_a_file = path.join('.','finals2000A.all')

log = logging.getLogger(__name__)

def set_astropy_config():
   """ Function to set astropy configuration for leap seconds, IERS table A, and downloads."""
   # Set astropy to not try to download files (with resulting error messages)
   iers.conf.auto_download = False

   # Load the files from the file paths above - this is a good file as of 04/22
   log.info("Loading leap seconds file")
   iers.LeapSeconds.from_iers_leap_seconds(leap_sec_path)
   log.info("Loading IERS A file")
   iers_a = iers.IERS_A.open(iers_a_file)
   iers.earth_orientation_table.set(iers_a)
    
def add_args_to_parser(parser_in):
    """Add arguments to parser."""

    parser.add_argument("--rxname1", help="Receiver name 1 Default: %default", default='dish')
    parser.add_argument("--rxname2", 
                        help="Receiver name 2 Default: %default", 
                        default='ltts')
    parser.add_argument("--rxpos1", 
                         help="Receiver position 1 if not known by name in locations module (as 'X Y Z') Default: %default", 
                         default='')
    parser.add_argument("--rxpos2", 
                        help="Receiver position 2 if not known by name in locations module (as 'X Y Z') Default: %default", 
                        default='')
    parser.add_argument("--centerFreqHz", 
                        help="Assume center frequency in Hz Default: %default", 
                        default=1440.0e6, 
                        type=float, 
                        metavar="HZ")
    parser.add_argument("--bandwidth", 
                        help="Bandwidth to be sampled in Hz: %default", 
                        default=20.0e6, 
                        type=float, 
                        metavar="BW")
    parser.add_argument("--searchRad", 
                        help="Search radius from RA, DEC in catalog (arcmin): %default", 
                        default=1.0, 
                        type=float, 
                        metavar="BW")
    parser.add_argument("--specRes", 
                        help="Spectral resolution (Hz): %default", 
                        default=0.1e6, 
                        type=float, 
                        metavar="BW")   
    parser.add_argument("--freqSample", 
                        help="Sample interval in Hz over the bandwidth: %default", 
                        default=1.0e5, 
                        type=float, 
                        metavar="FS")
    parser.add_argument("--time", 
                        help="Simulate at specified time (format: 2018-03-05T17:31:18) (may be specified multiple times)", 
                        default=[], 
                        action='append')
    parser.add_argument("--scanLen",
                        help="Specify a scan length in seconds, if this is given, --time should be a single value."\
                             +"A plot of flux vs. time will be produced if this option is given.",
                        default=0.0,
                        type=float,
                        metavar="SL")
    parser.add_argument("--source", 
                        help="Third catalogue (3c) name of source e.g. 3c123. Scale flux in image by Perley (2016) flux calibrator scale for given centerFreqHz.",
                        default = '', 
                        type=str
                        )  
    parser.add_argument('--ra',
                        type=float,
                        help="Specify right ascension of dish pointing vector/fringe stop location. [Degrees]")
    parser.add_argument('--dec',
                        type=float,
                        help="Specify declination of dish pointing vector/fringe stop location. [Degrees]")
    parser.add_argument("--Npt", 
                        help="Number of points to do simulations with (integer). A good starting point would be at least scanLen (1 point per second)."\
                        +" This strongly affects run time", 
                        default = 200, 
                        type=int
                        )  
    parser.add_argument("--genfringe", 
                        dest='genfringe',
                        help="Generate interferometer pattern projected over source area", 
                        default = False, 
                        action = 'store_true'
                        )
    parser.add_argument("--gainPattern", 
                        dest='gainPattern',
                        help="Generate gain pattern for Topcon CR-G5 labeled with source elevation", 
                        default = False, 
                        action = 'store_true'
                        ) 
    parser.add_argument("--debug", 
                        dest='debug',
                        help="Set this to generate debug information", 
                        default = False, 
                        action = 'store_true'
                        )                    
       
    srcGrp = parser.add_argument_group("Source descriptions")
    srcGrp.add_argument("--catalog", 
                        help="Load NVSS catalog of point sources, and simulate",                        
                        default=[], 
                        metavar="FILENAME", 
                        action='append')
    
def parseRXPosHelper(name, pos, ant_type = 'low_gain', date = ''):
    """ Function to return common ground site coordinates
        This simply invokes the locations module.  If the site is not in 
        the structure of known sites, it is added to the structure.
        
        Args:
            name (str) - The key for the site as a string,  Key matches 
            what the locations module knows
            pos (str) - A space delimited set of 'X Y Z' values in meters.
            May be empty if site "name" is known to the locations module.
            If site is known, pos will be ignored.
            
        Keywords:
            ant_type (str) - one of 'low_gain' or 'dish'
            date (str) - Date of site occupation 
        
        Returns
            astropy.coordinates.EarthLocation (object)
    """
    
    rx = loc.receive_sites()
    
    # If we don't recognize the name of the site - add it
    if name not in rx.site_keys:
        try:
            location = list(map(float, pos.split()))
        except ValueError:
            err_msg = 'Position string may not have enough values to unpack: {0}'.format(pos)
            raise ValueError(err_msg)
            
        rx.add_location(name, location, date, ant_type = ant_type)
        log.debug("Site name {0} not recognized. Position {1} will be added to receive sites".format(name, pos))
     
    return rx.get_location(name)
    

def doVisSimulation(tim, srcs, rxpos1, rxpos2, freqHz, right_asc=False, declination=False, N_pt=200, specRes=0, scanLen=0):
    """
    Simulate complex visibility

    Arguments:
        tim: time of simulation (astropy.time.Time instance)
        srcs: a numpy recarray of sources to simulate (with fields 'ra', 'dec', 'flux')
        rxpos1: receiver position 1 (astropy.coordinates.EarthLocation instance)
        rxpos2: receiver position 2 (astropy.coordinates.EarthLocation instance)
        freqHz: center frequency in Hz
        right_asc: right ascension to fringe stop on (degrees)
        declination: declination to fringe stop on (degrees)
        specRes: the spectral resolution of the channelization (Hz)
        scanLen: the integration time on the source (sec)
        N_pt: the number of points to simulate with

        Arguments may be scalar or 1-D numpy arrays

    Returns:
        complexVis: Complex-valued visibility array with shape (len(tim),len(srcs),len(rxpos1),len(rxpos2), len(centerFreqHz))
    """
    # Convert ra/dec points to ECEF unit vectors
    sc = coordinates.SkyCoord(srcs['ra'], srcs['dec'], frame='icrs', unit='deg')
    srcUnitVectorsECEF = sc[np.newaxis, ...].transform_to(coordinates.ITRS(obstime=tim[..., np.newaxis]))

    # Identify baselines
    baselineVectorECEF = rxpos1.get_itrs().cartesian[..., np.newaxis] - rxpos2.get_itrs().cartesian[np.newaxis, ...]
    # Project points onto baseline
    projVectorsMeters = srcUnitVectorsECEF[..., np.newaxis, np.newaxis].cartesian.dot(baselineVectorECEF[np.newaxis, np.newaxis, ...])

    # Compute correlation contribution for individual sources
    projVectorPhase = projVectorsMeters[..., np.newaxis]*2*np.pi*(freqHz * units.Hz)/constants.c

    complexVis =  np.exp(1j*projVectorPhase) * srcs['flux'][np.newaxis, ..., np.newaxis, np.newaxis, np.newaxis]

    if right_asc and declination:
        fringeStopCoord = coordinates.SkyCoord(right_asc, declination, frame='icrs', unit='deg')
        fringeStopUnit = fringeStopCoord[np.newaxis, ...].transform_to(coordinates.ITRS(obstime=tim[..., np.newaxis]))
        fringeStopProj = fringeStopUnit.cartesian.dot(baselineVectorECEF)
        fringeStopPhase = fringeStopProj*2*np.pi*(freqHz * units.Hz)/constants.c
        fringeStopVis = np.exp(1j*fringeStopPhase)

        if len(complexVis[0,0,0,:,0]) > 1:
            for i in np.arange(fringeStopVis.shape[0]): # number of times
                for j in np.arange(fringeStopVis.shape[1]): # number of baselines
                    complexVis[i,:,0,j,0] = complexVis[i,:,0,j,0]/fringeStopVis[i,j]
                    
                    # after fringe stopping, do smearing--see VLBI Imaging, Morgan et al. 2010 for details
                    if scanLen > 0: # do (cheap) time smearing - approximate dtau/dt as a line between 2 pts.
                        phase_del_rate_diff = findPDRDiff(tim, rxpos1, rxpos2[j], sc, fringeStopCoord, scanLen, N_pt)
                        # numpy sinc is a normalized sinc function, therefore remove pi
                        complexVis[i,:,0,j,0] = complexVis[i,:,0,j,0]*np.sinc(phase_del_rate_diff*scanLen)
                    
                    if specRes > 0: # do frequency smearing
                        fringeStopDelay = fringeStopProj[i,j]/constants.c
                        delay = projVectorsMeters[i,:,0,j]/constants.c
                        diff_delay = delay-fringeStopDelay
                        complexVis[i,:,0,j,0] = complexVis[i,:,0,j,0]*(1-diff_delay.value*specRes)
        else:
            complexVis = complexVis/fringeStopVis
            if specRes > 0: # do frequency smearing
                    fringeStopDelay = fringeStopProj/constants.c
                    delay = projVectorsMeters[0,:,0,0]/constants.c
                    diff_delay = delay-fringeStopDelay
                    complexVis[0,:,0,0,0] = complexVis[0,:,0,0,0]*(1-diff_delay.value*specRes)
    return complexVis


def findPDRDiff(tim, rxpos1, rxpos2,  skyCoord, fringeStopCoord, scanLen, N_pt):
    """
    Find average phase delay rate difference from antenna positions, pointing direction, scan length

    Arguments:
        tim: time of simulation (astropy.time.Time instance)
        rxpos1: receiver position 1 (astropy.coordinates.EarthLocation instance)
        rxpos2: receiver position 2 (astropy.coordinates.EarthLocation instance)
        freqHz: center frequency in Hz
        skyCoord: astropy sky coordinate instance holding right ascension and declination values
        fringeStopCoord: astropy sky coordinate instance holding the fringe stopped RA & Dec
        scanLen: the integration time on the source (sec)
        N_pt: number of points in the simulation

    Returns:
        phaseDelRateDiff: average difference in phase delay rate between pixel and fringe stop location, sec/sec
    """
    # Identify baselines
    baselineVectorECEF = rxpos1.get_itrs().cartesian[..., np.newaxis] - rxpos2.get_itrs().cartesian[np.newaxis, ...]

    tVecSeconds = np.linspace(0, scanLen, N_pt)
    tVec = time.Time(times[0].mjd+tVecSeconds/86400,format='mjd',scale='utc')

    fringeStopUnit = fringeStopCoord[np.newaxis, ...].transform_to(coordinates.ITRS(obstime=tVec[..., np.newaxis]))
    skyUnit = skyCoord[np.newaxis, ...].transform_to(coordinates.ITRS(obstime=tVec[..., np.newaxis]))
    delSky = skyUnit.cartesian.dot(baselineVectorECEF)/constants.c # [time,srcs]
    delFringeStop = fringeStopUnit.cartesian.dot(baselineVectorECEF)/constants.c # [time,1]
    delArr = delSky-delFringeStop # [time,srcs]
    phaseDelRateDiff = np.sum(np.diff(delArr,axis=0)/np.diff(tVecSeconds*units.s)[:,np.newaxis],axis=0)/(len(tVec)-1)
    return phaseDelRateDiff*units.rad

def findEllipseAvgRMS(tim, srcModel, rxpos1, rxpos2, wavelength):
    """
    Find average and RMS correlated flux density by processing an ellipse in u,v space
    l,m -- direction cosines in the plane tangent to the core of the source
    u,v -- spatial frequencies derived from the geometry of the baseline coordinates
    Arguments:
        time: time of simulation (astropy.time.Time instance)
        srcModel: Numpy rec array with RA, DEC, Flux Density
        rxpos1: receiver position 1 (astropy.coordinates.EarthLocation instance)
        rxpos2: receiver position 2 (astropy.coordinates.EarthLocation instance)
        wavelength: wavelength of center frequency in m
    Returns:
        None
    """
    t_vec_seconds = np.linspace(0,86164.0905,1000) # sidereal day in seconds 
    t_vec = time.Time(tim.mjd+t_vec_seconds/86400,format='mjd',scale='utc')
    rx1_gcrs = rxpos1.get_gcrs(obstime=t_vec).cartesian
    rx2_gcrs = rxpos2.get_gcrs(obstime=t_vec).cartesian
    baselineVectorGCRF = (rx1_gcrs.get_xyz()-rx2_gcrs.get_xyz())/wavelength
    
    
    sc = coordinates.SkyCoord(srcModel['ra'], srcModel['dec'], frame='icrs', unit='deg')
    srcUnitVectors = sc.cartesian.get_xyz() # inertial XYZ coords @ solar system barycenter
    
    # find the maximum flux density component of the source -- this will be l=0, m=0
    idx_max = np.argmax(srcModel['flux'])
    right_asc = srcModel['ra'][idx_max]
    declination = srcModel['dec'][idx_max]
   
    ra = -right_asc*np.pi/180
    dec = declination*np.pi/180
    src_vec_max = srcUnitVectors[:,idx_max] # unit vector of max. flux dens. source

    # compute the spatial frequencies u,v
    u_vec = np.zeros(len(baselineVectorGCRF[0,:]))
    v_vec = np.zeros(len(baselineVectorGCRF[0,:]))
    for idx in range(len(baselineVectorGCRF[0,:])):
        baseline_vec = baselineVectorGCRF[:,idx]
        #rot_matrix = np.matrix([[np.sin(ra), np.cos(ra), 0],\
        #                        [-np.sin(dec)*cos(ra), sin(dec)*sin(ra), cos(dec)],\
        #                        [cos(dec)*cos(ra), -cos(dec)*sin(ra), sin(dec)]])
        u = np.sin(ra)*baseline_vec[0] + np.cos(ra)*baseline_vec[1]
        v = -np.sin(dec)*np.cos(ra)*baseline_vec[0] + np.sin(dec)*np.sin(ra)*baseline_vec[1] + np.cos(dec)*baseline_vec[2]
        #w = np.cos(dec)*np.cos(ra)*baseline_vec[0] - np.cos(dec)*np.sin(ra)*baseline_vec[1] + np.sin(dec)*baseline_vec[2]
        u_vec[idx] = u.value
        v_vec[idx] = v.value
    
    # compute the direction cosines l, m for each source component
    l_vec = np.zeros(len(srcModel['ra']))
    m_vec = np.zeros(len(srcModel['ra']))
    for jdx in range(len(srcModel['ra'])):
        src_vec = srcUnitVectors[:,jdx]
        l = np.sin(ra)*src_vec[0] + np.cos(ra)*src_vec[1] 
        m = -np.sin(dec)*np.cos(ra)*src_vec[0] + np.sin(dec)*np.sin(ra)*src_vec[1] +\
             np.cos(dec)*src_vec[2] 
        l_vec[jdx] = l.value
        m_vec[jdx] = m.value
    
    my_fig = pl.figure(figsize=(8,8))
    my_ax = my_fig.add_subplot(111)
    sc_im = my_ax.scatter(l_vec, m_vec, c = srcModel['flux'], cmap = color_map)
    my_fig.colorbar(sc_im, label="Flux Density (Jy)", ax = my_ax)
    my_ax.set_xlabel('l')
    my_ax.set_ylabel('m')
    my_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_lm.png')
    
    # perform the discrete Fourier transform, accounting for each source component at each l,m for each u,v
    flux_uv = np.zeros(len(u_vec))
    for idx in range(len(u_vec)):
        u = u_vec[idx]
        v = v_vec[idx]
        f = 0 + 1j*0
        for jdx in range(len(l_vec)): # discrete Fourier transform
            flux_dens = srcModel['flux'][jdx]
            l = l_vec[jdx]
            m = m_vec[jdx]
            f = f + flux_dens*np.exp(-1j*2*np.pi*(u*l+v*m))
        flux_uv[idx] = np.abs(f)

    # generate UV ellipse figure
    my_fig = pl.figure(figsize=(8,8))
    my_ax = my_fig.add_subplot(111)
    sc_im = my_ax.scatter(u_vec, v_vec, c = flux_uv, cmap = color_map)
    my_fig.colorbar(sc_im, label="Flux Density (Jy)", ax = my_ax)
    xlim = my_ax.get_xlim()
    ylim = my_ax.get_ylim()
    my_ax.set_xlabel('u')
    my_ax.set_ylabel('v')
    my_ax.set_title('Ellipse avg. flux: ' + str(np.round(np.mean(flux_uv),3)) + '\n' + 'Ellipse RMS flux: ' + str(np.round(np.std(flux_uv),3)))
    my_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_uv.png')
    print('Ellipse avg. flux: ' + str(np.mean(flux_uv)))
    print('Ellipse RMS flux: ' + str(np.std(flux_uv)))


    uv_fig = pl.figure(figsize=(16,8))
    radec_ax = uv_fig.add_subplot(121)
    rd_im = radec_ax.scatter((srcModel['ra']-right_asc)*60, (srcModel['dec']-declination)*60, c = srcModel['flux'], cmap = color_map, norm=colors.Normalize(vmin=0.1, vmax=np.amax(flux_uv))) 
    radec_ax.set_xlabel('Right Ascension (arcmin)', fontsize = 25)
    radec_ax.set_ylabel('Declination (arcmin)', fontsize = 25)
    radec_ax.tick_params(labelsize=20)
   
    uv_ax = uv_fig.add_subplot(122)
    uv_ax.tick_params(labelsize=20)
    uv_im = uv_ax.scatter(u_vec, v_vec, c = flux_uv, cmap = color_map, norm=colors.Normalize(vmin=0.1, vmax=np.amax(flux_uv)))
    cb = uv_fig.colorbar(uv_im, ax = uv_ax)
    cb.set_label(label="Flux Density (Jy)", size=28)
    cb.ax.tick_params(labelsize=20)
    uv_ax.set_xlabel(r'$u$ ($\lambda$)', fontsize = 28)
    uv_ax.set_ylabel(r'$v$ ($\lambda$)', fontsize = 28)   
    uv_fig.tight_layout()
    uv_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_uv_full.pdf')

    # save data
    # create a list of the arrays
    arr_list = [(srcModel['ra']-right_asc)*60, (srcModel['dec']-declination)*60, srcModel['flux'], u_vec, v_vec, flux_uv] 
    # determine the maximum length of the arrays
    max_len = max(len(arr) for arr in arr_list)

    # save the data
    arr_list = [(srcModel['ra']-right_asc)*60, (srcModel['dec']-declination)*60, srcModel['flux'], u_vec, v_vec, flux_uv]
    col_names = ['Right ascension (arcmin)', 'Declination (arcmin)', 'Delta Fcn Flux Density (Jy)', 'u (wavelength)', 'v (wavelength)', 'Flux Density (Jy)']
    outfile_name = 'ellipse_fig.csv'
    utilities.save_data(outfile_name, arr_list, col_names)

    return 


class SimCatalog(object):
    """ Support simulation of visibilities from a point source catalog  """
    def __init__(self, filename):
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
                    # if -00, need to preserve negative sign
                    # b/c float(-00) = 0
                   if words[3] == '-00' or float(words[3]) < 0: 
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
        self.flux_arr = flux_arr[idxs_good]
        self.RA_deg_arr = RA_deg_arr[idxs_good]
        self.DEC_deg_arr = DEC_deg_arr[idxs_good]
        print('Max flux in set: ' + str(np.amax(flux_arr)) + ' Jy')
        print('\nNumber of sources considered: ' + str(len(flux_arr)))
        source_pts = np.vstack((RA_deg_arr,DEC_deg_arr)).T
        self.sources_tree = KDTree(source_pts)

    def processTree(self, RA, DEC, search_rad):
        """ Process the KDTree, keeping points roughly within the VLBA field of view.
            Produce a list of RA and DEC for simulation
            Arguments:
                self.sources_tree: KDTree object containing right ascension and declination
                of NVSS point sources, contains query method to find nearest neighbors
                RA: queried right ascension, decimal degrees
                DEC: queried declination, decimal degrees
                search_rad: radius of search in NVSS catalog, degrees
            Returns:
                numpy recarray with right ascension, declination, and flux density of found components
        """

        idxs_ball = self.sources_tree.query_ball_point(np.array([RA,DEC]).T, search_rad, workers=-1) 
        ra_use = np.array([self.RA_deg_arr[idx_ball] for idx_ball in idxs_ball])
        dec_use = np.array([self.DEC_deg_arr[idx_ball]  for idx_ball in idxs_ball])
        fluxes = np.array([self.flux_arr[idx_ball] for idx_ball in idxs_ball])
        print('Total NVSS Flux: ' + str(np.sum(fluxes)) + ' Jy')
        return np.rec.fromarrays([ra_use, dec_use, fluxes], names='ra,dec,flux')
 
    @property
    def totalFlux(self):
        return np.sum(self.jy[self.jy >= self.threshold])

    def plot(self, myax):
        return myax.imshow(self.jy, cmap = color_map, aspect = 'auto', origin = 'lower')

if __name__ == '__main__':
        
    #Set the astropy configuration early
    set_astropy_config()

    ### Parse command-line options

    parser = argparse.ArgumentParser(description='Simulation of the visibility of a source.')
    add_args_to_parser(parser)
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)

    ### Pull command-line options into internal variables

    rxpos1 = parseRXPosHelper(args.rxname1, args.rxpos1)
    rxpos2 = parseRXPosHelper(args.rxname2, args.rxpos2)
    
    right_asc = args.ra
    declination = args.dec
    specRes = args.specRes
    scanLen = args.scanLen
    searchRad = args.searchRad/60 # convert to deg
    N_pt = args.Npt

    centerFreqHz = args.centerFreqHz
    bw = args.bandwidth
    freq_int = args.freqSample
    source_name = args.source
    
    times = [time.Time(timestr, format='isot', scale='utc') for timestr in args.time]
    baseline = rxpos2.get_itrs().cartesian-rxpos1.get_itrs().cartesian
    baseline = np.round(baseline.norm())

    # Set pylab interactive mode on
    pl.ion()
    
    ## Load any source catalogs
    log.debug('Loading catalog')
    for filename in  args.catalog:
        
        # Grab the filename off of the path
        fname = path.basename(filename)
        
        ## Load file, generate  source model
        img = SimCatalog(filename)

        srcModel = img.processTree(right_asc, declination, searchRad)
        if source_name is not '':
            img.description = source_name
        
        print('Processing for source ' + img.description)
        # Compute the bounds for viewable window based on srcModel
        
        # Plot the source model as a scatter plot  
        my_fig = pl.figure(figsize=(8,8))
        my_ax = my_fig.add_subplot(111)
        sc_im = my_ax.scatter(srcModel['ra'], srcModel['dec'], c = srcModel['flux'], cmap = color_map)
        my_fig.colorbar(sc_im, label="Flux Density (Jy/pixel)", ax = my_ax)
        xlim = my_ax.get_xlim()
        ylim = my_ax.get_ylim()
        my_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_source_cc.png')
        
        if args.genfringe is True:
            N = 100
            ra, dec = np.meshgrid(np.linspace(right_asc-searchRad,right_asc+searchRad,N), \
                    np.linspace(declination-searchRad,declination+searchRad,N) , indexing='ij')
            ra = ra.flatten()
            dec = dec.flatten()
            flux = np.ones(N**2)
            srcs = np.rec.fromarrays([ra, dec, flux], names='ra,dec,flux')
            vectorSum = doVisSimulation(times[0], srcs, rxpos1, rxpos2, centerFreqHz)
            amp = vectorSum[0,:,0,0,0]
            amp = amp.reshape([N,N])
            normalized_sum = np.abs(np.sum(amp))/np.sum(np.abs(amp))

            pattern_fig = pl.figure(figsize = (8,8))
            pattern_fig.suptitle('Source {0} \n Interferometer Pattern \n Normalized Sum {1:.3e}'.format(img.description,normalized_sum))
            
            pattern_ax1 = pattern_fig.add_subplot(211)
            sc_im1 = pattern_ax1.imshow(np.real(amp), cmap = color_map, aspect = 'auto', origin = 'lower',\
                     extent = (right_asc-searchRad,right_asc+searchRad,declination-searchRad,declination+searchRad))  
            pattern_ax1.set_ylabel('Dec (deg)')
            pattern_ax1.set_xlabel('RA (deg)')
            pattern_fig.colorbar(sc_im1, label="Complex visibility real amplitude", ax = pattern_ax1)

            pattern_ax2 = pattern_fig.add_subplot(212)
            sc_im2 = pattern_ax2.imshow(np.imag(amp), cmap = color_map, aspect = 'auto', origin = 'lower',\
                    extent = (right_asc-searchRad,right_asc+searchRad,declination-searchRad,declination+searchRad)) 
            pattern_ax2.set_ylabel('Dec (deg)')
            pattern_ax2.set_xlabel('RA (deg)')
            pattern_fig.colorbar(sc_im2, label="Complex visibility imaginary amplitude", ax = pattern_ax2)
            pattern_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_interferometer_pattern.png')

    srcs = srcModel
    if len(srcs) == 0:
        raise Exception("No sources specified!")

    ### High level sanity checks
    log.info("Total simulated point sources: {} points totalling {} Janskys ".format(len(srcs), sum(srcs['flux'])))
        
    ### Plot equiv flux as a function of time
    if scanLen > 0.0:
        if len(times) > 1: 
            raise Exception('If scan length is given, only one time can be given')
        wavelength = const.c/centerFreqHz
        findEllipseAvgRMS(times[0], srcModel, rxpos1, rxpos2, wavelength)
        t_vec_seconds = np.linspace(0, scanLen, N_pt)
        t_vec = time.Time(times[0].mjd+t_vec_seconds/86400,format='mjd',scale='utc')
        visSum_t = np.zeros(N_pt)
        
        if right_asc and declination: 
            visSmear = np.zeros(len(srcs['flux']))

        for idx, tim in enumerate(t_vec):
            visTime = doVisSimulation(tim, srcs, rxpos1, rxpos2, centerFreqHz, right_asc, declination, N_pt, specRes)[0,:,0,0,0]
            visSum_t[idx] = np.abs(np.sum(visTime, axis=0))
            
            if right_asc and declination: 
                visSmear = visSmear + visTime
        time_fig = pl.figure()
        time_ax = time_fig.add_subplot(111)
        time_ax.plot(t_vec_seconds, visSum_t)
        time_ax.set_title("Flux vs time")
        time_ax.set_ylabel("Flux Density (Jy)")
        time_ax.set_xlabel("Time")
        # note that this time dependence figure incorporates frequency smearing
        time_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_time_dependence.png')
        
        if right_asc and declination:
            # calculate the flux density loss due to time smearing and frequency smearing
            smearFluxDens = np.abs(np.sum(visSmear))/N_pt
            print('Flux density incorporating time and frequency smearing for given baseline:' + str(np.round(smearFluxDens,3)) + ' Jy')
            # print the altitude of the source in the sky--low altitude sources will lose SNR due to patch antenna
            fringeStopCoord = coordinates.SkyCoord(right_asc, declination, frame='icrs', unit='deg')
            altFringe = fringeStopCoord.transform_to(coordinates.AltAz(obstime=t_vec[len(t_vec)//2],location=rxpos2))
            print(img.description+" approximate altitude in sky: " +str(np.round(altFringe.alt.value,3))+" deg")
            
            if args.gainPattern:
                # plot the gain vs elevation figure indicating where the source is
                antenna_file = 'antenna_pattern.csv'
                interp_fcn = utilities.gen_gain_pattern(antenna_file)
                gain_fig = pl.figure()
                gain_ax = gain_fig.add_subplot(polar=True)
                elev_angle_plot = np.linspace(0,90,2*N_pt)
                gain_interp = interp_fcn(elev_angle_plot,centerFreqHz)
                gain_ax.plot(elev_angle_plot*np.pi/180,gain_interp)
                gain_ax.set_thetamin(0)
                gain_ax.set_thetamax(90)
                gain_elev = altFringe.alt.value
                gain_val = interp_fcn(gain_elev, centerFreqHz)
                gain_ax.plot(gain_elev*np.pi/180, gain_val, marker='X')
                gain_ax.set_title("Gain pattern for Topcon CR-G5 (dB)\n"+"Elev: "+str(np.round(gain_elev,1))\
                        +" deg, Freq: " + str(np.round(centerFreqHz/1e6,1)) +" MHz, Gain: "+str(np.round(gain_val[0],3)) + " dB")
                gain_fig.tight_layout()
                gain_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_gain_fig.png')


    ### Run simulation
    log.info('Running simulations')
    for tim in times:
        log.info('Simulation time: {0}'.format(tim))
        vectorSum = doVisSimulation(tim, srcs, rxpos1, rxpos2, centerFreqHz)

        # Find equivalent vector if all flux was at a single point source
        equivPointSrcSum = np.sum(srcs['flux'])

        # Find brightest simulated source for use below
        peakFluxSrc = srcs[[np.argmax(srcs['flux'])]]

        ### Plot equiv flux as function of offset frequency
        flux_fig = pl.figure('Flux vs Frequency')
        flux_ax = flux_fig.add_subplot(211)
        f = centerFreqHz + np.arange(-bw, bw, freq_int)
        visByFreq = np.sum(doVisSimulation(tim, srcs, rxpos1, rxpos2, f)[0,:,0,0,:], axis=0)
        flux_ax.plot(f/1e6, abs(visByFreq.flatten()), label=str(tim))
        flux_ax.set_title("Narrowband flux vs frequency")
        flux_ax.axvline(centerFreqHz/1e6, linestyle='--', color='black')
        flux_ax.set_ylabel("Flux Density (Jy)")
        flux_ax.grid(True)
        flux_ax.legend(loc=0)

        fringe_ax = flux_fig.add_subplot(212, sharex=flux_ax)
        refVisByFreq = np.sum(doVisSimulation(tim, peakFluxSrc, rxpos1, rxpos2, f)[0,:,0,0,:], axis=0) # Fringe stop on brightest source
        fringe_ax.plot((f)/1e6, np.unwrap(np.angle(visByFreq.flatten()/refVisByFreq.flatten())), label=str(tim))
        fringe_ax.set_title("Fringe stopped phase vs frequency")
        fringe_ax.set_xlabel("Frequency (MHz)")
        fringe_ax.set_ylabel("Phase offset (radians)")

        fringe_ax.grid(True)
        
        pl.subplots_adjust(hspace=0.4)
        
        flux_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_flux_v_freq.png')

        ### Plot equiv flux as function of baseline length
        
        baselineVectorECEF = rxpos2.get_itrs().cartesian-rxpos1.get_itrs().cartesian
        baselineVec = baselineVectorECEF / baselineVectorECEF.norm()
        baselineLength = np.logspace(np.log10(10), np.log10(10000), 2*N_pt)*units.m
        idx = np.argwhere(baselineLength>baseline)[0]
        baselineLength[idx] = baseline
        #baselineLength = np.concatenate((baselineLength1, baselineLength2[1:]))*units.m
        rxpos2s = rxpos1.get_itrs().cartesian + baselineVec*baselineLength
        rxpos2s = coordinates.EarthLocation(x=rxpos2s.x, y=rxpos2s.y, z=rxpos2s.z)

        flux_v_baseline_fig = pl.figure('Flux vs BaselineLength')
        fvb_ax = flux_v_baseline_fig.add_subplot(111)
        flux_baseline = abs(np.sum(doVisSimulation(tim, srcs, rxpos1, rxpos2s, centerFreqHz, right_asc, declination, N_pt, specRes, scanLen)[0,:,0,:,0],axis=0)).flatten()
        fvb_ax.semilogx(baselineLength, flux_baseline, label=str(tim))
        fvb_ax.set_xlabel("Baseline (meters)", fontsize = '14')
        fvb_ax.set_title("Flux vs baseline length \n Intercept "+str(np.round(flux_baseline[idx],3))+" Jy", fontsize = '14')
        fvb_ax.set_ylabel("Flux Density (Jy)", fontsize = '14')
        fvb_ax.axvline(baselineVectorECEF.norm().value, linestyle='--', color='black')
        fvb_ax.legend(loc=0)
        fvb_ax.grid(True)
        
        flux_v_baseline_fig.savefig(img.description + '_baseline_' + str(baseline).replace(' ', '_') + '_flux_v_baseline.png')
        
        ### Plot equiv flux as function of baseline azimuth about rxpos1
        
        eastVec = coordinates.CartesianRepresentation(-np.sin(np.radians(rxpos1.lon)), np.cos(np.radians(rxpos1.lon)), 0)
        northVec = rxpos1.get_itrs().cartesian.cross(eastVec)
        northVec /= northVec.norm()

        # # rxpos, east, north vectors should be pairwise orthogonal
        assert abs(eastVec.dot(rxpos1.get_itrs().cartesian)) < 1*units.m
        assert abs(northVec.dot(rxpos1.get_itrs().cartesian)) < 1*units.m
        assert abs(northVec.dot(eastVec)) < 1e-6

        # # north vector should point north
        assert northVec.z > 0

        # # north/east should be unit vectors
        assert abs(northVec.norm()-1) < 1e-6
        assert abs(eastVec.norm()-1) < 1e-6

