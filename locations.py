#!/usr/bin/python
"""
Module which provides the standard set of locations for VLBI-like processing


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

from astropy import coordinates
from astropy.time import Time

class receiver(object):
    """ Class for a ground receiver at a specfic location computed at a specific
    date.
    """
    def __init__(self, name, location, date, units = 'meter', ant_type='low_gain'):
        """ Basic class representing a receiver.  For now this is very basic.
        The time stamp date_location_determined is currently a string and for informational
        purposes only.
        
            Args:
                name (str) - Name of the site
                location (3 element tuple, list of array) - The coordinates (xyz) 
                      of the site
                date - The date the coordinates were determined as a string YYYYMMDD
            
            Keywords:
                units - the units of the coordinates - Default is meters
                ant_type - 'dish' or 'low_gain'
            
            Returns:
                None
        """
        
        if len(name) == 0:
            raise ValueError('receiver object must have a valid name.  Blank string provided')
        else:
            self.name = name
            
        location = [float(loc) for loc in location]
        
        if len(location) != 3:
            raise ValueError('receiver object must have 3 coordinates.  {0} provided'.format(location))
        
        self.x = location[0]
        self.y = location[1]
        self.z = location[2]
        
        self.date_location_determined = date
        self.location_units = units
        self.ant_type = ant_type
        
    @property
    def astropy_rep(self):
        """ Get the astropy representation of coordinates 
        
            Args:
                None
                
            Returns:
                Astropy Earth Location object
        """
        return coordinates.EarthLocation.from_geocentric(self.x,self.y,self.z,self.location_units)
        

class receive_sites(object):
    """ Class responsible for registering and providing the receive sites"""
    def __init__(self):
        """ Initialization with know sites."""
        
        # Self.__locations in xyz form.  These are not what is returned.
        # Default units are in meters
        self.__locations = {
                            "vlba": receiver("VLBA",  [-1324095.2765, -5332177.8520, 3231908.2957], '20220101', ant_type='dish'),
                            "mgo": receiver("MGO",  [-1330763.3581, -5328087.3443, 3236455.1778], '20220101'),
                            "ft_davis_gnss": receiver("FT_DAVIS",  [-1324095.2765, -5332177.8520, 3231908.2957], '20220101')
                            }
        
    @property
    def site_names(self):
        """ List of all registered site names"""
        return [self.__locations[site].name for site in self.__locations]
    
    @property
    def site_keys(self):
        """ List of all registered keys for site names. Note the keys do not
        have to match the names
        """
        return [key for key in self.__locations]
    
    @property
    def ltts(self):
        """ Lake Travis Test Station site"""
        return self.__locations["ltts"].astropy_rep
    
    @property
    def building_187(self):
        """ RF Building site"""
        return self.__locations["187"].astropy_rep
    
    @property
    def dish(self):
        """ Dish location"""
        return self.__locations["dish"].astropy_rep
    
    def get_location(self, key):
        """ Gets the location for a given key."""
        try:
            rval = self.__locations[key].astropy_rep
        except KeyError:
            err_msg = 'Invalid key {0} used.  This is not in the list of \
                       locations: {1}.'.format(key, list(self.__locations.keys()))
            raise KeyError(err_msg)
        
        return  rval
    
    #add location to the dictionary
    def add_location(self, key, location, date, ant_type = 'low_gain', units = 'meter'):
        try:
            self.__locations[key] = receiver(key, location, date, ant_type = ant_type, units = units)   
        except Exception as e:
            raise e
            
        return self

class celestial_sources(object):
    """ Class responsible for registering and providing the standard celestial
        sources.
    """
    
    def __init__(self):
        """ Initialization with know sources."""
        
        # Self.__locations in J2000 RA/Dec form.  Many of these can be found
        # in the paper by Fey et al, Astronomical Journal 127:3587-3608 2004 June
        self.__sources = dict()
        self.__sources['Cyg_A'] = ["19h59m28.35663s", "40d44m02.0970s"]
        self.__sources['M87'] = ["12h30m49.42s","+12d23m28.0s"] # AKA Virgo
        self.__sources['Cas_A'] = ["23h23m24.00s", "+58d48m54.0s"] 
        self.__sources['3C273'] = ["12h29m06.69s", "+02d03m08.6s"] # Quasar
        self.__sources['Tycho'] = ["00h25m21.5s",  "+64d08m27s"]
        self.__sources['Centaurus_A'] = ["13h25m27.61507s", "-43d01m08.8053s"]
        
        # Set the default time for locating sources
        self.default_time = Time.now()
    
    @property
    def source_names(self):
        """ List of all registered site names"""
        return [src for src in self.__sources]
    
    def get_astropy_rep(self, ra, dec, frame = 'icrs', obstime='J2000'):
        """ Get the astropy representation of coordinates given as xyz triplet
        
            Args:
                ra - right ascension in hms
                dec = dec in units as dms
                
            Keywords: 
                frame - default frame (icrs)
                obstime - the observation time (default is J2000)
                
            Returns:
                Astropy Earth Location object
        """
        return coordinates.SkyCoord(ra, dec, frame=frame, obstime=obstime)
    
    @property
    def cyga(self):
        """ Cygnus A"""
        return self.get_astropy_rep(self.__sources['Cyg_A'][0], self.__sources['Cyg_A'][1])
    
    @property
    def casa(self):
        """ Cassiopeia A"""
        return self.get_astropy_rep(self.__sources['Cas_A'][0], self.__sources['Cas_A'][1])
    
    @property
    def m87(self):
        """ M87"""
        return self.get_astropy_rep(self.__sources['M87'][0], self.__sources['M87'][1])
     
    def get_location(self, key):
        """ Gets the location for a given key."""
        try:
            rval = self.get_astropy_rep(self.__sources[key][0], self.__sources[key][1])
        except KeyError:
            err_msg = 'Invalid key {0} used.  This is not in the list of \
                       sources.'.format(key)
            raise KeyError(err_msg)
        
        return  rval
    
    #add location to the dictionary
    def add_location(self, key, location):
        """ Add a celestial source.
        
        Args:
            key (string) - Name of the source
            location (tuple or list) - RA and Dec of the source.  
        """
        try:
            self.__sources[key] = [j for j in location]    
        except Exception as e:
            raise e
            
        return self
    
def check_objects(obj1, obj2):
    """ A test function to check objects at a deeper level"""
    for k in obj1.__dict__:
        if obj1.__dict__[k] != obj2.__dict__[k]:
            print('Key {0} mismatch'.format(k))
            print('Types: {0}, {1}'.format(type(obj1.__dict__[k]), type(obj2.__dict__[k])))
            for ky in obj1.__dict__[k].__dict__:
                if obj1.__dict__[k].__dict__[ky] != obj2.__dict__[k].__dict__[ky]:
                    print('Key {0} mismatch'.format(ky))
                    print(obj1.__dict__[k].__dict__[ky], 
                          obj2.__dict__[k].__dict__[ky])
            
    return
    
