# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division
try: # pragma: no cover
    from cStringIO import StringIO
except: # pragma: no cover
    from io import BytesIO as StringIO
import os
import gzip

import numpy as np
from fluids.core import F2K
from scipy.constants import mile, knot, inch
from scipy.spatial import KDTree, cKDTree
from scipy.stats import scoreatpercentile

#F2K = lambda F : convert_temperature(F, 'f', 'k')


try: # pragma: no cover
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError: # pragma: no cover
    from urllib2 import urlopen
    from urllib2 import HTTPError

folder = os.path.join(os.path.dirname(__file__), 'data')


def get_clean_isd_history(dest=os.path.join(folder, 'isd-history-cleaned.tsv'),
                          url="ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv"): # pragma: no cover
    '''Basic method to update the isd-history file from the NOAA. This is 
    useful as new weather stations are updated all the time.
    
    This function requires pandas to run. If fluids is installed for the 
    superuser, this method must be called in an instance of Python running
    as the superuser (administrator).
    
    Retrieving the file from ftp typically takes several seconds.
    Pandas reads the file in ~30 ms and writes it in ~220 ms. Reading it with 
    the code below takes ~220 ms but is necessary to prevent a pandas 
    dependency.
    
    Parameters
    ----------
    dest : str, optional
        The file to store the data retrieved; leave as the default argument
        for it to be accessible by fluids.
    url : str, optional
        The location of the data file; this can be anywhere that can be read
        by pandas, including a local file as would be useful in an offline
        situation.
    '''
    import pandas as pd
    df = pd.read_csv(url)
    df.to_csv(dest, sep='\t', index=False, header=False)


class IntegratedSurfaceDatabaseStation(object):
    '''Class to hold data on a weather station in the Integrated Surface
    Database.
    
    License information for the database can be found at the following link:
    https://data.noaa.gov/dataset/global-surface-summary-of-the-day-gsod

    Parameters
    ----------
    USAF : int or None if unassigned
        Air Force station ID. May contain a letter in the first position.
    WBAN : int or None if unassigned
        NCDC WBAN number
    NAME : str
        Name of the station; ex. 'CENTRAL COLORADO REGIONAL AP'
    CTRY : str or None if unspecified
        FIPS country ID
    ST : str or None if not in the US
        State for US stations
    ICAO : str or None if not an airport
        ICAO airport code
    LAT : float
        Latitude with a precision of one thousandths of a decimal degree, 
        [degrees]
    LON : float
        Longitude with a precision of one thousandths of a decimal degree, 
        [degrees]
    ELEV : float
        Elevation of weather station, [m]
    BEGIN : float
        Beginning Period Of Record (YYYYMMDD). There may be reporting gaps 
        within the P.O.R.
    END : Ending Period Of Record (YYYYMMDD). There may be reporting gaps
        within the P.O.R.
    '''
    __slots__ = ['USAF', 'WBAN', 'NAME', 'CTRY', 'ST', 'ICAO', 'LAT', 'LON',
                 'ELEV', 'BEGIN', 'END']
    
    def __repr__(self):
        s = ('<Weather station registered in the Integrated Surface Database, '
            'name %s, country %s, USAF %s, WBAN %s, coords (%s, %s) '
            'Weather data from %s to %s>' )
        return s%(self.NAME, self.CTRY, self.USAF, self.WBAN, self.LAT, self.LON, str(self.BEGIN)[0:4], str(self.END)[0:4])
    
    def __init__(self, USAF, WBAN, NAME, CTRY, ST, ICAO, LAT, LON, ELEV, BEGIN,
                 END):
        self.USAF = USAF
        self.WBAN = WBAN
        self.NAME = NAME
        self.CTRY = CTRY
        self.ST = ST
        self.ICAO = ICAO
        self.LAT = LAT
        self.LON = LON
        self.ELEV = ELEV
        self.BEGIN = BEGIN
        self.END = END


stations = []
_latlongs = []
'''Read in the parsed data into 
1) a list of latitudes and longitudes, temporary, which will get converted to
a numpy array for use in KDTree
2) a list of IntegratedSurfaceDatabaseStation objects; the query will return
the index of the nearest weather stations.
'''
with open(os.path.join(folder, 'isd-history-cleaned.tsv')) as f:
    for line in f:
        values = line.split('\t')
        for i in range(0, 11):
            v = values[i]
            if not v:
                values[i] = None # '' case
            else:
                try:
                    values[i] = float(v)
                    if int(v) == 99999:
                        values[i] = None
                except:
                    continue
        lat, lon = values[6], values[7]
        if lat and lon:
            # Some stations have no lat-long; this isn't useful
            stations.append(IntegratedSurfaceDatabaseStation(*values))
            _latlongs.append((lat, lon))
_latlongs = np.array(_latlongs)
station_count = len(stations)


kd_tree = cKDTree(_latlongs) # _latlongs must be unchanged as data is not copied


def get_closest_station(latitude, longitude, minumum_recent_data=20140000, 
                        match_max=100):
    '''Query function to find the nearest weather station to a particular 
    set of coordinates. Optionally allows for a recent date by which the 
    station is required to be still active at.
    
    Parameters
    ----------
    latitude : float
        Latitude to search for nearby weather stations at, [degrees]
    longitude : float
        Longitude to search for nearby weather stations at, [degrees]
    minumum_recent_data : int, optional
        Date that the weather station is required to have more recent
        weather data than; format YYYYMMDD; set this to 0 to not restrict data
        by date.
    match_max : int, optional
        The maximum number of results in the KDTree to search for before 
        applying the filtering criteria; an internal parameter which is
        increased automatically if the default value is insufficient [-]
        
    Returns
    -------
    station : IntegratedSurfaceDatabaseStation
        Instance of IntegratedSurfaceDatabaseStation which was nearest
        to the requested coordinates and with sufficiently recent data
        available [-]
        
    Notes
    -----
    Searching for 100 stations is a reasonable choice as it takes, ~70 
    microseconds vs 50 microsecond to find only 1 station. The search does get 
    slower as more points are requested. Bad data is returned from a KDTree
    search if more points are requested than are available.
    
    Examples
    --------
    >>> get_closest_station(51.02532675, -114.049868485806, 20150000)
    <Weather station registered in the Integrated Surface Database, name CALGARY INTL CS, country CA, USAF 713930.0, WBAN None, coords (51.1, -114.0) Weather data from 2004 to 2017>
    '''
    # Both station strings may be important
    # Searching for 100 stations is fine, 70 microseconds vs 50 microsecond for 1
    # but there's little point for more points, it gets slower.
    # bad data is returned if k > station_count
    distances, indexes = kd_tree.query([latitude, longitude], k=min(match_max, station_count)) 
    #
    for i in indexes:
        latlon = _latlongs[i]
        enddate = stations[i].END
        # Iterate for all indexes until one is found whose date is current
        if enddate > minumum_recent_data:
            return stations[i]
    if match_max < station_count:
        return get_closest_station(latitude, longitude, minumum_recent_data=minumum_recent_data, match_max=match_max*10)
    raise Exception('Could not find a station with more recent data than '
                    'specified near the specified coordinates.')


# This should be agressively cached
def get_station_year_text(station, year):
    '''Basic method to download data from the GSOD database, given a 
    station idenfifier and year. 

    Parameters
    ----------
    station : str
        Station format, as a string, "USAF-WBAN", [-]
    year : int
        Year data should be retrieved from, [year]
        
    Returns
    -------
    data : str
        Downloaded data file
    '''
    toget = ('ftp://ftp.ncdc.noaa.gov/pub/data/gsod/' + str(year) + '/' 
             + station + '-' + str(year) +'.op.gz')
    try:
        data = urlopen(toget)
    except Exception as e:
        raise Exception('Could not obtain desired data; check '
                        'if the year has data published for the '
                        'specified station and the station was specified '
                        'in the correct form. The full error is %s' %(e))
        
    data = data.read()
    data_thing = StringIO(data)

    with gzip.GzipFile(fileobj=data_thing, mode="r") as f:
        year_station_data = f.read()
        return year_station_data
    


