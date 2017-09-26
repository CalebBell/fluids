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
try:
    from cStringIO import StringIO
except:
    from io import BytesIO as StringIO
import os
import gzip

import numpy as np
from scipy.constants import convert_temperature, mile, knot, inch
from scipy.spatial import KDTree, cKDTree
from scipy.stats import scoreatpercentile

F2K = lambda F : convert_temperature(F, 'f', 'k')


try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    from urllib2 import urlopen
    from urllib2 import HTTPError

folder = os.path.join(os.path.dirname(__file__), 'data')


def get_clean_isd_history(dest=os.path.join(folder, 'isd-history-cleaned.tsv'), url="ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv"):
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

with open(os.path.join(folder, 'isd-history-cleaned.tsv'), encoding='utf-8') as f:
    for line in f:
        values = line.split('\t')
        for i in range(0, 11):
            v = values[i]
            if not v:
                values[i] = None # '' case
            else:
                try:
                    values[i] = float(v)
                    if v == 99999:
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
# Only end date is being used






def get_closest_station(coords, minumum_recent_data=20140000, match_max=100):
    # Both station strings may be important
    # Searching for 100 stations is fine, 70 microseconds vs 50 microsecond for 1
    # but there's little point for more points, it gets slower.
    # bad data is returned if k > station_count
    distances, indexes = kd_tree.query(coords, k=min(match_max, station_count)) 
    #
    for i in indexes:
        latlon = _latlongs[i]
        enddate = stations[i].END
        # Iterate for all indexes until one is found whose date is current
        if enddate > minumum_recent_data:
            return stations[i]
    if match_max < station_count:
        return get_closest_station(coords, minumum_recent_data=minumum_recent_data, match_max=match_max*10)
    raise Exception('Could not find a station with more recent data than '
                    'specified near the specified coordinates.')


#s = get_closest_station([51.02532675, -114.049868485806], 20150000)
#print([getattr(s, i) for i in s.__slots__])
#print(stations.index(s))

def get_station_year_text(station, year):
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
    
# test case for assert
#get_station_year_text('712650-99999', 1999)


