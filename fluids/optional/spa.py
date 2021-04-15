# -*- coding: utf-8 -*-
"""
irradiance.py from pvlib
========================
Stripped down, vendorized version from:
https://github.com/pvlib/pvlib-python/

Calculate the solar position using the NREL SPA algorithm either using
numpy arrays or compiling the code to machine language with numba.

The rational for not including this library as a strict dependency is to avoid
including a dependency on pandas, keeping load time low, and PyPy compatibility

Created by Tony Lorenzo (@alorenzo175), Univ. of Arizona, 2015

For a full list of contributors to this file, see the `pvlib` repository.

The copyright notice (BSD-3 clause) is as follows:

BSD 3-Clause License

Copyright (c) 2013-2018, Sandia National Laboratories and pvlib python
Development Team
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the {organization} nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from __future__ import division
import os
import time
from datetime import datetime
import math
from math import degrees, sin, cos, tan, radians, atan, asin, atan2, sqrt, acos
from fluids.constants import deg2rad, rad2deg
from fluids.numerics import sincos
__all__ = ['julian_day_dt', 'julian_day', 'julian_ephemeris_day', 'julian_century',
           'julian_ephemeris_century', 'julian_ephemeris_millennium', 'heliocentric_longitude',
           'heliocentric_latitude', 'heliocentric_radius_vector', 'geocentric_longitude',
           'geocentric_latitude', 'mean_elongation', 'mean_anomaly_sun', 'mean_anomaly_moon',
           'moon_argument_latitude', 'moon_ascending_longitude', 'longitude_nutation',
           'obliquity_nutation', 'mean_ecliptic_obliquity', 'true_ecliptic_obliquity',
           'aberration_correction', 'apparent_sun_longitude', 'mean_sidereal_time',
           'apparent_sidereal_time', 'geocentric_sun_right_ascension', 'geocentric_sun_declination',
           'local_hour_angle', 'equatorial_horizontal_parallax', 'uterm', 'xterm', 'yterm',
           'parallax_sun_right_ascension', 'topocentric_sun_right_ascension', 'topocentric_sun_declination',
           'topocentric_local_hour_angle', 'topocentric_elevation_angle_without_atmosphere',
           'atmospheric_refraction_correction', 'topocentric_elevation_angle', 'topocentric_zenith_angle',
           'topocentric_astronomers_azimuth', 'topocentric_azimuth_angle', 'sun_mean_longitude',
           'equation_of_time', 'calculate_deltat', 'longitude_obliquity_nutation',
           'transit_sunrise_sunset',
           ]
nan = float("nan")


HELIO_RADIUS_TABLE_LIST_0 = [[100013989.0, 0.0, 0.0],
 [1670700.0, 3.0984635, 6283.07585],
 [13956.0, 3.05525, 12566.1517],
 [3084.0, 5.1985, 77713.7715],
 [1628.0, 1.1739, 5753.3849],
 [1576.0, 2.8469, 7860.4194],
 [925.0, 5.453, 11506.77],
 [542.0, 4.564, 3930.21],
 [472.0, 3.661, 5884.927],
 [346.0, 0.964, 5507.553],
 [329.0, 5.9, 5223.694],
 [307.0, 0.299, 5573.143],
 [243.0, 4.273, 11790.629],
 [212.0, 5.847, 1577.344],
 [186.0, 5.022, 10977.079],
 [175.0, 3.012, 18849.228],
 [110.0, 5.055, 5486.778],
 [98.0, 0.89, 6069.78],
 [86.0, 5.69, 15720.84],
 [86.0, 1.27, 161000.69],
 [65.0, 0.27, 17260.15],
 [63.0, 0.92, 529.69],
 [57.0, 2.01, 83996.85],
 [56.0, 5.24, 71430.7],
 [49.0, 3.25, 2544.31],
 [47.0, 2.58, 775.52],
 [45.0, 5.54, 9437.76],
 [43.0, 6.01, 6275.96],
 [39.0, 5.36, 4694.0],
 [38.0, 2.39, 8827.39],
 [37.0, 0.83, 19651.05],
 [37.0, 4.9, 12139.55],
 [36.0, 1.67, 12036.46],
 [35.0, 1.84, 2942.46],
 [33.0, 0.24, 7084.9],
 [32.0, 0.18, 5088.63],
 [32.0, 1.78, 398.15],
 [28.0, 1.21, 6286.6],
 [28.0, 1.9, 6279.55],
 [26.0, 4.59, 10447.39]]

HELIO_RADIUS_TABLE_LIST_1 = [[103019.0, 1.10749, 6283.07585],
 [1721.0, 1.0644, 12566.1517],
 [702.0, 3.142, 0.0],
 [32.0, 1.02, 18849.23],
 [31.0, 2.84, 5507.55],
 [25.0, 1.32, 5223.69],
 [18.0, 1.42, 1577.34],
 [10.0, 5.91, 10977.08],
 [9.0, 1.42, 6275.96],
 [9.0, 0.27, 5486.78],
]
HELIO_RADIUS_TABLE_LIST_2 = [[4359.0, 5.7846, 6283.0758],
 [124.0, 5.579, 12566.152],
 [12.0, 3.14, 0.0],
 [9.0, 3.63, 77713.77],
 [6.0, 1.87, 5573.14],
 [3.0, 5.47, 18849.23]]
HELIO_RADIUS_TABLE_LIST_3 = [[145.0, 4.273, 6283.076],
 [7.0, 3.92, 12566.15]]
HELIO_RADIUS_TABLE_LIST_4 = [[4.0, 2.56, 6283.08]]

NUTATION_YTERM_LIST_0 = [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, -2.0, 0.0, 0.0, 2.0, -2.0, 0.0, -2.0, 0.0, 0.0, 2.0, 2.0, 0.0, -2.0, 0.0, 2.0, 2.0, -2.0, -2.0, 2.0, 2.0, 0.0, -2.0, -2.0, 0.0, -2.0, -2.0, 0.0, -1.0, -2.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 0.0, 2.0]
NUTATION_YTERM_LIST_1 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, -1.0]
NUTATION_YTERM_LIST_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 2.0, -2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, -2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -2.0, 1.0, 1.0, 1.0, -1.0, 3.0, 0.0]
NUTATION_YTERM_LIST_3 = [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
NUTATION_YTERM_LIST_4 = [1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]

NUTATION_ABCD_LIST = [[-171996.0, -174.2, 92025.0, 8.9],
 [-13187.0, -1.6, 5736.0, -3.1],
 [-2274.0, -0.2, 977.0, -0.5],
 [2062.0, 0.2, -895.0, 0.5],
 [1426.0, -3.4, 54.0, -0.1],
 [712.0, 0.1, -7.0, 0.0],
 [-517.0, 1.2, 224.0, -0.6],
 [-386.0, -0.4, 200.0, 0.0],
 [-301.0, 0.0, 129.0, -0.1],
 [217.0, -0.5, -95.0, 0.3],
 [-158.0, 0.0, 0.0, 0.0],
 [129.0, 0.1, -70.0, 0.0],
 [123.0, 0.0, -53.0, 0.0],
 [63.0, 0.0, 0.0, 0.0],
 [63.0, 0.1, -33.0, 0.0],
 [-59.0, 0.0, 26.0, 0.0],
 [-58.0, -0.1, 32.0, 0.0],
 [-51.0, 0.0, 27.0, 0.0],
 [48.0, 0.0, 0.0, 0.0],
 [46.0, 0.0, -24.0, 0.0],
 [-38.0, 0.0, 16.0, 0.0],
 [-31.0, 0.0, 13.0, 0.0],
 [29.0, 0.0, 0.0, 0.0],
 [29.0, 0.0, -12.0, 0.0],
 [26.0, 0.0, 0.0, 0.0],
 [-22.0, 0.0, 0.0, 0.0],
 [21.0, 0.0, -10.0, 0.0],
 [17.0, -0.1, 0.0, 0.0],
 [16.0, 0.0, -8.0, 0.0],
 [-16.0, 0.1, 7.0, 0.0],
 [-15.0, 0.0, 9.0, 0.0],
 [-13.0, 0.0, 7.0, 0.0],
 [-12.0, 0.0, 6.0, 0.0],
 [11.0, 0.0, 0.0, 0.0],
 [-10.0, 0.0, 5.0, 0.0],
 [-8.0, 0.0, 3.0, 0.0],
 [7.0, 0.0, -3.0, 0.0],
 [-7.0, 0.0, 0.0, 0.0],
 [-7.0, 0.0, 3.0, 0.0],
 [-7.0, 0.0, 3.0, 0.0],
 [6.0, 0.0, 0.0, 0.0],
 [6.0, 0.0, -3.0, 0.0],
 [6.0, 0.0, -3.0, 0.0],
 [-6.0, 0.0, 3.0, 0.0],
 [-6.0, 0.0, 3.0, 0.0],
 [5.0, 0.0, 0.0, 0.0],
 [-5.0, 0.0, 3.0, 0.0],
 [-5.0, 0.0, 3.0, 0.0],
 [-5.0, 0.0, 3.0, 0.0],
 [4.0, 0.0, 0.0, 0.0],
 [4.0, 0.0, 0.0, 0.0],
 [4.0, 0.0, 0.0, 0.0],
 [-4.0, 0.0, 0.0, 0.0],
 [-4.0, 0.0, 0.0, 0.0],
 [-4.0, 0.0, 0.0, 0.0],
 [3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0],
 [-3.0, 0.0, 0.0, 0.0]]

HELIO_LAT_TABLE_LIST_0 = [[280.0, 3.199, 84334.662],
 [102.0, 5.422, 5507.553],
 [80.0, 3.88, 5223.69],
 [44.0, 3.7, 2352.87],
 [32.0, 4.0, 1577.34]]

HELIO_LAT_TABLE_LIST_1 = [[9.0, 3.9, 5507.55],
 [6.0, 1.73, 5223.69]]

#HELIO_LONG_TABLE_LIST = HELIO_LONG_TABLE.tolist()
HELIO_LONG_TABLE_LIST_0 = [[175347046.0, 0.0, 0.0],
 [3341656.0, 4.6692568, 6283.07585],
 [34894.0, 4.6261, 12566.1517],
 [3497.0, 2.7441, 5753.3849],
 [3418.0, 2.8289, 3.5231],
 [3136.0, 3.6277, 77713.7715],
 [2676.0, 4.4181, 7860.4194],
 [2343.0, 6.1352, 3930.2097],
 [1324.0, 0.7425, 11506.7698],
 [1273.0, 2.0371, 529.691],
 [1199.0, 1.1096, 1577.3435],
 [990.0, 5.233, 5884.927],
 [902.0, 2.045, 26.298],
 [857.0, 3.508, 398.149],
 [780.0, 1.179, 5223.694],
 [753.0, 2.533, 5507.553],
 [505.0, 4.583, 18849.228],
 [492.0, 4.205, 775.523],
 [357.0, 2.92, 0.067],
 [317.0, 5.849, 11790.629],
 [284.0, 1.899, 796.298],
 [271.0, 0.315, 10977.079],
 [243.0, 0.345, 5486.778],
 [206.0, 4.806, 2544.314],
 [205.0, 1.869, 5573.143],
 [202.0, 2.458, 6069.777],
 [156.0, 0.833, 213.299],
 [132.0, 3.411, 2942.463],
 [126.0, 1.083, 20.775],
 [115.0, 0.645, 0.98],
 [103.0, 0.636, 4694.003],
 [102.0, 0.976, 15720.839],
 [102.0, 4.267, 7.114],
 [99.0, 6.21, 2146.17],
 [98.0, 0.68, 155.42],
 [86.0, 5.98, 161000.69],
 [85.0, 1.3, 6275.96],
 [85.0, 3.67, 71430.7],
 [80.0, 1.81, 17260.15],
 [79.0, 3.04, 12036.46],
 [75.0, 1.76, 5088.63],
 [74.0, 3.5, 3154.69],
 [74.0, 4.68, 801.82],
 [70.0, 0.83, 9437.76],
 [62.0, 3.98, 8827.39],
 [61.0, 1.82, 7084.9],
 [57.0, 2.78, 6286.6],
 [56.0, 4.39, 14143.5],
 [56.0, 3.47, 6279.55],
 [52.0, 0.19, 12139.55],
 [52.0, 1.33, 1748.02],
 [51.0, 0.28, 5856.48],
 [49.0, 0.49, 1194.45],
 [41.0, 5.37, 8429.24],
 [41.0, 2.4, 19651.05],
 [39.0, 6.17, 10447.39],
 [37.0, 6.04, 10213.29],
 [37.0, 2.57, 1059.38],
 [36.0, 1.71, 2352.87],
 [36.0, 1.78, 6812.77],
 [33.0, 0.59, 17789.85],
 [30.0, 0.44, 83996.85],
 [30.0, 2.74, 1349.87],
 [25.0, 3.16, 4690.48]]
HELIO_LONG_TABLE_LIST_1 = [[628331966747.0, 0.0, 0.0],
 [206059.0, 2.678235, 6283.07585],
 [4303.0, 2.6351, 12566.1517],
 [425.0, 1.59, 3.523],
 [119.0, 5.796, 26.298],
 [109.0, 2.966, 1577.344],
 [93.0, 2.59, 18849.23],
 [72.0, 1.14, 529.69],
 [68.0, 1.87, 398.15],
 [67.0, 4.41, 5507.55],
 [59.0, 2.89, 5223.69],
 [56.0, 2.17, 155.42],
 [45.0, 0.4, 796.3],
 [36.0, 0.47, 775.52],
 [29.0, 2.65, 7.11],
 [21.0, 5.34, 0.98],
 [19.0, 1.85, 5486.78],
 [19.0, 4.97, 213.3],
 [17.0, 2.99, 6275.96],
 [16.0, 0.03, 2544.31],
 [16.0, 1.43, 2146.17],
 [15.0, 1.21, 10977.08],
 [12.0, 2.83, 1748.02],
 [12.0, 3.26, 5088.63],
 [12.0, 5.27, 1194.45],
 [12.0, 2.08, 4694.0],
 [11.0, 0.77, 553.57],
 [10.0, 1.3, 6286.6],
 [10.0, 4.24, 1349.87],
 [9.0, 2.7, 242.73],
 [9.0, 5.64, 951.72],
 [8.0, 5.3, 2352.87],
 [6.0, 2.65, 9437.76],
 [6.0, 4.67, 4690.48],
 ]
HELIO_LONG_TABLE_LIST_2 = [[52919.0, 0.0, 0.0],
 [8720.0, 1.0721, 6283.0758],
 [309.0, 0.867, 12566.152],
 [27.0, 0.05, 3.52],
 [16.0, 5.19, 26.3],
 [16.0, 3.68, 155.42],
 [10.0, 0.76, 18849.23],
 [9.0, 2.06, 77713.77],
 [7.0, 0.83, 775.52],
 [5.0, 4.66, 1577.34],
 [4.0, 1.03, 7.11],
 [4.0, 3.44, 5573.14],
 [3.0, 5.14, 796.3],
 [3.0, 6.05, 5507.55],
 [3.0, 1.19, 242.73],
 [3.0, 6.12, 529.69],
 [3.0, 0.31, 398.15],
 [3.0, 2.28, 553.57],
 [2.0, 4.38, 5223.69],
 [2.0, 3.75, 0.98]]

HELIO_LONG_TABLE_LIST_3 = [[289.0, 5.844, 6283.076],
 [35.0, 0.0, 0.0],
 [17.0, 5.49, 12566.15],
 [3.0, 5.2, 155.42],
 [1.0, 4.72, 3.52],
 [1.0, 5.3, 18849.23],
 [1.0, 5.97, 242.73]
 ]
HELIO_LONG_TABLE_LIST_4 = [[114.0, 3.142, 0.0],
 [8.0, 4.13, 6283.08],
 [1.0, 3.84, 12566.15]]



def julian_day_dt(year, month, day, hour, minute, second, microsecond):
    """This is the original way to calculate the julian day from the NREL paper.

    However, it is much faster to convert to unix/epoch time and then convert to
    julian day. Note that the date must be UTC.
    """
    # Not used anywhere!
    if month <= 2:
        year = year-1
        month = month+12
    a = int(year/100)
    b = 2 - a + int(a * 0.25)
    frac_of_day = (microsecond + (second + minute * 60 + hour * 3600)
                   ) * 1.0 / (3600*24)
    d = day + frac_of_day
    jd = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + d +
          b - 1524.5)
    return jd


def julian_day(unixtime):
    jd = unixtime*1.1574074074074073e-05 + 2440587.5
#    jd = unixtime/86400.0 + 2440587.5
    return jd


def julian_ephemeris_day(julian_day, delta_t):
    jde = julian_day + delta_t*1.1574074074074073e-05
#    jde = julian_day + delta_t * 1.0 / 86400.0
    return jde


def julian_century(julian_day):
    jc = (julian_day - 2451545.0)*2.7378507871321012e-05# * 1.0 / 36525
    return jc


def julian_ephemeris_century(julian_ephemeris_day):
#    1/36525.0 =  2.7378507871321012e-05
    jce = (julian_ephemeris_day - 2451545.0)*2.7378507871321012e-05
    return jce


def julian_ephemeris_millennium(julian_ephemeris_century):
    jme = julian_ephemeris_century*0.1
    return jme


def heliocentric_longitude(jme):
    # Might be able to replace this with a pade approximation?
    # Looping over rows is probably still faster than (a, b, c)
    # Maximum optimization
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    l4 = 0.0
    l5 = 0.0
    for row in range(64):
        HELIO_LONG_TABLE_LIST_0_ROW = HELIO_LONG_TABLE_LIST_0[row]
        l0 += (HELIO_LONG_TABLE_LIST_0_ROW[0]
               * cos(HELIO_LONG_TABLE_LIST_0_ROW[1]
                        + HELIO_LONG_TABLE_LIST_0_ROW[2] * jme)
               )
    for row in range(34):
        HELIO_LONG_TABLE_LIST_1_ROW = HELIO_LONG_TABLE_LIST_1[row]
        l1 += (HELIO_LONG_TABLE_LIST_1_ROW[0]
               * cos(HELIO_LONG_TABLE_LIST_1_ROW[1]
                        + HELIO_LONG_TABLE_LIST_1_ROW[2] * jme)
               )
    for row in range(20):
        HELIO_LONG_TABLE_LIST_2_ROW = HELIO_LONG_TABLE_LIST_2[row]
        l2 += (HELIO_LONG_TABLE_LIST_2_ROW[0]
               * cos(HELIO_LONG_TABLE_LIST_2_ROW[1]
                        + HELIO_LONG_TABLE_LIST_2_ROW[2] * jme)
               )

    for row in range(7):
        HELIO_LONG_TABLE_LIST_3_ROW = HELIO_LONG_TABLE_LIST_3[row]
        l3 += (HELIO_LONG_TABLE_LIST_3_ROW[0]
               * cos(HELIO_LONG_TABLE_LIST_3_ROW[1]
                        + HELIO_LONG_TABLE_LIST_3_ROW[2] * jme)
               )
    for row in range(3):
        HELIO_LONG_TABLE_LIST_4_ROW = HELIO_LONG_TABLE_LIST_4[row]
        l4 += (HELIO_LONG_TABLE_LIST_4_ROW[0]
               * cos(HELIO_LONG_TABLE_LIST_4_ROW[1]
                        + HELIO_LONG_TABLE_LIST_4_ROW[2] * jme)
               )
#    l5 = (HELIO_LONG_TABLE_LIST_5[0][0]*cos(HELIO_LONG_TABLE_LIST_5[0][1]))
    l5 = -0.9999987317275395
    l_rad = (jme*(jme*(jme*(jme*(jme*l5 + l4) + l3) + l2) + l1) + l0)*1E-8
    l = rad2deg*l_rad
    return l % 360


def heliocentric_latitude(jme):
    b0 = 0.0
    b1 = 0.0
    for row in range(5):
        HELIO_LAT_TABLE_LIST_0_ROW = HELIO_LAT_TABLE_LIST_0[row]
        b0 += (HELIO_LAT_TABLE_LIST_0_ROW[0]
               * cos(HELIO_LAT_TABLE_LIST_0_ROW[1]
                        + HELIO_LAT_TABLE_LIST_0_ROW[2] * jme)
               )
    HELIO_LAT_TABLE_LIST_1_ROW = HELIO_LAT_TABLE_LIST_1[0]
    b1 += (HELIO_LAT_TABLE_LIST_1_ROW[0]
           * cos(HELIO_LAT_TABLE_LIST_1_ROW[1]
                    + HELIO_LAT_TABLE_LIST_1_ROW[2] * jme))

    HELIO_LAT_TABLE_LIST_1_ROW = HELIO_LAT_TABLE_LIST_1[1]
    b1 += (HELIO_LAT_TABLE_LIST_1_ROW[0]
           * cos(HELIO_LAT_TABLE_LIST_1_ROW[1]
                    + HELIO_LAT_TABLE_LIST_1_ROW[2] * jme))
    b_rad = (b0 + b1 * jme)*1E-8
    b = rad2deg*b_rad
    return b


def heliocentric_radius_vector(jme):
    # no optimizations can be thought of
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    r4 = 0.0
    # Would be possible to save a few multiplies of table1row[2]*jme, table1row[1]*jme as they are dups
    for row in range(40):
        table0row = HELIO_RADIUS_TABLE_LIST_0[row]
        r0 += (table0row[0]*cos(table0row[1] + table0row[2]*jme))
    for row in range(10):
        table1row = HELIO_RADIUS_TABLE_LIST_1[row]
        r1 += (table1row[0]*cos(table1row[1] + table1row[2]*jme))
    for row in range(6):
        table2row = HELIO_RADIUS_TABLE_LIST_2[row]
        r2 += (table2row[0]*cos(table2row[1] + table2row[2]*jme))

    table3row = HELIO_RADIUS_TABLE_LIST_3[0]
    r3 += (table3row[0]*cos(table3row[1] + table3row[2]*jme))
    table3row = HELIO_RADIUS_TABLE_LIST_3[1]
    r3 += (table3row[0]*cos(table3row[1] + table3row[2]*jme))

#    table4row = HELIO_RADIUS_TABLE_LIST_4[0]
#    r4 = (table4row[0]*cos(table4row[1] + table4row[2]*jme))
    r4 = (4.0*cos(2.56 + 6283.08*jme))
    return (jme*(jme*(jme*(jme*r4 + r3) + r2) + r1) + r0)*1E-8


def geocentric_longitude(heliocentric_longitude):
    theta = heliocentric_longitude + 180.0
    return theta % 360


def geocentric_latitude(heliocentric_latitude):
    beta = -heliocentric_latitude
    return beta


def mean_elongation(julian_ephemeris_century):
    return (julian_ephemeris_century*(julian_ephemeris_century
            *(5.27776898149614e-6*julian_ephemeris_century - 0.0019142)
            + 445267.11148) + 297.85036)
#    x0 = (297.85036
#          + 445267.111480 * julian_ephemeris_century
#          - 0.0019142 * julian_ephemeris_century**2
#          + julian_ephemeris_century**3 / 189474.0)
#    return x0


def mean_anomaly_sun(julian_ephemeris_century):
    return (julian_ephemeris_century*(julian_ephemeris_century*(
            -3.33333333333333e-6*julian_ephemeris_century - 0.0001603)
            + 35999.05034) + 357.52772)
#    x1 = (357.52772
#          + 35999.050340 * julian_ephemeris_century
#          - 0.0001603 * julian_ephemeris_century**2
#          - julian_ephemeris_century**3 / 300000.0)
#    return x1


def mean_anomaly_moon(julian_ephemeris_century):
    return (julian_ephemeris_century*(julian_ephemeris_century*(
            1.77777777777778e-5*julian_ephemeris_century + 0.0086972)
        + 477198.867398) + 134.96298)
#    x2 = (134.96298
#          + 477198.867398 * julian_ephemeris_century
#          + 0.0086972 * julian_ephemeris_century**2
#          + julian_ephemeris_century**3 / 56250)
#    return x2


def moon_argument_latitude(julian_ephemeris_century):
    return julian_ephemeris_century*(julian_ephemeris_century*(
            3.05558101873071e-6*julian_ephemeris_century - 0.0036825)
        + 483202.017538) + 93.27191
#    x3 = (93.27191
#          + 483202.017538 * julian_ephemeris_century
#          - 0.0036825 * julian_ephemeris_century**2
#          + julian_ephemeris_century**3 / 327270)
#    return x3


def moon_ascending_longitude(julian_ephemeris_century):
    return (julian_ephemeris_century*(julian_ephemeris_century*(
            2.22222222222222e-6*julian_ephemeris_century + 0.0020708)
            - 1934.136261) + 125.04452)
#    x4 = (125.04452
#          - 1934.136261 * julian_ephemeris_century
#          + 0.0020708 * julian_ephemeris_century**2
#          + julian_ephemeris_century**3 / 450000)
#    return x4


def longitude_obliquity_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4):
    x0, x1, x2, x3, x4 = deg2rad*x0, deg2rad*x1, deg2rad*x2, deg2rad*x3, deg2rad*x4
    delta_psi_sum = 0.0
    delta_eps_sum = 0.0
    # If the sincos formulation is used, the speed up is ~8% with numba.
    for row in range(63):
        arg = (NUTATION_YTERM_LIST_0[row]*x0 +
               NUTATION_YTERM_LIST_1[row]*x1 +
               NUTATION_YTERM_LIST_2[row]*x2 +
               NUTATION_YTERM_LIST_3[row]*x3 +
               NUTATION_YTERM_LIST_4[row]*x4)
        arr = NUTATION_ABCD_LIST[row]
        sinarg, cosarg = sincos(arg)
#        sinarg = sin(arg)
#        cosarg = sqrt(1.0 - sinarg*sinarg)
        t0 = (arr[0] + julian_ephemeris_century*arr[1])
        delta_psi_sum += t0*sinarg
#        delta_psi_sum += t0*sin(arg)
        t0 = (arr[2] + julian_ephemeris_century*arr[3])
        delta_eps_sum += t0*cosarg
#        delta_eps_sum += t0*cos(arg)
    delta_psi = delta_psi_sum/36000000.0
    delta_eps = delta_eps_sum/36000000.0
    res = [0.0]*2
    res[0] = delta_psi
    res[1] = delta_eps
    return res

def longitude_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4):
    x0, x1, x2, x3, x4 = deg2rad*x0, deg2rad*x1, deg2rad*x2, deg2rad*x3, deg2rad*x4
    delta_psi_sum = 0.0
    for row in range(63):
#       # None can be skipped but the multiplies can be with effort -2 to 2 with dict - just might be slower
        argsin = (NUTATION_YTERM_LIST_0[row]*x0 +
                  NUTATION_YTERM_LIST_1[row]*x1 +
                  NUTATION_YTERM_LIST_2[row]*x2 +
                  NUTATION_YTERM_LIST_3[row]*x3 +
                  NUTATION_YTERM_LIST_4[row]*x4)
        term = (NUTATION_ABCD_LIST[row][0] + NUTATION_ABCD_LIST[row][1]
                * julian_ephemeris_century)*sin(argsin)
        delta_psi_sum += term
    delta_psi = delta_psi_sum/36000000.0
    return delta_psi


def obliquity_nutation(julian_ephemeris_century, x0, x1, x2, x3, x4):
    delta_eps_sum = 0.0
    x0, x1, x2, x3, x4 = deg2rad*x0, deg2rad*x1, deg2rad*x2, deg2rad*x3, deg2rad*x4
    for row in range(63):
        argcos = (NUTATION_YTERM_LIST_0[row]*x0 +
                  NUTATION_YTERM_LIST_1[row]*x1 +
                  NUTATION_YTERM_LIST_2[row]*x2 +
                  NUTATION_YTERM_LIST_3[row]*x3 +
                  NUTATION_YTERM_LIST_4[row]*x4)
        term = (NUTATION_ABCD_LIST[row][2]
               + NUTATION_ABCD_LIST[row][3]*julian_ephemeris_century)*cos(argcos)
        delta_eps_sum += term
    delta_eps = delta_eps_sum/36000000.0
    return delta_eps


def mean_ecliptic_obliquity(julian_ephemeris_millennium):
    U = 0.1*julian_ephemeris_millennium
    e0 =  (U*(U*(U*(U*(U*(U*(U*(U*(U*(2.45*U + 5.79) + 27.87) + 7.12) - 39.05)
           - 249.67) - 51.38) + 1999.25) - 1.55) - 4680.93) + 84381.448)
    return e0


def true_ecliptic_obliquity(mean_ecliptic_obliquity, obliquity_nutation):
#    e0 = mean_ecliptic_obliquity
#    deleps = obliquity_nutation
    return mean_ecliptic_obliquity*0.0002777777777777778  + obliquity_nutation
#    e = e0/3600.0 + deleps
#    return e


def aberration_correction(earth_radius_vector):
    # -20.4898 / (3600)
    deltau = -0.005691611111111111/earth_radius_vector
    return deltau


def apparent_sun_longitude(geocentric_longitude, longitude_nutation,
                           aberration_correction):
    lamd = geocentric_longitude + longitude_nutation + aberration_correction
    return lamd


def mean_sidereal_time(julian_day, julian_century):
    julian_century2 = julian_century*julian_century
    v0 = (280.46061837 + 360.98564736629*(julian_day - 2451545.0)
          + 0.000387933*julian_century2
          - julian_century2*julian_century/38710000.0)
    return v0 % 360.0


def apparent_sidereal_time(mean_sidereal_time, longitude_nutation,
                           true_ecliptic_obliquity):
    v = mean_sidereal_time + longitude_nutation*cos(deg2rad*true_ecliptic_obliquity)
    return v


def geocentric_sun_right_ascension(apparent_sun_longitude,
                                   true_ecliptic_obliquity,
                                   geocentric_latitude):
    num = (sin(deg2rad*apparent_sun_longitude)
           * cos(deg2rad*true_ecliptic_obliquity)
           - tan(deg2rad*geocentric_latitude)
           * sin(deg2rad*true_ecliptic_obliquity))
    alpha = degrees(atan2(num, cos(
        deg2rad*apparent_sun_longitude)))
    return alpha % 360


def geocentric_sun_declination(apparent_sun_longitude, true_ecliptic_obliquity,
                               geocentric_latitude):
    delta = degrees(asin(sin(deg2rad*geocentric_latitude) *
                                 cos(deg2rad*true_ecliptic_obliquity) +
                                 cos(deg2rad*geocentric_latitude) *
                                 sin(deg2rad*true_ecliptic_obliquity) *
                                 sin(deg2rad*apparent_sun_longitude)))
    return delta


def local_hour_angle(apparent_sidereal_time, observer_longitude,
                     sun_right_ascension):
    """Measured westward from south."""
    H = apparent_sidereal_time + observer_longitude - sun_right_ascension
    return H % 360


def equatorial_horizontal_parallax(earth_radius_vector):
    return 8.794 / (3600.0 * earth_radius_vector)


def uterm(observer_latitude):
    u = atan(0.99664719*tan(deg2rad*observer_latitude))
    return u


def xterm(u, observer_latitude, observer_elevation):
    # 1/6378140.0 = const
    x = (cos(u) + observer_elevation*1.5678552054360676e-07*cos(deg2rad*observer_latitude))
    return x


def yterm(u, observer_latitude, observer_elevation):
    # 1/6378140.0 = const
    y = (0.99664719 * sin(u) + observer_elevation*1.5678552054360676e-07
         * sin(deg2rad*observer_latitude))
    return y


def parallax_sun_right_ascension(xterm, equatorial_horizontal_parallax,
                                 local_hour_angle, geocentric_sun_declination):
    x0 = sin(deg2rad*equatorial_horizontal_parallax)
    x1 = deg2rad*local_hour_angle
    num = -xterm*x0*sin(x1)
    denom = (cos(deg2rad*geocentric_sun_declination) - xterm*x0 * cos(x1))
    delta_alpha = degrees(atan2(num, denom))
    return delta_alpha


def topocentric_sun_right_ascension(geocentric_sun_right_ascension,
                                    parallax_sun_right_ascension):
    alpha_prime = geocentric_sun_right_ascension + parallax_sun_right_ascension
    return alpha_prime


def topocentric_sun_declination(geocentric_sun_declination, xterm, yterm,
                                equatorial_horizontal_parallax,
                                parallax_sun_right_ascension,
                                local_hour_angle):
    x0 = sin(deg2rad*equatorial_horizontal_parallax)
    num = ((sin(deg2rad*geocentric_sun_declination) - yterm
            * x0)
           * cos(deg2rad*parallax_sun_right_ascension))
    denom = (cos(deg2rad*geocentric_sun_declination) - xterm
             * x0
             * cos(deg2rad*local_hour_angle))
    delta = degrees(atan2(num, denom))
    return delta


def topocentric_local_hour_angle(local_hour_angle,
                                 parallax_sun_right_ascension):
    H_prime = local_hour_angle - parallax_sun_right_ascension
    return H_prime


def topocentric_elevation_angle_without_atmosphere(observer_latitude,
                                                   topocentric_sun_declination,
                                                   topocentric_local_hour_angle
                                                   ):
    observer_latitude = observer_latitude
    topocentric_sun_declination = topocentric_sun_declination
    topocentric_local_hour_angle = topocentric_local_hour_angle

    r_observer_latitude = deg2rad*observer_latitude
    r_topocentric_sun_declination = deg2rad*topocentric_sun_declination
    e0 = degrees(asin(
        sin(r_observer_latitude)
        * sin(r_topocentric_sun_declination)
        + cos(r_observer_latitude)
        * cos(r_topocentric_sun_declination)
        * cos(deg2rad*topocentric_local_hour_angle)))

    return e0


def atmospheric_refraction_correction(local_pressure, local_temp,
                                      topocentric_elevation_angle_wo_atmosphere,
                                      atmos_refract):
    # switch sets delta_e when the sun is below the horizon
    switch = topocentric_elevation_angle_wo_atmosphere >= -1.0 * (
        0.26667 + atmos_refract)
    delta_e = ((local_pressure / 1010.0) * (283.0 / (273.0 + local_temp))
               * 1.02 / (60.0 * tan(deg2rad*(
                   topocentric_elevation_angle_wo_atmosphere
                   + 10.3 / (topocentric_elevation_angle_wo_atmosphere
                             + 5.11))))) * switch
    return delta_e


def topocentric_elevation_angle(topocentric_elevation_angle_without_atmosphere,
                                atmospheric_refraction_correction):
    e = (topocentric_elevation_angle_without_atmosphere
         + atmospheric_refraction_correction)
    return e


def topocentric_zenith_angle(topocentric_elevation_angle):
    theta = 90.0 - topocentric_elevation_angle
    return theta


def topocentric_astronomers_azimuth(topocentric_local_hour_angle,
                                    topocentric_sun_declination,
                                    observer_latitude):
    num = sin(deg2rad*topocentric_local_hour_angle)
    denom = (cos(deg2rad*topocentric_local_hour_angle)
             * sin(deg2rad*observer_latitude)
             - tan(deg2rad*topocentric_sun_declination)
             * cos(deg2rad*observer_latitude))
    gamma = degrees(atan2(num, denom))

    return gamma % 360.0


def topocentric_azimuth_angle(topocentric_astronomers_azimuth):
    phi = topocentric_astronomers_azimuth + 180.0
    return phi % 360.0


def sun_mean_longitude(julian_ephemeris_millennium):
    M = julian_ephemeris_millennium*(julian_ephemeris_millennium*(
            julian_ephemeris_millennium*(julian_ephemeris_millennium*(
                    -5.0e-7*julian_ephemeris_millennium - 6.5359477124183e-5)
        + 2.00276381406341e-5) + 0.03032028) + 360007.6982779) + 280.4664567
    return M


#@jcompile('float64(float64, float64, float64, float64)', nopython=True)
def equation_of_time(sun_mean_longitude, geocentric_sun_right_ascension,
                     longitude_nutation, true_ecliptic_obliquity):
    term = cos(deg2rad*true_ecliptic_obliquity)
    E = (sun_mean_longitude - 0.0057183 - geocentric_sun_right_ascension +
         longitude_nutation * term)
    # limit between 0 and 360
    E = E % 360
    # convert to minutes
    E *= 4.0
    greater = E > 20.0
    less = E < -20.0
    other = (E <= 20.0) & (E >= -20.0)
    E = greater * (E - 1440.0) + less * (E + 1440.0) + other * E
    return E


def earthsun_distance(unixtime, delta_t):
    """Calculates the distance from the earth to the sun using the NREL SPA
    algorithm described in [1].

    Parameters
    ----------
    unixtime : numpy array
        Array of unix/epoch timestamps to calculate solar position for.
        Unixtime is the number of seconds since Jan. 1, 1970 00:00:00 UTC.
        A pandas.DatetimeIndex is easily converted using .astype(np.int64)/10**9
    delta_t : float
        Difference between terrestrial time and UT. USNO has tables.

    Returns
    -------
    R : array
        Earth-Sun distance in AU.

    References
    ----------
    [1] Reda, I., Andreas, A., 2003. Solar position algorithm for solar
    radiation applications. Technical report: NREL/TP-560- 34302. Golden,
    USA, http://www.nrel.gov.
    """
    jd = julian_day(unixtime)
    jde = julian_ephemeris_day(jd, delta_t)
    jce = julian_ephemeris_century(jde)
    jme = julian_ephemeris_millennium(jce)
    R = heliocentric_radius_vector(jme)
    return R


def solar_position(unixtime, lat, lon, elev, pressure, temp, delta_t,
                   atmos_refract, sst=False):
    """Calculate the solar position using the NREL SPA algorithm described in
    [1].

    If numba is installed, the functions can be compiled
    and the code runs quickly. If not, the functions
    still evaluate but use numpy instead.

    Parameters
    ----------
    unixtime : numpy array
        Array of unix/epoch timestamps to calculate solar position for.
        Unixtime is the number of seconds since Jan. 1, 1970 00:00:00 UTC.
        A pandas.DatetimeIndex is easily converted using .astype(np.int64)/10**9
    lat : float
        Latitude to calculate solar position for
    lon : float
        Longitude to calculate solar position for
    elev : float
        Elevation of location in meters
    pressure : int or float
        avg. yearly pressure at location in millibars;
        used for atmospheric correction
    temp : int or float
        avg. yearly temperature at location in
        degrees C; used for atmospheric correction
    delta_t : float, optional
        If delta_t is None, uses spa.calculate_deltat
        using time.year and time.month from pandas.DatetimeIndex.
        For most simulations specifying delta_t is sufficient.
        Difference between terrestrial time and UT1.
        *Note: delta_t = None will break code using nrel_numba,
        this will be fixed in a future version.
        By default, use USNO historical data and predictions
    atmos_refrac : float, optional
        The approximate atmospheric refraction (in degrees)
        at sunrise and sunset.
    numthreads: int, optional, default None
        Number of threads to use for computation if numba>=0.17
        is installed.
    sst : bool, default False
        If True, return only data needed for sunrise, sunset, and transit
        calculations.

    Returns
    -------
    list with elements:
        apparent zenith,
        zenith,
        elevation,
        apparent_elevation,
        azimuth,
        equation_of_time

    References
    ----------
    .. [1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
    applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    .. [2] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.
    """

    jd = julian_day(unixtime)
    jde = julian_ephemeris_day(jd, delta_t)
    jc = julian_century(jd)
    jce = julian_ephemeris_century(jde)
    jme = julian_ephemeris_millennium(jce)
    R = heliocentric_radius_vector(jme)
    L = heliocentric_longitude(jme)
    B = heliocentric_latitude(jme)
    Theta = geocentric_longitude(L)
    beta = geocentric_latitude(B)
    x0 = mean_elongation(jce)
    x1 = mean_anomaly_sun(jce)
    x2 = mean_anomaly_moon(jce)
    x3 = moon_argument_latitude(jce)
    x4 = moon_ascending_longitude(jce)
    delta_psi, delta_epsilon = longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4)

    epsilon0 = mean_ecliptic_obliquity(jme)
    epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
    delta_tau = aberration_correction(R)
    lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
    v0 = mean_sidereal_time(jd, jc)
    v = apparent_sidereal_time(v0, delta_psi, epsilon)
    alpha = geocentric_sun_right_ascension(lamd, epsilon, beta)
    delta = geocentric_sun_declination(lamd, epsilon, beta)
    if sst: # numba: delete
        return v, alpha, delta # numba: delete

    m = sun_mean_longitude(jme)
    eot = equation_of_time(m, alpha, delta_psi, epsilon)
    H = local_hour_angle(v, lon, alpha)
    xi = equatorial_horizontal_parallax(R)
    u = uterm(lat)
    x = xterm(u, lat, elev)
    y = yterm(u, lat, elev)
    delta_alpha = parallax_sun_right_ascension(x, xi, H, delta)
    delta_prime = topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
    H_prime = topocentric_local_hour_angle(H, delta_alpha)
    e0 = topocentric_elevation_angle_without_atmosphere(lat, delta_prime,
                                                        H_prime)
    delta_e = atmospheric_refraction_correction(pressure, temp, e0,
                                                atmos_refract)
    e = topocentric_elevation_angle(e0, delta_e)
    theta = topocentric_zenith_angle(e)
    theta0 = topocentric_zenith_angle(e0)
    gamma = topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
    phi = topocentric_azimuth_angle(gamma)
    return [theta, theta0, e, e0, phi, eot]


try:
    if IS_NUMBA:  # type: ignore
        try:
            import numpy as np
        except:
            pass
        import numba
        import numpy as np
        import threading
        # This is 3x slower without nogil
        @numba.njit(nogil=True)
        def solar_position_loop(unixtime, loc_args, out):
            """Loop through the time array and calculate the solar position."""
            lat = loc_args[0]
            lon = loc_args[1]
            elev = loc_args[2]
            pressure = loc_args[3]
            temp = loc_args[4]
            delta_t = loc_args[5]
            atmos_refract = loc_args[6]
            sst = loc_args[7]
            esd = loc_args[8]

            for i in range(len(unixtime)):
                utime = unixtime[i]
                jd = julian_day(utime)
                jde = julian_ephemeris_day(jd, delta_t)
                jc = julian_century(jd)
                jce = julian_ephemeris_century(jde)
                jme = julian_ephemeris_millennium(jce)
                R = heliocentric_radius_vector(jme)
                L = heliocentric_longitude(jme)
                B = heliocentric_latitude(jme)

                Theta = geocentric_longitude(L)
                beta = geocentric_latitude(B)
                x0 = mean_elongation(jce)
                x1 = mean_anomaly_sun(jce)
                x2 = mean_anomaly_moon(jce)
                x3 = moon_argument_latitude(jce)
                x4 = moon_ascending_longitude(jce)
#                delta_psi = longitude_nutation(jce, x0, x1, x2, x3, x4)
#                delta_epsilon = obliquity_nutation(jce, x0, x1, x2, x3, x4)
                delta_psi, delta_epsilon = longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4)
                epsilon0 = mean_ecliptic_obliquity(jme)
                epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
                delta_tau = aberration_correction(R)
                lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
                v0 = mean_sidereal_time(jd, jc)
                v = apparent_sidereal_time(v0, delta_psi, epsilon)
                alpha = geocentric_sun_right_ascension(lamd, epsilon, beta)
                delta = geocentric_sun_declination(lamd, epsilon, beta)
#                if sst:
#                    out[0, i] = v
#                    out[1, i] = alpha
#                    out[2, i] = delta
#                    continue
                m = sun_mean_longitude(jme)
                eot = equation_of_time(m, alpha, delta_psi, epsilon)
                H = local_hour_angle(v, lon, alpha)
                xi = equatorial_horizontal_parallax(R)
                u = uterm(lat)
                x = xterm(u, lat, elev)
                y = yterm(u, lat, elev)
                delta_alpha = parallax_sun_right_ascension(x, xi, H, delta)
                delta_prime = topocentric_sun_declination(delta, x, y, xi, delta_alpha,
                                                          H)
                H_prime = topocentric_local_hour_angle(H, delta_alpha)
                e0 = topocentric_elevation_angle_without_atmosphere(lat, delta_prime,
                                                                    H_prime)
                delta_e = atmospheric_refraction_correction(pressure, temp, e0,
                                                            atmos_refract)
                e = topocentric_elevation_angle(e0, delta_e)
                theta = topocentric_zenith_angle(e)
                theta0 = topocentric_zenith_angle(e0)
                gamma = topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
                phi = topocentric_azimuth_angle(gamma)
                out[0, i] = theta
                out[1, i] = theta0
                out[2, i] = e
                out[3, i] = e0
                out[4, i] = phi
                out[5, i] = eot


        def solar_position_numba(unixtime, lat, lon, elev, pressure, temp, delta_t,
                                 atmos_refract, numthreads, sst=False, esd=False):
            """Calculate the solar position using the numba compiled functions
            and multiple threads.

            Very slow if functions are not numba compiled.
            """
            # these args are the same for each thread
            loc_args = np.array([lat, lon, elev, pressure, temp, delta_t,
                                 atmos_refract, sst, esd])

            # construct dims x ulength array to put the results in
            ulength = unixtime.shape[0]
            if sst:
                dims = 3
            elif esd:
                dims = 1
            else:
                dims = 6
            result = np.empty((dims, ulength), dtype=np.float64)

            if unixtime.dtype != np.float64:
                unixtime = unixtime.astype(np.float64)

            if ulength < numthreads:
                warnings.warn('The number of threads is more than the length of '
                              'the time array. Only using %s threads.' %(length))
                numthreads = ulength

            if numthreads <= 1:
                solar_position_loop(unixtime, loc_args, result)
                return result

            # split the input and output arrays into numthreads chunks
            split0 = np.array_split(unixtime, numthreads)
            split2 = np.array_split(result, numthreads, axis=1)
            chunks = [[a0, loc_args, split2[i]] for i, a0 in enumerate(split0)]
            # Spawn one thread per chunk
            threads = [threading.Thread(target=solar_position_loop, args=chunk)
                       for chunk in chunks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return result


except:
    pass


def transit_sunrise_sunset(dates, lat, lon, delta_t):
    """Calculate the sun transit, sunrise, and sunset for a set of dates at a
    given location.

    Parameters
    ----------
    dates : array
        Numpy array of ints/floats corresponding to the Unix time
        for the dates of interest, must be midnight UTC (00:00+00:00)
        on the day of interest.
    lat : float
        Latitude of location to perform calculation for
    lon : float
        Longitude of location
    delta_t : float
        Difference between terrestrial time and UT. USNO has tables.

    Returns
    -------
    tuple : (transit, sunrise, sunset) localized to UTC

    >>> transit_sunrise_sunset(1523836800, 51.0486, -114.07, 70.68302220312503)
    (1523907360.3863413, 1523882341.570479, 1523932345.7781625)
    """
    condition = (dates % 86400) != 0.0
    if condition:
        raise ValueError('Input dates must be at 00:00 UTC')

    utday = (dates // 86400) * 86400
    ttday0 = utday - delta_t
    ttdayn1 = ttday0 - 86400.0
    ttdayp1 = ttday0 + 86400.0

    # index 0 is v, 1 is alpha, 2 is delta
    utday_res = solar_position(utday, 0, 0, 0, 0, 0, delta_t,
                               0, sst=True)
    v = utday_res[0]

    ttday0_res = solar_position(ttday0, 0, 0, 0, 0, 0, delta_t,
                                0, sst=True)
    ttdayn1_res = solar_position(ttdayn1, 0, 0, 0, 0, 0, delta_t,
                                 0, sst=True)
    ttdayp1_res = solar_position(ttdayp1, 0, 0, 0, 0, 0, delta_t,
                                 0, sst=True)
    m0 = (ttday0_res[1] - lon - v) / 360
    cos_arg = ((-0.014543315936696236 - sin(radians(lat)) # sin(radians(-0.8333)) = -0.0145...
               * sin(radians(ttday0_res[2]))) /
               (cos(radians(lat)) * cos(radians(ttday0_res[2]))))
    if abs(cos_arg) > 1:
        cos_arg = nan

    H0 = degrees(acos(cos_arg)) % 180

    m = [0.0]*3
    m[0] = m0 % 1
    m[1] = (m[0] - H0 / 360.0)
    m[2] = (m[0] + H0 / 360.0)

    # need to account for fractions of day that may be the next or previous
    # day in UTC
    add_a_day = m[2] >= 1
    sub_a_day = m[1] < 0
    m[1] = m[1] % 1
    m[2] = m[2] % 1
    vs = [0.0]*3
    for i in range(3):
        vs[i] = v + 360.985647*m[i]
    n = [0.0]*3
    for i in range(3):
        n[i] = m[i] + delta_t / 86400.0

    a = ttday0_res[1] - ttdayn1_res[1]

    if abs(a) > 2:
        a = a %1
    ap = ttday0_res[2] - ttdayn1_res[2]
    if (abs(ap) > 2):
        ap = ap % 1
    b = ttdayp1_res[1] - ttday0_res[1]
    if (abs(b) > 2):
        b = b % 1
    bp = ttdayp1_res[2] - ttday0_res[2]
    if abs(bp) > 2:
        bp = bp % 1


    c = b - a
    cp = bp - ap

    alpha_prime = [0.0]*3
    delta_prime = [0.0]*3
    Hp = [0.0]*3
    for i in range(3):
        alpha_prime[i] = ttday0_res[1] + (n[i] * (a + b + c * n[i]))*0.5
        delta_prime[i] = ttday0_res[2] + (n[i] * (ap + bp + cp * n[i]))*0.5
        Hp[i] = (vs[i] + lon - alpha_prime[i]) % 360
        if Hp[i] >= 180.0:
            Hp[i] = Hp[i] - 360.0


    #alpha_prime = ttday0_res[1] + (n * (a + b + c * n)) / 2 # this is vect
    #delta_prime = ttday0_res[2] + (n * (ap + bp + cp * n)) / 2 # this is vect
    #Hp = (vs + lon - alpha_prime) % 360
    #Hp[Hp >= 180] = Hp[Hp >= 180] - 360
    x1 = sin(radians(lat))
    x2 = cos(radians(lat))

    h = [0.0]*3
    for i in range(3):
        h[i] = degrees(asin(x1*sin(radians(delta_prime[i])) + x2 * cos(radians(delta_prime[i])) * cos(radians(Hp[i]))))

    T = float((m[0] - Hp[0] / 360.0) * 86400.0)
    R = float((m[1] + (h[1] + 0.8333) / (360.0 * cos(radians(delta_prime[1])) *
                                   cos(radians(lat)) *
                                   sin(radians(Hp[1])))) * 86400.0)
    S = float((m[2] + (h[2] + 0.8333) / (360.0 * cos(radians(delta_prime[2])) *
                                   cos(radians(lat)) *
                                   sin(radians(Hp[2])))) * 86400.0)

    if add_a_day:
        S += 86400.0
    if sub_a_day:
        R -= 86400.0

    transit = T + utday
    sunrise = R + utday
    sunset = S + utday

    return transit, sunrise, sunset


def calculate_deltat(year, month):
    y = year + (month - 0.5)/12
    if (2005 <= year) & (year < 2050):
        t1 = (y-2000.0)
        deltat = (62.92+0.32217*t1 + 0.005589*t1*t1)
    elif  (1986 <= year) & (year < 2005):
        t1 = y - 2000.0
        deltat = (63.86+0.3345*t1
                      - 0.060374*t1**2
                      + 0.0017275*t1**3
                      + 0.000651814*t1**4
                      + 0.00002373599*t1**5)
    elif (2050 <= year) & (year < 2150):
        deltat = (-20+32*((y-1820)/100)**2
                      - 0.5628*(2150-y))
    elif year < -500.0:
        deltat = -20.0 + 32*(0.01*(y-1820.0))**2

    elif (-500 <= year) & (year < 500):
        t1 = y/100
        deltat = (10583.6-1014.41*(y/100)
                      + 33.78311*(y/100)**2
                      - 5.952053*(y/100)**3
                      - 0.1798452*(y/100)**4
                      + 0.022174192*(y/100)**5
                      + 0.0090316521*(y/100)**6)
    elif (500 <= year) & (year < 1600):
        t1 = (y-1000)/100
        deltat = (1574.2-556.01*((y-1000)/100)
                      + 71.23472*((y-1000)/100)**2
                      + 0.319781*((y-1000)/100)**3
                      - 0.8503463*((y-1000)/100)**4
                      - 0.005050998*((y-1000)/100)**5
                      + 0.0083572073*((y-1000)/100)**6)
    elif (1600 <= year) & (year < 1700):
        t1 = (y-1600.0)
        deltat = (120-0.9808*(y-1600)
                      - 0.01532*(y-1600)**2
                      + (y-1600)**3/7129)
    elif (1700 <= year) & (year < 1800):
        t1 = (y - 1700.0)
        deltat = (8.83+0.1603*(y-1700)
                      - 0.0059285*(y-1700)**2
                      + 0.00013336*(y-1700)**3
                      - (y-1700)**4/1174000)
    elif (1800 <= year) & (year < 1860):
        t1 = y - 1800.0
        deltat = (13.72-0.332447*(y-1800)
                      + 0.0068612*(y-1800)**2
                      + 0.0041116*(y-1800)**3
                      - 0.00037436*(y-1800)**4
                      + 0.0000121272*(y-1800)**5
                      - 0.0000001699*(y-1800)**6
                      + 0.000000000875*(y-1800)**7)
    elif (1860 <= year) & (year < 1900):
        t1 = y-1860.0
        deltat = (7.62+0.5737*(y-1860)
                      - 0.251754*(y-1860)**2
                      + 0.01680668*(y-1860)**3
                      - 0.0004473624*(y-1860)**4
                      + (y-1860)**5/233174)
    elif (1900 <= year) & (year < 1920):
        t1 = y - 1900.0
        deltat = (-2.79+1.494119*(y-1900)
                      - 0.0598939*(y-1900)**2
                      + 0.0061966*(y-1900)**3
                      - 0.000197*(y-1900)**4)
    elif (1920 <= year) & (year < 1941):
        t1 = y - 1920.0
        deltat = (21.20+0.84493*(y-1920)
                      - 0.076100*(y-1920)**2
                      + 0.0020936*(y-1920)**3)
    elif (1941 <= year) & (year < 1961):
        t1 = y - 1950.0
        deltat = (29.07+0.407*(y-1950)
                      - (y-1950)**2/233
                      + (y-1950)**3/2547)
    elif (1961 <= year) & (year < 1986):
        t1 = y-1975
        deltat = (45.45+1.067*(y-1975)
                      - (y-1975)**2/260
                      - (y-1975)**3/718)

    elif year >= 2150:
        deltat = -20+32*((y-1820)/100)**2


    return deltat