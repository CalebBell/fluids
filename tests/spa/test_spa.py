# -*- coding: utf-8 -*-
"""
test_spa unit test suite from pvlib
===================================
Vendorized version from:
https://github.com/pvlib/pvlib-python/

The rational for not including this library as a strict dependency is to avoid
including a dependency on pandas, keeping load time low, and PyPy compatibility
.

.. moduleauthor :: Tony Lorenzo <atlorenzo@email.arizona.edu>
.. moduleauthor :: Will Holmgren <william.holmgren@gmail.com>
.. moduleauthor :: Volker Beutner < VolkerBeu@gmail.com >

Some tests were changed and added as well.


The copyright notice (BSD-3 clause) is as follows:

BSD 3-Clause License

Copyright (c) 2013-2018, Sandia National Laboratories and pvlib python Development Team
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



import os
import datetime as dt

try:
    from importlib import reload
except ImportError:
    try:
        from imp import reload
    except ImportError:
        pass

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd

import unittest
import pytest


try:
    from numba import __version__ as numba_version
    numba_version_int = int(numba_version.split('.')[0] +
                            numba_version.split('.')[1])
except ImportError:
    numba_version_int = 0
except:
    numba_version_int = -1

from fluids.optional import spa

times = (pd.date_range('2003-10-17 12:30:30', periods=1, freq='D')
           .tz_localize('MST'))
unixtimes = np.array(times.tz_convert('UTC').view(np.int64)*1.0/10**9)
unixtimes = float(np.array(times.tz_convert('UTC').view(np.int64)*1.0/10**9)[0])
lat = 39.742476
lon = -105.1786
elev = 1830.14
pressure = 820
temp = 11
delta_t = 67.0
atmos_refract= 0.5667

JD = 2452930.312847
JC = 0.0379277986858
JDE = 2452930.313623
JCE = 0.037927819916852
JME = 0.003792781991685
L = 24.0182616917
B = -0.0001011219
R = 0.9965422974
Theta = 204.0182616917
beta = 0.0001011219
X0 = 17185.861179
X1 = 1722.893218
X2 = 18234.075703
X3 = 18420.071012
X4 = 51.686951
dPsi = -0.00399840
dEpsilon = 0.00166657
epsilon0 = 84379.672625
epsilon = 23.440465
dTau = -0.005711
lamd = 204.0085519281
v0 = 318.515579
v = 318.511910
alpha = 202.227408
delta = -9.31434
H = 11.10590
xi = 0.002451
dAlpha = -0.000369
alpha_prime = 202.22704
delta_prime = -9.316179
H_prime = 11.10627
e0 = 39.872046
de = 0.016332
e = 39.888378
theta = 50.11162
theta0 = 90 - e0
Gamma = 14.340241
Phi = 194.340241
year = 1985
month = 2
year_array = np.array([-499, 500, 1000, 1500, 1800, 1900, 1950, 1970, 1985, 1990, 2000, 2005])
month_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dt_actual = 54.413442486
dt_actual_array = np.array([1.7184831e+04, 5.7088051e+03, 1.5730419e+03,
                          1.9801820e+02, 1.3596506e+01, -2.1171894e+00,
                          2.9289261e+01, 4.0824887e+01, 5.4724581e+01,
                          5.7426651e+01, 6.4108015e+01, 6.5038015e+01])
mix_year_array = np.full((10), year)
mix_month_array = np.full((10), month)
mix_year_actual = np.full((10), dt_actual)
mix_month_actual = mix_year_actual

class SpaBase(object):
    """Test functions common to numpy and numba spa"""
    def test_julian_day_dt(self):
        dt = times.tz_convert('UTC')[0]
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        microsecond = dt.microsecond
        assert_almost_equal(JD,
                             self.spa.julian_day_dt(year, month, day, hour,
                                           minute, second, microsecond), 6)

    def test_julian_ephemeris_day(self):
        assert_almost_equal(JDE, self.spa.julian_ephemeris_day(JD, delta_t), 5)

    def test_julian_century(self):
        assert_almost_equal(JC, self.spa.julian_century(JD), 6)

    def test_julian_ephemeris_century(self):
        assert_almost_equal(JCE, self.spa.julian_ephemeris_century(JDE), 10)

    def test_julian_ephemeris_millenium(self):
        assert_almost_equal(JME, self.spa.julian_ephemeris_millennium(JCE), 10)

    def test_heliocentric_longitude(self):
        assert_almost_equal(L, self.spa.heliocentric_longitude(JME), 6)

    def test_heliocentric_latitude(self):
        assert_almost_equal(B, self.spa.heliocentric_latitude(JME), 6)

    def test_heliocentric_radius_vector(self):
        assert_almost_equal(R, self.spa.heliocentric_radius_vector(JME), 6)

    def test_geocentric_longitude(self):
        assert_almost_equal(Theta, self.spa.geocentric_longitude(L), 6)

    def test_geocentric_latitude(self):
        assert_almost_equal(beta, self.spa.geocentric_latitude(B), 6)

    def test_mean_elongation(self):
        assert_almost_equal(X0, self.spa.mean_elongation(JCE), 5)

    def test_mean_anomaly_sun(self):
        assert_almost_equal(X1, self.spa.mean_anomaly_sun(JCE), 5)

    def test_mean_anomaly_moon(self):
        assert_almost_equal(X2, self.spa.mean_anomaly_moon(JCE), 5)

    def test_moon_argument_latitude(self):
        assert_almost_equal(X3, self.spa.moon_argument_latitude(JCE), 5)

    def test_moon_ascending_longitude(self):
        assert_almost_equal(X4, self.spa.moon_ascending_longitude(JCE), 6)

    def test_longitude_nutation(self):
        assert_almost_equal(dPsi, self.spa.longitude_nutation(JCE, X0, X1, X2,
                                                               X3, X4), 6)

    def test_obliquity_nutation(self):
        assert_almost_equal(dEpsilon, self.spa.obliquity_nutation(JCE, X0, X1,
                                                                   X2, X3, X4),
                             6)

    def test_mean_ecliptic_obliquity(self):
        assert_almost_equal(epsilon0, self.spa.mean_ecliptic_obliquity(JME), 6)

    def test_true_ecliptic_obliquity(self):
        assert_almost_equal(epsilon, self.spa.true_ecliptic_obliquity(
            epsilon0, dEpsilon), 6)

    def test_aberration_correction(self):
        assert_almost_equal(dTau, self.spa.aberration_correction(R), 6)

    def test_apparent_sun_longitude(self):
        assert_almost_equal(lamd, self.spa.apparent_sun_longitude(
            Theta, dPsi, dTau), 6)

    def test_mean_sidereal_time(self):
        assert_almost_equal(v0, self.spa.mean_sidereal_time(JD, JC), 3)

    def test_apparent_sidereal_time(self):
        assert_almost_equal(v, self.spa.apparent_sidereal_time(
            v0, dPsi, epsilon), 5)

    def test_geocentric_sun_right_ascension(self):
        assert_almost_equal(alpha, self.spa.geocentric_sun_right_ascension(
            lamd, epsilon, beta), 6)

    def test_geocentric_sun_declination(self):
        assert_almost_equal(delta, self.spa.geocentric_sun_declination(
            lamd, epsilon, beta), 6)

    def test_local_hour_angle(self):
        assert_almost_equal(H, self.spa.local_hour_angle(v, lon, alpha), 4)

    def test_equatorial_horizontal_parallax(self):
        assert_almost_equal(xi, self.spa.equatorial_horizontal_parallax(R), 6)

    def test_parallax_sun_right_ascension(self):
        u = self.spa.uterm(lat)
        x = self.spa.xterm(u, lat, elev)
        y = self.spa.yterm(u, lat, elev)
        assert_almost_equal(dAlpha, self.spa.parallax_sun_right_ascension(
            x, xi, H, delta), 4)

    def test_topocentric_sun_right_ascension(self):
        assert_almost_equal(alpha_prime,
                             self.spa.topocentric_sun_right_ascension(
                                 alpha, dAlpha), 5)

    def test_topocentric_sun_declination(self):
        u = self.spa.uterm(lat)
        x = self.spa.xterm(u, lat, elev)
        y = self.spa.yterm(u, lat, elev)
        assert_almost_equal(delta_prime, self.spa.topocentric_sun_declination(
            delta, x, y, xi, dAlpha,H), 5)

    def test_topocentric_local_hour_angle(self):
        assert_almost_equal(H_prime, self.spa.topocentric_local_hour_angle(
            H, dAlpha), 5)

    def test_topocentric_elevation_angle_without_atmosphere(self):
        assert_almost_equal(
            e0, self.spa.topocentric_elevation_angle_without_atmosphere(
                lat, delta_prime, H_prime), 6)

    def test_atmospheric_refraction_correction(self):
        assert_almost_equal(de, self.spa.atmospheric_refraction_correction(
            pressure, temp, e0, atmos_refract), 6)

    def test_topocentric_elevation_angle(self):
        assert_almost_equal(e, self.spa.topocentric_elevation_angle(e0, de), 6)

    def test_topocentric_zenith_angle(self):
        assert_almost_equal(theta, self.spa.topocentric_zenith_angle(e), 5)

    def test_topocentric_astronomers_azimuth(self):
        assert_almost_equal(Gamma, self.spa.topocentric_astronomers_azimuth(
            H_prime, delta_prime, lat), 5)

    def test_topocentric_azimuth_angle(self):
        assert_almost_equal(Phi, self.spa.topocentric_azimuth_angle(Gamma), 5)

    def test_solar_position(self):
        assert_almost_equal(np.array([theta, theta0, e, e0, Phi]),
                            self.spa.solar_position(unixtimes, lat, lon, elev, pressure, temp, delta_t, atmos_refract)[:-1], 5)
        assert_almost_equal(np.array([v, alpha, delta]),
                            self.spa.solar_position(unixtimes, lat, lon, elev, pressure, temp, delta_t, atmos_refract, sst=True)[:3], 5)

    def test_equation_of_time(self):
        eot = 14.64
        M = self.spa.sun_mean_longitude(JME)
        assert_almost_equal(eot, self.spa.equation_of_time(
            M, alpha, dPsi, epsilon), 2)

    def test_transit_sunrise_sunset(self):
        # tests at greenwich
        times = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 0),
                                  dt.datetime(2004, 12, 4, 0)]
                                 ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 7, 8, 15),
                                    dt.datetime(2004, 12, 4, 4, 38, 57)]
                                   ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 17, 1, 4),
                                   dt.datetime(2004, 12, 4, 19, 2, 2)]
                                  ).tz_localize('UTC').view(np.int64)*1.0/10**9
        times = np.array(times)
        sunrise = np.array(sunrise)
        sunset = np.array(sunset)
        result = [self.spa.transit_sunrise_sunset(t, -35.0, 0.0, 64.0) for t in times]
        for i in range(2):
            assert_almost_equal(sunrise[i]/1e3, result[i][1]/1e3, 3)
            assert_almost_equal(sunset[i]/1e3, result[i][2]/1e3, 3)


        times = pd.DatetimeIndex([dt.datetime(1994, 1, 2),]
                                 ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(1994, 1, 2, 16, 59, 55),]
                                  ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(1994, 1, 2, 7, 8, 12),]
                                   ).tz_localize('UTC').view(np.int64)*1.0/10**9
        times = np.array(times)
        sunrise = np.array(sunrise)
        sunset = np.array(sunset)
        result = [self.spa.transit_sunrise_sunset(t, 35.0, 0.0, 64.0) for t in times]
        for i in range(1):
            assert_almost_equal(sunrise[i]/1e3, result[i][1]/1e3, 3)
            assert_almost_equal(sunset[i]/1e3, result[i][2]/1e3, 3)

        # tests from USNO
        # Golden
        times = pd.DatetimeIndex([dt.datetime(2015, 1, 2),
                                  dt.datetime(2015, 4, 2),
                                  dt.datetime(2015, 8, 2),
                                  dt.datetime(2015, 12, 2),],
                                 ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 7, 19),
                                    dt.datetime(2015, 4, 2, 5, 43),
                                    dt.datetime(2015, 8, 2, 5, 1),
                                    dt.datetime(2015, 12, 2, 7, 1),],
                                   ).tz_localize('MST').view(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 16, 49),
                                   dt.datetime(2015, 4, 2, 18, 24),
                                   dt.datetime(2015, 8, 2, 19, 10),
                                   dt.datetime(2015, 12, 2, 16, 38),],
                                  ).tz_localize('MST').view(np.int64)*1.0/10**9
        times = np.array(times)
        sunrise = np.array(sunrise)
        sunset = np.array(sunset)
        result = [self.spa.transit_sunrise_sunset(t, 39.0, -105.0, 64.0) for t in times]
        for i in range(4):
            assert_almost_equal(sunrise[i]/1e3, result[i][1]/1e3, 1)
            assert_almost_equal(sunset[i]/1e3, result[i][2]/1e3, 1)

        # Beijing
        times = pd.DatetimeIndex([dt.datetime(2015, 1, 2),
                                  dt.datetime(2015, 4, 2),
                                  dt.datetime(2015, 8, 2),
                                  dt.datetime(2015, 12, 2),],
                                 ).tz_localize('UTC').view(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 7, 36),
                                    dt.datetime(2015, 4, 2, 5, 58),
                                    dt.datetime(2015, 8, 2, 5, 13),
                                    dt.datetime(2015, 12, 2, 7, 17),],
                                   ).tz_localize('Asia/Shanghai'
                                   ).view(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 17, 0),
                                   dt.datetime(2015, 4, 2, 18, 39),
                                   dt.datetime(2015, 8, 2, 19, 28),
                                   dt.datetime(2015, 12, 2, 16, 50),],
                                  ).tz_localize('Asia/Shanghai'
                                  ).view(np.int64)*1.0/10**9
        times = np.array(times)
        sunrise = np.array(sunrise)
        sunset = np.array(sunset)
        result = [self.spa.transit_sunrise_sunset(t, 39.917, 116.383, 64.0) for t in times]
        for i in range(4):
            assert_almost_equal(sunrise[i]/1e3, result[i][1]/1e3, 1)
            assert_almost_equal(sunset[i]/1e3, result[i][2]/1e3, 1)

    def test_earthsun_distance(self):
        times = (pd.date_range('2003-10-17 12:30:30', periods=1, freq='D')
           .tz_localize('MST'))
        unixtimes = times.tz_convert('UTC').view(np.int64)*1.0/10**9
        unixtimes = float(np.array(unixtimes)[0])
        result = self.spa.earthsun_distance(unixtimes, 64.0)
        assert_almost_equal(R, result, 6)

    def test_calculate_deltat(self):
        result_mix_year = [self.spa.calculate_deltat(t, month) for t in mix_year_array]
        assert_almost_equal(mix_year_actual, result_mix_year)

        result_mix_month = self.spa.calculate_deltat(year, mix_month_array)
        assert_almost_equal(mix_month_actual, result_mix_month)

        result_array = [self.spa.calculate_deltat(t, m) for t, m in zip(year_array, month_array)]
        assert_almost_equal(dt_actual_array, result_array, 3)

        result_scalar = self.spa.calculate_deltat(year,month)
        assert_almost_equal(dt_actual, result_scalar)

class NumpySpaTest(unittest.TestCase, SpaBase):
    """Import spa without compiling to numba then run tests"""
    @classmethod
    def setUpClass(self):
        from fluids.optional import spa
        spa = reload(spa)
        self.spa = spa

    @classmethod
    def tearDownClass(self):
        pass

    def test_julian_day(self):
        assert_almost_equal(JD, self.spa.julian_day(unixtimes), 6)


@pytest.mark.skipif(numba_version_int < 17,
                    reason='Numba not installed or version not >= 0.17.0')
class NumbaSpaTest(unittest.TestCase, SpaBase):
    """Import spa, compiling to numba, and run tests"""
    @classmethod
    def setUpClass(self):
        if numba_version_int >= 17:
            from fluids.optional import spa
            spa = reload(spa)
            self.spa = spa

    @classmethod
    def tearDownClass(self):
        pass

    def test_julian_day(self):
        assert_almost_equal(JD, self.spa.julian_day(unixtimes), 6)

    def test_solar_position_singlethreaded(self):
        assert_almost_equal(
            np.array([theta, theta0, e, e0, Phi]), self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t,
                atmos_refract)[:-1], 5)
        assert_almost_equal(
            np.array([v, alpha, delta]), self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t,
                atmos_refract, sst=True)[:3], 5)


try:
    import astropy
except:
    astropy = None
@pytest.mark.skipif(astropy is None, reason='Astropy is not installed')
def test_deltat_astropy():
    # Can't do a full range of tests because astropy doesn't have
    # answers before 1960, after 1999 in this version
    from astropy.time import Time
    from datetime import datetime
    def delta_t_astropy(dt):
        t = Time(dt, scale='utc')
        return -(dt - t.tt.value).total_seconds()

#    years = range(1, 3000, 100) + [3000]
    years = range(1960, 1999, 1)

    months = range(1, 13)
    for year in years:
        for month in months:
            delta_t_pvlib = spa.calculate_deltat(year, month)
            dt = datetime(year, month, 1)
            delta_t_external = delta_t_astropy(dt)
            assert_allclose(delta_t_pvlib, delta_t_external, atol=.5, rtol=.01)

#suite = unittest.TestSuite()
#suite.addTest(NumpySpaTest("testItIsHot"))
#runner = unittest.TextTestRunner()
#runner.run(suite)
#
#NumpySpaTest.test_calculate_deltat()

if __name__ == '__main__':
    unittest.main()
