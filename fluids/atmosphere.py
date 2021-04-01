# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.

This module contains models of earth's atmosphere. Models are empirical and
based on extensive research, primarily by NASA.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Atmospheres
-----------
.. autoclass:: ATMOSPHERE_1976
    :members:
.. autoclass:: ATMOSPHERE_NRLMSISE00
    :members:
.. autofunction:: airmass

Solar Radiation and Position
----------------------------
.. autofunction:: solar_position
.. autofunction:: solar_irradiation
.. autofunction:: sunrise_sunset
.. autofunction:: earthsun_distance

Wind Models (requires Fortran compiler!)
----------------------------------------
.. autofunction:: hwm93
.. autofunction:: hwm14
"""

from __future__ import division

from math import sqrt, exp, cos, radians, pi, sin
import time
import os
from fluids.constants import N_A, R, au
from fluids.numerics import brenth, quad, numpy as np
try:
    from datetime import datetime, timedelta
except:
    pass

__all__ = ['ATMOSPHERE_1976', 'ATMOSPHERE_NRLMSISE00', 'hwm93', 'hwm14',
           'earthsun_distance', 'solar_position', 'solar_irradiation',
           'sunrise_sunset']

no_gfortran_error = '''This function uses f2py to encapsulate a fortran \
routine. However, f2py did not detect one on installation and could not compile \
this routine. '''

try:
    # Needed by hwm14
    os.environ["HWMPATH"] = os.path.join(os.path.dirname(__file__), 'optional')
except:
    pass


H_std = [0.0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852.0]
T_grad = [-6.5E-3, 0.0, 1E-3, 2.8E-3, 0.0, -2.8E-3, -2E-3, 0.0]
T_std = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
P_std = [101325, 22632.06397346291, 5474.8886696777745, 868.0186847552279,
        110.90630555496608, 66.93887311868738, 3.956420428040732,
        0.3733835899762159]

r0 = 6356766.0
P0 = 101325.0
M0 = 28.9644
g0 = 9.80665
gamma = 1.400

def H_for_P_ATMOSPHERE_1976_err(H, P1):
    return ATMOSPHERE_1976(H).P - P1

def to_int_dP_ATMOSPHERE_1976(Z, dT):
    atm = ATMOSPHERE_1976(Z, dT=dT)
    return atm.g*atm.rho

class ATMOSPHERE_1976(object):
    r'''US Standard Atmosphere 1976 class, which calculates `T`, `P`,
    `rho`, `v_sonic`, `mu`, `k`, and `g` as a function of altitude above
    sea level. Designed to provide reasonable results up to an elevation
    of 86,000 m (0.4 Pa). The model is also valid under sea level, to
    -610 meters.

    Parameters
    ----------
    Z : float
        Elevation, [m]
    dT : float, optional
        Temperature difference from standard conditions used in determining
        the properties of the atmosphere, [K]

    Attributes
    ----------
    T : float
        Temperature of atmosphere at specified conditions, [K]
    P : float
        Pressure of atmosphere at specified conditions, [Pa]
    rho : float
        Mass density of atmosphere at specified conditions [kg/m^3]
    H : float
        Geopotential height, [m]
    g : float
        Acceleration due to gravity, [m/s^2]
    mu : float
        Viscosity of atmosphere at specified conditions, [Pa*s]
    k : float
        Thermal conductivity of atmosphere at specified conditions, [W/m/K]
    v_sonic : float
        Speed of sound of atmosphere at specified conditions, [m/s]

    Examples
    --------
    >>> five_km = ATMOSPHERE_1976(5000)
    >>> five_km.P, five_km.rho, five_km.mu
    (54048.28614576141, 0.7364284207799743, 1.628248135362207e-05)
    >>> five_km.k, five_km.g, five_km.v_sonic
    (0.02273190295142526, 9.791241076982665, 320.5455196704035)

    Notes
    -----
    Up to 32 km, the International Standard Atmosphere (ISA) and World
    Meteorological Organization (WMO) standard atmosphere are identical.

    This is a revision of the US 1962 atmosphere.

    References
    ----------
    .. [1] NOAA, NASA, and USAF. "U.S. Standard Atmosphere, 1976" October 15,
       1976. http://ntrs.nasa.gov/search.jsp?R=19770009539.
    .. [2] "ISO 2533:1975 - Standard Atmosphere." ISO.
       http://www.iso.org/iso/catalogue_detail.htm?csnumber=7472.
    .. [3] Yager, Robert J. "Calculating Atmospheric Conditions (Temperature,
       Pressure, Air Density, and Speed of Sound) Using C++," June 2013.
       http://www.dtic.mil/cgi-bin/GetTRDoc?AD=ADA588839
    '''

    def __init__(self, Z, dT=0.0):
        self.Z = Z
        self.dT = dT
        self.H = r0*Z/(r0+Z)

        i = self._get_ind_from_H(self.H)
        self.T_layer = T_std[i]
        self.T_increase = T_grad[i]
        self.P_layer = P_std[i]
        self.H_layer = H_std[i]

        self.H_above_layer = self.H - self.H_layer
        self.T = self.T_layer + self.T_increase*self.H_above_layer

        R = 8314.32
        if self.T_increase == 0.0:
            self.P = self.P_layer*exp(-g0*M0*(self.H_above_layer)/(R*self.T_layer))
        else:
            self.P = self.P_layer*(self.T_layer/self.T)**(g0*M0/(R*self.T_increase))

        # Affects only the following properties
        self.T += dT


        self.rho = self.density(self.T, self.P)
        self.v_sonic = self.sonic_velocity(self.T)
        self.mu = self.viscosity(self.T)
        self.k = self.thermal_conductivity(self.T)
        self.g = self.gravity(self.Z)

    @staticmethod
    def _get_ind_from_H(H):
        r'''Method defined in the US Standard Atmosphere 1976 for determining
        the index of the layer a specified elevation is above. Levels are
        0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852 meters respectively.
        '''
        if H <= 0.0:
            return 0
        for ind, Hi in enumerate(H_std):
            if Hi >= H :
                return ind - 1
        return 7 # case for > 84852 m.

    @staticmethod
    def thermal_conductivity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        thermal conductivity of air as a function of `T` only.

        .. math::
            k_g = \frac{2.64638\times10^{-3}T^{1.5}}
            {T + 245.4\cdot 10^{-12./T}}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        kg : float
            Thermal conductivity, [W/m/K]
        '''
        # 10**(-12./T) = exp(-12*log(10)/T) = -27.63102111...
        return 2.64638E-3*T*sqrt(T)/(T + 245.4*exp(-27.63102111592855/T))

    @staticmethod
    def viscosity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        viscosity of air as a function of `T` only.

        .. math::
            \mu_g = \frac{1.458\times10^{-6}T^{1.5}}{T+110.4}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        mug : float
            Viscosity, [Pa*s]
        '''
        return 1.458E-6*T*sqrt(T)/(T + 110.4)

    @staticmethod
    def density(T, P):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        density of air as a function of `T` and `P`. MW is defined as 28.9644
        g/mol, and R as 8314.32 J/kmol/K

        .. math::
            \rho_g = \frac{P\cdot MW}{T\cdot R\cdot 1000}

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]

        Returns
        -------
        rho : float
            Mass density, [kg/m^3]
        '''
        # 0.00348367635597379 = M0/R
        return P*0.00348367635597379/T

    @staticmethod
    def sonic_velocity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        the speed of sound in air as a function of `T` only.

        .. math::
            c = \left(\frac{\gamma R T}{MW}\right)^{0.5}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        c : float
            Speed of sound, [m/s]
        '''
        # 401.87... = gamma*R/MO
        return sqrt(401.87430086589046*T)

    @staticmethod
    def gravity(Z):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        the gravitational acceleration above earth as a function of elevation
        only.

        .. math::
            g = g_0\left(\frac{r_0}{r_0+Z}\right)^2

        Parameters
        ----------
        Z : float
            Elevation above sea level, [m]

        Returns
        -------
        g : float
            Acceleration due to gravity, [m/s^2]
        '''
        x0 = (r0/(r0+Z))
        return g0*x0*x0

    @staticmethod
    def pressure_integral(T1, P1, dH):
        r'''Method to compute an integral of the pressure differential of an
        elevation difference with a base elevation defined by temperature `T1`
        and pressure `P1`. This is
        similar to subtracting the pressures at two different elevations,
        except it allows for local conditions (temperature and pressure) to be
        taken into account. This is useful for e.x. evaluating the pressure
        difference between the top and bottom of a natural draft cooling tower.


        Parameters
        ----------
        T1 : float
            Temperature at the lower elevation condition, [K]
        P1 : float
            Pressure at the lower elevation condition, [Pa]
        dH : float
            Elevation difference for which to evaluate the pressure difference,
            [m]

        Returns
        -------
        delta_P : float
            Pressure difference between the elevations, [Pa]
        '''
        # Compute the elevation to obtain the pressure specified
        H_ref = brenth(H_for_P_ATMOSPHERE_1976_err, -610.0, 86000.0, args=(P1,))

        # Compute the temperature delta
        dT = T1 - ATMOSPHERE_1976(H_ref).T

        return quad(to_int_dP_ATMOSPHERE_1976, H_ref, H_ref+dH, args=(dT,))[0]



class ATMOSPHERE_NRLMSISE00(object):
    r'''NRLMSISE 00 model for calculating temperature and density of gases in
    the atmosphere, from ground level to 1000 km, as a function of time of year,
    longitude and latitude, solar activity and earth's geomagnetic disturbance.

    NRLMSISE stands for the `US Naval Research Laboratory Mass Spectrometer and
    Incoherent Scatter Radar Exosphere` model, released in 2001; see [1]_ for
    details.

    Parameters
    ----------
    Z : float
        Elevation, [m]
    latitude : float, optional
        Latitude, between -90 and 90 [degrees]
    longitude : float, optional
        Longitude, between -180 and 180 or 0 and 360, [degrees]
    day : float, optional
        Day of year, 0-366 [day]
    seconds : float, optional
        Seconds since start of day, in UT1 time; using UTC provides no loss in
        accuracy [s]
    f107 : float, optional
        Daily average 10.7 cm solar flux measurement of the strength of solar
        emissions on the 100 MHz band centered on 2800 MHz, averaged hourly; in
        sfu units, which are multiples of 10^-22 W/m^2/Hz; use 150 as a default
        [10^-22 W/m^2/Hz]
    f107_avg : float, optional
        81-day sfu average; centered on specified day if possible, otherwise
        use the previous days [10^-22 W/m^2/Hz]
    geomagnetic_disturbance_indices : list of float, optional
        List of the 7 following `Ap` indexes also known as planetary magnetic
        indexes. Has a negligible effect on the calculation. 4 is the default
        value often used for each of these values.

        * Average daily `Ap`.
        * 3-hour average `Ap` centered on the current time.
        * 3-hour average `Ap` before the current time.
        * 6-hour average `Ap` before the current time.
        * 9-hour average `Ap` before the current time.
        * Average `Ap` from 12 to 33 hours before the current time, based on
          eight 3-hour average `Ap` values.
        * Average `Ap` from 36 to 57 hours before the current time, based on
          eight 3-hour average `Ap` values.

    Attributes
    ----------
    rho : float
        Mass density [kg/m^3]
    T : float
        Temperature, [K]
    P : float
        Pressure, calculated with ideal gas law [Pa]
    He_density : float
        Density of helium atoms [count/m^3]
    O_density : float
        Density of monatomic oxygen [count/m^3]
    N2_density : float
        Density of nitrogen molecules [count/m^3]
    O2_density : float
        Density of oxygen molecules [count/m^3]
    Ar_density : float
        Density of Argon atoms [count/m^3]
    H_density : float
        Density of hydrogen atoms [count/m^3]
    N_density : float
        Density of monatomic nitrogen [count/m^3]
    O_anomalous_density : float
        Density of `anomalous` oxygen; see [1]_ for details [count/m^3]
    particle_density : float
        Total density of molecules [count/m^3]
    components : list[str]
        List of species making up the atmosphere [-]
    zs : list[float]
        Mole fractions of each molecule in the atmosphere, in order of
        `components` [-]

    Examples
    --------
    >>> atmosphere = ATMOSPHERE_NRLMSISE00(1E3, 45, 45, 150)
    >>> atmosphere.T, atmosphere.rho
    (285.5440860623, 1.10190620264)

    Notes
    -----
    No full description has been published of this model; it has been defined by
    its implementation only. It was written in FORTRAN, and is accessible
    at ftp://hanna.ccmc.gsfc.nasa.gov/pub/modelweb/atmospheric/msis/nrlmsise00/

    A C port of the model by Dominik Brodowskihas become popular, and is
    available on his website: http://www.brodo.de/space/nrlmsise/.

    In 2013 Joshua Milas ported the C port to Python. This is an interface to
    his excellent port. It is a 1000-sloc model, and has
    been rigorously tested against the C version, and the online calculation
    tool available at [3]_ for parametric inputs of latitude, longitude,
    altitude, time of day and day of year.

    This model is based on measurements other than gravity; it does not provide
    a calculation method for `g`. It does not provide transport properties.

    This model takes on the order of ~2 ms.

    References
    ----------
    .. [1] Picone, J. M., A. E. Hedin, D. P. Drob, and A. C. Aikin.
       "NRLMSISE-00 Empirical Model of the Atmosphere: Statistical Comparisons
       and Scientific Issues." Journal of Geophysical Research: Space Physics
       107, no. A12 (December 1, 2002): 1468. doi:10.1029/2002JA009430.
    .. [2] Tapping, K. F. "The 10.7 Cm Solar Radio Flux (F10.7)." Space Weather
       11, no. 7 (July 1, 2013): 394-406. doi:10.1002/swe.20064.
    .. [3] Natalia Papitashvili. "NRLMSISE-00 Atmosphere Model." Accessed
       November 27, 2016. http://ccmc.gsfc.nasa.gov/modelweb/models/nrlmsise00.php.
    '''
    components = ['N2', 'O2', 'Ar', 'He', 'O', 'H', 'N']
    atrrs = ['N2_density', 'O2_density', 'Ar_density', 'He_density',
             'O_density', 'H_density', 'N_density']
    MWs = [28.0134, 31.9988, 39.948, 4.002602, 15.9994, 1.00794, 14.0067]

    def __init__(self, Z, latitude=0.0, longitude=0.0, day=0, seconds=0.0,
                 f107=150., f107_avg=150., geomagnetic_disturbance_indices=None):
        self.Z = Z
        self.latitude = latitude
        self.longitude = longitude
        self.day = day
        self.seconds = seconds
        self.f107 = f107
        self.f107_avg = f107_avg
        self.geomagnetic_disturbance_indices = geomagnetic_disturbance_indices

        from fluids.nrlmsise00 import gtd7, nrlmsise_output, nrlmsise_input, nrlmsise_flags, ap_array
        alt = Z*1e-3
        output_obj = nrlmsise_output()
        input_obj = nrlmsise_input()
        flags = nrlmsise_flags()

        flags.switches = [0] + [1]*23

        if geomagnetic_disturbance_indices:
            aph = ap_array()
            aph.a = geomagnetic_disturbance_indices
            flags.switches[9] = -1
            input_obj.ap = geomagnetic_disturbance_indices[0]
            input_obj.ap_a = aph

        input_obj.doy = day
        input_obj.year = 0
        input_obj.sec = seconds
        input_obj.alt = alt
        input_obj.g_lat = latitude
        input_obj.g_long = longitude
        input_obj.lst = seconds/3600. + longitude/15.
        input_obj.f107A = f107_avg
        input_obj.f107 = f107
        gtd7(input_obj, flags, output_obj)

        self.He_density = output_obj.d[0]*1E6 # 1/cm^3 to 1/m^3
        self.O_density = output_obj.d[1]*1E6 # 1/cm^3 to 1/m^3
        self.N2_density = output_obj.d[2]*1E6 # 1/cm^3 to 1/m^3
        self.O2_density = output_obj.d[3]*1E6 # 1/cm^3 to 1/m^3
        self.Ar_density = output_obj.d[4]*1E6 # 1/cm^3 to 1/m^3
        self.rho = output_obj.d[5]*1000 # gram/cm^3 to kg/m^3
        self.H_density = output_obj.d[6]*1E6 # 1/cm^3 to 1/m^3
        self.N_density = output_obj.d[7]*1E6 # 1/cm^3 to 1/m^3
        self.O_anomalous_density = output_obj.d[8]*1E6 # 1/cm^3 to 1/m^3
        self.T_exospheric = output_obj.t[0]
        self.T = output_obj.t[1]

        # Calculate pressure with the ideal gas law PV = nRT with V = 1 m^3
        self.P = sum([getattr(self, a) for a in self.atrrs])*self.T*R/N_A
        # Calculate mass density with known MWs
        self.rho_calculated = sum([getattr(self, a)*MW for c, a, MW in
                                   zip(self.components, self.atrrs, self.MWs)])/(1000.*N_A)

        self.particle_density = sum(getattr(self, a) for a in self.atrrs)
        self.zs = [getattr(self, a)/self.particle_density for a in self.atrrs]


def hwm93(Z, latitude=0, longitude=0, day=0, seconds=0, f107=150.,
          f107_avg=150., geomagnetic_disturbance_index=4):
    r'''Horizontal Wind Model 1993, for calculating wind velocity in the
    atmosphere as a function of time of year, longitude and latitude, solar
    activity and earth's geomagnetic disturbance.

    The model is described across the publications [1]_, [2]_, and [3]_.

    Parameters
    ----------
    Z : float
        Elevation, [m]
    latitude : float, optional
        Latitude, between -90 and 90 [degrees]
    longitude : float, optional
        Longitude, between -180 and 180 or 0 and 360, [degrees]
    day : float, optional
        Day of year, 0-366 [day]
    seconds : float, optional
        Seconds since start of day, in UT1 time; using UTC provides no loss in
        accuracy [s]
    f107 : float, optional
        Daily average 10.7 cm solar flux measurement of the strength of solar
        emissions on the 100 MHz band centered on 2800 MHz, averaged hourly; in
        sfu units, which are multiples of 10^-22 W/m^2/Hz; use 150 as a default
        [W/m^2/Hz]
    f107_avg : float, optional
        81-day sfu average; centered on specified day if possible, otherwise
        use the previous days [W/m^2/Hz]
    geomagnetic_disturbance_index : float, optional
        Average daily `Ap` or also known as planetary magnetic index.

    Returns
    -------
    v_north : float
        Wind velocity, meridional (Northward) [m/s]
    v_east : float
        Wind velocity, zonal (Eastward) [m/s]

    Examples
    --------
    >>> hwm93(5E5, 45, 50, 365) # doctest: +SKIP
    (-73.00312042236328, 0.1485661268234253)

    Notes
    -----
    No full description has been published of this model; it has been defined by
    its implementation only. It was written in FORTRAN, and is accessible
    at ftp://hanna.ccmc.gsfc.nasa.gov/pub/modelweb/atmospheric/hwm93/.

    F2PY auto-compilation support is not yet currently supported.
    To compile this file, run the following command in a shell after navigating
    to $FLUIDSPATH/fluids/optional/. This should generate the file hwm93.so
    in that directory.

    .. code-block:: bash

        f2py -c hwm93.pyf hwm93.for --f77flags="-std=legacy"

    If the module is not compiled, an import error will be raised.

    References
    ----------
    .. [1] Hedin, A. E., N. W. Spencer, and T. L. Killeen. "Empirical Global
       Model of Upper Thermosphere Winds Based on Atmosphere and Dynamics
       Explorer Satellite Data." Journal of Geophysical Research: Space Physics
       93, no. A9 (September 1, 1988): 9959-78. doi:10.1029/JA093iA09p09959.
    .. [2] Hedin, A. E., M. A. Biondi, R. G. Burnside, G. Hernandez, R. M.
       Johnson, T. L. Killeen, C. Mazaudier, et al. "Revised Global Model of
       Thermosphere Winds Using Satellite and Ground-Based Observations."
       Journal of Geophysical Research: Space Physics 96, no. A5 (May 1, 1991):
       7657-88. doi:10.1029/91JA00251.
    .. [3] Hedin, A. E., E. L. Fleming, A. H. Manson, F. J. Schmidlin, S. K.
       Avery, R. R. Clark, S. J. Franke, et al. "Empirical Wind Model for the
       Upper, Middle and Lower Atmosphere." Journal of Atmospheric and
       Terrestrial Physics 58, no. 13 (September 1996): 1421-47.
       doi:10.1016/0021-9169(95)00122-0.
    '''
    try:
        from fluids.optional.hwm93 import gws5
    except: # pragma: no cover
        raise ImportError(no_gfortran_error)
    slt_hour = seconds/3600. + longitude/15.
    ans = gws5(day, seconds, Z/1000., latitude, longitude, slt_hour, f107,
               f107_avg, geomagnetic_disturbance_index)
    return tuple(ans.tolist())


def hwm14(Z, latitude=0, longitude=0, day=0, seconds=0,
          geomagnetic_disturbance_index=4):
    r'''Horizontal Wind Model 2014, for calculating wind velocity in the
    atmosphere as a function of time of year, longitude and latitude, and
    earth's geomagnetic disturbance. The model is described in [1]_.

    The model no longer accounts for solar flux.

    Parameters
    ----------
    Z : float
        Elevation, [m]
    latitude : float, optional
        Latitude, between -90 and 90 [degrees]
    longitude : float, optional
        Longitude, between -180 and 180 or 0 and 360, [degrees]
    day : float, optional
        Day of year, 0-366 [day]
    seconds : float, optional
        Seconds since start of day, in UT1 time; using UTC provides no loss in
        accuracy [s]
    geomagnetic_disturbance_index : float, optional
        Average daily `Ap` or also known as planetary magnetic index.

    Returns
    -------
    v_north : float
        Wind velocity, meridional (Northward) [m/s]
    v_east : float
        Wind velocity, zonal (Eastward) [m/s]


    Examples
    --------
    >>> hwm14(5E5, 45, 50, 365) # doctest: +SKIP
    (-38.64341354370117, 12.871272087097168)

    Notes
    -----
    No full description has been published of this model; it has been defined by
    its implementation only. It was written in FORTRAN, and is accessible
    at http://onlinelibrary.wiley.com/store/10.1002/2014EA000089/asset/supinfo/ess224-sup-0002-supinfo.tgz?v=1&s=2a957ba70b7cf9dd0612d9430076297c3634ea75.

    F2PY auto-compilation support is not yet currently supported.
    To compile this file, run the following commands in a shell after navigating
    to $FLUIDSPATH/fluids/optional/. This should generate the file hwm14.so
    in that directory.


    Generate a .pyf signature file:

    .. code-block:: bash

        f2py -m hwm14 -h hwm14.pyf hwm14.f90

    Compile the interface:

    .. code-block:: bash

        f2py -c hwm14.pyf hwm14.f90


    If the module is not compiled, an import error will be raised.

    No patches were necessary to either the generated pyf or hwm14.f90 file,
    as the authors of [1]_ have made it F2PY compatible.

    Developed using 73 million data points taken by 44 instruments over 60
    years.

    References
    ----------
    .. [1] Drob, Douglas P., John T. Emmert, John W. Meriwether, Jonathan J.
       Makela, Eelco Doornbos, Mark Conde, Gonzalo Hernandez, et al. "An Update
       to the Horizontal Wind Model (HWM): The Quiet Time Thermosphere." Earth
       and Space Science 2, no. 7 (July 1, 2015): 2014EA000089.
       doi:10.1002/2014EA000089.
    '''
    # Needed by hwm14
    os.environ["HWMPATH"] = os.path.join(os.path.dirname(__file__), 'optional')
    try:
        try:
            from fluids.optional import hwm14
        except:
            import optional.hwm14
    except: # pragma: no cover
        raise ImportError(no_gfortran_error)
    ans = hwm14.hwm14(day, seconds, Z*1e-3, latitude, longitude, 0, 0,
               0, np.array([np.nan, geomagnetic_disturbance_index]))
    return tuple(ans.tolist())


def to_int_airmass(Z, c1, c2, angle_term, R_planet_inv, func):
    rho = func(Z)
    t1 = c2 - rho*c1
    x0 = angle_term/(1.0 + Z*R_planet_inv)
    t2 = x0*x0
    t3 = 1.0/sqrt(1.0 - t1*t2)
    return rho*t3

def airmass(func, angle, H_max=86400.0, R_planet=6.371229E6, RI=1.000276):
    r'''Calculates mass of air per square meter in the atmosphere using a
    provided atmospheric model. The lowest air mass is calculated straight up;
    as the angle is lowered to nearer and nearer the horizon, the air mass
    increases, and can approach 40x or more the minimum airmass.

    .. math::
        m(\gamma) = \int_0^\infty \rho \left\{1 - \left[1 + 2(\text{RI}-1)
        (1-\rho/\rho_0)\right]
        \left[\frac{\cos \gamma}{(1+h/R)}\right]^2\right\}^{-1/2} dH

    Parameters
    ----------
    func : float
        Function which returns the density of the atmosphere as a function of
        elevation
    angle : float
        Degrees above the horizon (90 = straight up), [degrees]
    H_max : float, optional
        Maximum height to compute the integration up to before the contribution
        of density becomes negligible, [m]
    R_planet : float, optional
        The radius of the planet for which the integration is being performed,
        [m]
    RI : float, optional
        The refractive index of the atmosphere (air on earth at 0.7 um as
        default) assumed a constant, [-]

    Returns
    -------
    m : float
        Mass of air per square meter in the atmosphere, [kg/m^2]

    Notes
    -----
    Numerical integration via SciPy's `quad` is used to perform the
    calculation.

    Examples
    --------
    >>> airmass(lambda Z : ATMOSPHERE_1976(Z).rho, 90)
    10356.12

    References
    ----------
    .. [1] Kasten, Fritz, and Andrew T. Young. "Revised Optical Air Mass Tables
       and Approximation Formula." Applied Optics 28, no. 22 (November 15,
       1989): 4735-38. https://doi.org/10.1364/AO.28.004735.
    '''
    delta0 = RI - 1.0
    rho0_inv = 1.0/func(0.0)
    angle_term = cos(radians(angle))
    R_planet_inv = 1.0/R_planet

    c0 = delta0 + delta0
    c1 = c0*rho0_inv
    c2 = 1.0 + c0
    return quad(to_int_airmass, 0.0, 86400.0, args=(c1, c2, angle_term, R_planet_inv, func))[0]



PVLIB_MISSING_MSG = 'The module pvlib is required for this function; install it first'


def earthsun_distance(moment):
    r'''Calculates the distance between the earth and the sun as a function
    of date and time. Uses the Reda and Andreas (2004) model described in [1]_,
    originally incorporated into the excellent
    `pvlib library <https://github.com/pvlib/pvlib-python>`_

    Parameters
    ----------
    moment : datetime
        Time and date for the calculation, in UTC time (or GMT, which is
        almost the same thing); OR a timezone-aware datetime instance
        which will be internally converted to UTC, [-]

    Returns
    -------
    distance : float
        Distance between the center of the earth and the center of the sun,
        [m]

    Examples
    --------
    >>> earthsun_distance(datetime(2003, 10, 17, 13, 30, 30))
    149090925951.18338

    The distance at perihelion, which occurs at 4:21 according to this
    algorithm. The real value is 04:38 (January 2nd).

    >>> earthsun_distance(datetime(2013, 1, 2, 4, 21, 50))
    147098089490.67123

    The distance at aphelion, which occurs at 14:44 according to this
    algorithm. The real value is dead on - 14:44 (July 5).

    >>> earthsun_distance(datetime(2013, 7, 5, 14, 44, 51, 0))
    152097354414.36044

    Using a timezone-aware date:

    >>> import pytz
    >>> earthsun_distance(pytz.timezone('America/Edmonton').localize(datetime(2020, 6, 6, 10, 0, 0, 0)))
    151817805599.67142

    This has a slightly different value than the value without a timezone;
    almost 5000 km further away!

    >>> earthsun_distance(datetime(2020, 6, 6, 10, 0, 0, 0))
    151812898579.44104

    Notes
    -----
    This function is quite accurate. The difference comes from the impact of
    the moon.

    Note this function is not continuous; the sun-earth distance is not
    sufficiently accurately modeled for the change to be continuous throughout
    each day.

    References
    ----------
    .. [1] Reda, Ibrahim, and Afshin Andreas. "Solar Position Algorithm for
       Solar Radiation Applications." Solar Energy 76, no. 5 (January 1, 2004):
       577-89. https://doi.org/10.1016/j.solener.2003.12.003.
    '''
    from fluids.optional import spa
    delta_t = spa.calculate_deltat(moment.year, moment.month)
    import calendar
    unixtime = calendar.timegm(moment.utctimetuple())
    # Convert datetime object to unixtime
    return spa.earthsun_distance(unixtime, delta_t=delta_t)*au


def solar_position(moment, latitude, longitude, Z=0.0, T=298.15, P=101325.0,
                   atmos_refract=0.5667):
    r'''Calculate the position of the sun in the sky. It is defined in terms of
    two angles - the zenith and the azimith. The azimuth tells where a sundial
    would see the sun as coming from; the zenith tells how high in the sky it
    is. The solar elevation angle is returned for convenience; it is the
    complimentary angle of the zenith.

    The sun's refraction changes how high it appears as though the sun is;
    so values are returned with an optional conversion to the apparent angle.
    This impacts only the zenith/elevation.

    Uses the Reda and Andreas (2004) model described in [1]_,
    originally incorporated into the excellent
    `pvlib library <https://github.com/pvlib/pvlib-python>`_

    Parameters
    ----------
    moment : datetime, optionally with pytz info
        Time and date for the calculation, in UTC time OR in the time zone
        of the latitude/longitude specified BUT WITH A TZINFO ATTACHED!
        Please be careful with this argument, time zones are confusing. [-]
    latitude : float
        Latitude, between -90 and 90 [degrees]
    longitude : float
        Longitude, between -180 and 180, [degrees]
    Z : float, optional
        Elevation above sea level for the solar position calculation, [m]
    T : float, optional
        Temperature of atmosphere at ground level, [K]
    P : float, optional
        Pressure of atmosphere at ground level, [Pa]
    atmos_refract : float, optional
        Atmospheric refractivity, [degrees]

    Returns
    -------
    apparent_zenith : float
        Zenith of the sun as observed from the ground based after accounting
        for atmospheric refraction, [degrees]
    zenith : float
        Actual zenith of the sun (ignores atmospheric refraction), [degrees]
    apparent_altitude : float
        Altitude of the sun as observed from the ground based after accounting
        for atmospheric refraction, [degrees]
    altitude : float
        Actual altitude of the sun (ignores atmospheric refraction), [degrees]
    azimuth : float
        The azimuth of the sun, [degrees]
    equation_of_time : float
        Equation of time - the number of seconds to be added to the day's
        mean solar time to obtain the apparent solar noon time, [seconds]

    Examples
    --------
    >>> import pytz

    Perth, Australia - sunrise

    >>> solar_position(pytz.timezone('Australia/Perth').localize(datetime(2020, 6, 6, 7, 10, 57)), -31.95265, 115.85742)
    [90.89617025931, 90.89617025931, -0.896170259317, -0.896170259317, 63.6016017691, 79.0711232143]

    Perth, Australia - Comparing against an online source
    https://www.suncalc.org/#/-31.9526,115.8574,9/2020.06.06/14:30/1/0

    >>> solar_position(pytz.timezone('Australia/Perth').localize(datetime(2020, 6, 6, 14, 30, 0)), -31.95265, 115.85742)
    [63.4080568623, 63.4400018158, 26.59194313766, 26.55999818417, 325.121376246, 75.7467475485]

    Perth, Australia - time input without timezone; must be converted by user to UTC!

    >>> solar_position(datetime(2020, 6, 6, 14, 30, 0) - timedelta(hours=8), -31.95265, 115.85742)
    [63.4080568623, 63.4400018158, 26.59194313766, 26.55999818417, 325.121376246, 75.7467475485]

    Sunrise occurs when the zenith is 90 degrees (Calgary, AB):

    >>> local_time = datetime(2018, 4, 15, 6, 43, 5)
    >>> local_time = pytz.timezone('America/Edmonton').localize(local_time)
    >>> solar_position(local_time, 51.0486, -114.07)[0]
    90.0005468548

    Sunset occurs when the zenith is 90 degrees (13.5 hours later in this case):

    >>> solar_position(pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 20, 30, 28)), 51.0486, -114.07)
    [89.999569566, 90.5410381216, 0.000430433876, -0.541038121618, 286.831378190, 6.63142952587]

    Notes
    -----
    If you were standing at the same longitude of the sun such that it was no
    further east or west than you were, the amount of angle it was south or
    north of you is the *zenith*. If it were directly overhead it would be 0°;
    a little north or south and it would be a little positive;
    near sunset or sunrise, near 90°; and at night, between 90° and 180°.

    The *solar altitude angle* is defined as 90° -`zenith`.
    Note the *elevation* angle is just another name for the *altitude* angle.

    The *azimuth* the angle in degrees that the sun is East of the North angle.
    It is positive North eastwards 0° to 360°. Other conventions may be used.

    Note that due to differences in atmospheric refractivity, estimation of
    sunset and sunrise are accuract to no more than one minute. Refraction
    conditions truly vary across the atmosphere; so characterizing it by an
    average value is limiting as well.

    References
    ----------
    .. [1] Reda, Ibrahim, and Afshin Andreas. "Solar Position Algorithm for
       Solar Radiation Applications." Solar Energy 76, no. 5 (January 1, 2004):
       577-89. https://doi.org/10.1016/j.solener.2003.12.003.
    .. [2] "Navigation - What Azimuth Description Systems Are in Use? -
       Astronomy Stack Exchange."
       https://astronomy.stackexchange.com/questions/237/what-azimuth-description-systems-are-in-use?rq=1.
    '''
    from fluids.optional import spa
    import calendar
    tt = moment.utctimetuple()
    delta_t = spa.calculate_deltat(tt.tm_year, tt.tm_mon)
    unixtime = calendar.timegm(tt)
    # Input pressure in milibar; input temperature in deg C
#    print(dict(unixtime=unixtime, lat=latitude, lon=longitude, elev=Z,
#                          pressure=P*1E-2, temp=T-273.15, delta_t=delta_t,
#                          atmos_refract=atmos_refract, sst=False))
    result = spa.solar_position(unixtime, lat=latitude, lon=longitude, elev=Z,
                          pressure=P*1E-2, temp=T-273.15, delta_t=delta_t,
                          atmos_refract=atmos_refract, sst=False)
    # confirmed equation of time https://www.minasi.com/figeot.asp
    # Convert minutes to seconds; sometimes negative, sometimes positive

    result[-1] = result[-1]*60.0
    return result


def sunrise_sunset(moment, latitude, longitude):
    r'''Calculates the times at which the sun is at sunset; sunrise; and
    halfway between sunrise and sunset (transit).

    Uses the Reda and Andreas (2004) model described in [1]_,
    originally incorporated into the excellent
    `pvlib library <https://github.com/pvlib/pvlib-python>`_

    Parameters
    ----------
    moment : datetime
        Date for the calculation; needs to contain only the year, month, and
        day; if it is timezone-aware, the return values will be localized to
        this timezone [-]
    latitude : float
        Latitude, between -90 and 90 [degrees]
    longitude : float
        Longitude, between -180 and 180, [degrees]

    Returns
    -------
    sunrise : datetime
        The time at the specified day when the sun rises **IN UTC IF MOMENT
        DOES NOT HAVE A TIMEZONE, OTHERWISE THE TIMEZONE GIVEN WITH IT**, [-]
    sunset : datetime
        The time at the specified day when the sun sets **IN UTC IF MOMENT
        DOES NOT HAVE A TIMEZONE, OTHERWISE THE TIMEZONE GIVEN WITH IT**, [-]
    transit : datetime
        The time at the specified day when the sun is at solar noon - halfway
        between sunrise and sunset **IN UTC IF MOMENT
        DOES NOT HAVE A TIMEZONE, OTHERWISE THE TIMEZONE GIVEN WITH IT**, [-]

    Examples
    --------
    >>> sunrise, sunset, transit = sunrise_sunset(datetime(2018, 4, 17),
    ... 51.0486, -114.07)
    >>> sunrise
    datetime.datetime(2018, 4, 17, 12, 36, 55, 782660)
    >>> sunset
    datetime.datetime(2018, 4, 18, 2, 34, 4, 249326)
    >>> transit
    datetime.datetime(2018, 4, 17, 19, 35, 46, 686265)

    Example with time zone:

    >>> import pytz
    >>> sunrise_sunset(pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 17)), 51.0486, -114.07)
    (datetime.datetime(2018, 4, 16, 6, 39, 1, 570479, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>), datetime.datetime(2018, 4, 16, 20, 32, 25, 778162, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>), datetime.datetime(2018, 4, 16, 13, 36, 0, 386341, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>))

    Note that the year/month/day as input with a timezone, is converted to UTC
    time as well.


    Notes
    -----
    This functions takes on the order of 2 ms per calculation.

    References
    ----------
    .. [1] Reda, Ibrahim, and Afshin Andreas. "Solar Position Algorithm for
       Solar Radiation Applications." Solar Energy 76, no. 5 (January 1, 2004):
       577-89. https://doi.org/10.1016/j.solener.2003.12.003.
    '''
    from fluids.optional import spa
    import calendar
    if moment.utcoffset() is not None:
        moment_utc = moment + moment.utcoffset()
    else:
        moment_utc = moment

    delta_t = spa.calculate_deltat(moment_utc.year, moment_utc.month)
    # Strip the part of the day
    ymd_moment_utc = datetime(moment_utc.year, moment_utc.month, moment_utc.day)
    unixtime = calendar.timegm(ymd_moment_utc.utctimetuple())

    unixtime = unixtime - unixtime % (86400) # Remove the remainder of the value, rounding it to the day it is
    transit, sunrise, sunset = spa.transit_sunrise_sunset(unixtime, lat=latitude, lon=longitude, delta_t=delta_t)

    transit = datetime.utcfromtimestamp(transit)
    sunrise = datetime.utcfromtimestamp(sunrise)
    sunset = datetime.utcfromtimestamp(sunset)

    if moment.tzinfo is not None:
        sunrise = moment.tzinfo.fromutc(sunrise)
        sunset = moment.tzinfo.fromutc(sunset)
        transit = moment.tzinfo.fromutc(transit)
    return sunrise, sunset, transit


apparent_zenith_airmass_models = set(['simple', 'kasten1966', 'kastenyoung1989',
                                   'gueymard1993', 'pickering2002'])
true_zenith_airmass_models = set(['youngirvine1967', 'young1994'])


def _get_extra_radiation_shim(datetime_or_doy, solar_constant=1366.1,
    method='spencer', epoch_year=2014, **kwargs):
    if method == 'spencer':
        if not isinstance(datetime_or_doy, (float, int)):
            dayofyear = datetime_or_doy.timetuple().tm_yday
        else:
            dayofyear = datetime_or_doy
        B = (2.*pi/365.)*(dayofyear - 1)
        RoverR0sqrd = (1.00011 + 0.034221*cos(B) + 0.00128*sin(B) +
        0.000719*cos(2.0*B) + 7.7e-05*sin(2.0*B))

        Ea = solar_constant * RoverR0sqrd
        return Ea
    from pvlib.irradiance import get_extra_radiation
    return get_extra_radiation(datetime_or_doy=datetime_or_doy,
                              solar_constant=solar_constant,
                              method=method,
                              epoch_year=epoch_year,
                              **kwargs)


def solar_irradiation(latitude, longitude, Z, moment, surface_tilt,
                      surface_azimuth, T=None, P=None, solar_constant=1366.1,
                      atmos_refract=0.5667, albedo=0.25, linke_turbidity=None,
                      extraradiation_method='spencer',
                      airmass_model='kastenyoung1989',
                      cache=None):
    r'''Calculates the amount of solar radiation and radiation reflected back
    the atmosphere which hits a surface at a specified tilt, and facing a
    specified azimuth.

    This functions is a wrapper for the incredibly
    comprehensive `pvlib library <https://github.com/pvlib/pvlib-python>`_,
    and requires it to be installed.

    Parameters
    ----------
    latitude : float
        Latitude, between -90 and 90 [degrees]
    longitude : float
        Longitude, between -180 and 180, [degrees]
    Z : float, optional
        Elevation above sea level for the position, [m]
    moment : datetime, optionally with pytz info
        Time and date for the calculation, in UTC time OR in the time zone
        of the latitude/longitude specified BUT WITH A TZINFO ATTACHED!
        Please be careful with this argument, time zones are confusing. [-]
    surface_tilt : float
        The angle above the horizontal of the object being hit by radiation,
        [degrees]
    surface_azimuth : float
        The angle the object is facing (positive, North eastwards 0° to 360°),
        [degrees]
    T : float, optional
        Temperature of atmosphere at ground level, [K]
    P : float, optional
        Pressure of atmosphere at ground level, [Pa]
    solar_constant : float, optional
        The amount of solar radiation which reaches earth's disk (at a
        standardized distance of 1 AU); this constant is independent of
        activity or conditions on earth, but will vary throughout the sun's
        lifetime and may increase or decrease slightly due to solar activity,
        [W/m^2]
    atmos_refract : float, optional
        Atmospheric refractivity at sunrise/sunset (0.5667 deg is an often used
        value; this varies substantially and has an impact of a few minutes on
        when sunrise and sunset is), [degrees]
    albedo : float, optional
        The average amount of reflection of the terrain surrounding the object
        at quite a distance; this impacts how much sunlight reflected off the
        ground, gets reflected back off clouds, [-]
    linke_turbidity : float, optional
        The amount of pollution/water in the sky versus a perfect clear sky;
        If not specified, this will be retrieved from a historical grid;
        typical values are 3 for cloudy, and 7 for severe pollution around a
        city, [-]
    extraradiation_method : str, optional
        The specified method to calculate the effect of earth's position on the
        amount of radiation which reaches earth according to the methods
        available in the `pvlib` library, [-]
    airmass_model : str, optional
        The specified method to calculate the amount of air the sunlight
        needs to travel through to reach the earth according to the methods
        available in the `pvlib` library, [-]
    cache : dict, optional
        Dictionary to to check for values to use to skip some calculations;
        `apparent_zenith`, `zenith`, `azimuth` supported, [-]

    Returns
    -------
    poa_global : float
        The total irradiance in the plane of the surface, [W/m^2]
    poa_direct : float
        The total beam irradiance in the plane of the surface, [W/m^2]
    poa_diffuse : float
        The total diffuse irradiance in the plane of the surface, [W/m^2]
    poa_sky_diffuse : float
        The sky component of the diffuse irradiance, excluding the impact
        from the ground, [W/m^2]
    poa_ground_diffuse : float
        The ground-sky diffuse irradiance component, [W/m^2]

    Examples
    --------
    >>> import pytz
    >>> solar_irradiation(Z=1100.0, latitude=51.0486, longitude=-114.07, linke_turbidity=3,
    ... moment=pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 13, 43, 5)), surface_tilt=41.0,
    ... surface_azimuth=180.0)
    (1065.7621896280, 945.2656564506, 120.49653317744, 95.31535344213, 25.181179735317)

    >>> cache = {'apparent_zenith': 41.099082295767545, 'zenith': 41.11285376417578, 'azimuth': 182.5631874250523}
    >>> solar_irradiation(Z=1100.0, latitude=51.0486, longitude=-114.07,
    ... moment=pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 13, 43, 5)), surface_tilt=41.0,
    ... linke_turbidity=3, T=300, P=1E5,
    ... surface_azimuth=180.0, cache=cache)
    (1042.567770367, 918.237754854, 124.3300155131, 99.622865737, 24.7071497753)

    At night, there is no solar radiation and this function returns zeros:

    >>> solar_irradiation(Z=1100.0, latitude=51.0486, longitude=-114.07, linke_turbidity=3,
    ... moment=pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 2, 43, 5)), surface_tilt=41.0,
    ... surface_azimuth=180.0)
    (0.0, -0.0, 0.0, 0.0, 0.0)


    Notes
    -----
    The retrieval of `linke_turbidity` requires the pytables library (and
    Pandas); if it is not installed, specify a value of `linke_turbidity` to
    avoid the dependency.

    There is some redundancy of the calculated results, according to the
    following relations. The total irradiance is normally that desired for
    engineering calculations.

    poa_diffuse = poa_ground_diffuse + poa_sky_diffuse

    poa_global = poa_direct + poa_diffuse

    For a surface such as a pipe or vessel, an approach would be to split it
    into a number of rectangles and sum up the radiation absorbed by each.

    This calculation is fairly slow.

    References
    ----------
    .. [1] Will Holmgren, Calama-Consulting, Tony Lorenzo, Uwe Krien, bmu,
       DaCoEx, mayudong, et al. Pvlib/Pvlib-Python: 0.5.1. Zenodo, 2017.
       https://doi.org/10.5281/zenodo.1016425.
    '''
    # Atmospheric refraction at sunrise/sunset (0.5667 deg is an often used value)
    import calendar
    from fluids.optional import spa
    from fluids.optional.irradiance import (get_relative_airmass, get_absolute_airmass,
                                            ineichen, get_relative_airmass,
                                            get_absolute_airmass, get_total_irradiance)

    moment_timetuple = moment.timetuple()
    moment_arg_dni = (moment_timetuple.tm_yday if
                      extraradiation_method == 'spencer' else moment)

    dni_extra = _get_extra_radiation_shim(moment_arg_dni, solar_constant=solar_constant,
                               method=extraradiation_method,
                               epoch_year=moment.year)

    if T is None or P is None:
        atmosphere = ATMOSPHERE_NRLMSISE00(Z=Z, latitude=latitude,
                                           longitude=longitude,
                                           day=moment_timetuple.tm_yday)
        if T is None:
            T = atmosphere.T
        if P is None:
            P = atmosphere.P

    if cache is not None and 'zenith' in cache:
        zenith = cache['zenith']
        apparent_zenith = cache['apparent_zenith']
        azimuth = cache['azimuth']
    else:
        apparent_zenith, zenith, _, _, azimuth, _ = solar_position(moment=moment,
                                                                   latitude=latitude,
                                                                   longitude=longitude,
                                                                   Z=Z, T=T, P=P,
                                                                   atmos_refract=atmos_refract)

    if linke_turbidity is None:
        try:
            import pvlib
        except:
            raise ImportError(PVLIB_MISSING_MSG)
        from pvlib.clearsky import lookup_linke_turbidity
        import pandas as pd
        linke_turbidity = float(lookup_linke_turbidity(
            pd.DatetimeIndex([moment]), latitude, longitude).values)


    if airmass_model in apparent_zenith_airmass_models:
        used_zenith = apparent_zenith
    elif airmass_model in true_zenith_airmass_models:
        used_zenith = zenith
    else:
        raise ValueError('Unrecognized airmass model')

    relative_airmass = get_relative_airmass(used_zenith, model=airmass_model)
    airmass_absolute = get_absolute_airmass(relative_airmass, pressure=P)


    ans = ineichen(apparent_zenith=apparent_zenith,
                   airmass_absolute=airmass_absolute,
                   linke_turbidity=linke_turbidity,
                   altitude=Z, dni_extra=solar_constant, perez_enhancement=True)
    ghi = ans['ghi']
    dni = ans['dni']
    dhi = ans['dhi']


#    from pvlib.irradiance import get_total_irradiance
    ans = get_total_irradiance(surface_tilt=surface_tilt,
                      surface_azimuth=surface_azimuth,
                      solar_zenith=apparent_zenith, solar_azimuth=azimuth,
                      dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra,
                      airmass=airmass_absolute, albedo=albedo)
    poa_global = float(ans['poa_global'])
    poa_direct = float(ans['poa_direct'])
    poa_diffuse = float(ans['poa_diffuse'])
    poa_sky_diffuse = float(ans['poa_sky_diffuse'])
    poa_ground_diffuse = float(ans['poa_ground_diffuse'])
    return (poa_global, poa_direct, poa_diffuse, poa_sky_diffuse,
            poa_ground_diffuse)
