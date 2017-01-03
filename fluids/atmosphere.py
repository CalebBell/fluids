# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import os
from math import exp
import numpy as np
from scipy.constants import N_A, R
from .nrlmsise00 import gtd7, nrlmsise_output, nrlmsise_input, nrlmsise_flags, ap_array

__all__ = ['ATMOSPHERE_1976', 'ATMOSPHERE_NRLMSISE00', 'hwm93', 'hwm14']

no_gfortran_error = '''This function uses f2py to encapsulate a fortran \
routine. However, f2py did not detect one on installation and could not compile \
this routine. '''


# Needed by hwm14
os.environ["HWMPATH"] = os.path.join(os.path.dirname(__file__), 'optional')


H_std = [0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852]
T_grad = [-6.5E-3, 0, 1E-3, 2.8E-3, 0, -2.8E-3, -2E-3, 0]
T_std = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
P_std = [101325, 22632.06397346291, 5474.8886696777745, 868.0186847552279,
        110.90630555496608, 66.93887311868738, 3.956420428040732,
        0.3733835899762159]

r0 = 6356766.0
P0 = 101325.0
M0 = 28.9644
g0 = 9.80665
gamma = 1.400


class ATMOSPHERE_1976(object):
    r'''US Standard Atmosphere 1976 class, which calculates `T`, `P`,
    `rho`, `v_sonic`, `mu`, `k`, and `g` as a function of altitude above 
    sea level. 
    
    Parameters
    ----------
    Z : float
        Elevation, [m]
    dT : float, optional
        Temperature difference from standard conditions used in determining
        the properties of the atmosphere, [K]
    
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
    R = 8314.32

    @staticmethod
    def get_ind_from_H(H):
        r'''Method defined in the US Standard Atmosphere 1976 for determining
        the index of the layer a specified elevation is above. Levels are 
        0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852 meters respectively.
        '''
        if H <= 0:
            return 0
        for ind, Hi in enumerate(H_std):
            if Hi >= H :
                return ind-1
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
        return 2.64638E-3*T**1.5/(T + 245.4*10**(-12./T))
    
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
        return 1.458E-6*T**1.5/(T+110.4)
    
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
        return (401.87430086589046*T)**0.5
    
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
        return g0*(r0/(r0+Z))**2


    def __init__(self, Z, dT=0):
        self.Z = Z
        self.H = r0*self.Z/(r0+self.Z)

        i = self.get_ind_from_H(self.H)
        self.T_layer = T_std[i]
        self.T_increase = T_grad[i]
        self.P_layer = P_std[i]
        self.H_layer = H_std[i]

        self.H_above_layer = self.H - self.H_layer
        self.T = self.T_layer + self.T_increase*self.H_above_layer

        if self.T_increase == 0:
            self.P = self.P_layer*exp(-g0*M0*(self.H_above_layer)/self.R/self.T_layer)
        else:
            self.P = self.P_layer*(self.T_layer/self.T)**(g0*M0/self.R/self.T_increase)

        if dT: # Affects only the following properties
            self.T += dT
            
        self.rho = self.density(self.T, self.P)
        self.v_sonic = self.sonic_velocity(self.T)
        self.mu = self.viscosity(self.T)
        self.k = self.thermal_conductivity(self.T)
        self.g = self.gravity(self.Z)
    

class ATMOSPHERE_NRLMSISE00(object):
    r'''NRLMSISE 00 model for calculating temperature and density of gases in
    the atmosphere, from groud level to 1000 km, as a function of time of year,
    longitude and latitude, solar activity and earth's geomagnetic disturbance.
    
    NRLMSISE standa for the `US Naval Research Laboratory Mass Spectrometer and
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
        [sfu]
    f107_avg : float, optional
        81-day sfu average; centered on specified day if possible, otherwise
        use the previous days [sfu]
    geomagnetic_disturbance_indices : list[float], optional
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
        Pressure, calculated with ideal gas law [P]
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

    Examples
    --------
    >>> atmosphere = ATMOSPHERE_NRLMSISE00(1E3, 45, 45, 150)
    >>> atmosphere.T, atmosphere.rho
    (285.54408606237405, 1.1019062026405517)
    
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
    
    def __init__(self, Z, latitude=0, longitude=0, day=0, seconds=0, 
                 f107=150., f107_avg=150., geomagnetic_disturbance_indices=None):
        alt = Z/1000.
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
        [sfu]
    f107_avg : float, optional
        81-day sfu average; centered on specified day if possible, otherwise
        use the previous days [sfu]
    geomagnetic_disturbance_index : float, optional
        Average daily `Ap` or also known as planetary magnetic index.
        
    Returns
    -------
    w : list[float]
        Wind velocity, meridional (m/sec Northward) and zonal (m/sec Eastward)

    Examples
    --------
    >>> hwm93(5E5, 45, 50, 365)
    [-73.00312042236328, 0.1485661268234253]
    
    Notes
    -----
    No full description has been published of this model; it has been defined by
    its implementation only. It was written in FORTRAN, and is accessible
    at ftp://hanna.ccmc.gsfc.nasa.gov/pub/modelweb/atmospheric/hwm93/.
    
    F2PY auto-compilation support is not yet currently supported.
    
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
        from .optional.hwm93 import gws5
    except:
        raise ImportError(no_gfortran_error)
    slt_hour = seconds/3600. + longitude/15.
    ans = gws5(day, seconds, Z/1000., latitude, longitude, slt_hour, f107, 
               f107_avg, geomagnetic_disturbance_index)
    return ans.tolist()


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
    w : list[float]
        Wind velocity, meridional (m/sec Northward) and zonal (m/sec Eastward)

    Examples
    --------
    >>> hwm14(5E5, 45, 50, 365)
    [-38.64341354370117, 12.871272087097168]

    Notes
    -----
    No full description has been published of this model; it has been defined by
    its implementation only. It was written in FORTRAN, and is accessible
    at http://onlinelibrary.wiley.com/store/10.1002/2014EA000089/asset/supinfo/ess224-sup-0002-supinfo.tgz?v=1&s=2a957ba70b7cf9dd0612d9430076297c3634ea75.
    
    F2PY auto-compilation support is not yet currently supported.
    
    No patches were necessary to either the generated pyf or hwm14.f90 file,
    as the authors of [1]_ have made it F2PY compatible.
    
    References
    ----------
    .. [1] Drob, Douglas P., John T. Emmert, John W. Meriwether, Jonathan J. 
       Makela, Eelco Doornbos, Mark Conde, Gonzalo Hernandez, et al. "An Update
       to the Horizontal Wind Model (HWM): The Quiet Time Thermosphere." Earth
       and Space Science 2, no. 7 (July 1, 2015): 2014EA000089. 
       doi:10.1002/2014EA000089.
    '''
    try:
        import optional.hwm14
    except:
        raise ImportError(no_gfortran_error)
    ans = optional.hwm14.hwm14(day, seconds, Z/1000., latitude, longitude, 0, 0, 
               0, np.array([np.nan, geomagnetic_disturbance_index]))
    return ans.tolist()
