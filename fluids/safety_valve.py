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
from math import exp
from scipy.constants import psi, F2K, inch, atm, C2K
from fluids.compressible import is_critical_flow
from scipy.interpolate import interp1d, interp2d

__all__ = ['API526_A_sq_inch', 'API526_letters', 'API526_A',
'API520_round_size', 'API520_C', 'API520_F2', 'API520_Kv', 'API520_N',
'API520_SH', 'API520_B', 'API520_W', 'API520_A_g', 'API520_A_steam']

API526_A_sq_inch = [0.110, 0.196, 0.307, 0.503, 0.785, 1.287, 1.838, 2.853, 3.60,
             4.34, 6.38, 11.05, 16.00, 26.00] # square inches
API526_letters = ['D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R','T']
API526_A = [i*inch**2 for i in API526_A_sq_inch]


def API520_round_size(A):
    r'''Rounds up the area from an API 520 calculation to an API526 standard
    valve area. The returned area is always larger or equal to the input area.

    Parameters
    ----------
    A : float
        Minimum discharge area [m^2]

    Returns
    -------
    area : float
        Actual discharge area [m^2]

    Notes
    -----
    To obtain the letter designation of an input area, lookup the area with
    the following:

    API526_letters[API526_A.index(area)]

    An exception is raised if the required relief area is larger than any of
    the API 526 sizes.

    Examples
    --------
    From [1]_, checked with many points on Table 8.

    >>> API520_round_size(1E-4)
    0.00012645136
    >>> API526_letters[API526_A.index(API520_round_size(1E-4))]
    'E'

    References
    ----------
    .. [1] API Standard 526.
    '''
    for area in API526_A:
        if area >= A:
            return area
    raise Exception('Required relief area is larger than can be provided with one valve')


def API520_C(k):
    r'''Calculates coefficient C for use in API 520 critical flow relief valve
    sizing.

    .. math::
        C = 0.03948\sqrt{k\left(\frac{2}{k+1}\right)^\frac{k+1}{k-1}}

    Parameters
    ----------
    k : float
        Isentropic coefficient or ideal gas heat capacity ratio [-]

    Returns
    -------
    C : float
        Coefficient `C` [-]

    Notes
    -----
    If C cannot be established, assume a coefficient of 0.0239,
    the highest value possible for C.

    Although not dimensional, C varies with the units used.

    If k is exactly equal to 1, the expression is undefined, and the formula
    must be simplified as follows from an application of L'Hopital's rule.

    .. math::
        C = 0.03948\sqrt{\frac{1}{e}}

    Examples
    --------
    From [1]_, checked with many points on Table 8.

    >>> API520_C(1.35)
    0.02669419967057233

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    if k != 1:
        C = 0.03948*( k*(2./(k+1.))**((k+1.)/(k-1.)) )**0.5
    else:
        C = 0.03948*(1./exp(1))**0.5
    return C


def API520_F2(k, P1, P2):
    r'''Calculates coefficient F2 for subcritical flow for use in API 520
    subcritical flow relief valve sizing.

    .. math::
        F_2 = \sqrt{\left(\frac{k}{k-1}\right)r^\frac{2}{k}
        \left[\frac{1-r^\frac{k-1}{k}}{1-r}\right]}

        r = \frac{P_2}{P_1}

    Parameters
    ----------
    k : float
        Isentropic coefficient or ideal gas heat capacity ratio [-]
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]
    P2 : float
        Built-up backpressure; the increase in pressure during flow at the
        outlet of a pressure-relief device after it opens, [Pa]

    Returns
    -------
    F2 : float
        Subcritical flow coefficient `F2` [-]

    Notes
    -----
    F2 is completely dimensionless.

    Examples
    --------
    From [1]_ example 2, matches.

    >>> API520_F2(1.8, 1E6, 7E5)
    0.8600724121105563

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    r = P2/P1
    F2 = ( k/(k-1)*r**(2./k) * ((1-r**((k-1.)/k))/(1.-r)) )**0.5
    return F2


def API520_Kv(Re):
    r'''Calculates correction due to viscosity for liquid flow for use in
    API 520 relief valve sizing.

    .. math::
        K_v = \left(0.9935 + \frac{2.878}{Re^{0.5}} + \frac{342.75}
        {Re^{1.5}}\right)^{-1}

    Parameters
    ----------
    Re : float
        Reynolds number for flow out the valve [-]

    Returns
    -------
    Kv : float
        Correction due to viscosity [-]

    Notes
    -----
    Reynolds number in the standard is defined as follows, with Q in L/min, G1
    as specific gravity, mu in centipoise, and area in mm^2:

    .. math::
        Re = \frac{Q(18800G_1)}{\mu \sqrt{A}}

    It is unclear how this expression was derived with a constant of 18800;
    the following code demonstrates what the constant should be:

    >>> from scipy.constants import *
    >>> liter/minute*1000./(0.001*(milli**2)**0.5)
    16666.666666666668

    Examples
    --------
    From [1]_, checked with example 5.

    >>> API520_Kv(100)
    0.6157445891444229

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    Kv = (0.9935 + 2.878/Re**0.5 + 342.75/Re**1.5)**-1.0
    return Kv


def API520_N(P1):
    r'''Calculates correction due to steam pressure for steam flow for use in
    API 520 relief valve sizing.

    .. math::
        K_N = \frac{0.02764P_1-1000}{0.03324P_1-1061}

    Parameters
    ----------
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]

    Returns
    -------
    KN : float
        Correction due to steam temperature [-]

    Notes
    -----
    Although not dimensional, KN varies with the units used.

    For temperatures above 922 K or 22057 kPa, KN is not defined.

    Internally, units of kPa are used to match the equation in the standard.

    Examples
    --------
    Custom example:

    >>> API520_N(1774700)
    0.9490406958152466

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    P1 = P1/1000. # Pa to kPa
    KN = (0.02764*P1-1000.)/(0.03324*P1-1061)
    return KN


_KSH_psigs = [15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
              280, 300, 350, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000,
              2500, 3000]
_KSH_tempFs = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
_KSH_Pa = [i*psi+101325 for i in _KSH_psigs]
_KSH_tempKs = [F2K(i) for i in _KSH_tempFs]
_KSH_factors = [[1, 0.98, 0.93, 0.88, 0.84, 0.8, 0.77, 0.74, 0.72, 0.7],
[1, 0.98, 0.93, 0.88, 0.84, 0.8, 0.77, 0.74, 0.72, 0.7],
[1, 0.99, 0.93, 0.88, 0.84, 0.81, 0.77, 0.74, 0.72, 0.7],
[1, 0.99, 0.93, 0.88, 0.84, 0.81, 0.77, 0.75, 0.72, 0.7],
[1, 0.99, 0.93, 0.88, 0.84, 0.81, 0.77, 0.75, 0.72, 0.7],
[1, 0.99, 0.94, 0.89, 0.84, 0.81, 0.77, 0.75, 0.72, 0.7],
[1, 0.99, 0.94, 0.89, 0.84, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 0.99, 0.94, 0.89, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 0.99, 0.94, 0.89, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 0.99, 0.94, 0.89, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 0.99, 0.95, 0.89, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 0.99, 0.95, 0.89, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.95, 0.9, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.95, 0.9, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.96, 0.9, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.96, 0.9, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.96, 0.9, 0.86, 0.82, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.96, 0.91, 0.86, 0.82, 0.78, 0.75, 0.72, 0.7],
[1, 1, 0.96, 0.92, 0.86, 0.82, 0.78, 0.75, 0.73, 0.7],
[1, 1, 0.97, 0.92, 0.87, 0.82, 0.79, 0.75, 0.73, 0.7],
[1, 1, 1, 0.95, 0.88, 0.83, 0.79, 0.76, 0.73, 0.7],
[1, 1, 1, 0.96, 0.89, 0.84, 0.78, 0.76, 0.73, 0.71],
[1, 1, 1, 0.97, 0.91, 0.85, 0.8, 0.77, 0.74, 0.71],
[1, 1, 1, 1, 0.93, 0.86, 0.81, 0.77, 0.74, 0.71],
[1, 1, 1, 1, 0.94, 0.86, 0.81, 0.77, 0.73, 0.7],
[1, 1, 1, 1, 0.95, 0.86, 0.8, 0.76, 0.72, 0.69],
[1, 1, 1, 1, 0.95, 0.85, 0.78, 0.73, 0.69, 0.66],
[1, 1, 1, 1, 1, 0.82, 0.74, 0.69, 0.65, 0.62]]
API520_KSH = interp2d(_KSH_tempKs, _KSH_Pa, _KSH_factors)


def API520_SH(T1, P1):
    r'''Calculates correction due to steam superheat for steam flow for use in
    API 520 relief valve sizing. 2D interpolation among a table with 28
    pressures and 10 temperatures is performed.


    Parameters
    ----------
    T1 : float
        Temperature of the fluid entering the valve [K]
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]

    Returns
    -------
    KSH : float
        Correction due to steam superheat [-]

    Notes
    -----
    For P above 20679 kPag, use the critical flow model.
    Superheat cannot be above 649 degrees Celcius.
    If T1 is above 149 degrees Celcius, returns 1.

    Examples
    --------
    Custom example from table 9:

    >>> API520_SH(593+273.15, 1066.325E3)
    0.7201800000000002

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    if P1 > 20679E3+atm:
        raise Exception('For P above 20679 kPag, use the critical flow model')
    if T1 > C2K(649):
        raise Exception('Superheat cannot be above 649 degrees Celcius')
    if T1 < C2K(149):
        return 1. # No superheat under 15 psig
    KSH = float(API520_KSH(T1, P1))
    return KSH



# Kw, for liquids. Applicable for all overpressures.
_Kw_x = [15., 16.5493, 17.3367, 18.124, 18.8235, 19.5231, 20.1351, 20.8344, 21.4463, 22.0581, 22.9321, 23.5439, 24.1556, 24.7674, 25.0296, 25.6414, 26.2533, 26.8651, 27.7393, 28.3511, 28.9629, 29.6623, 29.9245, 30.5363, 31.2357, 31.8475, 32.7217, 33.3336, 34.0329, 34.6448, 34.8196, 35.4315, 36.1308, 36.7428, 37.7042, 38.3162, 39.0154, 39.7148, 40.3266, 40.9384, 41.6378, 42.7742, 43.386, 43.9978, 44.6098, 45.2216, 45.921, 46.5329, 47.7567, 48.3685, 49.0679, 49.6797, 50.]
_Kw_y = [1, 0.996283, 0.992565, 0.987918, 0.982342, 0.976766, 0.97119, 0.964684, 0.958178, 0.951673, 0.942379, 0.935874, 0.928439, 0.921933, 0.919145, 0.912639, 0.906134, 0.899628, 0.891264, 0.884758, 0.878253, 0.871747, 0.868959, 0.862454, 0.855948, 0.849442, 0.841078, 0.834572, 0.828067, 0.821561, 0.819703, 0.814126, 0.806691, 0.801115, 0.790892, 0.785316, 0.777881, 0.771375, 0.76487, 0.758364, 0.751859, 0.740706, 0.734201, 0.727695, 0.722119, 0.715613, 0.709108, 0.702602, 0.69052, 0.684015, 0.677509, 0.671004, 0.666357]
API520_Kw = interp1d(_Kw_x, _Kw_y)


def API520_W(Pset, Pback):
    r'''Calculates capacity correction due to backpressure on balanced
    spring-loaded PRVs in liquid service. For pilot operated valves,
    this is always 1. Applicable up to 50% of the percent gauge backpressure,
    For use in API 520 relief valve sizing. 1D interpolation among a table with
    53 backpressures is performed.

    Parameters
    ----------
    Pset : float
        Set pressure for relief [Pa]
    Pback : float
        Backpressure, [Pa]

    Returns
    -------
    KW : float
        Correction due to liquid backpressure [-]

    Notes
    -----
    If the calculated gauge backpressure is less than 15%, a value of 1 is
    returned.

    Examples
    --------
    Custom example from figure 31:

    >>> API520_W(1E6, 3E5) # 22% overpressure
    0.9511471848008564

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    gauge_backpressure = (Pback-atm)/(Pset-atm)*100 # in percent
    if gauge_backpressure < 15:
        return 1
    KW = float(API520_Kw(gauge_backpressure))
    return KW



# Kb Backpressure correction factor, for gases
_16_over_x = [37.6478, 38.1735, 38.6991, 39.2904, 39.8817, 40.4731, 40.9987, 41.59, 42.1156, 42.707, 43.2326, 43.8239, 44.4152, 44.9409, 45.5322, 46.0578, 46.6491, 47.2405, 47.7661, 48.3574, 48.883, 49.4744, 50]
_16_over_y = [0.998106, 0.994318, 0.99053, 0.985795, 0.982008, 0.97822, 0.973485, 0.96875, 0.964962, 0.961174, 0.956439, 0.951705, 0.947917, 0.943182, 0.939394, 0.935606, 0.930871, 0.926136, 0.921402, 0.918561, 0.913826, 0.910038, 0.90625]
API520_Kb_16 = interp1d(_16_over_x, _16_over_y)

_10_over_x = [30.0263, 30.6176, 31.1432, 31.6689, 32.1945, 32.6544, 33.18, 33.7057, 34.1656, 34.6255, 35.0854, 35.5453, 36.0053, 36.4652, 36.9251, 37.385, 37.8449, 38.2392, 38.6334, 39.0276, 39.4875, 39.9474, 40.4074, 40.8016, 41.1958, 41.59, 42.0499, 42.4442, 42.8384, 43.2326, 43.6925, 44.0867, 44.4809, 44.8752, 45.2694, 45.6636, 46.0578, 46.452, 46.8463, 47.2405, 47.6347, 48.0289, 48.4231, 48.883, 49.2773, 49.6715]
_10_over_y = [0.998106, 0.995265, 0.99053, 0.985795, 0.981061, 0.975379, 0.969697, 0.963068, 0.957386, 0.950758, 0.945076, 0.938447, 0.930871, 0.925189, 0.918561, 0.910985, 0.904356, 0.897727, 0.891098, 0.883523, 0.876894, 0.870265, 0.862689, 0.856061, 0.848485, 0.840909, 0.83428, 0.827652, 0.820076, 0.8125, 0.805871, 0.798295, 0.79072, 0.783144, 0.775568, 0.768939, 0.762311, 0.754735, 0.747159, 0.739583, 0.732008, 0.724432, 0.716856, 0.70928, 0.701705, 0.695076]
API520_Kb_10 = interp1d(_10_over_x, _10_over_y)



def API520_B(Pset, Pback, overpressure=0.1):
    r'''Calculates capacity correction due to backpressure on balanced
    spring-loaded PRVs in vapor service. For pilot operated valves,
    this is always 1. Applicable up to 50% of the percent gauge backpressure,
    For use in API 520 relief valve sizing. 1D interpolation among a table with
    53 backpressures is performed.

    Parameters
    ----------
    Pset : float
        Set pressure for relief [Pa]
    Pback : float
        Backpressure, [Pa]
    overpressure : float, optional
        The maximum fraction overpressure; one of 0.1, 0.16, or 0.21, []

    Returns
    -------
    Kb : float
        Correction due to vapor backpressure [-]

    Notes
    -----
    If the calculated gauge backpressure is less than 30%, 38%, or 50% for
    overpressures of 0.1, 0.16, or 0.21, a value of 1 is returned.

    Percent gauge backpressure must be under 50%.

    Examples
    --------
    Custom examples from figure 30:

    >>> API520_B(1E6, 5E5)
    0.7929945420944432

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    gauge_backpressure = (Pback-atm)/(Pset-atm)*100 # in percent
    if overpressure not in [0.1, 0.16, 0.21]:
        raise Exception('Only overpressure of 10%, 16%, or 21% are permitted')
    if (overpressure == 0.1 and gauge_backpressure < 30) or (
        overpressure == 0.16 and gauge_backpressure < 38) or (
        overpressure == 0.21 and gauge_backpressure < 50):
        return 1
    elif gauge_backpressure > 50:
        raise Exception('Gauge pressure must be < 50%')
    if overpressure == 0.16:
         Kb = float(API520_Kb_16(gauge_backpressure))
    elif overpressure == 0.1:
         Kb = float(API520_Kb_10(gauge_backpressure))
    return Kb

#print [API520_B(1E6, 3E5), API520_B(1E6, 5E5), API520_B(1E6, 5E5, overpressure=.16), API520_B(1E6, 5E5, overpressure=.21)]

def API520_A_g(m, T, Z, MW, k, P1, P2=101325, Kd=0.975, Kb=1, Kc=1):
    r'''Calculates required relief valve area for an API 520 valve passing
    a gas or a vapor, at either critical or sub-critical flow.

    For Critical flow:

    .. math::
        A = \frac{m}{CK_dP_1K_bK_c}\sqrt{\frac{TZ}{M}}

    For sub-critical flow:

    .. math::
        A = \frac{17.9m}{F_2K_dK_c}\sqrt{\frac{TZ}{MP_1(P_1-P_2)}}

    Parameters
    ----------
    m : float
        Mass flow rate of vapor through the valve, [kg/s]
    T : float
        Temperature of vapor entering the valve, [K]
    Z : float
        Compressibility factor of the vapor, [-]
    MW : float
        Molecular weight of the vapor, [g/mol]
    k : float
        Isentropic coefficient or ideal gas heat capacity ratio [-]
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]
    P2 : float, optional
        Built-up backpressure; the increase in pressure during flow at the
        outlet of a pressure-relief device after it opens, [Pa]
    Kd : float, optional
        The effective coefficient of discharge, from the manufacturer or for
        preliminary sizing, using 0.975 normally or 0.62 when used with a
        rupture disc as described in [1]_, []
    Kb : float, optional
        Correction due to vapor backpressure [-]
    Kc : float, optional
        Combination correction factor for installation with a ruture disk
        upstream of the PRV, []

    Returns
    -------
    A : float
        Minimum area for relief valve according to [1]_, [m^2]

    Notes
    -----
    Units are interlally kg/hr, kPa, and mm^2 to match [1]_.

    Examples
    --------
    Example 1 from [1]_ for critical flow, matches:

    >>> API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)
    0.0036990460646834414

    Example 2 from [2]_ for sub-critical flow, matches:

    >>> API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1, Kc=1)
    0.004248358775943481

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    P1, P2 = P1/1000., P2/1000. # Pa to Kpa in the standard
    m = m*3600. # kg/s to kg/hr
    if is_critical_flow(P1, P2, k):
        C = API520_C(k)
        A = m/(C*Kd*Kb*Kc*P1)*(T*Z/MW)**0.5
    else:
        F2 = API520_F2(k, P1, P2)
        A = 17.9*m/(F2*Kd*Kc)*(T*Z/(MW*P1*(P1-P2)))**0.5
    A = A*0.001**2 # convert mm^2 to m^2
    return A

#print [API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)]
#print [API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1, Kc=1)]

def API520_A_steam(m, T, P1, Kd=0.975, Kb=1, Kc=1):
    r'''Calculates required relief valve area for an API 520 valve passing
    a steam, at either saturation or superheat but not partially condensed.

    .. math::
        A = \frac{190.5m}{P_1 K_d K_b K_c K_N K_{SH}}

    Parameters
    ----------
    m : float
        Mass flow rate of steam through the valve, [kg/s]
    T : float
        Temperature of steam entering the valve, [K]
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]
    Kd : float, optional
        The effective coefficient of discharge, from the manufacturer or for
        preliminary sizing, using 0.975 normally or 0.62 when used with a
        rupture disc as described in [1]_, []
    Kb : float, optional
        Correction due to vapor backpressure [-]
    Kc : float, optional
        Combination correction factor for installation with a ruture disk
        upstream of the PRV, []

    Returns
    -------
    A : float
        Minimum area for relief valve according to [1]_, [m^2]

    Notes
    -----
    Units are interlally kg/hr, kPa, and mm^2 to match [1]_.
    With the provided temperature and pressure, the KN coefficient is
    calculated with the function API520_N; as is the superheat correction KSH,
    with the function API520_SH.

    Examples
    --------
    Example 4 from [1]_, matches:

    >>> API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1)
    0.0011034712423692733

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    KN = API520_N(P1)
    KSH = API520_SH(T, P1)
    P1 = P1/1000. # Pa to kPa
    m = m*3600. # kg/s to kg/hr
    A = 190.5*m/(P1*Kd*Kb*Kc*KN*KSH)
    A = A*0.001**2 # convert mm^2 to m^2
    return A

#print [API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1)]
