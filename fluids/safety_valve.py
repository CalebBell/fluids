"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
SOFTWARE.

This module contains functions for sizing and rating pressure relief valves.
At present, this consists of several functions from API 520.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Interfaces
----------
.. autofunction:: API520_A_g
.. autofunction:: API520_A_steam
.. autofunction:: API520_A_l
.. autofunction:: API521_noise

Functions and Data
------------------
.. autofunction:: API520_round_size
.. autofunction:: API520_C
.. autofunction:: API520_F2
.. autofunction:: API520_Kv
.. autofunction:: API520_N
.. autofunction:: API520_SH
.. autofunction:: API520_B
.. autofunction:: API520_W
.. autofunction:: API521_noise_graph
.. autofunction:: VDI_3732_noise_ground_flare
.. autofunction:: VDI_3732_noise_elevated_flare
.. autodata:: API526_letters
.. autodata:: API526_A_sq_inch
.. autodata:: API526_A

"""

from math import log10, pi, sqrt

from fluids.compressible import is_critical_flow
from fluids.constants import atm, inch
from fluids.numerics import bisplev, interp, tck_interp2d_linear

__all__ = ['API526_A_sq_inch', 'API526_letters', 'API526_A',
'API520_round_size', 'API520_C', 'API520_F2', 'API520_Kv', 'API520_N',
'API520_SH', 'API520_B', 'API520_W', 'API520_A_g', 'API520_A_steam',
'API521_noise', 'API521_noise_graph', 'VDI_3732_noise_ground_flare',
'VDI_3732_noise_elevated_flare', 'API520_A_l']

API526_A_sq_inch = [0.110, 0.196, 0.307, 0.503, 0.785, 1.287, 1.838, 2.853, 3.60,
             4.34, 6.38, 11.05, 16.00, 26.00] # square inches
"""list: Nominal relief area in for different valve sizes in API 520, [in^2]"""
API526_letters = ['D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R','T']
"""list: Letter size designations for different valve sizes in API 520"""
inch2 = inch*inch
API526_A = [i*inch2 for i in API526_A_sq_inch]
"""list: Nominal relief area in for different valve sizes in API 520, [m^2]"""
del inch2

TENTH_EDITION = '10E'
SEVENTH_EDITION = '7E'

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
    raise ValueError('Required relief area is larger than can be provided with one valve')


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
        kp1 = k+1
        return 0.03948*sqrt(k*(2./kp1)**(kp1/(k-1.)))
    else:
        return 0.023945830445454768
        # return 0.03948*sqrt(1./exp(1))


def API520_F2(k, P1, P2):
    r'''Calculates coefficient F2 for subcritical flow for use in API 520
    subcritical flow relief valve sizing.

    .. math::
        F_2 = \sqrt{\left(\frac{k}{k-1}\right)r^\frac{2}{k}
        \left[\frac{1-r^\frac{k-1}{k}}{1-r}\right]}

    .. math::
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
    return sqrt(k/(k-1.0)*r**(2./k) * ((1-r**((k-1.)/k))/(1.-r)))


def API520_N(P1):
    r'''Calculates correction due to steam pressure for steam flow for use in
    API 520 relief valve sizing.

    For pressures below 10339 kPa, the correction factor is 1.

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

    For temperatures above 922 K or pressures above 22057 kPa, KN is not defined.

    Internally, units of kPa are used to match the equation in the standard.

    Examples
    --------
    >>> API520_N(10500e3)
    0.9969100255

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    P1 = P1*1e-3 # Pa to kPa
    if P1 <= 10339.0:
        KN = 1.0
    else:
        KN = (0.02764*P1 - 1000.)/(0.03324*P1 - 1061.0)
    return KN



# Values from API 520 7th edition through 9th edition
_KSH_psigs_7E = [15, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
              280, 300, 350, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000,
              2500, 3000]
_KSH_tempFs_7E = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

# _KSH_psigs_7E converted from psig to Pa
_KSH_Pa_7E = [204746.3593975254, 239220.14586336722, 377115.29172673443,
           515010.4375901016, 652905.5834534689, 790800.7293168361,
           928695.8751802032, 1066591.0210435705, 1204486.1669069377,
           1342381.312770305, 1480276.4586336722, 1618171.6044970395,
           1756066.7503604065, 1893961.8962237737, 2031857.042087141,
           2169752.187950508, 2514490.0526089263, 2859227.9172673444,
           3548703.64658418, 4238179.375901016, 5617130.834534689,
           6996082.29316836, 8719771.616460452, 10443460.939752541,
           12167150.263044631, 13890839.58633672, 17338218.232920904,
           20785596.879505083]

# _KSH_tempFs_7E converted from F to K
_KSH_tempKs_7E = [422.03888888888889, 477.59444444444443, 533.14999999999998,
               588.70555555555552, 644.26111111111106, 699.81666666666661,
               755.37222222222226, 810.92777777777769, 866.48333333333335,
               922.03888888888889]

_KSH_factors_7E = [[1, 0.98, 0.93, 0.88, 0.84, 0.8, 0.77, 0.74, 0.72, 0.7],
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

_KSH_Pa_10E = [500000.0, 750000.0, 1000000.0, 1250000.0, 1500000.0, 1750000.0,
               2000000.0, 2250000.0, 2500000.0, 2750000.0, 3000000.0, 3250000.0,
               3500000.0, 3750000.0, 4000000.0, 4250000.0, 4500000.0, 4750000.0,
               5000000.0, 5250000.0, 5500000.0, 5750000.0, 6000000.0, 6250000.0,
               6500000.0, 6750000.0, 7000000.0, 7250000.0, 7500000.0, 7750000.0,
               8000000.0, 8250000.0, 8500000.0, 8750000.0, 9000000.0, 9250000.0,
               9500000.0, 9750000.0, 10000000.0, 10250000.0, 10500000.0,
               10750000.0, 11000000.0, 11250000.0, 11500000.0, 11750000.0,
               12000000.0, 12250000.0, 12500000.0, 12750000.0, 13000000.0,
               13250000.0, 13500000.0, 14000000.0, 14250000.0, 14500000.0,
               14750000.0, 15000000.0, 15250000.0, 15500000.0, 15750000.0,
               16000000.0, 16250000.0, 16500000.0, 16750000.0, 17000000.0,
               17250000.0, 17500000.0, 17750000.0, 18000000.0, 18250000.0,
               18500000.0, 18750000.0, 19000000.0, 19250000.0, 19500000.0,
               19750000.0, 20000000.0, 20250000.0, 20500000.0, 20750000.0,
               21000000.0, 21250000.0, 21500000.0, 21750000.0, 22000000.0, ]

_KSH_K_10E = [478.15, 498.15, 523.15, 548.15, 573.15, 598.15, 623.15, 648.15,
              673.15, 698.15, 723.15, 748.15, 773.15, 798.15, 823.15, 848.15,
              873.15, 898.15]

_KSH_factors_10E = [[0.991, 0.968, 0.942, 0.919, 0.896, 0.876, 0.857, 0.839, 0.823, 0.807, 0.792, 0.778, 0.765, 0.752, 0.74, 0.728, 0.717, 0.706],
[0.995, 0.972, 0.946, 0.922, 0.899, 0.878, 0.859, 0.841, 0.824, 0.808, 0.793, 0.779, 0.766, 0.753, 0.74, 0.729, 0.717, 0.707],
[0.985, 0.973, 0.95, 0.925, 0.902, 0.88, 0.861, 0.843, 0.825, 0.809, 0.794, 0.78, 0.766, 0.753, 0.741, 0.729, 0.718, 0.707],
[0.981, 0.976, 0.954, 0.928, 0.905, 0.883, 0.863, 0.844, 0.827, 0.81, 0.795, 0.781, 0.767, 0.754, 0.741, 0.729, 0.718, 0.707],
[1, 1, 0.957, 0.932, 0.907, 0.885, 0.865, 0.846, 0.828, 0.812, 0.796, 0.782, 0.768, 0.755, 0.742, 0.73, 0.718, 0.708],
[1, 1, 0.959, 0.935, 0.91, 0.887, 0.866, 0.847, 0.829, 0.813, 0.797, 0.782, 0.769, 0.756, 0.743, 0.731, 0.719, 0.708],
[1, 1, 0.96, 0.939, 0.913, 0.889, 0.868, 0.849, 0.831, 0.814, 0.798, 0.784, 0.769, 0.756, 0.744, 0.731, 0.72, 0.708],
[1, 1, 0.963, 0.943, 0.916, 0.892, 0.87, 0.85, 0.832, 0.815, 0.799, 0.785, 0.77, 0.757, 0.744, 0.732, 0.72, 0.709],
[1, 1, 1, 0.946, 0.919, 0.894, 0.872, 0.852, 0.834, 0.816, 0.8, 0.785, 0.771, 0.757, 0.744, 0.732, 0.72, 0.71],
[1, 1, 1, 0.948, 0.922, 0.897, 0.874, 0.854, 0.835, 0.817, 0.801, 0.786, 0.772, 0.758, 0.745, 0.733, 0.721, 0.71],
[1, 1, 1, 0.949, 0.925, 0.899, 0.876, 0.855, 0.837, 0.819, 0.802, 0.787, 0.772, 0.759, 0.746, 0.733, 0.722, 0.71],
[1, 1, 1, 0.951, 0.929, 0.902, 0.879, 0.857, 0.838, 0.82, 0.803, 0.788, 0.773, 0.759, 0.746, 0.734, 0.722, 0.711],
[1, 1, 1, 0.953, 0.933, 0.905, 0.881, 0.859, 0.84, 0.822, 0.804, 0.789, 0.774, 0.76, 0.747, 0.734, 0.722, 0.711],
[1, 1, 1, 0.956, 0.936, 0.908, 0.883, 0.861, 0.841, 0.823, 0.806, 0.79, 0.775, 0.761, 0.748, 0.735, 0.723, 0.711],
[1, 1, 1, 0.959, 0.94, 0.91, 0.885, 0.863, 0.842, 0.824, 0.807, 0.791, 0.776, 0.762, 0.748, 0.735, 0.723, 0.712],
[1, 1, 1, 0.961, 0.943, 0.913, 0.887, 0.864, 0.844, 0.825, 0.808, 0.792, 0.776, 0.762, 0.749, 0.736, 0.724, 0.713],
[1, 1, 1, 1, 0.944, 0.917, 0.89, 0.866, 0.845, 0.826, 0.809, 0.793, 0.777, 0.763, 0.749, 0.737, 0.725, 0.713],
[1, 1, 1, 1, 0.946, 0.919, 0.892, 0.868, 0.847, 0.828, 0.81, 0.793, 0.778, 0.764, 0.75, 0.737, 0.725, 0.713],
[1, 1, 1, 1, 0.947, 0.922, 0.894, 0.87, 0.848, 0.829, 0.811, 0.794, 0.779, 0.765, 0.751, 0.738, 0.725, 0.714],
[1, 1, 1, 1, 0.949, 0.926, 0.897, 0.872, 0.85, 0.83, 0.812, 0.795, 0.78, 0.765, 0.752, 0.738, 0.726, 0.714],
[1, 1, 1, 1, 0.952, 0.93, 0.899, 0.874, 0.851, 0.831, 0.813, 0.797, 0.78, 0.766, 0.752, 0.739, 0.727, 0.714],
[1, 1, 1, 1, 0.954, 0.933, 0.902, 0.876, 0.853, 0.833, 0.815, 0.798, 0.782, 0.767, 0.753, 0.739, 0.727, 0.715],
[1, 1, 1, 1, 0.957, 0.937, 0.904, 0.878, 0.855, 0.834, 0.816, 0.798, 0.783, 0.768, 0.753, 0.74, 0.727, 0.716],
[1, 1, 1, 1, 0.96, 0.94, 0.907, 0.88, 0.856, 0.836, 0.817, 0.799, 0.783, 0.768, 0.754, 0.74, 0.728, 0.716],
[1, 1, 1, 1, 0.964, 0.944, 0.91, 0.882, 0.859, 0.837, 0.818, 0.801, 0.784, 0.769, 0.754, 0.741, 0.729, 0.716],
[1, 1, 1, 1, 0.966, 0.946, 0.913, 0.885, 0.86, 0.839, 0.819, 0.802, 0.785, 0.769, 0.755, 0.742, 0.729, 0.717],
[1, 1, 1, 1, 1, 0.947, 0.916, 0.887, 0.862, 0.84, 0.82, 0.802, 0.786, 0.77, 0.756, 0.742, 0.729, 0.717],
[1, 1, 1, 1, 1, 0.949, 0.919, 0.889, 0.863, 0.842, 0.822, 0.803, 0.787, 0.771, 0.756, 0.743, 0.73, 0.717],
[1, 1, 1, 1, 1, 0.951, 0.922, 0.891, 0.865, 0.843, 0.823, 0.805, 0.788, 0.772, 0.757, 0.744, 0.73, 0.718],
[1, 1, 1, 1, 1, 0.953, 0.925, 0.893, 0.867, 0.844, 0.824, 0.806, 0.788, 0.772, 0.758, 0.744, 0.731, 0.719],
[1, 1, 1, 1, 1, 0.955, 0.928, 0.896, 0.869, 0.846, 0.825, 0.806, 0.789, 0.773, 0.758, 0.744, 0.732, 0.719],
[1, 1, 1, 1, 1, 0.957, 0.932, 0.898, 0.871, 0.847, 0.827, 0.807, 0.79, 0.774, 0.759, 0.745, 0.732, 0.719],
[1, 1, 1, 1, 1, 0.96, 0.935, 0.901, 0.873, 0.849, 0.828, 0.809, 0.791, 0.775, 0.76, 0.746, 0.732, 0.72],
[1, 1, 1, 1, 1, 0.963, 0.939, 0.903, 0.875, 0.85, 0.829, 0.81, 0.792, 0.776, 0.76, 0.746, 0.733, 0.721],
[1, 1, 1, 1, 1, 0.966, 0.943, 0.906, 0.877, 0.852, 0.83, 0.811, 0.793, 0.776, 0.761, 0.747, 0.734, 0.721],
[1, 1, 1, 1, 1, 0.97, 0.947, 0.909, 0.879, 0.853, 0.832, 0.812, 0.794, 0.777, 0.762, 0.747, 0.734, 0.721],
[1, 1, 1, 1, 1, 0.973, 0.95, 0.911, 0.881, 0.855, 0.833, 0.813, 0.795, 0.778, 0.763, 0.748, 0.734, 0.722],
[1, 1, 1, 1, 1, 0.977, 0.954, 0.914, 0.883, 0.857, 0.834, 0.814, 0.796, 0.779, 0.763, 0.749, 0.735, 0.722],
[1, 1, 1, 1, 1, 0.981, 0.957, 0.917, 0.885, 0.859, 0.836, 0.815, 0.797, 0.78, 0.764, 0.749, 0.735, 0.722],
[1, 1, 1, 1, 1, 0.984, 0.959, 0.92, 0.887, 0.86, 0.837, 0.816, 0.798, 0.78, 0.764, 0.75, 0.736, 0.723],
[1, 1, 1, 1, 1, 1, 0.961, 0.923, 0.889, 0.862, 0.838, 0.817, 0.799, 0.781, 0.765, 0.75, 0.737, 0.723],
[1, 1, 1, 1, 1, 1, 0.962, 0.925, 0.891, 0.863, 0.839, 0.818, 0.799, 0.782, 0.766, 0.751, 0.737, 0.724],
[1, 1, 1, 1, 1, 1, 0.963, 0.928, 0.893, 0.865, 0.84, 0.819, 0.8, 0.782, 0.766, 0.751, 0.737, 0.724],
[1, 1, 1, 1, 1, 1, 0.964, 0.93, 0.893, 0.865, 0.84, 0.819, 0.799, 0.781, 0.765, 0.75, 0.736, 0.723],
[1, 1, 1, 1, 1, 1, 0.964, 0.931, 0.894, 0.865, 0.84, 0.818, 0.798, 0.78, 0.764, 0.749, 0.735, 0.722],
[1, 1, 1, 1, 1, 1, 0.965, 0.932, 0.894, 0.865, 0.839, 0.817, 0.797, 0.78, 0.763, 0.748, 0.734, 0.721],
[1, 1, 1, 1, 1, 1, 0.966, 0.933, 0.894, 0.864, 0.839, 0.817, 0.797, 0.779, 0.762, 0.747, 0.733, 0.719],
[1, 1, 1, 1, 1, 1, 0.967, 0.935, 0.895, 0.864, 0.839, 0.816, 0.796, 0.778, 0.761, 0.746, 0.732, 0.718],
[1, 1, 1, 1, 1, 1, 0.967, 0.936, 0.896, 0.864, 0.838, 0.816, 0.796, 0.777, 0.76, 0.745, 0.731, 0.717],
[1, 1, 1, 1, 1, 1, 0.968, 0.937, 0.896, 0.864, 0.838, 0.815, 0.795, 0.776, 0.759, 0.744, 0.729, 0.716],
[1, 1, 1, 1, 1, 1, 0.969, 0.939, 0.896, 0.864, 0.837, 0.814, 0.794, 0.775, 0.758, 0.743, 0.728, 0.715],
[1, 1, 1, 1, 1, 1, 0.971, 0.94, 0.897, 0.864, 0.837, 0.813, 0.792, 0.774, 0.757, 0.741, 0.727, 0.713],
[1, 1, 1, 1, 1, 1, 0.972, 0.942, 0.897, 0.863, 0.837, 0.813, 0.792, 0.773, 0.756, 0.74, 0.725, 0.712],
[1, 1, 1, 1, 1, 1, 0.976, 0.946, 0.897, 0.863, 0.835, 0.811, 0.79, 0.771, 0.753, 0.737, 0.723, 0.709],
[1, 1, 1, 1, 1, 1, 0.978, 0.947, 0.898, 0.862, 0.834, 0.81, 0.789, 0.77, 0.752, 0.736, 0.721, 0.707],
[1, 1, 1, 1, 1, 1, 1, 0.948, 0.898, 0.862, 0.833, 0.809, 0.787, 0.768, 0.751, 0.734, 0.72, 0.706],
[1, 1, 1, 1, 1, 1, 1, 0.948, 0.898, 0.862, 0.832, 0.808, 0.786, 0.767, 0.749, 0.733, 0.719, 0.704],
[1, 1, 1, 1, 1, 1, 1, 0.948, 0.899, 0.861, 0.832, 0.807, 0.785, 0.766, 0.748, 0.732, 0.717, 0.703],
[1, 1, 1, 1, 1, 1, 1, 0.947, 0.899, 0.861, 0.831, 0.806, 0.784, 0.764, 0.746, 0.73, 0.716, 0.702],
[1, 1, 1, 1, 1, 1, 1, 0.947, 0.899, 0.861, 0.83, 0.804, 0.782, 0.763, 0.745, 0.728, 0.714, 0.7],
[1, 1, 1, 1, 1, 1, 1, 0.946, 0.899, 0.86, 0.829, 0.803, 0.781, 0.761, 0.743, 0.727, 0.712, 0.698],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.9, 0.859, 0.828, 0.802, 0.779, 0.759, 0.741, 0.725, 0.71, 0.696],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.9, 0.859, 0.827, 0.801, 0.778, 0.757, 0.739, 0.723, 0.708, 0.694],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.9, 0.858, 0.826, 0.799, 0.776, 0.756, 0.738, 0.721, 0.706, 0.692],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.9, 0.857, 0.825, 0.797, 0.774, 0.754, 0.736, 0.719, 0.704, 0.69],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.9, 0.856, 0.823, 0.796, 0.773, 0.752, 0.734, 0.717, 0.702, 0.688],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.9, 0.855, 0.822, 0.794, 0.771, 0.75, 0.732, 0.715, 0.7, 0.686],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.9, 0.854, 0.82, 0.792, 0.769, 0.748, 0.73, 0.713, 0.698, 0.684],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.9, 0.853, 0.819, 0.791, 0.767, 0.746, 0.728, 0.711, 0.696, 0.681],
[1, 1, 1, 1, 1, 1, 1, 0.944, 0.901, 0.852, 0.817, 0.789, 0.765, 0.744, 0.725, 0.709, 0.694, 0.679],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.901, 0.851, 0.815, 0.787, 0.763, 0.742, 0.723, 0.706, 0.691, 0.677],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.901, 0.85, 0.814, 0.785, 0.761, 0.739, 0.72, 0.704, 0.689, 0.674],
[1, 1, 1, 1, 1, 1, 1, 0.945, 0.901, 0.849, 0.812, 0.783, 0.758, 0.737, 0.718, 0.701, 0.686, 0.671],
[1, 1, 1, 1, 1, 1, 1, 0.946, 0.901, 0.847, 0.81, 0.781, 0.756, 0.734, 0.715, 0.698, 0.683, 0.669],
[1, 1, 1, 1, 1, 1, 1, 0.948, 0.901, 0.846, 0.808, 0.778, 0.753, 0.732, 0.713, 0.696, 0.681, 0.666],
[1, 1, 1, 1, 1, 1, 1, 0.95, 0.9, 0.844, 0.806, 0.776, 0.75, 0.729, 0.71, 0.693, 0.677, 0.663],
[1, 1, 1, 1, 1, 1, 1, 0.952, 0.899, 0.842, 0.803, 0.773, 0.748, 0.726, 0.707, 0.69, 0.674, 0.66],
[1, 1, 1, 1, 1, 1, 1, 1, 0.899, 0.84, 0.801, 0.77, 0.745, 0.723, 0.704, 0.687, 0.671, 0.657],
[1, 1, 1, 1, 1, 1, 1, 1, 0.899, 0.839, 0.798, 0.767, 0.742, 0.72, 0.701, 0.683, 0.668, 0.654],
[1, 1, 1, 1, 1, 1, 1, 1, 0.899, 0.837, 0.795, 0.764, 0.738, 0.717, 0.697, 0.68, 0.665, 0.651],
[1, 1, 1, 1, 1, 1, 1, 1, 0.898, 0.834, 0.792, 0.761, 0.735, 0.713, 0.694, 0.677, 0.661, 0.647],
[1, 1, 1, 1, 1, 1, 1, 1, 0.896, 0.832, 0.79, 0.758, 0.732, 0.71, 0.691, 0.673, 0.658, 0.643],
[1, 1, 1, 1, 1, 1, 1, 1, 0.894, 0.829, 0.786, 0.754, 0.728, 0.706, 0.686, 0.669, 0.654, 0.64],
[1, 1, 1, 1, 1, 1, 1, 1, 0.892, 0.826, 0.783, 0.75, 0.724, 0.702, 0.682, 0.665, 0.65, 0.636],
[1, 1, 1, 1, 1, 1, 1, 1, 0.891, 0.823, 0.779, 0.746, 0.72, 0.698, 0.679, 0.661, 0.646, 0.631],
[1, 1, 1, 1, 1, 1, 1, 1, 0.887, 0.82, 0.776, 0.743, 0.716, 0.694, 0.674, 0.657, 0.641, 0.627]]
API520_KSH_tck_7E = tck_interp2d_linear(_KSH_tempKs_7E, _KSH_Pa_7E, _KSH_factors_7E)

API520_KSH_tck_10E = tck_interp2d_linear(_KSH_K_10E, _KSH_Pa_10E, _KSH_factors_10E)


def API520_SH(T1, P1, edition=TENTH_EDITION):
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
    edition : str, optional
        One of '10E', '7E', [-]

    Returns
    -------
    KSH : float
        Correction due to steam superheat [-]

    Notes
    -----
    For P above 20679 kPag, use the critical flow model.
    Superheat cannot be above 649 degrees Celsius.
    If T1 is above 149 degrees Celsius, returns 1.

    Examples
    --------
    Custom example from table 9, 7th edition:

    >>> API520_SH(593+273.15, 1066.325E3, '7E')
    0.7201800000

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    if T1 > 922.15:
        raise ValueError('Superheat cannot be above 649 degrees Celcius')
    if edition == SEVENTH_EDITION:
        if P1 > 20780325.0: # 20679E3+atm
            raise ValueError('For P above 20679 kPag, use the gas flow model')
        if T1 < 422.15:
            return 1. # No superheat under 15 psig
        return float(bisplev(T1, P1, API520_KSH_tck_7E))
    elif edition == TENTH_EDITION:
        if T1 < 478.15:
            # Avoid extrapolating above 1.0
            return 1.0
        if P1 > 22063223.338138755:
            raise ValueError('For P1 above 22.06 MPa, use the gas flow model')
        return float(bisplev(T1, P1, API520_KSH_tck_10E))
    else:
        raise ValueError("Acceptable editions are '7E', '10E'")



# Kb Backpressure correction factor, for gases
Kb_16_over_x = [37.6478, 38.1735, 38.6991, 39.2904, 39.8817, 40.4731, 40.9987,
                41.59, 42.1156, 42.707, 43.2326, 43.8239, 44.4152, 44.9409,
                45.5322, 46.0578, 46.6491, 47.2405, 47.7661, 48.3574, 48.883,
                49.4744, 50.0]
Kb_16_over_y = [0.998106, 0.994318, 0.99053, 0.985795, 0.982008, 0.97822,
                0.973485, 0.96875, 0.964962, 0.961174, 0.956439, 0.951705,
                0.947917, 0.943182, 0.939394, 0.935606, 0.930871, 0.926136,
                0.921402, 0.918561, 0.913826, 0.910038, 0.90625]

Kb_10_over_x = [30.0263, 30.6176, 31.1432, 31.6689, 32.1945, 32.6544, 33.18,
                33.7057, 34.1656, 34.6255, 35.0854, 35.5453, 36.0053, 36.4652,
                36.9251, 37.385, 37.8449, 38.2392, 38.6334, 39.0276, 39.4875,
                39.9474, 40.4074, 40.8016, 41.1958, 41.59, 42.0499, 42.4442,
                42.8384, 43.2326, 43.6925, 44.0867, 44.4809, 44.8752, 45.2694,
                45.6636, 46.0578, 46.452, 46.8463, 47.2405, 47.6347, 48.0289,
                48.4231, 48.883, 49.2773, 49.6715]
Kb_10_over_y = [0.998106, 0.995265, 0.99053, 0.985795, 0.981061, 0.975379,
                0.969697, 0.963068, 0.957386, 0.950758, 0.945076, 0.938447,
                0.930871, 0.925189, 0.918561, 0.910985, 0.904356, 0.897727,
                0.891098, 0.883523, 0.876894, 0.870265, 0.862689, 0.856061,
                0.848485, 0.840909, 0.83428, 0.827652, 0.820076, 0.8125,
                0.805871, 0.798295, 0.79072, 0.783144, 0.775568, 0.768939,
                0.762311, 0.754735, 0.747159, 0.739583, 0.732008, 0.724432,
                0.716856, 0.70928, 0.701705, 0.695076]


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
        The maximum fraction overpressure; one of 0.1, 0.16, or 0.21, [-]

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
    gauge_backpressure = (Pback-atm)/(Pset-atm)*100.0 # in percent
    if overpressure not in (0.1, 0.16, 0.21):
        raise ValueError('Only overpressure of 10%, 16%, or 21% are permitted')
    if (overpressure == 0.1 and gauge_backpressure < 30.0) or (
        overpressure == 0.16 and gauge_backpressure < 38.0) or (
        overpressure == 0.21 and gauge_backpressure <= 50.0):
        return 1.0
    elif gauge_backpressure > 50.0:
        raise ValueError('Gauge pressure must be < 50%')
    if overpressure == 0.16:
        Kb = interp(gauge_backpressure, Kb_16_over_x, Kb_16_over_y)
    elif overpressure == 0.1:
        Kb = interp(gauge_backpressure, Kb_10_over_x, Kb_10_over_y)
    return Kb


def API520_A_g(m, T, Z, MW, k, P1, P2=101325, Kd=0.975, Kb=1, Kc=1):
    r'''Calculates required relief valve area for an API 520 valve passing
    a gas or a vapor, at either critical or sub-critical flow.

    For critical flow:

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
        Combination correction factor for installation with a rupture disk
        upstream of the PRV; 1.0 when a rupture disk is not installed, and
        0.9 if a rupture disk is present and the combination has not been
        certified, []

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

    Example 2 from [1]_ for sub-critical flow, matches:

    >>> API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1, Kc=1)
    0.004248358775943481

    The mass flux in (kg/(s*m^2)) can be found by dividing the specified mass
    flow by the calculated area:

    >>> (24270/3600.)/API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)
    1822.541960488834

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    P1, P2 = P1*1e-3, P2*1e-3 # Pa to Kpa in the standard
    m = m*3600. # kg/s to kg/hr
    if is_critical_flow(P1, P2, k):
        C = API520_C(k)
        A = m/(C*Kd*Kb*Kc*P1)*sqrt(T*Z/MW)
    else:
        F2 = API520_F2(k, P1, P2)
        A = 17.9*m/(F2*Kd*Kc)*sqrt(T*Z/(MW*P1*(P1-P2)))
    return A*1e-6# convert mm^2 to m^2


def API520_A_steam(m, T, P1, Kd=0.975, Kb=1, Kc=1, edition=TENTH_EDITION):
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
        Correction due to backpressure, see :obj:`API520_B` [-]
    Kc : float, optional
        Combination correction factor for installation with a rupture disk
        upstream of the PRV; 1.0 when a rupture disk is not installed, and
        0.9 if a rupture disk is present and the combination has not been
        certified, []
    edition : str, optional
        One of '10E', '7E', [-]

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
    Example 4 from [1]_ 7th edition, matches:

    >>> API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1, edition='7E')
    0.001103471242369

    Example 4 from the 10th edition of [1]_:

    >>> API520_A_steam(m=69615/3600., T=707.0389, P1=12236E3, Kd=0.975, Kb=1, Kc=1, edition='10E')
    0.00128518893191

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    KN = API520_N(P1)
    KSH = API520_SH(T, P1, edition)
    P1 = P1*1e-3 # Pa to kPa
    m = m*3600. # kg/s to kg/hr
    A = 190.5*m/(P1*Kd*Kb*Kc*KN*KSH)
    return A*1e-6# convert mm^2 to m^2

### Liquids

def API520_Kv(Re, edition=TENTH_EDITION):
    r'''Calculates correction due to viscosity for liquid flow for use in
    API 520 relief valve sizing.

    From the 7th to 9th editions, the formula for this calculation is as
    follows:

    .. math::
        K_v = \left(0.9935 + \frac{2.878}{Re^{0.5}} + \frac{342.75}
        {Re^{1.5}}\right)^{-1}

    Startign in the 10th edition, the formula is

    .. math::
        K_v = \left(1 + \frac{170}{Re}\right)^{-0.5}

    In the 10th edition, the formula is applicable for Re > 80. It is also
    recommended there that if the viscosity is < 0.1 Pa*s, this correction
    should be set to 1.

    Parameters
    ----------
    Re : float
        Reynolds number for flow out the valve [-]
    edition : str, optional
        One of '10E', '7E', [-]

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

    The constant 18800 is derived as follows, combining multiple unit
    conversions and the formula from diameter from area together. The precise
    value is shown below.

    >>> from scipy.constants import *
    >>> liter/minute*1000./(0.001*(milli**2)**0.5)*sqrt(4/pi)
    18806.319451591

    Note that 4 formulas are provided in API 520 part 1; two metric and two
    imperial. One pair of formulas uses viscosity in conventional units; the
    other uses it in Saybolt Universal Seconds. A conversion is essentially
    embedded in the the Saybolt Universal Seconds formula. A more precise
    conversion can be obtained from
    :obj:`chemicals.viscosity.viscosity_converter`.

    In both editions, if the formula is used below the recommended Re range
    and into the very low Re region this correction tends towards 0.

    In the 10th edition, the formula tends to 1 exactly as Re increases. In the
    7th edition, the formula can actually produce corrections above 1; this is
    handled by truncating the factor to 1.

    Examples
    --------
    From [1]_ 7E, checked with example 5.

    >>> API520_Kv(100, edition='7E')
    0.615744589

    From [2]_ 10E, checked with example 5:

    >>> API520_Kv(4525, edition='10E')
    0.9817287137013179

    Example in [3]_, using the 7th edition formula:

    >>> API520_Kv(2110, edition='7E')
    0.943671807

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection, 7E
    .. [2] API Standard 520, Part 1 - Sizing and Selection, 10E
    .. [3] CCPS. Guidelines for Pressure Relief and Effluent Handling Systems.
       2nd edition. New York, NY: Wiley-AIChE, 2017.
    '''
    if edition == SEVENTH_EDITION:
        factor = 1.0/(0.9935 + 2.878/sqrt(Re) + 342.75/(Re*sqrt(Re)))
        if factor > 1.0:
            factor = 1.0
        return factor
    elif edition == TENTH_EDITION:
        return 1.0/sqrt(170.0/Re + 1.0)
    else:
        raise ValueError("Acceptable editions are '7E', '10E'")



# Kw, for liquids. Applicable for all overpressures.
Kw_x = [15., 16.5493, 17.3367, 18.124, 18.8235, 19.5231, 20.1351, 20.8344,
        21.4463, 22.0581, 22.9321, 23.5439, 24.1556, 24.7674, 25.0296, 25.6414,
        26.2533, 26.8651, 27.7393, 28.3511, 28.9629, 29.6623, 29.9245, 30.5363,
        31.2357, 31.8475, 32.7217, 33.3336, 34.0329, 34.6448, 34.8196, 35.4315,
        36.1308, 36.7428, 37.7042, 38.3162, 39.0154, 39.7148, 40.3266, 40.9384,
        41.6378, 42.7742, 43.386, 43.9978, 44.6098, 45.2216, 45.921, 46.5329,
        47.7567, 48.3685, 49.0679, 49.6797, 50.]
Kw_y = [1, 0.996283, 0.992565, 0.987918, 0.982342, 0.976766, 0.97119, 0.964684,
        0.958178, 0.951673, 0.942379, 0.935874, 0.928439, 0.921933, 0.919145,
        0.912639, 0.906134, 0.899628, 0.891264, 0.884758, 0.878253, 0.871747,
        0.868959, 0.862454, 0.855948, 0.849442, 0.841078, 0.834572, 0.828067,
        0.821561, 0.819703, 0.814126, 0.806691, 0.801115, 0.790892, 0.785316,
        0.777881, 0.771375, 0.76487, 0.758364, 0.751859, 0.740706, 0.734201,
        0.727695, 0.722119, 0.715613, 0.709108, 0.702602, 0.69052, 0.684015,
        0.677509, 0.671004, 0.666357]


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
    Custom example from figure 31 in [1]_:

    >>> API520_W(1E6, 3E5) # 22% overpressure
    0.95114718480085

    Example 5 from [2]_, set pressure 250 psig and backpressure up to 50 psig:

    >>> API520_W(Pset=1825014, Pback=446062)
    0.97242133397677

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection. 7E
    .. [2] API Standard 520, Part 1 - Sizing and Selection. 10E
    '''
    gauge_backpressure = (Pback-atm)/(Pset-atm)*100.0 # in percent
    if gauge_backpressure < 15.0:
        return 1.0
    return interp(gauge_backpressure, Kw_x, Kw_y)



rho0 = 999.0107539518483

def API520_A_l(m, rho, P1, P2, overpressure, Kd=0.65, Kc=1.0,
               Kw=None, Kv=None, edition=TENTH_EDITION, mu=None):
    r'''Calculates required relief valve area for an API 520 valve passing
    a liquid in sub-critical flow.

    .. math::
        A = \frac{11.78Q}{K_d K_w K_c K_v}\left(\frac{G_1}{P1 - P2}\right)^{0.5}

    Parameters
    ----------
    m : float
        Mass flow rate of liquid through the valve, [kg/s]
    rho : float
        Liquid density, [kg/m^3]
    P1 : float
        Upstream relieving pressure; the set pressure plus the allowable
        overpressure, plus atmospheric pressure, [Pa]
    P2 : float
        Built-up backpressure; the increase in pressure during flow at the
        outlet of a pressure-relief device after it opens, [Pa]
    overpressure : float
        The maximum fraction overpressure; used if `Kw` is not specified, [-]
    Kd : float, optional
        The effective coefficient of discharge, from the manufacturer or for
        preliminary sizing, using 0.65 normally or 0.62 when used with a
        rupture disc as described in [1]_, []
    Kc : float, optional
        Combination correction factor for installation with a rupture disk
        upstream of the PRV; 1.0 when a rupture disk is not installed, and
        0.9 if a rupture disk is present and the combination has not been
        certified, []
    Kw : float, optional
        Correction due to liquid backpressure [-]
    Kv : float, optional
        Correction due to viscosity [-]
    edition : str, optional
        One of '10E', '7E', [-]
    mu : float, optional
        If provided and `Kv` is None, `Kv` will be calculated automatically,
        [Pa*s]

    Returns
    -------
    A : float
        Minimum area for relief valve according to [1]_, [m^2]

    Notes
    -----
    Units are interlally kg/hr, kPa, and mm^2 to match [1]_.

    This expression is essentially a form of the Loss coefficient `K`
    expression, with many factors and unit conversions. The raw expression in
    SI units, with `K` the true loss coefficient, is as follows:

    .. math::
        A = \frac{\sqrt{2} m \sqrt{\frac{K}{\rho \left(P_{1} - P_{2}\right)}}}{2}

    The constant 11.78 is the result of the following conversions:

        * 60000, converting from m^3/s to L/min
        * sqrt(2)/2 as a factor from algebra
        * 1e6 converting from m^2 to mm^2
        * sqrt(1e-3*(rho0)) converting from Pa to kPa and kg/m^3 to specific gravity

    The full precise value is (depending on the reference density chosen)

    >>> sqrt(1e-3*(999.0107539518483))/60000*sqrt(2)/2*1e6
    11.779282389196

    The K value from a relief valve sized with this method can be calculated
    as follows:

    .. math::
        K = \frac{2 A^{2} \rho \left(P_{1} - P_{2}\right)}{m^{2}}


    The K value can also be directly calculated from the coefficients Kd, Kc,
    Kw, and Kv. The calculation is as follows, making use of the correction
    above.

    .. math::
        K = \left(\frac{1}{K_d K_w K_c K_v\cdot (11.779282389196/11.78)}\right)^2

    Examples
    --------
    Example 5 in [1]_, 10th edition. The calculation involves numerous steps,
    shown below and ending with a recalculation with a viscosity correction.

    >>> Q = 6814*1.6666666666666667e-05 # L/min to m^3/s
    >>> rho = 0.9*999 # specific gravity times density of water kg/m^3
    >>> m = rho*Q # mass flow rate, kg/s
    >>> overpressure = 0.1
    >>> P_design_g = 1724E3 # design pressure, guage
    >>> P1 = (1+overpressure)*P_design_g + 101325.0 # upstream relieving pressure, Pa
    >>> backpressure = 0.2
    >>> mu = 0.388 # viscosity, Pa*s, converted from 2000 Saybolt Universal Seconds
    >>> P2 = backpressure*P_design_g + 101325.0 # backpressure, Pa

    Do the first calculation, using the value of Kw=0.97 shown in [1]

    >>> A0 = API520_A_l(m=m, rho=rho, P1=P1, P2=P2, overpressure=overpressure, Kd=0.65, Kw=0.97, Kc=1.0, Kv=1.0)
    >>> A0
    0.0030661356203

    This value matches the 3066 mm^2 shown in the example calculation.

    Do the same calculation but allow the calculation of `Kw` automatically:

    >>> A0 = API520_A_l(m=m, rho=rho, P1=P1, P2=P2, overpressure=overpressure, Kd=0.65, Kc=1.0, Kv=1.0)
    >>> A0
    0.0030585022573

    There is a slight deviation with a more precise `Kw` value.

    Compute Reynolds number from this original area

    >>> from math import pi
    >>> D = (A0*4/pi)**0.5
    >>> v = Q/A0
    >>> Re = rho*v*D/mu
    >>> Re
    5369.4253339

    The reynolds number shown in [1] is 4525; the difference comes from the less
    precise Saybolt Universal Seconds conversion.

    Compute the viscosity correction:

    >>> Kv = API520_Kv(Re, '10E')
    >>> Kv
    0.984535878488

    Compute the final area

    >>> A = API520_A_l(m=m, rho=rho, P1=P1, P2=P2, overpressure=overpressure, Kd=0.65, Kc=1.0, Kv=Kv)
    >>> A
    0.003106542203

    The final answer given in API 520 example 5 is 3122 mm^2, a very similar
    value despite the small differences.

    If is also possible to have `Kv` be calculated by this routine
    automatically, by setting `Kv` to None and providing the fluid's viscosity.

    >>> A = API520_A_l(m=m, rho=rho, P1=P1, P2=P2, overpressure=overpressure, Kd=0.65, Kc=1.0, Kv=None, mu=mu)
    >>> A
    0.003106542203

    As described in the note, an overall K value can be calculated for the
    valve

    >>> K = 2*A**2*rho*(P1 - P2)/m**2
    >>> K
    2.5825844233354602

    We can check the calculation

    >>> from fluids.core import dP_from_K
    >>> v = Q/A
    >>> dP_from_K(K=K, rho=rho, V=v), P1-P2
    (1551600.000, 1551600.00)

    References
    ----------
    .. [1] API Standard 520, Part 1 - Sizing and Selection.
    '''
    G1 = rho/rho0
    Q = m/rho # m^3/s
    Q *= 60000.0 # m^3/s to L/min in the original equation

    P_set_guage = (P1 - atm)/(1.0 + overpressure)
    P_set = P_set_guage + atm
    if Kw is None:
        Kw = API520_W(P_set, P2)
    if Kv is None and mu is not None:
        A0 = API520_A_l(m=m, rho=rho, P1=P1, P2=P2, overpressure=overpressure, Kd=Kd, Kc=Kc, Kv=1.0, Kw=Kw)
        D = sqrt(A0*4.0/pi)
        v = (Q/60000.0)/A0
        Re = rho*v*D/mu
        Kv = API520_Kv(Re, edition)
    P1 = P1*1e-3 # Pa to kPa
    P2 = P2*1e-3 # Pa to kPa
    A = 11.78*Q*sqrt(G1/(P1-P2))/(Kd*Kw*Kc*Kv)
    A = A*1e-6# convert mm^2 to m^2
    return A

def API521_noise_graph(P_ratio):
    r'''Calculate the `L` parameter used in the API 521
    noise calculation, from their Figure 18, Sound
    Pressure Level at 30 m from the stack tip.

    Parameters
    ----------
    P_ratio : float
        The ratio of relieving pressure to atmospheric pressure [-]

    Returns
    -------
    L : float
        Sound pressure level at 30 m from the stack tip [decibels]

    Notes
    -----
    Two logarithmic linear polynomials are used. The function is
    continious throughout. The pressure ratio should be more than 1
    for physical reasons; the value is checked for this case.

    References
    ----------
    .. [1] API Standard 521.
    '''
    if P_ratio < 1.0:
        P_ratio = 1.0
    lgX = log10(P_ratio)
    # Small curve fit
    lower_value = 87.9084*lgX + 12.7647
    higher_value = 4.8239*lgX + 51.6217
    if P_ratio < 2.92:
        value = lower_value
    elif P_ratio < 2.93:
        # interpolate between the two curves to keep the function continuous
        value = interp(P_ratio, [2.92, 2.93], [lower_value, higher_value])
    else:
        value = higher_value
    return value

def API521_noise(m, P1, P2, c, r):
    r'''Calculate the the noise coming from a flare tip at a
    specified distance according to API 521. A graphical technique
    is used to get the noise at 30 m from the tip, and it is then
    adjusted for distance.

    .. math::
        L_{30 \text{m}} = L - 10 \log_{10}(0.5 m c^2)

    .. math::
        L_p = L_{30 \text{m}} - 20 \log_{10}(r/(30 \text{m}))

    Parameters
    ----------
    m : float
        Mass flow rate of relieving fluid, [kg/s]
    P1 : float
        Upstream pressure at the source, before the relieving
        device [Pa]
    P2 : float
        Atmospheric pressure, [Pa]
    c : float
        Speed of sound of the fluid at the relieving device [m/s]
    r : float
        Distance from the flare stack, [m]

    Returns
    -------
    L : float
        Sound pressure level at the specified distance from the
        stack tip [decibels]

    Notes
    -----

    Examples
    --------
    Example as shown in [1]_:

    >>> API521_noise(m=14.6, P1=330E3, P2=101325, c=353.0, r=30)
    113.6841057

    References
    ----------
    .. [1] API Standard 521.
    '''
    P_ratio = P1/P2
    L = API521_noise_graph(P_ratio) # from chart, hardcoded for now
    L30 = L + 10.0*log10(0.5*m*c*c)
    Lp = L30 - 20.0*log10(r*(1.0/30.0))
    return Lp


def VDI_3732_noise_ground_flare(m):
    r'''Calculate the the noise at the flare tip of a ground flare
    [1]_, [2]_.

    .. math::
        L = 100 + 15\log_{10}\left(\frac{m}{\text{tonne/hour}}\right)

    Parameters
    ----------
    m : float
        Mass flow rate of relieving fluid, [kg/s]

    Returns
    -------
    noise : float
        Sound pressure level at the relieving flare stack [decibels]

    Notes
    -----

    Examples
    --------
    >>> VDI_3732_noise_ground_flare(3.0)
    145.501356332

    References
    ----------
    .. [1] VDI 3732 - Standard Noise Levels of Technical Sound
       Sources - Flares, 1999.
       https://www.vdi.de/en/home/vdi-standards/details/vdi-3732-standard-noise-levels-of-technical-sound-sources-flares.
    .. [2] AdminFlare Noise Calculator. WKC Group (blog).
       https://www.wkcgroup.com/tools-room/flare-noise-calculator/.

    '''
    m *= 360.0
    return 100.0 + 15.0*log10(m)

def VDI_3732_noise_elevated_flare(m):
    r'''Calculate the the noise at the flare tip of an elevated flare stack
    [1]_, [2]_.

    .. math::
        L = 112 + 17\log_{10}\left(\frac{m}{\text{tonne/hour}}\right)

    Parameters
    ----------
    m : float
        Mass flow rate of relieving fluid, [kg/s]

    Returns
    -------
    noise : float
        Sound pressure level at the relieving flare stack [decibels]

    Notes
    -----

    Examples
    --------
    >>> VDI_3732_noise_elevated_flare(3.0)
    163.56820384

    References
    ----------
    .. [1] VDI 3732 - Standard Noise Levels of Technical Sound
       Sources - Flares, 1999.
       https://www.vdi.de/en/home/vdi-standards/details/vdi-3732-standard-noise-levels-of-technical-sound-sources-flares.
    .. [2] AdminFlare Noise Calculator. WKC Group (blog).
       https://www.wkcgroup.com/tools-room/flare-noise-calculator/.
    '''
    m *= 360.0
    return 112.0 + 17.0*log10(m)
