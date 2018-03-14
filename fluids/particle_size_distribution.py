# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
SOFTWARE.'''
from __future__ import division

__all__ = ['ParticleSizeDistribution', 'ParticleSizeDistributionContinuous',
           'PSDLognormal', 'PSDGatesGaudinSchuhman', 'PSDRosinRammler',
           'PSDInterpolated', 'PSDCustom', 'psd_spacing',
           'pdf_lognormal', 'cdf_lognormal', 'pdf_lognormal_basis_integral',
           'pdf_Gates_Gaudin_Schuhman', 'cdf_Gates_Gaudin_Schuhman',
           'pdf_Gates_Gaudin_Schuhman_basis_integral',
           'pdf_Rosin_Rammler', 'cdf_Rosin_Rammler', 
           'pdf_Rosin_Rammler_basis_integral',
           'ASTM_E11_sieves', 'ISO_3310_1_sieves', 'Sieve',
           'ISO_3310_1_R20_3', 'ISO_3310_1_R20', 'ISO_3310_1_R10', 
           'ISO_3310_1_R40_3']

from math import log, exp, pi, log10
from io import open
import os
from sys import float_info
from numpy.random import lognormal
import numpy as np
from scipy.optimize import brenth, minimize
from scipy.integrate import quad
from scipy.special import erf, gammaincc, gamma
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, PchipInterpolator
import scipy.stats

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except:
    has_matplotlib = False


folder = os.path.join(os.path.dirname(__file__), 'data')

ROOT_TWO_PI = (2.0*pi)**0.5


class Sieve(object):
    r'''Class for storing data on sieves. If a property is not available, it is
    set to None.

    Attributes
    ----------
    designation : str
        The standard name of the sieve - its opening's length in units of 
        milimeters
    old_designation : str
        The older, imperial-esque name of the sieve; in Numbers, or inches for 
        large sieves
    opening : float
        The opening length of the sieve holes, [m]
    opening_inch : float
        The opening length of the sieve holes in the rounded inches as stated
        in common tables (not exactly equal to the `opening`), [inch]
    Y_variation_avg : float
        The allowable average variation in the Y direction of the sieve 
        openings, [m]
    X_variation_max : float
        The allowable maximum variation in the X direction of the sieve 
        openings, [m]
    max_opening : float
        The maximum allowable opening of the sieve, [m]
    calibration_samples : float
        The number of opening sample inspections required for `calibration`-
        type sieve openings (per 100 ft^2 of sieve material), [1/(ft^2)]
    compliance_sd : float
        The maximum standard deviation of `compliance`-type sieve openings, 
        [-]
    inspection_samples : float
        The number of opening sample inspections required for `inspection`-
        type sieve openings (based on an 8-inch sieve), [-]
    inspection_sd : float
        The maximum standard deviation of `inspection`-type sieve openings, 
        [-]
    calibration_samples : float
        The number of opening sample inspections required for `calibration`-
        type sieve openings (based on an 8-inch sieve), [-]
    calibration_sd : float
        The maximum standard deviation of `calibration`-type sieve openings, 
        [-]
    d_wire : float
        Typical wire diameter of the specified sieve size, [m] 
    d_wire_min : float
        Permissible minimum wire diameter of specified sieve size, [m]
    d_wire_max : float
        Permissible maximum wire diameter of specified sieve size, [m]
        
    '''
    def __init__(self, designation, old_designation=None, opening=None,
                 opening_inch=None, Y_variation_avg=None, X_variation_max=None,
                 max_opening=None, compliance_samples=None, compliance_sd=None,
                 inspection_samples=None, inspection_sd=None, calibration_samples=None,
                 calibration_sd=None, d_wire=None, d_wire_min=None, d_wire_max=None):
        
        self.designation = designation
        self.old_designation = old_designation
        self.opening_inch = opening_inch
        self.opening = opening
        
        self.Y_variation_avg = Y_variation_avg
        self.X_variation_max = X_variation_max
        self.max_opening = max_opening
        
        self.compliance_samples = compliance_samples
        self.compliance_sd = compliance_sd
        
        self.inspection_samples = inspection_samples 
        self.inspection_sd = inspection_sd
        
        self.calibration_samples = calibration_samples
        self.calibration_sd = calibration_sd
        
        self.d_wire = d_wire
        self.d_wire_min = d_wire_min
        self.d_wire_max = d_wire_max
    

ASTM_E11_sieves = {}
ASTM_E11_sieve_list = []

with open(os.path.join(folder, 'ASTM E11 sieves.csv'), encoding='utf-8') as f:    
    # All sieves are read in from large to small
    lines = f.readlines()[1:]
    for line in lines:
        values = line.strip().split('\t')
        designation, old_designation, opening = values[0], values[1], float(values[0])*1e-3
        args = []
        for arg in values[2:]:
            try:
                arg = float(arg)
            except:
                arg = None
            args.append(arg)
        # First three and last three arguments have units to be changed from mm to m
        for i in (0, 1, 2, -1, -2, -3):
            args[i] = args[i]*1e-3
        
        # Store the Sieve object
        s = Sieve(designation, old_designation, opening, *args)
        ASTM_E11_sieves[designation] = s
        ASTM_E11_sieve_list.append(s)




ISO_3310_1_sieves = {}
ISO_3310_1_sieve_list = []

with open(os.path.join(folder, 'ISO 3310-1 sieves.csv'), encoding='utf-8') as f:    
    lines = f.readlines()[1:]
    for line in lines:
        values = line.strip().split('\t')
        args = []
        for arg in values:
            try:
                arg = float(arg)
            except:
                arg = None
            args.append(arg)
        # Key should be size in mm as a string
        designation = '%g' %(round(args[0]*1000.0, 4))
        
        s = Sieve(designation=designation, opening=args[0],
                  X_variation_max=args[1], Y_variation_avg=args[2], 
                  compliance_sd=args[3], d_wire=args[4], 
                  d_wire_max=args[5], d_wire_min=args[6])
        ISO_3310_1_sieves[designation] = s
        ISO_3310_1_sieve_list.append(s)


ISO_3310_1_R20_3 = ['125', '90', '63', '45', '31.5', '22.4', '16', '11.2', '8', '5.6', '4', '2.8', '2', '1.4', '1', '0.71', '0.5', '0.355', '0.25', '0.18', '0.125', '0.09', '0.063', '0.045']
ISO_3310_1_R20_3 = [ISO_3310_1_sieves[i] for i in ISO_3310_1_R20_3]

ISO_3310_1_R20 = ['125', '112', '100', '90', '80', '71', '63', '56', '50', '45', '40', '35.5', '31.5', '28', '25', '22.4', '20', '18', '16', '14', '12.5', '11.2', '10', '9', '8', '7.1', '6.3', '5.6', '5', '4.5', '4', '3.55', '3.15', '2.8', '2.5', '2.24', '2', '1.8', '1.6', '1.4', '1.25', '1.12', '1', '0.9', '0.8', '0.71', '0.63', '0.56', '0.5', '0.45', '0.4', '0.355', '0.315', '0.28', '0.25', '0.224', '0.2', '0.18', '0.16', '0.14', '0.125', '0.112', '0.1', '0.09', '0.08', '0.071', '0.063', '0.056', '0.05', '0.045', '0.04', '0.036']
ISO_3310_1_R20 = [ISO_3310_1_sieves[i] for i in ISO_3310_1_R20]

ISO_3310_1_R40_3 = ['125', '106', '90', '75', '63', '53', '45', '37.5', '31.5', '26.5', '22.4', '19', '16', '13.2', '11.2', '9.5', '8', '6.7', '5.6', '4.75', '4', '3.35', '2.8', '2.36', '2', '1.7', '1.4', '1.18', '1', '0.85', '0.71', '0.6', '0.5', '0.425', '0.355', '0.3', '0.25', '0.212', '0.18', '0.15', '0.125', '0.106', '0.09', '0.075', '0.063', '0.053', '0.045', '0.038']
ISO_3310_1_R40_3 = [ISO_3310_1_sieves[i] for i in ISO_3310_1_R40_3]

ISO_3310_1_R10 = ['0.036', '0.032', '0.025', '0.02']
ISO_3310_1_R10 = [ISO_3310_1_sieves[i] for i in ISO_3310_1_R10]

sieve_spacing_options = {'ISO 3310-1': ISO_3310_1_sieve_list,
                         'ISO 3310-1 R20': ISO_3310_1_R20,
                         'ISO 3310-1 R20/3': ISO_3310_1_R20_3,
                         'ISO 3310-1 R40/3': ISO_3310_1_R40_3,
                         'ISO 3310-1 R10': ISO_3310_1_R10,
                         'ASTM E11': ASTM_E11_sieve_list,}


def psd_spacing(dmin=None, dmax=None, pts=20, method='logarithmic'):
    r'''Create a particle spacing mesh in one of several ways for use in
    modeling discrete particle size distributions. The allowable meshes are
    'linear', 'logarithmic', a geometric series specified by a Renard number
    such as 'R10', or the meshes available in one of several sieve standards.
    
    Parameters
    ----------
    dmin : float, optional
        The minimum diameter at which the mesh starts, [m]
    dmax : float, optional
        The maximum diameter at which the mesh ends, [m]
    pts : int, optional
        The number of points to return for the mesh (note this is not respected
        by sieve meshes), [-]
    method : str, optional
        Either 'linear', 'logarithmic', a Renard number like 'R10' or 'R5' or
        'R2.5', or one of the sieve standards 'ISO 3310-1 R40/3', 
        'ISO 3310-1 R20', 'ISO 3310-1 R20/3', 'ISO 3310-1', 'ISO 3310-1 R10', 
        'ASTM E11', [-]

    Returns
    -------
    ds : list[float]
        The generated mesh diameters, [m]

    Notes
    -----
    Note that when specifying a Renard series, only one of `dmin` or `dmax` can
    be respected! Provide only one of those numbers. 
    
    Note that when specifying a sieve standard the number of points is not
    respected!

    Examples
    --------
    >>> psd_spacing(dmin=5e-5, dmax=5e-4, method='ISO 3310-1 R20/3')
    [6.3e-05, 9e-05, 0.000125, 0.00018, 0.00025, 0.000355, 0.0005]

    References
    ----------
    .. [1] ASTM E11 - 17 - Standard Specification for Woven Wire Test Sieve 
       Cloth and Test Sieves.
    .. [2] ISO 3310-1:2016 - Test Sieves -- Technical Requirements and Testing
       -- Part 1: Test Sieves of Metal Wire Cloth.
    '''
    if method == 'logarithmic':
        return np.logspace(log10(dmin), log10(dmax), pts).tolist()
    elif method == 'linear':
        return np.linspace(dmin, dmax, pts).tolist()
    elif method[0] in ('R', 'r'):
        ratio = 10**(1.0/float(method[1:]))
        if dmin is not None and dmax is not None:
            raise Exception('For geometric (Renard) series, only '
                            'one of `dmin` and `dmax` should be provided')
        if dmin is not None:
            ds = [dmin]
            for i in range(pts-1):
                ds.append(ds[-1]*ratio)
            return ds
        elif dmax is not None:
            ds = [dmax]
            for i in range(pts-1):
                ds.append(ds[-1]/ratio)
            return list(reversed(ds))
    elif method in sieve_spacing_options:
        l = sieve_spacing_options[method]
        ds = []
        for sieve in l:
           if  dmin <= sieve.opening <= dmax:
               ds.append(sieve.opening)
        return list(reversed(ds))
    else:
        raise Exception('Method not recognized')


def pdf_lognormal(d, d_characteristic, s):
    r'''Calculates the probability density function of a lognormal particle
    distribution given a particle diameter `d`, characteristic particle
    diameter `d_characteristic`, and distribution standard deviation `s`.
    
    .. math::
        q(d) = \frac{1}{ds\sqrt{2\pi}} \exp\left[-0.5\left(\frac{
        \ln(d/d_{characteristic})}{s}\right)^2\right]
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; often D[3, 3] is used for this
        purpose but not always, [m]
    s : float
        Distribution standard deviation, [-]    

    Returns
    -------
    pdf : float
        Lognormal probability density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted 
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.
        
    The standard distribution (i.e. the one used in Scipy) can perform the same
    computation with  `d_characteristic` as the value of `scale`.

    >>> scipy.stats.lognorm.pdf(x=1E-4, s=1.1, scale=1E-5)
    405.5420921156425
    
    Scipy's calculation is over 300 times slower however, and this expression
    is numerically integrated so speed is required.

    Examples
    --------
    >>> pdf_lognormal(d=1E-4, d_characteristic=1E-5, s=1.1)
    405.5420921156425

    References
    ----------
    .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size 
       Analysis - Part 2: Calculation of Average Particle Sizes/Diameters and 
       Moments from Particle Size Distributions.
    '''
    try:
        log_term = log(d/d_characteristic)/s
    except ValueError:
        return 0.0
    return 1./(d*s*ROOT_TWO_PI)*exp(-0.5*log_term*log_term)


def cdf_lognormal(d, d_characteristic, s):
    r'''Calculates the cumulative distribution function of a lognormal particle
    distribution given a particle diameter `d`, characteristic particle
    diameter `d_characteristic`, and distribution standard deviation `s`.
    
    .. math::
        Q(d) = 0.5\left(1 + \text{err}\left[\left(\frac{\ln(d/d_c)}{s\sqrt{2}}
        \right)\right]\right)
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; often D[3, 3] is used for this
        purpose but not always, [m]
    s : float
        Distribution standard deviation, [-]    

    Returns
    -------
    cdf : float
        Lognormal cummulative density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted 
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.
        
    The standard distribution (i.e. the one used in Scipy) can perform the same
    computation with  `d_characteristic` as the value of `scale`.

    >>> scipy.stats.lognorm.cdf(x=1E-4, s=1.1, scale=1E-5)
    0.98183698757981774
    
    Scipy's calculation is over 100 times slower however.

    Examples
    --------
    >>> cdf_lognormal(d=1E-4, d_characteristic=1E-5, s=1.1)
    0.98183698757981763

    References
    ----------
    .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size 
       Analysis - Part 2: Calculation of Average Particle Sizes/Diameters and 
       Moments from Particle Size Distributions.
    '''
    try:
        return 0.5*(1.0 + erf((log(d/d_characteristic))/(s*2.0**0.5)))
    except:
        # math error at cdf = 0 (x going as low as possible)
        return 0.0


def pdf_lognormal_basis_integral(d, d_characteristic, s, n):
    r'''Calculates the integral of the multiplication of d^n by the lognormal
    pdf, given a particle diameter `d`, characteristic particle
    diameter `d_characteristic`, distribution standard deviation `s`, and
    exponent `n`.
    
    .. math::
        \int d^n\cdot q(d)\; dd = -\frac{1}{2} \exp\left(\frac{s^2 n^2}{2}
        \right)d^n \left(\frac{d}{d_{characteristic}}\right)^{-n}
        \text{erf}\left[\frac{s^2 n - \log(d/d_{characteristic})}
        {\sqrt{2} s} \right]
        
    This is the crucial integral required for interconversion between different
    bases such as number density (denoted :math:`q_0(d)`), length density 
    (:math:`q_1(d)`), surface area density (:math:`q_2(d)`), or volume density 
    (:math:`q_3(d)`).
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; often D[3, 3] is used for this
        purpose but not always, [m]
    s : float
        Distribution standard deviation, [-]    
    n : int
        Exponent of the multiplied n

    Returns
    -------
    pdf_basis_integral : float
        Integral of lognormal pdf multiplied by d^n, [-]

    Notes
    -----
    This integral has been verified numerically. This integral is itself
    integrated, so it is crucial to obtain an analytical form for at least
    this integral.
    
    Note overflow or zero division issues may occur for very large values of 
    `s`, larger than 10. No mathematical limit was able to be obtained with 
    a CAS.

    Examples
    --------
    >>> pdf_lognormal_basis_integral(d=1E-4, d_characteristic=1E-5, s=1.1, n=-2)
    56228306549.263626
    '''
    try:
        s2 = s*s
        t0 = exp(s2*n*n*0.5)
        d_ratio = d/d_characteristic
        t1 = (d/(d_ratio))**n
        t2 = erf((s2*n - log(d_ratio))/(2.**0.5*s))
        return -0.5*t0*t1*t2
    except (OverflowError, ZeroDivisionError, ValueError):
        return pdf_lognormal_basis_integral(d=1E-80, d_characteristic=d_characteristic, s=s, n=n)


def pdf_Gates_Gaudin_Schuhman(d, d_characteristic, m):
    r'''Calculates the probability density of a particle
    distribution following the Gates, Gaudin and Schuhman (GGS) model given a 
    particle diameter `d`, characteristic (maximum) particle
    diameter `d_characteristic`, and exponent `m`.
    
    .. math::
        q(d) = \frac{n}{d}\left(\frac{d}{d_{characteristic}}\right)^m 
        \text{ if } d < d_{characteristic} \text{ else } 0
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; in this model, it is the largest 
        particle size diameter in the distribution, [m]
    m : float
        Particle size distribution exponent, [-]    

    Returns
    -------
    pdf : float
        GGS probability density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted 
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.

    Examples
    --------
    >>> pdf_Gates_Gaudin_Schuhman(d=2E-4, d_characteristic=1E-3, m=2.3)
    283.8355768512045

    References
    ----------
    .. [1] Schuhmann, R., 1940. Principles of Comminution, I-Size Distribution 
       and Surface Calculations. American Institute of Mining, Metallurgical 
       and Petroleum Engineers Technical Publication 1189. Mining Technology, 
       volume 4, p. 1-11.
    .. [2] Bayat, Hossein, Mostafa Rastgo, Moharram Mansouri Zadeh, and Harry 
       Vereecken. "Particle Size Distribution Models, Their Characteristics and
       Fitting Capability." Journal of Hydrology 529 (October 1, 2015): 872-89.
    '''
    if d <= d_characteristic:
        return m/d*(d/d_characteristic)**m
    else:
        return 0.0


def cdf_Gates_Gaudin_Schuhman(d, d_characteristic, m):
    r'''Calculates the cumulative distribution function of a particle
    distribution following the Gates, Gaudin and Schuhman (GGS) model given a 
    particle diameter `d`, characteristic (maximum) particle
    diameter `d_characteristic`, and exponent `m`.
    
    .. math::
        Q(d) = \left(\frac{d}{d_{characteristic}}\right)^m \text{ if } 
        d < d_{characteristic} \text{ else } 1
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; in this model, it is the largest 
        particle size diameter in the distribution, [m]
    m : float
        Particle size distribution exponent, [-]    

    Returns
    -------
    cdf : float
        GGS cummulative density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted 
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.

    Examples
    --------
    >>> cdf_Gates_Gaudin_Schuhman(d=2E-4, d_characteristic=1E-3, m=2.3)
    0.024681354508800397

    References
    ----------
    .. [1] Schuhmann, R., 1940. Principles of Comminution, I-Size Distribution 
       and Surface Calculations. American Institute of Mining, Metallurgical 
       and Petroleum Engineers Technical Publication 1189. Mining Technology, 
       volume 4, p. 1-11.
    .. [2] Bayat, Hossein, Mostafa Rastgo, Moharram Mansouri Zadeh, and Harry 
       Vereecken. "Particle Size Distribution Models, Their Characteristics and
       Fitting Capability." Journal of Hydrology 529 (October 1, 2015): 872-89.
    '''
    if d <= d_characteristic:
        return (d/d_characteristic)**m
    else:
        return 1.0


def pdf_Gates_Gaudin_Schuhman_basis_integral(d, d_characteristic, m, n):
    r'''Calculates the integral of the multiplication of d^n by the Gates, 
    Gaudin and Schuhman (GGS) model given a particle diameter `d`,
    characteristic (maximum) particle diameter `d_characteristic`, and exponent
    `m`.
    
    .. math::
        \int d^n\cdot q(d)\; dd =\frac{m}{m+n} d^n \left(\frac{d}
        {d_{characteristic}}\right)^m
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    d_characteristic : float
        Characteristic particle diameter; in this model, it is the largest 
        particle size diameter in the distribution, [m]
    m : float
        Particle size distribution exponent, [-]    
    n : int
        Exponent of the multiplied n, [-]

    Returns
    -------
    pdf_basis_integral : float
        Integral of Rosin Rammler pdf multiplied by d^n, [-]

    Notes
    -----
    This integral does not have any numerical issues as `d` approaches 0.

    Examples
    --------
    >>> pdf_Gates_Gaudin_Schuhman_basis_integral(d=2E-4, d_characteristic=1E-3, m=2.3, n=-3)
    -10136984887.543015
    '''
    return m/(m+n)*d**n*(d/d_characteristic)**m


def pdf_Rosin_Rammler(d, k, m):
    r'''Calculates the probability density of a particle
    distribution following the Rosin-Rammler (RR) model given a 
    particle diameter `d`, and the two parameters `k` and `m`.
    
    .. math::
        q(d) = k m d^{(m-1)} \exp(- k d^{m})
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    k : float
        Parameter in the model, [(1/m)^m]
    m : float
        Parameter in the model, [-]

    Returns
    -------
    pdf : float
        RR probability density function, [-]

    Notes
    -----

    Examples
    --------
    >>> pdf_Rosin_Rammler(1E-3, 200, 2)
    0.3999200079994667

    References
    ----------
    .. [1] Rosin, P. "The Laws Governing the Fineness of Powdered Coal." J. 
       Inst. Fuel. 7 (1933): 29-36.
    .. [2] Bayat, Hossein, Mostafa Rastgo, Moharram Mansouri Zadeh, and Harry 
       Vereecken. "Particle Size Distribution Models, Their Characteristics and
       Fitting Capability." Journal of Hydrology 529 (October 1, 2015): 872-89.
    '''
    return d**(m - 1)*k*m*exp(-d**m*k)


def cdf_Rosin_Rammler(d, k, m):
    r'''Calculates the cumulative distribution function of a particle
    distribution following the Rosin-Rammler (RR) model given a 
    particle diameter `d`, and the two parameters `k` and `m`.
    
    .. math::
        Q(d) = 1 - \exp\left(-k d^m\right)
        
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    k : float
        Parameter in the model, [(1/m)^m]
    m : float
        Parameter in the model, [-]

    Returns
    -------
    cdf : float
        RR cummulative density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted 
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.

    Examples
    --------
    >>> cdf_Rosin_Rammler(5E-2, 200, 2)
    0.3934693402873667

    References
    ----------
    .. [1] Rosin, P. "The Laws Governing the Fineness of Powdered Coal." J. 
       Inst. Fuel. 7 (1933): 29-36.
    .. [2] Bayat, Hossein, Mostafa Rastgo, Moharram Mansouri Zadeh, and Harry 
       Vereecken. "Particle Size Distribution Models, Their Characteristics and
       Fitting Capability." Journal of Hydrology 529 (October 1, 2015): 872-89.
    '''
    return 1.0 - exp(-k*d**m)


def pdf_Rosin_Rammler_basis_integral(d, k, m, n):
    r'''Calculates the integral of the multiplication of d^n by the Rosin
    Rammler (RR) pdf, given a particle diameter `d`, and the two parameters `k`
    and `m`.
    
    .. math::
        \int d^n\cdot q(d)\; dd =-d^{m+n} k(d^mk)^{-\frac{m+n}{m}}\Gamma
        \left(\frac{m+n}{m}\right)\text{gammaincc}\left[\left(\frac{m+n}{m}
        \right), kd^m\right]
    
    Parameters
    ----------
    d : float
        Specified particle diameter, [m]
    k : float
        Parameter in the model, [(1/m)^m]
    m : float
        Parameter in the model, [-]
    n : int
        Exponent of the multiplied n, [-]

    Returns
    -------
    pdf_basis_integral : float
        Integral of Rosin Rammler pdf multiplied by d^n, [-]

    Notes
    -----
    This integral was derived using a CAS, and verified numerically.
    The `gammaincc` function is that from scipy.special, and `gamma` from the
    same.
    
    For very high powers of `n` or `m` when the diameter is very low, 
    execeptions may occur.

    Examples
    --------
    >>> pdf_Rosin_Rammler_basis_integral(5E-2, 200, 2, 3)
    -0.00045239898439007338
    '''
    # Also not able to compute the limit for d approaching 0.
    try:
        a = (m+n)/m
        x = d**m*k
        t1 = gamma(a)*(gammaincc(a, x))
        return -d**(m+n)*k*(d**m*k)**(-a)*t1
    except (OverflowError, ZeroDivisionError) as e:
        if d == 1E-40:
            raise e
        return pdf_Rosin_Rammler_basis_integral(1E-40, k, m, n)


names = {0: 'Number distribution', 1: 'Length distribution',
         2: 'Area distribution', 3: 'Volume/Mass distribution'}

def _label_distribution_n(n):
    if n in names:
        return names[n]
    else:
        return 'Order %s distribution' %n

_mean_size_docstring = r'''Calculates the mean particle size according to moment-ratio 
        notation. This is the more common and often convenient definition.
            
        .. math::
            \left[\bar D_{p,q} \right]^{(p-q)} = \frac{\sum_i n_i  D_i^p }
            {\sum_i n_i D_i^q}
            
            \left[\bar D_{p,p} \right] = \exp\left[\frac{\sum_i n_i  D_i^p\ln 
            D_i }{\sum_i n_i D_i^p}\right]  \text{, if p = q}
    
        Note that :math:`n_i` in the above equation is replaceable with
        the fraction of particles in that bin.
    
        Parameters
        ----------
        p : int
            Power and/or substript of D moment in the above equations, [-]
        q : int
            Power and/or substript of D moment in the above equations, [-]
            
        Returns
        -------
        d_pq : float
            Mean particle size according to the specified p and q, [m]
    
        Notes
        -----
        The following is a list of common names for specific mean diameters.
        
        * **D[-3, 0]**: arithmetic harmonic mean volume diameter 
        * **D[-2, 1]**: size-weighted harmonic mean volume diameter 
        * **D[-1, 2]**: area-weighted harmonic mean volume diameter 
        * **D[-2, 0]**: arithmetic harmonic mean area diameter 
        * **D[-1, 1]**: size-weighted harmonic mean area diameter 
        * **D[-1, 0]**: arithmetic harmonic mean diameter 
        * **D[0, 0]**: arithmetic geometric mean diameter 
        * **D[1, 1]**: size-weighted geometric mean diameter 
        * **D[2, 2]**: area-weighted geometric mean diameter 
        * **D[3, 3]**: volume-weighted geometric mean diameter 
        * **D[1, 0]**: arithmetic mean diameter 
        * **D[2, 1]**: size-weighted mean diameter 
        * **D[3, 2]**: area-weighted mean diameter, **Sauter mean diameter**
        * **D[4, 3]**: volume-weighted mean diameter, **De Brouckere diameter**
        * **D[2, 0]**: arithmetic mean area diameter 
        * **D[3, 1]**: size-weighted mean area diameter 
        * **D[4, 2]**: area-weighted mean area diameter 
        * **D[5, 3]**: volume-weighted mean area diameter 
        * **D[3, 0]**: arithmetic mean volume diameter
        * **D[4, 1]**: size-weighted mean volume diameter
        * **D[5, 2]**: area-weighted mean volume diameter 
        * **D[6, 3]**: volume-weighted mean volume diameter 

        This notation was first introduced in [1]_.
        
        The sum of p and q is called the order of the mean size [3]_.
        
        .. math::
            \bar D_{p,q}  \equiv \bar D_{q, p} 
    
        Examples
        --------
%s
        
        References
        ----------
        .. [1] Mugele, R. A., and H. D. Evans. "Droplet Size Distribution in 
           Sprays." Industrial & Engineering Chemistry 43, no. 6 (June 1951):
           1317-24. https://doi.org/10.1021/ie50498a023.  
        .. [2] ASTM E799 - 03(2015) - Standard Practice for Determining Data 
           Criteria and Processing for Liquid Drop Size Analysis.    
        .. [3] ISO 9276-2:2014 - Representation of Results of Particle Size 
           Analysis - Part 2: Calculation of Average Particle Sizes/Diameters  
           and Moments from Particle Size Distributions.
'''

_mean_size_iso_docstring =  r'''Calculates the mean particle size according to moment 
        notation (ISO). This system is related to the moment-ratio notation 
        as follows; see the `mean_size` method for the full formulas.
        
        .. math::
            \bar x_{p-q, q} \equiv \bar x_{k+r, r}  \equiv \bar D_{p,q} 
            
        Parameters
        ----------
        k : int
            Power and/or substript of D moment in the above equations, [-]
        r : int
            Power and/or substript of D moment in the above equations, [-]
            
        Returns
        -------
        x_kr : float
            Mean particle size according to the specified k and r in the ISO
            series, [m]
    
        Notes
        -----
        The following is a list of common names for specific mean diameters in
        the ISO naming convention.

        * **x[-3, 0]**: arithmetic harmonic mean volume diameter 
        * **x[-3, 1]**: size-weighted harmonic mean volume diameter 
        * **x[-3, 2]**: area-weighted harmonic mean volume diameter 
        * **x[-2, 0]**: arithmetic harmonic mean area diameter 
        * **x[-2, 1]**: size-weighted harmonic mean area diameter 
        * **x[-1, 0]**: arithmetic harmonic mean diameter 
        * **x[0, 0]**: arithmetic geometric mean diameter 
        * **x[0, 1]**: size-weighted geometric mean diameter 
        * **x[0, 2]**: area-weighted geometric mean diameter 
        * **x[0, 3]**: volume-weighted geometric mean diameter 
        * **x[1, 0]**: arithmetic mean diameter 
        * **x[1, 1]**: size-weighted mean diameter 
        * **x[1, 2]**: area-weighted mean diameter, **Sauter mean diameter**
        * **x[1, 3]**: volume-weighted mean diameter, **De Brouckere diameter**
        * **x[2, 0]**: arithmetic mean area diameter 
        * **x[1, 1]**: size-weighted mean area diameter 
        * **x[2, 2]**: area-weighted mean area diameter 
        * **x[2, 3]**: volume-weighted mean area diameter 
        * **x[3, 0]**: arithmetic mean volume diameter
        * **x[3, 1]**: size-weighted mean volume diameter
        * **x[3, 2]**: area-weighted mean volume diameter 
        * **x[3, 3]**: volume-weighted mean volume diameter 
        
        When working with continuous distributions, the ISO series must be used
        to perform the actual calculations.

        Examples
        --------
%s
        
        References
        ----------
        .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size 
           Analysis - Part 2: Calculation of Average Particle Sizes/Diameters  
           and Moments from Particle Size Distributions.
        '''




class ParticleSizeDistributionContinuous(object):
    r'''Base class representing a continuous particle size distribution
    specified by a mathematical/statistical function. This class holds the 
    common methods only.
 
    Notes
    -----
    Although the stated units of input are in meters, this class is actually
    independent of the units provided; all results will be consistent with the
    provided unit.

    Examples
    --------
    Example problem from [1]_.
    
    >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)

    References
    ----------
    .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size 
       Analysis - Part 2: Calculation of Average Particle Sizes/Diameters and 
       Moments from Particle Size Distributions.
    '''
    def _pdf_basis_integral_definite(self, dmin, dmax, n):
        # Needed as an api for numerical integrals
        return (self._pdf_basis_integral(d=dmax, n=n) 
                - self._pdf_basis_integral(d=dmin, n=n))
    
    def pdf(self, d, n=None):
        r'''Computes the probability density funtion of a
        continuous particle size distribution at a specified particle diameter,
        an optionally in a specified basis. The evaluation function varies with
        the distribution chosen. The interconversion between distribution 
        orders is performed using the following formula [1]_:
            
        .. math::
            q_s(d) = \frac{x^{(s-r)} q_r(d) dd}
            { \int_0^\infty d^{(s-r)} q_r(d) dd}
        
        Parameters
        ----------
        d : float
            Particle size diameter, [m]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        pdf : float
            The probability density function at the specified diameter and
            order, [-]
            
        Notes
        -----
        The pdf order conversions are typically available analytically after 
        some work. They have been verified numerically. See the various
        functions with names ending with 'basis_integral' for the formulations.
        The distributions normally do not have analytical limits for diameters
        of 0 or infinity, but large values suffice to capture the area of the
        integral.
        
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> psd.pdf(1e-5)
        30522.765209509154
        >>> psd.pdf(1e-5, n=3)
        30522.765209509154
        >>> psd.pdf(1e-5, n=0)
        1238.6613794833429
        
        References
        ----------
        .. [1] Masuda, Hiroaki, Ko Higashitani, and Hideto Yoshida. Powder 
           Technology: Fundamentals of Particles, Powder Beds, and Particle 
           Generation. CRC Press, 2006.
        '''
        ans = self._pdf(d=d)
        if n is not None and n != self.order:
            power = n - self.order
            numerator = d**power*ans
            denominator = self._pdf_basis_integral_definite(dmin=0.0, dmax=self.d_excessive, n=power)
            ans = numerator/denominator
        # Handle splines which might go below zero
        return max(ans, 0.0)

    def cdf(self, d, n=None):
        r'''Computes the cumulative distribution density funtion of a
        continuous particle size distribution at a specified particle diameter,
        an optionally in a specified basis. The evaluation function varies with
        the distribution chosen.
        
        .. math::
            Q_n(d) = \int_0^d q_n(d)
        
        Parameters
        ----------
        d : float
            Particle size diameter, [m]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        cdf : float
            The cumulative distribution function at the specified diameter and
            order, [-]
            
        Notes
        -----
        Analytical integrals can be found for most distributions even when 
        order conversions are necessary.
                
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> for n in (0, 1, 2, 3):
        ...     print(psd.cdf(5e-6, n))
        0.933192798731
        0.841344746069
        0.691462461274
        0.5
        '''
        if n is not None and n != self.order:
            power = n - self.order
            # One of the pdf_basis_integral calls could be saved except for 
            # support for numerical integrals
            numerator = self._pdf_basis_integral_definite(dmin=0.0, dmax=d, n=power)
            denominator = self._pdf_basis_integral_definite(dmin=0.0, dmax=self.d_excessive, n=power)
            return max(numerator/denominator, 0.0)
        # Handle splines which might go below zero
        return max(self._cdf(d=d), 0.0)

    def delta_cdf(self, dmin, dmax, n=None):
        r'''Computes the difference in cumulative distribution function between
        two particle size diameters.
                    
        .. math::
            \Delta Q_n = Q_n(d_{max}) - Q_n(d_{min}) 
                    
        Parameters
        ----------
        dmin : float
            Lower particle size diameter, [m]
        dmax : float
            Upper particle size diameter, [m]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        delta_cdf : float
            The difference in the cumulative distribution function for the two
            diameters specified, [-]
        
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> psd.delta_cdf(1e-6, 1e-5)
        0.91652800998538764
        '''
        return self.cdf(dmax, n=n) - self.cdf(dmin, n=n)

    def dn(self, fraction, n=None):
        r'''Computes the diameter at which a specified `fraction` of the 
        distribution falls under. Utilizes a bounded solver to search for the
        desired diameter.

        Parameters
        ----------
        fraction : float
            Fraction of the distribution which should be under the calculated
            diameter, [-]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        d : float
            Particle size diameter, [m]
        
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> psd.dn(.5)
        5e-06
        >>> psd.dn(1)
        0.00029474365335233776
        >>> psd.dn(0)
        0.0
        '''
        if fraction == 1.0:
            # Avoid returning the maximum value of the search interval
            fraction = 1.0 - float_info.epsilon
        if fraction < 0:
            raise ValueError('Fraction must be more than 0')
        elif fraction == 0: # pragma : no cover
            return 0.0
            # Solve to float prevision limit - works well, but is there a real
            # point when with mpmath it woule never happen?
            # dist.cdf(dist.dn(0)-1e-35) == 0
            # dist.cdf(dist.dn(0)-1e-36) == input
            # dn(0) == 1.9663615597466143e-20
            def err(d): 
                cdf = self.cdf(d, n=n)
                if cdf == 0:
                    cdf = -1
                return cdf
            return brenth(err, 0.0, self.d_excessive, maxiter=1000, xtol=1E-200)

        elif fraction > 1:
            raise ValueError('Fraction less than 1')
        # As the dn may be incredibly small, it is required for the absolute 
        # tolerance to not be happy - it needs to continue iterating as long
        # as necessary to pin down the answer
        return brenth(lambda d:self.cdf(d, n=n) -fraction, 
                      0.0, self.d_excessive, maxiter=1000, xtol=1E-200)
    
    def ds_discrete(self, dmin=None, dmax=None, pts=20, limit=1e-9, 
                    method='logarithmic'):
        r'''Create a particle spacing mesh to perform calculations with, 
        according to one of several ways. The allowable meshes are
        'linear', 'logarithmic', a geometric series specified by a Renard 
        number such as 'R10', or the meshes available in one of several sieve 
        standards.
        
        Parameters
        ----------
        dmin : float, optional
            The minimum diameter at which the mesh starts, [m]
        dmax : float, optional
            The maximum diameter at which the mesh ends, [m]
        pts : int, optional
            The number of points to return for the mesh (note this is not 
            respected by sieve meshes), [-]
        limit : float
            If `dmin` or `dmax` is not specified, it will be calculated as the
            `dn` at which this limit or 1-limit exists (this is ignored for 
            Renard numbers), [-]
        method : str, optional
            Either 'linear', 'logarithmic', a Renard number like 'R10' or 'R5' 
            or'R2.5', or one of the sieve standards 'ISO 3310-1 R40/3', 
            'ISO 3310-1 R20', 'ISO 3310-1 R20/3', 'ISO 3310-1', 
            'ISO 3310-1 R10', 'ASTM E11', [-]
    
        Returns
        -------
        ds : list[float]
            The generated mesh diameters, [m]
    
        Notes
        -----
        Note that when specifying a Renard series, only one of `dmin` or `dmax` can
        be respected! Provide only one of those numbers. 
        
        Note that when specifying a sieve standard the number of points is not
        respected!
    
        References
        ----------
        .. [1] ASTM E11 - 17 - Standard Specification for Woven Wire Test Sieve 
           Cloth and Test Sieves.
        .. [2] ISO 3310-1:2016 - Test Sieves -- Technical Requirements and Testing
           -- Part 1: Test Sieves of Metal Wire Cloth.
        '''
        if method[0] not in ('R', 'r'):
            if dmin is None:
                dmin = self.dn(limit)
            if dmax is None:
                dmax = self.dn(1.0 - limit)
        return psd_spacing(dmin=dmin, dmax=dmax, pts=pts, method=method)
    
    def fractions_discrete(self, ds, n=None):
        r'''Computes the fractions of the cumulative distribution functions 
        which lie between the specified specified particle diameters. The first
        diameter contains the cdf from 0 to it. 
        
        Parameters
        ----------
        ds : list[float]
            Particle size diameters, [m]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        fractions : float
            The differences in the cumulative distribution functions at the 
            specified diameters and order, [-]
        
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> psd.fractions_discrete([1e-6, 1e-5, 1e-4, 1e-3])
        [0.00064347101291384323, 0.9165280099853876, 0.08282851796190027, 1.039798247504109e-09]
        '''
        cdfs = [self.cdf(d, n=n) for d in ds]
        return [cdfs[0]] + np.diff(cdfs).tolist()
    
    def cdf_discrete(self, ds, n=None):
        r'''Computes the cumulative distribution functions for a list of 
        specified particle diameters.
                                        
        Parameters
        ----------
        ds : list[float]
            Particle size diameters, [m]
        n : int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer, [-]

        Returns
        -------
        cdfs : float
            The cumulative distribution functions at the specified diameters 
            and order, [-]
        
        Examples
        --------
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6, order=3)
        >>> psd.cdf_discrete([1e-6, 1e-5, 1e-4, 1e-3])
        [0.00064347101291384323, 0.91717148099830148, 0.99999999896020175, 1.0]
        '''
        return [self.cdf(d, n=n) for d in ds]
    
    def mean_size(self, p, q):
        '''        
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)
        >>> psd.mean_size(3, 2)
        4.4124845129229773e-06
        '''
        if p == q:
            raise Exception(NotImplemented)
        pow1 = q - self.order 
        
        denominator = self._pdf_basis_integral_definite(dmin=0.0, dmax=self.d_excessive, n=pow1)
        root_power = p - q
        pow3 = p - self.order
        numerator = self._pdf_basis_integral_definite(dmin=0.0, dmax=self.d_excessive, n=pow3)
        return (numerator/denominator)**(1.0/(root_power))
    
    def mean_size_ISO(self, k, r):
        '''        
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)
        >>> psd.mean_size_ISO(1, 2)
        4.4124845129229773e-06
        '''
        p = k + r
        q = r
        return self.mean_size(p=p, q=q)    
    
    def pdf_plot(self, n=(0, 1, 2, 3), dmin=None, dmax=None, pts=500): # pragma : no cover
        r'''Plot the probability density function of the particle size 
        distribution. The plotted range can be specified using `dmin` and 
        `dmax`, or estimated automatically. One or more order can be plotted,
        by providing an iterable of ints as the value of `n` or just one int.
                
        Parameters
        ----------
        n : tuple(int) or int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer; as many as desired may be specified, [-]
        dmin : float, optional
            Lower particle size diameter, [m]
        dmax : float, optional
            Upper particle size diameter, [m]
        pts : int
            The number of points for values to be calculated, [-]
        '''
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
            
        ds = self.ds_discrete(dmin=dmin, dmax=dmax, pts=pts)
        try:
            for ni in n:
                fractions = self.fractions_discrete(ds=ds, n=ni)
                plt.semilogx(ds, fractions, label=_label_distribution_n(ni))
        except:
            fractions = self.fractions_discrete(ds=ds, n=n)
            plt.semilogx(ds, fractions, label=_label_distribution_n(n))
        plt.ylabel('Probability density function, [-]')
        plt.xlabel('Particle diameter, [m]')
        plt.title('Probability density function of %s distribution with '
                  'parameters %s' %(self.name, self.parameters))
        plt.legend()
        plt.show()
    
    def cdf_plot(self, n=(0, 1, 2, 3), dmin=None, dmax=None, pts=500):  # pragma : no cover
        r'''Plot the cumulative distribution function of the particle size 
        distribution. The plotted range can be specified using `dmin` and 
        `dmax`, or estimated automatically. One or more order can be plotted,
        by providing an iterable of ints as the value of `n` or just one int.
                
        Parameters
        ----------
        n : tuple(int) or int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer; as many as desired may be specified, [-]
        dmin : float, optional
            Lower particle size diameter, [m]
        dmax : float, optional
            Upper particle size diameter, [m]
        pts : int
            The number of points for values to be calculated, [-]
        '''
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')

        ds = self.ds_discrete(dmin=dmin, dmax=dmax, pts=pts)
        try:
            for ni in n:
                cdfs = self.cdf_discrete(ds=ds, n=ni)
                plt.semilogx(ds, cdfs, label=_label_distribution_n(ni))
        except:
            cdfs = self.cdf_discrete(ds=ds, n=n)
            plt.semilogx(ds, cdfs, label=_label_distribution_n(n))
        if self.points:
            plt.plot(self.ds, self.cdf_fractions, '+', label='Volume/Mass points')
            
            if hasattr(self, 'area_fractions'):
                plt.plot(self.ds, np.cumsum(self.area_fractions), '+', label='Area points')
            if hasattr(self, 'length_fractions'):
                plt.plot(self.ds, np.cumsum(self.length_fractions), '+', label='Length points')
            if hasattr(self, 'number_fractions'):
                plt.plot(self.ds, np.cumsum(self.number_fractions), '+', label='Number points')
                
        plt.ylabel('Cumulative density function, [-]')
        plt.xlabel('Particle diameter, [m]')
        plt.title('Cumulative density function of %s distribution with '
                  'parameters %s' %(self.name, self.parameters))
        plt.legend()
        plt.show()


class ParticleSizeDistribution(ParticleSizeDistributionContinuous):
    r'''Class representing a discrete particle size distribution specified by a
    series of diameter bins, and the quantity of particles in each bin. The
    quantities may be specified as numbers, volume/mass/mole fractions, or
    numbers fractions.
    All parameters are also attributes.
    
    In addition to the diameter bins, one of `fractions`, `number_fractions`,
    or `numbers` must be specified.
            
    Parameters
    ----------
    ds : list[float]
        Diameter bins; length of the specified quantities, optionally +1 that
        length to specify a cutoff diameter for the smallest diameter bin, [m]
    fractions : list[float], optional
        The mass/mole/volume fractions of each particles in each diameter bin
        (this class represents particles of the same density and molecular
        weight, so each of the fractions will be the same), [-]
    number_fractions : list[float], optional
        The number fractions by actual number of particles in each bin, [-]
    numbers : lists[float], optional
        The actual counted number of particles in each bin, [-]
 
    Attributes
    ----------
    length_fractions : list[float]
        The length fractions of particles in each bin, [-]
    area_fractions : list[float]
        The area fractions of particles in each bin, [-]
    size_classes : bool
        Whether or not the diameter bins were set as size classes (as length
        of fractions + 1), [-]
    N : int
        The number of provided points, [-]

    Notes
    -----
    Although the stated units of input are in meters, this class is actually
    independent of the units provided; all results will be consistent with the
    provided unit.

    Examples
    --------
    Example problem from [1]_, calculating several diameters and the cumulative
    distribution.
    
    >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
    >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    >>> psd = ParticleSizeDistribution(ds=ds, numbers=numbers)
    >>> psd
    <Particle Size Distribution, points=14, D[3, 3]=0.002451 m>

    References
    ----------
    .. [1] ASTM E799 - 03(2015) - Standard Practice for Determining Data 
       Criteria and Processing for Liquid Drop Size Analysis.    
    .. [2] ISO 9276-2:2014 - Representation of Results of Particle Size 
       Analysis - Part 2: Calculation of Average Particle Sizes/Diameters and 
       Moments from Particle Size Distributions.
    '''
    def __repr__(self):
        txt = '<Particle Size Distribution, points=%d, D[3, 3]=%f m>'
        return txt %(self.N, self.mean_size(p=3, q=3))
    
    size_classes = False
    _interpolated = None
    points = True
    name = 'Discrete'
    def __init__(self, ds, fractions=None, number_fractions=None, numbers=None, 
                 length_fractions=None, area_fractions=None, 
                 monotonic=True, cdf=False, order=None):
        '''If given numbers or flows, convert to fractions immediately and move 
        forward with them.
        TODO: specify only `fraction`, and also `order`. Then `counts` becomes `flows`.
        TODO: allow cdf to be specified as well. Use cdf flag to convert 
        TODO: normalize i nputs to allow for flows being specified easier.
        # fractions to other basis if given.
        TODO: support any basis of input i.e. length, area.
        '''
        self.monotonic = monotonic
        self.ds = ds
        self.order = order
        
        # The use of locals means this cannot be a list comprehension
        specified_quantities = []
        for i in ('fractions', 'number_fractions', 'numbers', 'area_fractions', 'length_fractions'):
            if locals()[i] is not None:
                specified_quantities.append(i)
        
        if len(specified_quantities) > 1:
            raise Exception('More than one distribution specified')
        elif len(specified_quantities) == 0:
            raise Exception('No distribution specified')
        else:
            spec = specified_quantities[0]
        
        if ds is not None and (len(ds) == len(locals()[spec]) + 1):
            self.size_classes = True
        else:
            self.size_classes = False
            
        if cdf:
            s = locals()[spec]
            if len(s)+1 == len(ds):
#                print('This is a test', [s[0]] + np.diff(s).tolist())
                locals()[spec] = [s[0]] + np.diff(s).tolist()
            else:
                
                s = np.diff(s).tolist()
                s.insert(0, 0.0)
                locals()[spec] = s
            
        a = locals()[spec]
#        print('begin spec', a, 'hi', cdf, spec is fractions)
        spec = a
        self.N = len(spec)
        
        if spec is area_fractions:
            d3s = [self.di_power(i, power=1)*area_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            spec = fractions = [i/tot_d3 for i in d3s]
        elif spec is length_fractions:
            d3s = [self.di_power(i, power=2)*length_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            spec = fractions = [i/tot_d3 for i in d3s]

        if spec is numbers:
            self.numbers = numbers
            self.number_sum = sum(self.numbers)
            self.number_fractions = [i/self.number_sum for i in self.numbers]
        
            d3s = [self.di_power(i, power=3)*self.number_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            self.fractions = [i/tot_d3 for i in d3s]
        
        elif spec is number_fractions:
            self.number_fractions = number_fractions
            d3s = [self.di_power(i, power=3)*self.number_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            self.fractions = [i/tot_d3 for i in d3s]
        elif spec is fractions:
            self.fractions = fractions
            basis = 100 # m^3
            Vis = [basis*fi for fi in fractions]
            D3s = [self.di_power(i, power=3) for i in range(self.N)]
            Vps = [pi/6*Di for Di in D3s]
            numbers = [Vi/Vp for Vi, Vp in zip(Vis, Vps)]
            number_sum = sum(numbers)
            self.number_fractions = [i/number_sum for i in numbers]
        # Set the length fractions
        D3s = [self.di_power(i, power=2) for i in range(self.N)]
        numbers = [Vi/Vp for Vi, Vp in zip(self.fractions, D3s)]
        number_sum = sum(numbers)
        self.length_fractions = [i/number_sum for i in numbers]
        
        # Set the surface area fractions
        D3s = [self.di_power(i, power=1) for i in range(self.N)]
        numbers = [Vi/Vp for Vi, Vp in zip(self.fractions, D3s)]
        number_sum = sum(numbers)
        self.area_fractions = [i/number_sum for i in numbers]
        # Length and surface area fractions verified numerically
        
        # Things for interoperability with the Continuous distribution
        self.d_excessive = self.ds[-1]
        self.parameters = {}
        self.order = 3
        self.cdf_fractions = self.volume_cdf = np.cumsum(self.fractions)
        self.area_cdf = np.cumsum(self.area_fractions)
        self.length_cdf = np.cumsum(self.length_fractions)
        self.number_cdf = np.cumsum(self.number_fractions)

    @property
    def interpolated(self):
        if not self._interpolated:
            self._interpolated = PSDInterpolated(ds=self.ds, 
                                                 fractions=self.fractions, 
                                                 order=3, 
                                                 monotonic=self.monotonic)            
        return self._interpolated
        
    def _pdf(self, d):
        return self.interpolated._pdf(d)
    
    def _cdf(self, d):
        return self.interpolated._cdf(d)
    
    def _pdf_basis_integral(self, d, n):
        return self.interpolated._pdf_basis_integral(d, n)
        
    def _fit_obj_function(self, vals, distribution, n):
        err = 0.0
        dist = distribution(*list(vals))
        l = len(self.fractions) if self.size_classes else len(self.fractions) - 1
        for i in range(l):
            delta_cdf = dist.delta_cdf(dmin=self.ds[i], dmax=self.ds[i+1])
            err += abs(delta_cdf - self.fractions[i])
        return err
        
    def fit(self, x0=None, distribution='lognormal', n=None, **kwargs):
        '''Incomplete method to fit experimental values to a curve. It is very
        hard to get good initial guesses, which are really required for this.
        Differential evolution is promissing. This API is likely to change in
        the future.
        '''
        dist = {'lognormal': PSDLognormal, 
                'GGS': PSDGatesGaudinSchuhman, 
                'RR': PSDRosinRammler}[distribution]
        
        if distribution == 'lognormal':
            if x0 is None:
                d_characteristic = sum([fi*di for fi, di in zip(self.fractions, self.Dis)])
                s = 0.4
                x0 = [d_characteristic, s]
        elif distribution == 'GGS':
            if x0 is None:
                d_characteristic = sum([fi*di for fi, di in zip(self.fractions, self.Dis)])
                m = 1.5
                x0 = [d_characteristic, m]
        elif distribution == 'RR':
            if x0 is None:
                x0 = [5E-6, 1e-2]
        return minimize(self._fit_obj_function, x0, args=(dist, n), **kwargs)

    @property
    def Dis(self):
        '''Representative diameters of each bin.
        '''
        return [self.di_power(i, power=1) for i in range(self.N)]
    
    def di_power(self, i, power=1):
        r'''Method to calculate a power of a particle class/bin in a generic
        way so as to support when there are as many `ds` as `fractions`,
        or one more diameter spec than `fractions`.
        
        When each bin has a lower and upper bound, the formula is as follows
        [1]_.
        
        .. math::
            D_i^r = \frac{D_{i, ub}^{(r+1)} - D_{i, lb}^{(r+1)}}
            {(D_{i, ub} - D_{i, lb})(r+1)}
            
        Where `ub` represents the upper bound, and `lb` represents the lower
        bound. Otherwise, the standard definition is used:
            
        .. math::
            D_i^r = D_i^r
        
        Parameters
        ----------
        i : int
            The index of the diameter for the calculation, [-]
        power : int
            The exponent, [-]

        Returns
        -------
        di_power : float
            The representative bin diameter raised to  `power`, [m^power] 

        References
        ----------
        .. [1] ASTM E799 - 03(2015) - Standard Practice for Determining Data 
           Criteria and Processing for Liquid Drop Size Analysis.    
        '''
        if self.size_classes:
            rt = power + 1
            return ((self.ds[i+1]**rt - self.ds[i]**rt)/((self.ds[i+1] - self.ds[i])*rt))
        else:
            return self.ds[i]**power
        
    def mean_size(self, p, q):
        '''        
        >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, numbers=numbers)
        >>> psd.mean_size(3, 2)
        0.0022693210317450449
        '''
        if p != q:
            # Note: D(p, q) = D(q, p); in ISO and proven experimentally
            numerator = sum(self.di_power(i=i, power=p)*self.number_fractions[i] for i in range(self.N))
            denominator = sum(self.di_power(i=i, power=q)*self.number_fractions[i] for i in range(self.N))
            return (numerator/denominator)**(1.0/(p-q))
        else:
            numerator = sum(log(self.di_power(i=i, power=1))*self.di_power(i=i, power=p)*self.number_fractions[i] for i in range(self.N))
            denominator = sum(self.di_power(i=i, power=q)*self.number_fractions[i] for i in range(self.N))
            return exp(numerator/denominator)
        
    def mean_size_ISO(self, k, r):
        r'''
        >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, numbers=numbers)
        >>> psd.mean_size_ISO(1, 2)
        0.0022693210317450449
        '''
        p = k + r
        q = r
        return self.mean_size(p=p, q=q)

#from numpy.testing import *
#ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
#numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
##dist = ParticleSizeDistribution(ds=ds, numbers=numbers)
#
## this is calculated from (Ds, numbers)
#number_fractions = [0.010640039286298903, 0.01947945653953184, 0.03797675560648224, 0.06711409395973154, 0.102962841708954, 0.13897528237027337, 0.16205598297593715, 0.160582746767065, 0.13504665247994763, 0.09477819610410869, 0.048616794892781146, 0.01816991324275659, 0.0034375511540350305, 0.0001636929120969062]
#length_fractions = [0.0022265080273913248, 0.005405749400984079, 0.013173675010801534, 0.02909808308708846, 0.05576732372469186, 0.09403390879219536, 0.1370246122004729, 0.16966553692650058, 0.17831420382670332, 0.15641421494054603, 0.10028800800464328, 0.046849963047687335, 0.011078803825079166, 0.0006594091852147985]
#area_fractions = [0.0003643458522227456, 0.0011833425086503686, 0.0036047198267710797, 0.009951607879295004, 0.023826910138492176, 0.05018962198499494, 0.09139246506396961, 0.1414069073893575, 0.18572285033413602, 0.20362023102799823, 0.16318760564859225, 0.09528884410476045, 0.028165197280747324, 0.0020953509600122053]
#fractions = [4.8560356399310335e-05, 0.00021291794698947167, 0.0008107432330218852, 0.0027975134942445257, 0.00836789808490677, 0.02201901107895143, 0.05010399231412809, 0.0968727835386488, 0.15899879607747244, 0.2178784903712532, 0.21825921197532888, 0.159302671180342, 0.05885464261922434, 0.0054727677290887945]
#
#cdf_fractions = [4.856035639931034e-05, 0.0002614783033887821, 0.0010722215364106676, 0.003869735030655194, 0.012237633115561966, 0.0342566441945134, 0.0843606365086415, 0.18123342004729032, 0.34023221612476284, 0.5581107064960161, 0.7763699184713451, 0.9356725896516871, 0.9945272322709114, 1.0000000000000002]
#area_cdf = [0.00036434585222274563, 0.0015476883608731143, 0.005152408187644195, 0.015104016066939202, 0.038930926205431385, 0.08912054819042634, 0.18051301325439598, 0.3219199206437535, 0.5076427709778896, 0.7112630020058879, 0.8744506076544801, 0.9697394517592406, 0.9979046490399879, 1.0]
#length_cdf = [0.0022265080273913248, 0.007632257428375404, 0.020805932439176937, 0.0499040155262654, 0.10567133925095726, 0.1997052480431526, 0.3367298602436255, 0.506395397170126, 0.6847096009968294, 0.8411238159373755, 0.9414118239420188, 0.9882617869897061, 0.9993405908147853, 1.0000000000000002]
#
#opts = [
##        {'numbers': numbers, 'cdf': False, 'order': 0},
##        {'number_fractions': number_fractions, 'cdf': False, 'order': 0},  
##        {'fractions': fractions, 'cdf': False, 'order': 3},
##        {'length_fractions': length_fractions, 'cdf': False, 'order': 1},
##        {'area_fractions': area_fractions, 'cdf': False, 'order': 2},
#        
#        {'fractions': cdf_fractions, 'cdf': True, 'order': 3}]
##        {'area_cdf': area_cdf, 'cdf': True, 'order': 2},
##        {'length_cdf': length_cdf, 'cdf': True, 'order': 1}]
#
#for opt in opts:
#    asme_e799 = ParticleSizeDistribution(ds=ds, **opt)
#    
#    d10 = asme_e799.mean_size(1, 0)
#    assert_allclose(d10, 1459.3725650679328)
#    
#    d21 = asme_e799.mean_size(2, 1)
#    assert_allclose(d21, 1857.7888572055529)
#    d20 = asme_e799.mean_size(2, 0)
#    assert_allclose(d20, 1646.5740462835831)
#    
#    d32 = asme_e799.mean_size(3, 2)
#    assert_allclose(d32, 2269.3210317450453)
#    # This one is rounded to 2280 in ASME - weird
#    
#    d31 = asme_e799.mean_size(3, 1)
#    assert_allclose(d31, 2053.2703977309357)
#    # This one is rounded to 2060 in ASME - weird
#    
#    d30 = asme_e799.mean_size(3, 0)
#    assert_allclose(d30, 1832.39665294744)
#    
#    d43 = asme_e799.mean_size(4, 3)
#    assert_allclose(d43, 2670.751954612969)
#    # The others are, rounded to the nearest 10, correct.
#    # There's something weird about the end points of some intermediate values of
#    #  D3 and D4. Likely just rounding issues.
#    
#    vol_percents_exp = [0.005, 0.021, 0.081, 0.280, 0.837, 2.202, 5.010, 9.687, 15.900, 21.788, 21.826, 15.930, 5.885, 0.547]
#    assert vol_percents_exp == [round(i*100, 3) for i in asme_e799.fractions]
#    
#    assert_allclose(asme_e799.fractions, fractions)
#    assert_allclose(asme_e799.number_fractions, number_fractions)
#    
#    # i, i distributions
#    d00 = asme_e799.mean_size(0, 0)
#    assert_allclose(d00, 1278.7057976023061)
#    
#    d11 = asme_e799.mean_size(1, 1)
#    assert_allclose(d11, 1654.6665309027303)
#    
#    d22 = asme_e799.mean_size(2, 2)
#    assert_allclose(d22, 2054.3809583432208)
#    
#    d33 = asme_e799.mean_size(3, 3)
#    assert_allclose(d33, 2450.886241250387)
#    
#    d44 = asme_e799.mean_size(4, 4)
#    assert_allclose(d44, 2826.0471682278476)

try:
    # Python 2
    ParticleSizeDistributionContinuous.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__func__.__doc__)
    ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__)
    ParticleSizeDistribution.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__func__.__doc__)
    ParticleSizeDistribution.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__func__.__doc__)
except AttributeError:
    # Python 3
    ParticleSizeDistributionContinuous.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__doc__)
    ParticleSizeDistributionContinuous.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__doc__)
    ParticleSizeDistribution.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__doc__)
    ParticleSizeDistribution.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__doc__)

class PSDLognormal(ParticleSizeDistributionContinuous):
    name = 'Lognormal'
    points = False
    truncated = False
    def __init__(self, d_characteristic, s, order=3, d_min=None, d_max=None):
        self.s = s
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'s': s, 'd_characteristic': d_characteristic}
        self.d_min = d_min
        self.d_max = d_max
        # Pick an upper bound for the search algorithm of 15 orders of magnitude larger than
        # the characteristic diameter; should never be a problem, as diameters can only range
        # so much, physically.
        if self.d_max is not None:
            self.d_excessive = self.d_max
        else:
            self.d_excessive = 1E15*self.d_characteristic
        
        # Just do full truncation for now and to begin
        
    def _pdf(self, d):
        return pdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)

    def _cdf(self, d):
        return cdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)
    
    def _pdf_basis_integral(self, d, n):
        return pdf_lognormal_basis_integral(d, d_characteristic=self.d_characteristic, s=self.s, n=n)
    

class PSDGatesGaudinSchuhman(ParticleSizeDistributionContinuous):
    name = 'Gates Gaudin Schuhman'
    points = False
    def __init__(self, d_characteristic, m, order=3):
        self.m = m
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'m': m, 'd_characteristic': d_characteristic}
        # PDF above this is zero
        self.d_excessive = self.d_characteristic

    def _pdf(self, d):
        return pdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _cdf(self, d):
        return cdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _pdf_basis_integral(self, d, n):
        return pdf_Gates_Gaudin_Schuhman_basis_integral(d, d_characteristic=self.d_characteristic, m=self.m, n=n)


class PSDRosinRammler(ParticleSizeDistributionContinuous):
    name = 'Rosin Rammler'
    points = False
    def __init__(self, k, m, order=3):
        self.m = m
        self.k = k
        self.order = order
        self.parameters = {'m': m, 'k': k}
        
        # PDF above this is zero - todo?
        self.d_excessive = 1e3 

    def _pdf(self, d):
        return pdf_Rosin_Rammler(d, k=self.k, m=self.m)

    def _cdf(self, d):
        return cdf_Rosin_Rammler(d, k=self.k, m=self.m)
    
    def _pdf_basis_integral(self, d, n):
        return pdf_Rosin_Rammler_basis_integral(d, k=self.k, m=self.m, n=n)


'''# These are all brutally slow!
from scipy.stats import *
from fluids import *
distribution = lognorm(s=0.5, scale=5E-6)
psd = PSDCustom(distribution)

# psd.dn(0.5, n=2.0) # Doesn't work at all, but the main things do including plots
'''

class PSDCustom(ParticleSizeDistributionContinuous):
    name = ''
    points = False
    
    def __init__(self, distribution, order=3.0, d_excessive=1.0, name=None):
        if name:
            self.name = name
        else:
            try:
                self.name = distribution.dist.__class__.__name__
            except:
                pass
        try:
            self.parameters = distribution.kwds
        except:
            self.parameters = {}
            
        self.d_excessive = d_excessive
        self.distribution = distribution
        self.order = order
        
    def _pdf(self, d):
        return self.distribution.pdf(d)

    def _cdf(self, d):
        return self.distribution.cdf(d)
    
    def _pdf_basis_integral_definite(self, dmin, dmax, n):
        # Needed as an api for numerical integrals
        n = float(n)
        if dmin == 0:
            dmin = dmax*1E-12
        to_int = lambda d : d**n*self._pdf(d)
        points = np.logspace(np.log10(max(dmax/1000, dmin)), np.log10(dmax*.999), 40)
        return quad(to_int, dmin, dmax, points=points)[0] # 
            
    
class PSDInterpolated(ParticleSizeDistributionContinuous):
    name = 'Interpolated'
    points = True
    def __init__(self, ds, fractions, order=3, monotonic=True):
        self.order = order
        self.monotonic = monotonic
        self.parameters = {}
        
        ds = list(ds)
        fractions = list(fractions)
        
        if ds[0] != 0:
            ds = [0] + ds
            if len(ds) != len(fractions):
                fractions = [0] + fractions
            
        self.ds = ds
        self.fractions = fractions

        self.d_excessive = max(ds)
            
        self.cdf_fractions = np.cumsum(fractions)
        if self.monotonic:
            self.cdf_spline = PchipInterpolator(ds, self.cdf_fractions, extrapolate=True)
            self.pdf_spline = PchipInterpolator(ds, self.cdf_fractions, extrapolate=True).derivative(1)
        else:
            self.cdf_spline = InterpolatedUnivariateSpline(ds, self.cdf_fractions, ext=3)
            self.pdf_spline = InterpolatedUnivariateSpline(ds, self.cdf_fractions, ext=3).derivative(1)

        # The pdf basis integral splines will be stored here
        self.basis_integrals = {}
        
                
    def _pdf(self, d):
        return max(0.0, float(self.pdf_spline(d)))
    
    def _cdf(self, d):
        return max(0.0, float(self.cdf_spline(d)))
    
    def _pdf_basis_integral(self, d, n):
        # there are slight errors with this approach - but they are OK to 
        # ignore. 
        # DO NOT evaluate the first point as it leads to inf values; just set
        # it to zero
        if n not in self.basis_integrals:
            ds = np.array(self.ds[1:])
            pdf_vals = self.pdf_spline(ds)
            basis_integral = ds**n*pdf_vals
            if self.monotonic:
                self.basis_integrals[n] = PchipInterpolator(ds, basis_integral, extrapolate=True).antiderivative(1)
            else:
                self.basis_integrals[n] = UnivariateSpline(ds, basis_integral, ext=3, s=0).antiderivative(n=1)
        return max(float(self.basis_integrals[n](d)), 0.0)
    
