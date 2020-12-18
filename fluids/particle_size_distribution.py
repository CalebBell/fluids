# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains particle distribution characterization, fitting,
interpolating, and manipulation functions. It may be used with discrete
particle size distributions, or with statistical ones with parameters
specified.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Particle Size Distribution Base Class
-------------------------------------
.. autoclass:: ParticleSizeDistributionContinuous
    :members:

Discrete Particle Size Distributions
------------------------------------
.. autoclass:: ParticleSizeDistribution
    :members:
.. autoclass:: PSDInterpolated
    :members:

Statistical Particle Size Distributions
---------------------------------------
.. autoclass:: PSDLognormal
    :members:
.. autoclass:: PSDGatesGaudinSchuhman
    :members:
.. autoclass:: PSDRosinRammler
    :members:
.. autoclass:: PSDCustom
    :members:

Helper functions: Lognormal Distribution
----------------------------------------
.. autofunction:: pdf_lognormal
.. autofunction:: cdf_lognormal
.. autofunction:: pdf_lognormal_basis_integral

Helper functions: Gates Gaudin Schuhman Distribution
----------------------------------------------------
.. autofunction:: pdf_Gates_Gaudin_Schuhman
.. autofunction:: cdf_Gates_Gaudin_Schuhman
.. autofunction:: pdf_Gates_Gaudin_Schuhman_basis_integral

Helper functions: Rosin Rammler Distribution
--------------------------------------------
.. autofunction:: pdf_Rosin_Rammler
.. autofunction:: cdf_Rosin_Rammler
.. autofunction:: pdf_Rosin_Rammler_basis_integral

Sieves
------
.. autoclass:: Sieve
.. autodata:: ASTM_E11_sieves
.. autodata:: ISO_3310_1_sieves
.. autodata:: ISO_3310_1_R20
.. autodata:: ISO_3310_1_R20_3
.. autodata:: ISO_3310_1_R40_3
.. autodata:: ISO_3310_1_R10

Point Spacing
-------------
.. autofunction:: psd_spacing
"""
from __future__ import division

__all__ = ['ParticleSizeDistribution', 'ParticleSizeDistributionContinuous',
           'PSDLognormal', 'PSDGatesGaudinSchuhman', 'PSDRosinRammler',
           'PSDInterpolated', 'PSDCustom',
           'psd_spacing',

           'pdf_lognormal', 'cdf_lognormal', 'pdf_lognormal_basis_integral',

           'pdf_Gates_Gaudin_Schuhman', 'cdf_Gates_Gaudin_Schuhman',
           'pdf_Gates_Gaudin_Schuhman_basis_integral',

           'pdf_Rosin_Rammler', 'cdf_Rosin_Rammler',
           'pdf_Rosin_Rammler_basis_integral',
           'Sieve',

#           'ASTM_E11_sieves', 'ISO_3310_1_sieves',
#           'ISO_3310_1_R20_3', 'ISO_3310_1_R20', 'ISO_3310_1_R10',
#           'ISO_3310_1_R40_3'
           ]

from math import log, exp, pi, log10, sqrt
from fluids.numerics import (brenth, epsilon, gamma, erf, gammaincc,
                             linspace, logspace, cumsum, diff, normalize, quad)

ROOT_TWO_PI = sqrt(2.0*pi)

NO_MATPLOTLIB_MSG = 'Optional dependency matplotlib is required for plotting'

def __getattr__(name):
    global ASTM_E11_sieves, ISO_3310_1_sieves, ISO_3310_1_R20_3, ISO_3310_1_R20, ISO_3310_1_R10, ISO_3310_1_R40_3, sieve_spacing_options
    if name in ('ASTM_E11_sieves', 'ISO_3310_1_sieves',
           'ISO_3310_1_R20_3', 'ISO_3310_1_R20', 'ISO_3310_1_R10',
           'ISO_3310_1_R40_3'):
        from fluids.particle_size_distribution_data import ASTM_E11_sieves, ISO_3310_1_sieves, ISO_3310_1_R20_3, ISO_3310_1_R20, ISO_3310_1_R10, ISO_3310_1_R40_3, sieve_spacing_options
        return globals()[name]
    raise AttributeError("module %s has no attribute %s" %(__name__, name))
sieve_spacing_options = None

class Sieve(object):
    r'''Class for storing data on sieves. If a property is not available, it is
    set to None.

    Attributes
    ----------
    designation : str
        The standard name of the sieve - its opening's length in units of
        millimeters
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
    __slots__ = ('designation', 'old_designation', 'opening', 'opening_inch',
                 'Y_variation_avg', 'X_variation_max', 'max_opening',
                 'calibration_samples', 'compliance_sd', 'inspection_samples',
                 'inspection_sd', 'calibration_samples', 'calibration_sd',
                 'd_wire', 'd_wire_min', 'd_wire_max', 'compliance_samples')

#    def __repr__(self):
#        s = 'Sieve(%s)'
#        s2 = ''
#        for attr, value in self.__dict__.items():
#            if value is not None:
#                if type(value) == float:
#                    value = round(value, 8)
#                elif type(value) == str:
#                    value = "'" + value + "'"
#                s2 += '%s=%s, '%(attr, value)
#        s2 = s2[0:-2]
#        return s %(s2)

    def __repr__(self):
        return '<Sieve, designation %s mm, opening %g m>' %(self.designation, self.opening)

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




def psd_spacing(d_min=None, d_max=None, pts=20, method='logarithmic'):
    r'''Create a particle spacing mesh in one of several ways for use in
    modeling discrete particle size distributions. The allowable meshes are
    'linear', 'logarithmic', a geometric series specified by a Renard number
    such as 'R10', or the meshes available in one of several sieve standards.

    Parameters
    ----------
    d_min : float, optional
        The minimum diameter at which the mesh starts, [m]
    d_max : float, optional
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
    Note that when specifying a Renard series, only one of `d_min` or `d_max` can
    be respected! Provide only one of those numbers.

    Note that when specifying a sieve standard the number of points is not
    respected!

    Examples
    --------
    >>> psd_spacing(d_min=5e-5, d_max=5e-4, method='ISO 3310-1 R20/3')
    [6.3e-05, 9e-05, 0.000125, 0.00018, 0.00025, 0.000355, 0.0005]

    References
    ----------
    .. [1] ASTM E11 - 17 - Standard Specification for Woven Wire Test Sieve
       Cloth and Test Sieves.
    .. [2] ISO 3310-1:2016 - Test Sieves -- Technical Requirements and Testing
       -- Part 1: Test Sieves of Metal Wire Cloth.
    '''
    global sieve_spacing_options
    if d_min is not None:
        d_min = float(d_min)
    if d_max is not None:
        d_max = float(d_max)
    if method == 'logarithmic':
        return logspace(log10(d_min), log10(d_max), pts)
    elif method == 'linear':
        return linspace(d_min, d_max, pts)
    elif method[0] in ('R', 'r'):
        ratio = 10**(1.0/float(method[1:]))
        if d_min is not None and d_max is not None:
            raise ValueError('For geometric (Renard) series, only '
                            'one of `d_min` and `d_max` should be provided')
        if d_min is not None:
            ds = [d_min]
            for i in range(pts-1):
                ds.append(ds[-1]*ratio)
            return ds
        elif d_max is not None:
            ds = [d_max]
            for i in range(pts-1):
                ds.append(ds[-1]/ratio)
            return list(reversed(ds))
    if sieve_spacing_options is None:
        from fluids.particle_size_distribution_data import sieve_spacing_options
    if method in sieve_spacing_options:
        l = sieve_spacing_options[method]
        ds = []
        for sieve in l:
           if  d_min <= sieve.opening <= d_max:
               ds.append(sieve.opening)
        return list(reversed(ds))
    else:
        raise ValueError('Method not recognized')


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

    >>> import scipy.stats
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
        Lognormal cumulative density function, [-]

    Notes
    -----
    The characteristic diameter can be in terns of number density (denoted
    :math:`q_0(d)`), length density (:math:`q_1(d)`), surface area density
    (:math:`q_2(d)`), or volume density (:math:`q_3(d)`). Volume density is
    most often used. Interconversions among the distributions is possible but
    tricky.

    The standard distribution (i.e. the one used in Scipy) can perform the same
    computation with  `d_characteristic` as the value of `scale`.

    >>> import scipy.stats
    >>> scipy.stats.lognorm.cdf(x=1E-4, s=1.1, scale=1E-5)
    0.9818369875798177

    Scipy's calculation is over 100 times slower however.

    Examples
    --------
    >>> cdf_lognormal(d=1E-4, d_characteristic=1E-5, s=1.1)
    0.9818369875798

    References
    ----------
    .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size
       Analysis - Part 2: Calculation of Average Particle Sizes/Diameters and
       Moments from Particle Size Distributions.
    '''
    try:
        return 0.5*(1.0 + erf((log(d/d_characteristic))/(s*sqrt(2.0))))
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
    56228306549.26362
    '''
    try:
        s2 = s*s
        t0 = exp(s2*n*n*0.5)
        d_ratio = d/d_characteristic
        t1 = (d/(d_ratio))**n
        t2 = erf((s2*n - log(d_ratio))/(sqrt(2.)*s))
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
        GGS cumulative density function, [-]

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
    return d**(m - 1.0)*k*m*exp(-d**m*k)


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
        RR cumulative density function, [-]

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
    exceptions may occur.

    Examples
    --------
    >>> "{:g}".format(pdf_Rosin_Rammler_basis_integral(5E-2, 200, 2, 3))
    '-0.000452399'
    '''
    # Also not able to compute the limit for d approaching 0.
    try:
        a = (m + n)/m
        x = d**m*k
        t1 = float(gamma(a))*float(gammaincc(a, x))
        return (-d**(m+n)*k*(d**m*k)**(-a))*t1
    except (OverflowError, ZeroDivisionError) as e:
        if d == 1E-40:
            raise e
        return pdf_Rosin_Rammler_basis_integral(1E-40, k, m, n)


names = {0: 'Number distribution', 1: 'Length distribution',
         2: 'Area distribution', 3: 'Volume/Mass distribution'}

def _label_distribution_n(n):  # pragma: no cover
    if n in names:
        return names[n]
    else:
        return 'Order %s distribution' %str(n)

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
            Power and/or subscript of D moment in the above equations, [-]
        q : int
            Power and/or subscript of D moment in the above equations, [-]

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
            Power and/or subscript of D moment in the above equations, [-]
        r : int
            Power and/or subscript of D moment in the above equations, [-]

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
    def _pdf_basis_integral_definite(self, d_min, d_max, n):
        # Needed as an api for numerical integrals
        return (self._pdf_basis_integral(d=d_max, n=n)
                - self._pdf_basis_integral(d=d_min, n=n))

    def pdf(self, d, n=None):
        r'''Computes the probability density function of a
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
        1238.661379483343

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
            denominator = self._pdf_basis_integral_definite(d_min=0.0, d_max=self.d_excessive, n=power)
            ans = numerator/denominator
        # Handle splines which might go below zero
        ans = max(ans, 0.0)
        if self.truncated:
            if d < self.d_min or d > self.d_max:
                return 0.0
            ans = (ans)/(self._cdf_d_max - self._cdf_d_min)
        return ans

    def cdf(self, d, n=None):
        r'''Computes the cumulative distribution density function of a
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
        >>> [psd.cdf(5e-6, n) for n in range(4)]
        [0.933192798731, 0.8413447460685, 0.6914624612740, 0.5]
        '''
        if n is not None and n != self.order:
            power = n - self.order
            # One of the pdf_basis_integral calls could be saved except for
            # support for numerical integrals
            numerator = self._pdf_basis_integral_definite(d_min=0.0, d_max=d, n=power)
            denominator = self._pdf_basis_integral_definite(d_min=0.0, d_max=self.d_excessive, n=power)
            ans =  max(numerator/denominator, 0.0)
        # Handle splines which might go below zero
        else:
            ans = max(self._cdf(d=d), 0.0)
        if self.truncated:
            if d <= self.d_min:
                return 0.0
            elif d >= self.d_max:
                return 1.0
            ans = (ans - self._cdf_d_min)/(self._cdf_d_max - self._cdf_d_min)
        return ans

    def delta_cdf(self, d_min, d_max, n=None):
        r'''Computes the difference in cumulative distribution function between
        two particle size diameters.

        .. math::
            \Delta Q_n = Q_n(d_{max}) - Q_n(d_{min})

        Parameters
        ----------
        d_min : float
            Lower particle size diameter, [m]
        d_max : float
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
        0.9165280099853876
        '''
        return self.cdf(d_max, n=n) - self.cdf(d_min, n=n)

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
        0.0002947436533523378
        >>> psd.dn(0)
        0.0
        '''
        if fraction == 1.0:
            # Avoid returning the maximum value of the search interval
            fraction = 1.0 - epsilon
        if fraction < 0:
            raise ValueError('Fraction must be more than 0')
        elif fraction == 0:  # pragma: no cover
            if self.truncated:
                return self.d_min
            return 0.0
            # Solve to float prevision limit - works well, but is there a real
            # point when with mpmath it would never happen?
            # dist.cdf(dist.dn(0)-1e-35) == 0
            # dist.cdf(dist.dn(0)-1e-36) == input
            # dn(0) == 1.9663615597466143e-20
#            def err(d):
#                cdf = self.cdf(d, n=n)
#                if cdf == 0:
#                    cdf = -1
#                return cdf
#            return brenth(err, self.d_minimum, self.d_excessive, maxiter=1000, xtol=1E-200)

        elif fraction > 1:
            raise ValueError('Fraction less than 1')
        # As the dn may be incredibly small, it is required for the absolute
        # tolerance to not be happy - it needs to continue iterating as long
        # as necessary to pin down the answer
        return brenth(lambda d:self.cdf(d, n=n) -fraction,
                      self.d_minimum, self.d_excessive, maxiter=1000, xtol=1E-200)

    def ds_discrete(self, d_min=None, d_max=None, pts=20, limit=1e-9,
                    method='logarithmic'):
        r'''Create a particle spacing mesh to perform calculations with,
        according to one of several ways. The allowable meshes are
        'linear', 'logarithmic', a geometric series specified by a Renard
        number such as 'R10', or the meshes available in one of several sieve
        standards.

        Parameters
        ----------
        d_min : float, optional
            The minimum diameter at which the mesh starts, [m]
        d_max : float, optional
            The maximum diameter at which the mesh ends, [m]
        pts : int, optional
            The number of points to return for the mesh (note this is not
            respected by sieve meshes), [-]
        limit : float
            If `d_min` or `d_max` is not specified, it will be calculated as the
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
        Note that when specifying a Renard series, only one of `d_min` or `d_max` can
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
            if d_min is None:
                d_min = self.dn(limit)
            if d_max is None:
                d_max = self.dn(1.0 - limit)
        return psd_spacing(d_min=d_min, d_max=d_max, pts=pts, method=method)

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
        [0.00064347101291, 0.916528009985, 0.0828285179619, 1.039798e-09]
        '''
        cdfs = [self.cdf(d, n=n) for d in ds]
        return [cdfs[0]] + diff(cdfs)

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
        [0.000643471012913, 0.917171480998, 0.999999998960, 1.0]
        '''
        return [self.cdf(d, n=n) for d in ds]

    def mean_size(self, p, q):
        '''
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)
        >>> psd.mean_size(3, 2)
        4.412484512922977e-06

        Note that for the case where p == q, a different set of formulas are
        required - which do not have analytical results for many distributions.
        Therefore, a close numerical approximation is used instead, to
        perturb the values of p and q so they are 1E-9 away from each other.
        This leads only to slight errors, as in the example below where the
        correct answer is 5E-6.

        >>> psd.mean_size(3, 3)
        4.9999999304923345e-06
        '''
        if p == q:
            p -= 1e-9
            q += 1e-9
        pow1 = q - self.order

        denominator = self._pdf_basis_integral_definite(d_min=self.d_minimum, d_max=self.d_excessive, n=pow1)
        root_power = p - q
        pow3 = p - self.order
        numerator = self._pdf_basis_integral_definite(d_min=self.d_minimum, d_max=self.d_excessive, n=pow3)
        return (numerator/denominator)**(1.0/(root_power))

    def mean_size_ISO(self, k, r):
        '''
        >>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)
        >>> psd.mean_size_ISO(1, 2)
        4.412484512922977e-06
        '''
        p = k + r
        q = r
        return self.mean_size(p=p, q=q)

    @property
    def vssa(self):
        r'''The volume-specific surface area of a particle size distribution.

        .. math::
            \text{VSSA} = \frac{6}{\bar x_{1,2}}

        Returns
        -------
        VSSA : float
            The volume-specific surface area of the distribution, [m^2/m^3]

        Examples
        --------
        >>> PSDLognormal(s=0.5, d_characteristic=5E-6).vssa
        1359778.1436801916

        References
        ----------
        .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size
           Analysis - Part 2: Calculation of Average Particle Sizes/Diameters
           and Moments from Particle Size Distributions.
        '''
        return 6/self.mean_size(3, 2)


    def plot_pdf(self, n=(0, 1, 2, 3), d_min=None, d_max=None, pts=500,
                 normalized=False, method='linear'):  # pragma: no cover
        r'''Plot the probability density function of the particle size
        distribution. The plotted range can be specified using `d_min` and
        `d_max`, or estimated automatically. One or more order can be plotted,
        by providing an iterable of ints as the value of `n` or just one int.

        Parameters
        ----------
        n : tuple(int) or int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer; as many as desired may be specified, [-]
        d_min : float, optional
            Lower particle size diameter, [m]
        d_max : float, optional
            Upper particle size diameter, [m]
        pts : int, optional
            The number of points for values to be calculated, [-]
        normalized : bool, optional
            Whether to display the actual probability density function, which
            may have a huge magnitude - or to divide each point by the sum
            of all the points. Doing this is a common practice, but the values
            at each point are dependent on the number of points being plotted,
            and the distribution of the points;
            [-]
        method : str, optional
            Either 'linear', 'logarithmic', a Renard number like 'R10' or 'R5'
            or'R2.5', or one of the sieve standards 'ISO 3310-1 R40/3',
            'ISO 3310-1 R20', 'ISO 3310-1 R20/3', 'ISO 3310-1',
            'ISO 3310-1 R10', 'ASTM E11', [-]
        '''
        try:
            import matplotlib.pyplot as plt
        except:  # pragma: no cover
            raise ValueError(NO_MATPLOTLIB_MSG)
        ds = self.ds_discrete(d_min=d_min, d_max=d_max, pts=pts, method=method)
        try:
            for ni in n:
                fractions = [self.pdf(d, n=ni) for d in ds]
                if normalized:
                    fractions = normalize(fractions)
                plt.semilogx(ds, fractions, label=_label_distribution_n(ni))
        except Exception as e:
            fractions = [self.pdf(d, n=n) for d in ds]
            if normalized:
                fractions = normalize(fractions)
            plt.semilogx(ds, fractions, label=_label_distribution_n(n))
        plt.ylabel('Probability density function, [-]')
        plt.xlabel('Particle diameter, [m]')
        plt.title('Probability density function of %s distribution with '
                  'parameters %s' %(self.name, self.parameters))
        plt.legend()
        plt.show()
        return fractions

    def plot_cdf(self, n=(0, 1, 2, 3), d_min=None, d_max=None, pts=500,
                 method='logarithmic'):   # pragma: no cover
        r'''Plot the cumulative distribution function of the particle size
        distribution. The plotted range can be specified using `d_min` and
        `d_max`, or estimated automatically. One or more order can be plotted,
        by providing an iterable of ints as the value of `n` or just one int.

        Parameters
        ----------
        n : tuple(int) or int, optional
            None (for the `order` specified when the distribution was created),
            0 (number), 1 (length), 2 (area), 3 (volume/mass),
            or any integer; as many as desired may be specified, [-]
        d_min : float, optional
            Lower particle size diameter, [m]
        d_max : float, optional
            Upper particle size diameter, [m]
        pts : int, optional
            The number of points for values to be calculated, [-]
        method : str, optional
            Either 'linear', 'logarithmic', a Renard number like 'R10' or 'R5'
            or'R2.5', or one of the sieve standards 'ISO 3310-1 R40/3',
            'ISO 3310-1 R20', 'ISO 3310-1 R20/3', 'ISO 3310-1',
            'ISO 3310-1 R10', 'ASTM E11', [-]
        '''
        try:
            import matplotlib.pyplot as plt
        except:  # pragma: no cover
            raise ValueError(NO_MATPLOTLIB_MSG)

        ds = self.ds_discrete(d_min=d_min, d_max=d_max, pts=pts, method=method)
        try:
            for ni in n:
                cdfs = self.cdf_discrete(ds=ds, n=ni)
                plt.semilogx(ds, cdfs, label=_label_distribution_n(ni))
        except:
            cdfs = self.cdf_discrete(ds=ds, n=n)
            plt.semilogx(ds, cdfs, label=_label_distribution_n(n))
        if self.points:
            plt.plot(self.ds, self.fraction_cdf, '+', label='Volume/Mass points')

            if hasattr(self, 'area_fractions'):
                plt.plot(self.ds, cumsum(self.area_fractions), '+', label='Area points')
            if hasattr(self, 'length_fractions'):
                plt.plot(self.ds, cumsum(self.length_fractions), '+', label='Length points')
            if hasattr(self, 'number_fractions'):
                plt.plot(self.ds, cumsum(self.number_fractions), '+', label='Number points')

        plt.ylabel('Cumulative density function, [-]')
        plt.xlabel('Particle diameter, [m]')
        plt.title('Cumulative density function of %s distribution with '
                  'parameters %s' %(self.name, self.parameters))
        plt.legend()
        plt.show()


class ParticleSizeDistribution(ParticleSizeDistributionContinuous):
    r'''Class representing a discrete particle size distribution specified by a
    series of diameter bins, and the quantity of particles in each bin. The
    quantities may be specified as either the fraction of particles in each
    bin, or as cumulative distributions. The input fractions can be
    specified to be in a mass basis (`order=3`), number basis (`order=0`),
    or the orders in between for length basis or area basis. If the
    fractions do not sum to 1, and `cdf` is False, then the fractions are
    normalized. This allows flow rates or counts of size bins to be given as
    well.

    Parameters
    ----------
    ds : list[float]
        Diameter bins; length of the specified quantities, optionally +1 that
        length to specify a cutoff diameter for the smallest diameter bin, [m]
    fractions : list[float], optional
        The mass/mole/volume/length/area/count fractions or cumulative
        distributions or counts of each particle size in
        each diameter bin (the type is specified by `order`), [-]
    order : int, optional
        0 for a number distribution as input; 1 for length distribution;
        2 for area distribution; 3 for mass, mole, or volume distribution, [-]
    cdf : bool, optional
        If the distribution is given as increasing fractions with 1 as the last
        result, `cdf` must be set to True, [-]
    monotonic : bool, optional
        If True, for interpolated quanties, monotonic splines will be used
        instead of the standard splines, [-]

    Attributes
    ----------
    fractions : list[float]
        The mass/mole/volume basis fractions of particles in each bin, [-]
    area_fractions : list[float]
        The area fractions of particles in each bin, [-]
    length_fractions : list[float]
        The length fractions of particles in each bin, [-]
    number_fractions : list[float]
        The number fractions of particles in each bin, [-]
    fraction_cdf : list[float]
        The cumulative mass/mole/volume basis fractions of particles in each
        bin, [-]
    area_cdf : list[float]
        The cumulative area fractions of particles in each bin, [-]
    length_cdf : list[float]
        The cumulative length fractions of particles in each bin, [-]
    number_cdf : list[float]
        The cumulative number fractions of particles in each bin, [-]
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

    >>> import numpy as np
    >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
    >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    >>> psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)
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
    truncated = False
    name = 'Discrete'
    def __init__(self, ds, fractions, cdf=False, order=3, monotonic=True):
        self.monotonic = monotonic
        self.ds = ds
        self.order = order

        if ds is not None and (len(ds) == len(fractions) + 1):
            self.size_classes = True
        else:
            self.size_classes = False

        if cdf:
            # Convert a cdf to fraction set
            if len(fractions)+1 == len(ds):
                fractions = [fractions[0]] + diff(fractions)
            else:
                fractions = diff(fractions)
                fractions.insert(0, 0.0)
        elif sum(fractions) != 1.0:
            # Normalize flow inputs
            tot_inv = 1.0/sum(fractions)
            fractions = [i*tot_inv if i != 0.0 else 0.0 for i in fractions]

        self.N = len(fractions)

        # This will always be in base-3 basis
        if self.order != 3:
            power = 3 - self.order
            d3s = [self.di_power(i, power=power)*fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            self.fractions = [i/tot_d3 for i in d3s]
        else:
            self.fractions = fractions
        # Set the number fractions
        D3s = [self.di_power(i, power=3) for i in range(self.N)]
        numbers = [Vi/Vp for Vi, Vp in zip(self.fractions, D3s)]
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


        # Things for interoperability with the Continuous distribution
        self.d_excessive = self.ds[-1]
        self.d_minimum = 0.0
        self.parameters = {}
        self.order = 3
        self.fraction_cdf = self.volume_cdf = cumsum(self.fractions)
        self.area_cdf = cumsum(self.area_fractions)
        self.length_cdf = cumsum(self.length_fractions)
        self.number_cdf = cumsum(self.number_fractions)

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
            delta_cdf = dist.delta_cdf(d_min=self.ds[i], d_max=self.ds[i+1])
            err += abs(delta_cdf - self.fractions[i])
        return err

    def fit(self, x0=None, distribution='lognormal', n=None, **kwargs):
        """Incomplete method to fit experimental values to a curve.

        It is very hard to get good initial guesses, which are really required
        for this. Differential evolution is promissing. This API is likely to
        change in the future.
        """
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
        from scipy.optimize import minimize
        return minimize(self._fit_obj_function, x0, args=(dist, n), **kwargs)

    @property
    def Dis(self):
        """Representative diameters of each bin."""
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
        >>> import numpy as np
        >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)
        >>> psd.mean_size(3, 2)
        0.002269321031745045
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
        >>> import numpy as np
        >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        >>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)
        >>> psd.mean_size_ISO(1, 2)
        0.002269321031745045
        '''
        p = k + r
        q = r
        return self.mean_size(p=p, q=q)

    @property
    def vssa(self):
        r'''The volume-specific surface area of a particle size distribution.
        Note this uses the diameters provided by the method `Dis`.

        .. math::
            \text{VSSA} = \sum_i \text{fraction}_i \frac{SA_i}{V_i}

        Returns
        -------
        VSSA : float
            The volume-specific surface area of the distribution, [m^2/m^3]

        References
        ----------
        .. [1] ISO 9276-2:2014 - Representation of Results of Particle Size
           Analysis - Part 2: Calculation of Average Particle Sizes/Diameters
           and Moments from Particle Size Distributions.
        '''
        ds = self.Dis
        Vs = [pi/6*di**3 for di in ds]
        SAs = [pi*di**2 for di in ds]
        SASs = [SA/V for SA, V in zip(SAs, Vs)]
        VSSA = sum([fi*SASi for fi, SASi in zip(self.fractions, SASs)])
        return VSSA

try:  # pragma: no cover
    # Python 2
    ParticleSizeDistributionContinuous.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__func__.__doc__)
    ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__)
    ParticleSizeDistribution.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__func__.__doc__)
    ParticleSizeDistribution.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__func__.__doc__)
except AttributeError:  # pragma: no cover
    try:
        # Python 3
        ParticleSizeDistributionContinuous.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__doc__)
        ParticleSizeDistributionContinuous.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__doc__)
        ParticleSizeDistribution.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__doc__)
        ParticleSizeDistribution.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__doc__)
    except:
        pass # micropython
del _mean_size_iso_docstring
del _mean_size_docstring

class PSDLognormal(ParticleSizeDistributionContinuous):
    name = 'Lognormal'
    points = False
    truncated = False
    def __init__(self, d_characteristic, s, order=3, d_min=None, d_max=None):
        self.s = s
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'s': s, 'd_characteristic': d_characteristic,
                           'd_min': d_min, 'd_max': d_max}
        self.d_min = d_min
        self.d_max = d_max
        # Pick an upper bound for the search algorithm of 15 orders of magnitude larger than
        # the characteristic diameter; should never be a problem, as diameters can only range
        # so much, physically.
        if self.d_max is not None:
            self.d_excessive = self.d_max
        else:
            self.d_excessive = 1E15*self.d_characteristic
        if self.d_min is not None:
            self.d_minimum = self.d_min
        else:
            self.d_minimum = 0.0

        if self.d_min is not None or self.d_max is not None:
            self.truncated = True
            if self.d_max is None:
                self.d_max = self.d_excessive
            if self.d_min is None:
                self.d_min = 0.0

            self._cdf_d_max = self._cdf(self.d_max)
            self._cdf_d_min = self._cdf(self.d_min)

    def _pdf(self, d):
        return pdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)

    def _cdf(self, d):
        return cdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)

    def _pdf_basis_integral(self, d, n):
        return pdf_lognormal_basis_integral(d, d_characteristic=self.d_characteristic, s=self.s, n=n)


class PSDGatesGaudinSchuhman(ParticleSizeDistributionContinuous):
    name = 'Gates Gaudin Schuhman'
    points = False
    truncated = False
    def __init__(self, d_characteristic, m, order=3, d_min=None, d_max=None):
        self.m = m
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'m': m, 'd_characteristic': d_characteristic,
                           'd_min': d_min, 'd_max': d_max}

        if self.d_max is not None:
            # PDF above this is zero
            self.d_excessive = self.d_max
        else:
            self.d_excessive = self.d_characteristic
        if self.d_min is not None:
            self.d_minimum = self.d_min
        else:
            self.d_minimum = 0.0

        if self.d_min is not None or self.d_max is not None:
            self.truncated = True
            if self.d_max is None:
                self.d_max = self.d_excessive
            if self.d_min is None:
                self.d_min = 0.0

            self._cdf_d_max = self._cdf(self.d_max)
            self._cdf_d_min = self._cdf(self.d_min)



    def _pdf(self, d):
        return pdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _cdf(self, d):
        return cdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _pdf_basis_integral(self, d, n):
        return pdf_Gates_Gaudin_Schuhman_basis_integral(d, d_characteristic=self.d_characteristic, m=self.m, n=n)


class PSDRosinRammler(ParticleSizeDistributionContinuous):
    name = 'Rosin Rammler'
    points = False
    truncated = False
    def __init__(self, k, m, order=3, d_min=None, d_max=None):
        self.m = m
        self.k = k
        self.order = order
        self.parameters = {'m': m, 'k': k, 'd_min': d_min, 'd_max': d_max}

        if self.d_max is not None:
            self.d_excessive = self.d_max
        else:
            self.d_excessive = 1e15 # TODO
        if self.d_min is not None:
            self.d_minimum = self.d_min
        else:
            self.d_minimum = 0.0

        if self.d_min is not None or self.d_max is not None:
            self.truncated = True
            if self.d_max is None:
                self.d_max = self.d_excessive
            if self.d_min is None:
                self.d_min = 0.0

            self._cdf_d_max = self._cdf(self.d_max)
            self._cdf_d_min = self._cdf(self.d_min)

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
    truncated = False
    def __init__(self, distribution, order=3.0, d_excessive=1.0, name=None,
                 d_min=None, d_max=None):
        if name:
            self.name = name
        else:
            try:
                self.name = distribution.dist.__class__.__name__
            except:
                pass
        try:
            self.parameters = dict(distribution.kwds)
            self.parameters.update({'d_min': d_min, 'd_max': d_max})
        except:
            self.parameters = {'d_min': d_min, 'd_max': d_max}

        self.distribution = distribution
        self.order = order
        self.d_max = d_max
        self.d_min = d_min

        if self.d_max is not None:
            self.d_excessive = self.d_max
        else:
            self.d_excessive = d_excessive
        if self.d_min is not None:
            self.d_minimum = self.d_min
        else:
            self.d_minimum = 0.0

        if self.d_min is not None or self.d_max is not None:
            self.truncated = True
            if self.d_max is None:
                self.d_max = self.d_excessive
            if self.d_min is None:
                self.d_min = 0.0

            self._cdf_d_max = self._cdf(self.d_max)
            self._cdf_d_min = self._cdf(self.d_min)



    def _pdf(self, d):
        return self.distribution.pdf(d)

    def _cdf(self, d):
        return self.distribution.cdf(d)

    def _pdf_basis_integral_definite(self, d_min, d_max, n):
        # Needed as an api for numerical integrals
        n = float(n)
        if d_min == 0:
            d_min = d_max*1E-12

        if n == 0:
            to_int = lambda d : self._pdf(d)
        elif n == 1:
            to_int = lambda d : d*self._pdf(d)
        elif n == 2:
            to_int = lambda d : d*d*self._pdf(d)
        elif n == 3:
            to_int = lambda d : d*d*d*self._pdf(d)
        else:
            to_int = lambda d : d**n*self._pdf(d)

#        points = logspace(log10(max(d_max*1e-3, d_min)), log10(d_max*.999), 40)
        points = [d_max*1e-3] # d_min*.999 d_min
        return float(quad(to_int, d_min, d_max, points=points)[0]) #


class PSDInterpolated(ParticleSizeDistributionContinuous):
    name = 'Interpolated'
    points = True
    truncated = False
    def __init__(self, ds, fractions, order=3, monotonic=True):
        self.order = order
        self.monotonic = monotonic
        self.parameters = {}

        ds = list(ds)
        fractions = list(fractions)

        if len(ds) == len(fractions)+1:
            # size classes, the last point will be zero
            fractions.insert(0, 0.0)
            self.d_minimum = min(ds)
        elif ds[0] != 0:
            ds = [0] + ds
            if len(ds) != len(fractions):
                fractions = [0] + fractions
            self.d_minimum = 0.0

        self.ds = ds
        self.fractions = fractions

        self.d_excessive = max(ds)


        self.fraction_cdf = cumsum(fractions)
        if self.monotonic:
            from scipy.interpolate import PchipInterpolator
            globals()['PchipInterpolator'] = PchipInterpolator

            self.cdf_spline = PchipInterpolator(ds, self.fraction_cdf, extrapolate=True)
            self.pdf_spline = PchipInterpolator(ds, self.fraction_cdf, extrapolate=True).derivative(1)
        else:
            from scipy.interpolate import UnivariateSpline
            globals()['UnivariateSpline'] = UnivariateSpline

            self.cdf_spline = UnivariateSpline(ds, self.fraction_cdf, ext=3, s=0)
            self.pdf_spline = UnivariateSpline(ds, self.fraction_cdf, ext=3, s=0).derivative(1)

        # The pdf basis integral splines will be stored here
        self.basis_integrals = {}


    def _pdf(self, d):
        return max(0.0, float(self.pdf_spline(d)))

    def _cdf(self, d):
        if d > self.d_excessive:
            # Handle spline values past 1 that decrease to zero
            return 1.0
        return max(0.0, float(self.cdf_spline(d)))

    def _pdf_basis_integral(self, d, n):
        # there are slight errors with this approach - but they are OK to
        # ignore.
        # DO NOT evaluate the first point as it leads to inf values; just set
        # it to zero
        from fluids.numerics import numpy as np
        if n not in self.basis_integrals:
            ds = np.array(self.ds[1:])
            pdf_vals = self.pdf_spline(ds)
            basis_integral = ds**n*pdf_vals
            if self.monotonic:
                from scipy.interpolate import PchipInterpolator
                self.basis_integrals[n] = PchipInterpolator(ds, basis_integral, extrapolate=True).antiderivative(1)
            else:
                from scipy.interpolate import UnivariateSpline
                self.basis_integrals[n] = UnivariateSpline(ds, basis_integral, ext=3, s=0).antiderivative(n=1)
        return max(float(self.basis_integrals[n](d)), 0.0)

