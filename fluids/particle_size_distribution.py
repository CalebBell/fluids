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
           'pdf_lognormal', 'cdf_lognormal', 'pdf_lognormal_basis_integral',
           'pdf_Gates_Gaudin_Schuhman', 'cdf_Gates_Gaudin_Schuhman',
           'pdf_Gates_Gaudin_Schuhman_basis_integral',
           'pdf_Rosin_Rammler', 'cdf_Rosin_Rammler', 
           'pdf_Rosin_Rammler_basis_integral']

from math import log, exp, pi, log10
from sys import float_info
from scipy.optimize import brenth, minimize
from scipy.integrate import quad
from scipy.special import erf, gammaincc, gamma
import scipy.stats
from numpy.random import lognormal
import numpy as np

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except:
    has_matplotlib = False

ROOT_TWO_PI = (2.0*pi)**0.5


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
    computation with :math:`x = d/d_{characteristic}`, `s` unchanged, and
    the result divided by `d_characteristic` to obtain a compatible answer.

    >>> scipy.stats.lognorm.pdf(x=1E-4/1E-5, s=1.1)/1E-5
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
    computation with :math:`x = d/d_{characteristic}`, and `s` unchanged to 
    obtain a compatible answer.

    >>> scipy.stats.lognorm.cdf(x=1E-4/1E-5, s=1.1)
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


class ParticleSizeDistribution(object):
    r'''Class representing a discrete particle size distribution specified by a
    series of diameter bins, and the quantity of particles in each bin. The
    quantities may be specified as counts, volume/mass/mole fractions, or count
    fractions.
    All parameters are also attributes.
    
    In addition to the diameter bins, one of `fractions`, `count_fractions`,
    or `counts` must be specified.
            
    Parameters
    ----------
    ds : list[float]
        Diameter bins; length of the specified quantities, optionally +1 that
        length to specify a cutoff diameter for the smallest diameter bin, [m]
    fractions : list[float], optional
        The mass/mole/volume fractions of each particles in each diameter bin
        (this class represents particles of the same density and molecular
        weight, so each of the fractions will be the same), [-]
    count_fractions : list[float], optional
        The number fractions by actual number of particles in each bin, [-]
    counts : lists[float], optional
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
    >>> counts = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    >>> psd = ParticleSizeDistribution(ds=ds, counts=counts)
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
    def __init__(self, ds, fractions=None, count_fractions=None, counts=None, 
                 flows=None, rho=None, MW=None):
        '''If given counts or flows, convert to fractions immediately and move 
        forward with them.
        '''
        self.ds = ds
        
        specified_quantities = [i for i in (fractions, count_fractions, counts, flows) if i is not None]
        if len(specified_quantities) > 1:
            raise Exception('More than one distribution specified')
        elif len(specified_quantities) == 0:
            raise Exception('No distribution specified')
        else:
            spec = specified_quantities[0]
        
        if ds is not None and (len(ds) == len(spec) + 1):
            self.size_classes = True
        else:
            self.size_classes = False
            
        self.N = len(spec)
            
        if spec is counts:
            self.counts = counts
            self.count_sum = sum(self.counts)
            self.count_fractions = [i/self.count_sum for i in self.counts]
        
            d3s = [self.di_power(i, power=3)*self.count_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            self.fractions = [i/tot_d3 for i in d3s]
        
        elif spec is count_fractions:
            self.count_fractions = count_fractions
            d3s = [self.di_power(i, power=3)*self.count_fractions[i] for i in range(self.N)]
            tot_d3 = sum(d3s)
            self.fractions = [i/tot_d3 for i in d3s]
        elif spec is fractions:
            self.fractions = fractions
            basis = 100 # m^3
            Vis = [basis*fi for fi in fractions]
            D3s = [self.di_power(i, power=3) for i in range(self.N)]
            Vps = [pi/6*Di for Di in D3s]
            counts = [Vi/Vp for Vi, Vp in zip(Vis, Vps)]
            count_sum = sum(counts)
            self.count_fractions = [i/count_sum for i in counts]
        elif spec is flows:
            raise Exception(NotImplemented('Flows are not yet supported - TODO'))
        
        # Set the length fractions
        D3s = [self.di_power(i, power=2) for i in range(self.N)]
        counts = [Vi/Vp for Vi, Vp in zip(self.fractions, D3s)]
        count_sum = sum(counts)
        self.length_fractions = [i/count_sum for i in counts]
        
        # Set the surface area fractions
        D3s = [self.di_power(i, power=1) for i in range(self.N)]
        counts = [Vi/Vp for Vi, Vp in zip(self.fractions, D3s)]
        count_sum = sum(counts)
        self.area_fractions = [i/count_sum for i in counts]
        # Length and surface area fractions verified numerically

        self.flows = flows
        self.rho = rho
        self.MW = MW
        
        
    def _fit_obj_function(self, vals, distribution, n):
        err = 0.0
        dist = distribution(*list(vals))
        for i in range(len(self.fractions)):
            delta_cdf = dist.delta_cdf(dmin=self.ds[i], dmax=self.ds[i+1])
            err += abs(delta_cdf - self.fractions[i])
        return err
        
    def fit(self, x0=None, distribution='lognormal', n=None, **kwargs):
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
        return minimize(self._fit_obj_function, x0, args=(dist, n), **kwargs)['x']

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
        >>> counts = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, counts=counts)
        >>> psd.mean_size(3, 2)
        0.0022693210317450449
        '''
        if p != q:
            # Note: D(p, q) = D(q, p); in ISO and proven experimentally
            numerator = sum(self.di_power(i=i, power=p)*self.count_fractions[i] for i in range(self.N))
            denominator = sum(self.di_power(i=i, power=q)*self.count_fractions[i] for i in range(self.N))
            return (numerator/denominator)**(1.0/(p-q))
        else:
            numerator = sum(log(self.di_power(i=i, power=1))*self.di_power(i=i, power=p)*self.count_fractions[i] for i in range(self.N))
            denominator = sum(self.di_power(i=i, power=q)*self.count_fractions[i] for i in range(self.N))
            return exp(numerator/denominator)
        
    def mean_size_ISO(self, k, r):
        r'''
        >>> ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        >>> counts = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        >>> psd = ParticleSizeDistribution(ds=ds, counts=counts)
        >>> psd.mean_size_ISO(1, 2)
        0.0022693210317450449
        '''
        p = k + r
        q = r
        return self.mean_size(p=p, q=q)

try:
    # Python 2
    ParticleSizeDistribution.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__func__.__doc__)
    ParticleSizeDistribution.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__func__.__doc__)
except AttributeError:
    # Python 3
    ParticleSizeDistribution.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistribution.mean_size.__doc__)
    ParticleSizeDistribution.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistribution.mean_size_ISO.__doc__)


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
            0 (number pdf), 1 (length pdf), 2 (area pdf), 3 (volume/mass pdf),
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
        ans = self._pdf(d=d, n=n)
        if n is not None:
            power = n - self.order
            numerator = d**power*ans
            
            denominator = (self._pdf_basis_integral(d=self.d_excessive, n=power) 
                            - self._pdf_basis_integral(d=0.0, n=power))
            ans = numerator/denominator
        return ans

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
            0 (number pdf), 1 (length pdf), 2 (area pdf), 3 (volume/mass pdf),
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
        if n is not None:
            power = n - self.order
            t1 = self._pdf_basis_integral(d=0.0, n=power)
            numerator = (self._pdf_basis_integral(d=d, n=power) - t1)
            
            denominator = (self._pdf_basis_integral(d=self.d_excessive, n=power) 
                            - t1)
            return numerator/denominator        
        return self._cdf(d=d, n=n)

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
            0 (number pdf), 1 (length pdf), 2 (area pdf), 3 (volume/mass pdf),
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
        if fraction == 1.0:
            # Avoid returning the maximum value of the search interval
            fraction = 1.0 - float_info.epsilon
        if fraction < 0:
            raise ValueError('Fraction must be more than 0')
        elif fraction == 0: # pragma : no cover
            return 0
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
    
    def ds_discrete(self, dmin=None, dmax=None, pts=20, limit=1e-9):
        if dmin is None:
            dmin = self.dn(limit)
        if dmax is None:
            dmax = self.dn(1.0 - limit)
        #  method=('logarithmic', 'geometric', 'linear' 'R5', 'R10')
        return np.logspace(log10(dmin), log10(dmax), pts).tolist()
    
    def fractions_discrete(self, ds, n=None):
        # I dislike the 0 point - it should could from 0
        cdfs = [self.cdf(d, n=n) for d in ds]
        fractions = [cdfs[0]] + np.diff(cdfs).tolist()
        return fractions
    
    def cdf_discrete(self, ds, n=None):
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
        denominator = self._pdf_basis_integral(d=self.d_excessive, n=pow1) - self._pdf_basis_integral(d=1E-9, n=pow1)
        root_power = p - q
        pow3 = p - self.order
        numerator = self._pdf_basis_integral(d=self.d_excessive, n=pow3) - self._pdf_basis_integral(d=1E-9, n=pow3)
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
    
    def pdf_plot(self, n=(0, 1, 2, 3), dmin=None, dmax=None, pts=500):
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
    
    def cdf_plot(self, n=(0, 1, 2, 3), dmin=None, dmax=None, pts=200):
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
        plt.ylabel('Cumulative density function, [-]')
        plt.xlabel('Particle diameter, [m]')
        plt.title('Cumulative density function of %s distribution with '
                  'parameters %s' %(self.name, self.parameters))
        plt.legend()
        plt.show()

try:
    # Python 2
    ParticleSizeDistributionContinuous.mean_size.__func__.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__func__.__doc__)
    ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__func__.__doc__)
except AttributeError:
    # Python 3
    ParticleSizeDistributionContinuous.mean_size.__doc__ = _mean_size_docstring %(ParticleSizeDistributionContinuous.mean_size.__doc__)
    ParticleSizeDistributionContinuous.mean_size_ISO.__doc__ = _mean_size_iso_docstring %(ParticleSizeDistributionContinuous.mean_size_ISO.__doc__)

class PSDLognormal(ParticleSizeDistributionContinuous):
    name = 'Lognormal'
    def __init__(self, d_characteristic, s, order=3):
        self.s = s
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'s': s, 'd_characteristic': d_characteristic}
        # Pick an upper bound for the search algorithm of 15 orders of magnitude larger than
        # the characteristic diameter; should never be a problem, as diameters can only range
        # so much, physically.
        self.d_excessive = 1E15*self.d_characteristic
        
    def _pdf(self, d, n=None):
        return pdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)

    def _cdf(self, d, n=None):
        return cdf_lognormal(d, d_characteristic=self.d_characteristic, s=self.s)
    
    def _pdf_basis_integral(self, d, n):
        return pdf_lognormal_basis_integral(d, d_characteristic=self.d_characteristic, s=self.s, n=n)
    

class PSDGatesGaudinSchuhman(ParticleSizeDistributionContinuous):
    name = 'Gates Gaudin Schuhman'
    def __init__(self, d_characteristic, m, order=3):
        self.m = m
        self.d_characteristic = d_characteristic
        self.order = order
        self.parameters = {'m': m, 'd_characteristic': d_characteristic}
        # PDF above this is zero
        self.d_excessive = self.d_characteristic

    def _pdf(self, d, n=None):
        return pdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _cdf(self, d, n=None):
        return cdf_Gates_Gaudin_Schuhman(d, d_characteristic=self.d_characteristic, m=self.m)

    def _pdf_basis_integral(self, d, n):
        return pdf_Gates_Gaudin_Schuhman_basis_integral(d, d_characteristic=self.d_characteristic, m=self.m, n=n)


class PSDRosinRammler(ParticleSizeDistributionContinuous):
    name = 'Rosin Rammler'
    def __init__(self, k, m, order=3):
        self.m = m
        self.k = k
        self.order = order
        self.parameters = {'m': m, 'k': k}
        
        # PDF above this is zero - todo?
        self.d_excessive = 1e3 

    def _pdf(self, d, n=None):
        return pdf_Rosin_Rammler(d, k=self.k, m=self.m)

    def _cdf(self, d, n=None):
        return cdf_Rosin_Rammler(d, k=self.k, m=self.m)
    
    def _pdf_basis_integral(self, d, n):
        return pdf_Rosin_Rammler_basis_integral(d, k=self.k, m=self.m, n=n)
