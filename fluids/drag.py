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

This module contains correlations for the drag coefficient `Cd` of a particle
moving in a fluid. Numerical solvers for terminal velocity and an
integrator for particle position over time are included also.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Interfaces to Drag Models
-------------------------
.. autofunction:: drag_sphere
.. autofunction:: v_terminal
.. autofunction:: integrate_drag_sphere
.. autofunction:: time_v_terminal_Stokes
.. autofunction:: drag_sphere_methods

Drag Correlations
-----------------
.. autofunction:: Stokes
.. autofunction:: Barati
.. autofunction:: Barati_high
.. autofunction:: Khan_Richardson
.. autofunction:: Morsi_Alexander
.. autofunction:: Rouse
.. autofunction:: Engelund_Hansen
.. autofunction:: Clift_Gauvin
.. autofunction:: Graf
.. autofunction:: Flemmer_Banks
.. autofunction:: Swamee_Ojha
.. autofunction:: Yen
.. autofunction:: Haider_Levenspiel
.. autofunction:: Cheng
.. autofunction:: Terfous
.. autofunction:: Mikhailov_Freire
.. autofunction:: Clift
.. autofunction:: Ceylan
.. autofunction:: Almedeij
.. autofunction:: Morrison
.. autofunction:: Song_Xu
"""

from __future__ import division
from math import sqrt, exp, log, log10, tanh
from fluids.constants import g
from fluids.numerics import secant
from fluids.core import Reynolds

__all__ = ['drag_sphere', 'drag_sphere_methods', 'v_terminal', 'integrate_drag_sphere',
'time_v_terminal_Stokes', 'Stokes',
'Barati', 'Barati_high', 'Rouse', 'Engelund_Hansen',
'Clift_Gauvin', 'Morsi_Alexander', 'Graf', 'Flemmer_Banks', 'Khan_Richardson',
'Swamee_Ojha', 'Yen', 'Haider_Levenspiel', 'Cheng', 'Terfous',
'Mikhailov_Freire', 'Clift', 'Ceylan', 'Almedeij', 'Morrison', 'Song_Xu']

def Stokes(Re):
    r'''Calculates drag coefficient of a smooth sphere using Stoke's law.

    .. math::
        C_D = 24/Re

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 0.3

    Examples
    --------
    >>> Stokes(0.1)
    240.0

    References
    ----------
    .. [1] Rhodes, Martin J. Introduction to Particle Technology. Wiley, 2013.
    '''
    return 24./Re


def Barati(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_.

    .. math::
        C_D = 5.4856\times10^9\tanh(4.3774\times10^{-9}/Re)
        + 0.0709\tanh(700.6574/Re) + 0.3894\tanh(74.1539/Re)
        - 0.1198\tanh(7429.0843/Re) + 1.7174\tanh[9.9851/(Re+2.3384)] + 0.4744

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    Matching example in [1]_, in a table of calculated values.

    >>> Barati(200.)
    0.7682237950389874

    References
    ----------
    .. [1] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Re_inv = 1.0/Re
    Cd = (5.4856E9*tanh(4.3774E-9*Re_inv) + 0.0709*tanh(700.6574*Re_inv)
    + 0.3894*tanh(74.1539*Re_inv) - 0.1198*tanh(7429.0843*Re_inv)
    + 1.7174*tanh(9.9851/(Re + 2.3384)) + 0.4744)
    return Cd


def Barati_high(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_.

    .. math::
        C_D = 8\times 10^{-6}\left[(Re/6530)^2 + \tanh(Re) - 8\ln(Re)/\ln(10)\right]
        - 0.4119\exp(-2.08\times10^{43}/[Re + Re^2]^4)
        -2.1344\exp(-\{[\ln(Re^2 + 10.7563)/\ln(10)]^2 + 9.9867\}/Re)
        +0.1357\exp(-[(Re/1620)^2 + 10370]/Re)
        - 8.5\times 10^{-3}\{2\ln[\tanh(\tanh(Re))]/\ln(10) - 2825.7162\}/Re
        + 2.4795

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 1E6
    This model is the wider-range model the authors developed.
    At sufficiently low diameters or Re values, drag is no longer a phenomena.

    Examples
    --------
    Matching example in [1]_, in a table of calculated values.

    >>> Barati_high(200.)
    0.7730544082789523

    References
    ----------
    .. [1] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Re2 = Re*Re
    t0 = 1.0/Re
    t1 = (Re/6530.)
    t2 = (Re/1620.)
    t3 = log10(Re2 + 10.7563)
    tanhRe = tanh(Re)
    Cd = (8E-6*(t1*t1 + tanhRe - 8.0*log10(Re))
    - 0.4119*exp(-2.08E43/(Re+Re2)**4)
    - 2.1344*exp(-t0*(t3*t3 + 9.9867))
    + 0.1357*exp(-t0*(t2*t2 + 10370.))
    - 8.5E-3*t0*(2.0*log10(tanh(tanhRe)) - 2825.7162) + 2.4795)
    return Cd


def Rouse(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + \frac{3}{Re^{0.5}} + 0.34

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Rouse(200.)
    0.6721320343559642

    References
    ----------
    .. [1] H. Rouse, Fluid Mechanics for Hydraulic Engineers, Dover,
       New York, N.Y., 1938
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re + 3/sqrt(Re) + 0.34


def Engelund_Hansen(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + 1.5

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Engelund_Hansen(200.)
    1.62

    References
    ----------
    .. [1] F. Engelund, E. Hansen, Monograph on Sediment Transport in Alluvial
       Streams, Monograpsh Denmark Technical University, Hydraulic Lab,
       Denmark, 1967.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re + 1.5


def Clift_Gauvin(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re}(1 + 0.152Re^{0.677}) + \frac{0.417}
        {1 + 5070Re^{-0.94}}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Clift_Gauvin(200.)
    0.7905400398000133

    References
    ----------
    .. [1] R. Clift, W.H. Gauvin, The motion of particles in turbulent gas
       streams, Proc. Chemeca, 70, 1970, pp. 14-28.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re*(1 + 0.152*Re**0.677) + 0.417/(1 + 5070*Re**-0.94)


def Morsi_Alexander(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    If Re < 0.1:

    .. math::
        C_D = \frac{24}{Re}

    If 0.1 < Re < 1:

    .. math::
        C_D = \frac{22.73}{Re}+\frac{0.0903}{Re^2} + 3.69

    If 1 < Re < 10:

    .. math::
        C_D = \frac{29.1667}{Re}-\frac{3.8889}{Re^2} + 1.2220

    If 10 < Re < 100:

    .. math::
        C_D =\frac{46.5}{Re}-\frac{116.67}{Re^2} + 0.6167

    If 100 < Re < 1000:

    .. math::
        C_D = \frac{98.33}{Re}-\frac{2778}{Re^2} + 0.3644

    If 1000 < Re < 5000:

    .. math::
        C_D =  \frac{148.62}{Re}-\frac{4.75\times10^4}{Re^2} + 0.3570

    If 5000 < Re < 10000:

    .. math::
        C_D = \frac{-490.5460}{Re}+\frac{57.87\times10^4}{Re^2} + 0.46

    If 10000 < Re < 50000:

    .. math::
        C_D = \frac{-1662.5}{Re}+\frac{5.4167\times10^6}{Re^2} + 0.5191

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5.
    Original was reviewed, and confirmed to contain the cited equations.

    Examples
    --------
    >>> Morsi_Alexander(200)
    0.7866

    References
    ----------
    .. [1] Morsi, S. A., and A. J. Alexander. "An Investigation of Particle
       Trajectories in Two-Phase Flow Systems." Journal of Fluid Mechanics
       55, no. 02 (September 1972): 193-208. doi:10.1017/S0022112072001806.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    if Re < 0.1:
        return 24./Re
    elif Re < 1:
        return 22.73/Re + 0.0903/Re**2 + 3.69
    elif Re < 10:
        return 29.1667/Re - 3.8889/Re**2 + 1.222
    elif Re < 100:
        return 46.5/Re - 116.67/Re**2 + 0.6167
    elif Re < 1000:
        return 98.33/Re - 2778./Re**2 + 0.3644
    elif Re < 5000:
        return 148.62/Re - 4.75E4/Re**2 + 0.357
    elif Re < 10000:
        return -490.546/Re + 57.87E4/Re**2 + 0.46
    else:
        return -1662.5/Re + 5.4167E6/Re**2 + 0.5191


def Graf(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + \frac{7.3}{1+Re^{0.5}} + 0.25

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Graf(200.)
    0.8520984424785725

    References
    ----------
    .. [1] W.H. Graf, Hydraulics of Sediment Transport, Water Resources
       Publications, Littleton, Colorado, 1984.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re + 7.3/(1 + sqrt(Re)) + 0.25


def Flemmer_Banks(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re}10^E

    .. math::
        E = 0.383Re^{0.356}-0.207Re^{0.396} - \frac{0.143}{1+(\log_{10} Re)^2}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Flemmer_Banks(200.)
    0.7849169609270039

    References
    ----------
    .. [1] Flemmer, R. L. C., and C. L. Banks. "On the Drag Coefficient of a
       Sphere." Powder Technology 48, no. 3 (November 1986): 217-21.
       doi:10.1016/0032-5910(86)80044-4.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    E = 0.383*Re**0.356 - 0.207*Re**0.396 - 0.143/(1 + (log10(Re))**2)
    return 24./Re*10**E


def Khan_Richardson(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = (2.49Re^{-0.328} + 0.34Re^{0.067})^{3.18}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Khan_Richardson(200.)
    0.7747572379211097

    References
    ----------
    .. [1] Khan, A. R., and J. F. Richardson. "The Resistance to Motion of a
       Solid Sphere in a Fluid." Chemical Engineering Communications 62,
       no. 1-6 (December 1, 1987): 135-50. doi:10.1080/00986448708912056.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return (2.49*Re**-0.328 + 0.34*Re**0.067)**3.18


def Swamee_Ojha(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = 0.5\left\{16\left[(\frac{24}{Re})^{1.6} + (\frac{130}{Re})^{0.72}
        \right]^{2.5}+ \left[\left(\frac{40000}{Re}\right)^2 + 1\right]^{-0.25}
        \right\}^{0.25}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 1.5E5

    Examples
    --------
    >>> Swamee_Ojha(200.)
    0.8490012397545713

    References
    ----------
    .. [1] Swamee, P. and Ojha, C. (1991). "Drag Coefficient and Fall Velocity
       of nonspherical particles." J. Hydraul. Eng., 117(5), 660-667.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Cd = 0.5*sqrt(sqrt(16*((24./Re)**1.6 + (130./Re)**0.72)**2.5 + 1.0/sqrt(sqrt((40000./Re)**2 + 1))))
    return Cd


def Yen(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re}\left(1 + 0.15\sqrt{Re} + 0.017Re\right)
        - \frac{0.208}{1+10^4Re^{-0.5}}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Yen(200.)
    0.7822647002187014

    References
    ----------
    .. [1] B.C. Yen, Sediment Fall Velocity in Oscillating Flow, University of
       Virginia, Department of Civil Engineering, 1992.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re*(1 + 0.15*sqrt(Re) + 0.017*Re) - 0.208/(1 + 1E4*1.0/sqrt(Re))


def Haider_Levenspiel(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D=\frac{24}{Re}(1+0.1806Re^{0.6459})+\left(\frac{0.4251}{1
        +\frac{6880.95}{Re}}\right)

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5
    An improved version of this correlation is in Brown and Lawler.

    Examples
    --------
    >>> Haider_Levenspiel(200.)
    0.7959551680251666

    References
    ----------
    .. [1] Haider, A., and O. Levenspiel. "Drag Coefficient and Terminal
       Velocity of Spherical and Nonspherical Particles." Powder Technology
       58, no. 1 (May 1989): 63-70. doi:10.1016/0032-5910(89)80008-7.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re*(1 + 0.1806*Re**0.6459) + (0.4251/(1 + 6880.95/Re))


def Cheng(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D=\frac{24}{Re}(1+0.27Re)^{0.43}+0.47[1-\exp(-0.04Re^{0.38})]

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    >>> Cheng(200.)
    0.7939143028294227

    References
    ----------
    .. [1] Cheng, Nian-Sheng. "Comparison of Formulas for Drag Coefficient and
       Settling Velocity of Spherical Particles." Powder Technology 189, no. 3
       (February 13, 2009): 395-398. doi:10.1016/j.powtec.2008.07.006.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 24./Re*(1. + 0.27*Re)**0.43 + 0.47*(1. - exp(-0.04*Re**0.38))


def Terfous(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = 2.689 + \frac{21.683}{Re} + \frac{0.131}{Re^2}
        - \frac{10.616}{Re^{0.1}} + \frac{12.216}{Re^{0.2}}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is 0.1 < Re <= 5E4

    Examples
    --------
    >>> Terfous(200.)
    0.7814651149769638

    References
    ----------
    .. [1] Terfous, A., A. Hazzab, and A. Ghenaim. "Predicting the Drag
       Coefficient and Settling Velocity of Spherical Particles." Powder
       Technology 239 (May 2013): 12-20. doi:10.1016/j.powtec.2013.01.052.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    return 2.689 + 21.683/Re + 0.131/Re**2 - 10.616/Re**0.1 + 12.216/Re**0.2


def Mikhailov_Freire(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{3808[(1617933/2030) + (178861/1063)Re + (1219/1084)Re^2]}
        {681Re[(77531/422) + (13529/976)Re - (1/71154)Re^2]}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 118300

    Examples
    --------
    >>> Mikhailov_Freire(200.)
    0.7514111388018659

    References
    ----------
    .. [1] Mikhailov, M. D., and A. P. Silva Freire. "The Drag Coefficient of
       a Sphere: An Approximation Using Shanks Transform." Powder Technology
       237 (March 2013): 432-35. doi:10.1016/j.powtec.2012.12.033.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Cd = (3808.*((1617933./2030.) + (178861./1063.)*Re + (1219./1084.)*Re**2)
          /(681.*Re*((77531./422.) + (13529./976.)*Re - (1./71154.)*Re**2)))
    return Cd


def Clift(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    If Re < 0.01:

    .. math::
        C_D = \frac{24}{Re} + \frac{3}{16}

    If 0.01 < Re < 20:

    .. math::
        C_D = \frac{24}{Re}(1 + 0.1315Re^{0.82 - 0.05\log_{10} Re})

    If 20 < Re < 260:

    .. math::
        C_D = \frac{24}{Re}(1 + 0.1935Re^{0.6305})

    If 260 < Re < 1500:

    .. math::
        C_D = 10^{[1.6435 - 1.1242\log_{10} Re + 0.1558[\log_{10} Re]^2}

    If 1500 < Re < 12000:

    .. math::
        C_D = 10^{[-2.4571 + 2.5558\log_{10} Re - 0.9295[\log_{10} Re]^2 + 0.1049[\log_{10} Re]^3}

    If 12000 < Re < 44000:

    .. math::
        C_D = 10^{[-1.9181 + 0.6370\log_{10} Re - 0.0636[\log_{10} Re]^2}

    If 44000 < Re < 338000:

    .. math::
        C_D = 10^{[-4.3390 + 1.5809\log_{10} Re - 0.1546[\log_{10} Re]^2}

    If 338000 < Re < 400000:

    .. math::
        C_D = 9.78 - 5.3\log_{10} Re

    If 400000 < Re < 1000000:

    .. math::
        C_D = 0.19\log_{10} Re - 0.49

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 1E6.

    Examples
    --------
    >>> Clift(200)
    0.7756342422322543

    References
    ----------
    .. [1] R. Clift, J.R. Grace, M.E. Weber, Bubbles, Drops, and Particles,
       Academic, New York, 1978.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    if Re < 0.01:
        return 24./Re + 3/16.
    elif Re < 20:
        return 24./Re*(1 + 0.1315*Re**(0.82 - 0.05*log10(Re)))
    elif Re < 260:
        return 24./Re*(1 + 0.1935*Re**(0.6305))
    elif Re < 1500:
        return 10**(1.6435 - 1.1242*log10(Re) + 0.1558*(log10(Re))**2)
    elif Re < 12000:
        return 10**(-2.4571 + 2.5558*log10(Re) - 0.9295*(log10(Re))**2 + 0.1049*log10(Re)**3)
    elif Re < 44000:
        return 10**(-1.9181 + 0.6370*log10(Re) - 0.0636*(log10(Re))**2)
    elif Re < 338000:
        return 10**(-4.3390 + 1.5809*log10(Re) - 0.1546*(log10(Re))**2)
    elif Re < 400000:
        return 29.78 - 5.3*log10(Re)
    else:
        return 0.19*log10(Re) - 0.49


def Ceylan(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = 1 - 0.5\exp(0.182) + 10.11Re^{-2/3}\exp(0.952Re^{-1/4})
        - 0.03859Re^{-4/3}\exp(1.30Re^{-1/2})
        + 0.037\times10^{-4}Re\exp(-0.125\times10^{-4}Re)
        -0.116\times10^{-10}Re^2\exp(-0.444\times10^{-5}Re)

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is 0.1 < Re <= 1E6
    Original article reviewed.

    Examples
    --------
    >>> Ceylan(200.)
    0.7816735980280175

    References
    ----------
    .. [1] Ceylan, Kadim, Ayşe Altunbaş, and Gudret Kelbaliyev. "A New Model
       for Estimation of Drag Force in the Flow of Newtonian Fluids around
       Rigid or Deformable Particles." Powder Technology 119, no. 2-3
       (September 24, 2001): 250-56. doi:10.1016/S0032-5910(01)00261-3.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Cd = (1 - 0.5*exp(0.182) + 10.11*Re**(-2/3.)*exp(0.952/sqrt(sqrt(Re)))
    - 0.03859*Re**(-4/3.)*exp(1.30/sqrt(Re)) + 0.037E-4*Re*exp(-0.125E-4*Re)
    - 0.116E-10*Re**2*exp(-0.444E-5*Re))
    return Cd


def Almedeij(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \left[\frac{1}{(\phi_1 + \phi_2)^{-1} + (\phi_3)^{-1}} + \phi_4\right]^{0.1}

    .. math::
        \phi_1 = (24Re^{-1})^{10} + (21Re^{-0.67})^{10} + (4Re^{-0.33})^{10} + 0.4^{10}

    .. math::
        \phi_2 = \left[(0.148Re^{0.11})^{-10} + (0.5)^{-10}\right]^{-1}

    .. math::
        \phi_3 = (1.57\times10^8Re^{-1.625})^{10}

    .. math::
        \phi_4 = \left[(6\times10^{-17}Re^{2.63})^{-10} + (0.2)^{-10}\right]^{-1}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 1E6.
    Original work has been reviewed.

    Examples
    --------
    >>> Almedeij(200.)
    0.7114768646813396

    References
    ----------
    .. [1] Almedeij, Jaber. "Drag Coefficient of Flow around a Sphere: Matching
       Asymptotically the Wide Trend." Powder Technology 186, no. 3
       (September 10, 2008): 218-23. doi:10.1016/j.powtec.2007.12.006.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    phi4 = ((6E-17*Re**2.63)**-10 + 0.2**-10)**-1
    phi3 = (1.57E8*Re**-1.625)**10
    phi2 = ((0.148*Re**0.11)**-10 + 0.5**-10)**-1
    phi1 = (24*Re**-1)**10 + (21*Re**-0.67)**10 + (4*Re**-0.33)**10 + 0.4**10
    return (1/((phi1 + phi2)**-1 + phi3**-1) + phi4)**0.1


def Morrison(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + \frac{2.6Re/5}{1 + \left(\frac{Re}{5}\right)^{1.52}}
        + \frac{0.411 \left(\frac{Re}{263000}\right)^{-7.94}}{1
        + \left(\frac{Re}{263000}\right)^{-8}} + \frac{Re^{0.8}}{461000}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 1E6.

    Examples
    --------
    >>> Morrison(200.)
    0.767731559965325

    References
    ----------
    .. [1] Morrison, Faith A. An Introduction to Fluid Mechanics.
       Cambridge University Press, 2013.
    .. [2] Barati, Reza, Seyed Ali Akbar Salehi Neyshabouri, and Goodarz
       Ahmadi. "Development of Empirical Models with High Accuracy for
       Estimation of Drag Coefficient of Flow around a Smooth Sphere: An
       Evolutionary Approach." Powder Technology 257 (May 2014): 11-19.
       doi:10.1016/j.powtec.2014.02.045.
    '''
    Cd = (24./Re + 2.6*Re/5./(1 + (Re/5.)**1.52) + 0.411*(Re/263000.)**-7.94/(1 + (Re/263000.)**-8)
    + Re**0.8/461000.)
    return Cd


def Song_Xu(Re, sphericity=1., S=1.):
    r'''Calculates drag coefficient of a particle using the method in
    [1]_. Developed with data for spheres, cubes, and cylinders. Claims 3.52%
    relative error for 0.001 < Re < 100 based on 336 tests data.

    .. math::
        C_d = \frac{24}{Re\phi^{0.65}S^{0.3}}\left(1 + 0.35Re\right)^{0.44}

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]
    sphericity : float, optional
        Sphericity of the particle
    S : float, optional
        Ratio of equivalent sphere area and the projected area in the particle
        settling direction [-]

    Returns
    -------
    Cd : float
        Drag coefficient of particle [-]

    Notes
    -----
    Notable as its experimental data and analysis is included in their
    supporting material.

    Examples
    --------
    >>> Song_Xu(30.)
    2.3431335190092444

    References
    ----------
    .. [1] Song, Xianzhi, Zhengming Xu, Gensheng Li, Zhaoyu Pang, and Zhaopeng
       Zhu. "A New Model for Predicting Drag Coefficient and Settling Velocity
       of Spherical and Non-Spherical Particle in Newtonian Fluid." Powder
       Technology 321 (November 2017): 242-50.
       doi:10.1016/j.powtec.2017.08.017.
    '''
    return 24/(Re*sphericity**0.65*S**0.3)*(1+0.35*Re)**0.44


drag_sphere_correlations = {
    'Stokes': (Stokes, None, 0.3),
    'Barati': (Barati, None, 2E5),
    'Barati_high': (Barati_high, None, 1E6),
    'Rouse': (Rouse, None, 2E5),
    'Engelund_Hansen': (Engelund_Hansen, None, 2E5),
    'Clift_Gauvin': (Clift_Gauvin, None, 2E5),
    'Morsi_Alexander': (Morsi_Alexander, None, 2E5),
    'Graf': (Graf, None, 2E5),
    'Flemmer_Banks': (Flemmer_Banks, None, 2E5),
    'Khan_Richardson': (Khan_Richardson, None, 2E5),
    'Swamee_Ojha': (Swamee_Ojha, None, 1.5E5),
    'Yen': (Yen, None, 2E5),
    'Haider_Levenspiel': (Haider_Levenspiel, None, 2E5),
    'Cheng': (Cheng, None, 2E5),
    'Terfous': (Terfous, 0.1, 5E4),
    'Mikhailov_Freire': (Mikhailov_Freire, None, 118300),
    'Clift': (Clift, None, 1E6),
    'Ceylan': (Ceylan, 0.1, 1E6),
    'Almedeij': (Almedeij, None, 1E6),
    'Morrison': (Morrison, None, 1E6),
    'Song_Xu': (Song_Xu, None, 1E3)
}

def drag_sphere_methods(Re, check_ranges=True):
    r'''This function returns a list of methods that can be used to calculate
    the drag coefficient of a sphere.
    Twenty one methods are available, all requiring only the Reynolds number of
    the sphere. Most methods are valid from Re=0 to Re=200,000.

    Examples
    --------
    >>> len(drag_sphere_methods(200))
    20
    >>> len(drag_sphere_methods(200000, check_ranges=False))
    21
    >>> len(drag_sphere_methods(200000, check_ranges=True))
    5

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]
    check_ranges : bool, optional
        Whether to return only correlations claiming to be valid for the given
        `Re` or not, [-]

    Returns
    -------
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate `Cd` with the given `Re`
    '''
    methods = []
    for key, (func, Re_min, Re_max) in drag_sphere_correlations.items():
        if ((Re_min is None or Re > Re_min) and (Re_max is None or Re < Re_max)) or not check_ranges:
            methods.append(key)
    return methods

def drag_sphere(Re, Method=None):
    r'''This function handles calculation of drag coefficient on spheres.
    Twenty methods are available, all requiring only the Reynolds number of the
    sphere. Most methods are valid from Re=0 to Re=200,000. A correlation will
    be automatically selected if none is specified.
    If no correlation is selected, the following rules are used:

        * If Re < 0.01, use Stoke's solution.
        * If 0.01 <= Re < 0.1, linearly combine 'Barati' with Stokes's solution
          such that at Re = 0.1 the solution is 'Barati', and at Re = 0.01 the
          solution is 'Stokes'.
        * If 0.1 <= Re <= ~212963, use the 'Barati' solution.
        * If ~212963 < Re <= 1E6, use the 'Barati_high' solution.
        * For Re > 1E6, raises an exception; no valid results have been found.

    Examples
    --------
    >>> drag_sphere(200)
    0.7682237950389874

    Parameters
    ----------
    Re : float
        Particle Reynolds number of the sphere using the surrounding fluid
        density and viscosity, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        drag_sphere_correlations
    '''
    if Method is None:
        if Re > 0.1:
            # Smooth transition point between the two models
            if Re <= 212963.26847812787:
                return Barati(Re)
            elif Re <= 1E6:
                return Barati_high(Re)
            else:
                raise ValueError('No models implement a solution for Re > 1E6')
        elif Re >= 0.01:
            # Re from 0.01 to 0.1
            ratio = (Re - 0.01)/(0.1 - 0.01)
            # Ensure a smooth transition by linearly switching to Stokes' law
            return ratio*Barati(Re) + (1-ratio)*Stokes(Re)
        else:
            return Stokes(Re)

    if Method == "Stokes":
        return Stokes(Re)
    elif Method == "Barati":
        return Barati(Re)
    elif Method == "Barati_high":
        return Barati_high(Re)
    elif Method == "Rouse":
        return Rouse(Re)
    elif Method == "Engelund_Hansen":
        return Engelund_Hansen(Re)
    elif Method == "Clift_Gauvin":
        return Clift_Gauvin(Re)
    elif Method == "Morsi_Alexander":
        return Morsi_Alexander(Re)
    elif Method == "Graf":
        return Graf(Re)
    elif Method == "Flemmer_Banks":
        return Flemmer_Banks(Re)
    elif Method == "Khan_Richardson":
        return Khan_Richardson(Re)
    elif Method == "Swamee_Ojha":
        return Swamee_Ojha(Re)
    elif Method == "Yen":
        return Yen(Re)
    elif Method == "Haider_Levenspiel":
        return Haider_Levenspiel(Re)
    elif Method == "Cheng":
        return Cheng(Re)
    elif Method == "Terfous":
        return Terfous(Re)
    elif Method == "Mikhailov_Freire":
        return Mikhailov_Freire(Re)
    elif Method == "Clift":
        return Clift(Re)
    elif Method == "Ceylan":
        return Ceylan(Re)
    elif Method == "Almedeij":
        return Almedeij(Re)
    elif Method == "Morrison":
        return Morrison(Re)
    elif Method == "Song_Xu":
        return Song_Xu(Re)
    else:
        raise ValueError('Unrecognized method')


def _v_terminal_err(V, Method, Re_almost, main):
    Cd = drag_sphere(Re_almost*V, Method=Method)
    return V - sqrt(main/Cd)

def v_terminal(D, rhop, rho, mu, Method=None):
    r'''Calculates terminal velocity of a falling sphere using any drag
    coefficient method supported by `drag_sphere`. The laminar solution for
    Re < 0.01 is first tried; if the resulting terminal velocity does not
    put it in the laminar regime, a numerical solution is used.

    .. math::
        v_t = \sqrt{\frac{4 g d_p (\rho_p-\rho_f)}{3 C_D \rho_f }}

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    rhop : float
        Particle density, [kg/m^3]
    rho : float
        Density of the surrounding fluid, [kg/m^3]
    mu : float
        Viscosity of the surrounding fluid [Pa*s]
    Method : string, optional
        A string of the function name to use, as in the dictionary
        drag_sphere_correlations

    Returns
    -------
    v_t : float
        Terminal velocity of falling sphere [m/s]

    Notes
    -----
    As there are no correlations implemented for Re > 1E6, an error will be
    raised if the numerical solver seeks a solution above that limit.

    The laminar solution is given in [1]_ and is:

    .. math::
        v_t = \frac{g d_p^2 (\rho_p - \rho_f)}{18 \mu_f}

    Examples
    --------
    >>> v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3)
    0.004142497244531304

    Example 7-1 in GPSA handbook, 13th edition:

    >>> from scipy.constants import *
    >>> v_terminal(D=150E-6, rhop=31.2*lb/foot**3, rho=2.07*lb/foot**3,  mu=1.2e-05)/foot
    0.4491992020345101

    The answer reported there is 0.46 ft/sec.

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Rushton, Albert, Anthony S. Ward, and Richard G. Holdich.
       Solid-Liquid Filtration and Separation Technology. 1st edition. Weinheim ;
       New York: Wiley-VCH, 1996.
    '''
    '''The following would be the ideal implementation. The actual function is
    optimized for speed, not readability
    def err(V):
        Re = rho*V*D/mu
        Cd = Barati_high(Re)
        V2 = (4/3.*g*D*(rhop-rho)/rho/Cd)**0.5
        return (V-V2)
    return fsolve(err, 1.)'''
    v_lam = g*D*D*(rhop-rho)/(18*mu)
    Re_lam = Reynolds(V=v_lam, D=D, rho=rho, mu=mu)
    if Re_lam < 0.01 or Method == 'Stokes':
        return v_lam

    Re_almost = rho*D/mu
    main = 4/3.*g*D*(rhop-rho)/rho
    V_max = 1E6/rho/D*mu  # where the correlation breaks down, Re=1E6

    # Begin the solver with 1/100 th the velocity possible at the maximum
    # Reynolds number the correlation is good for
    return secant(_v_terminal_err, V_max/100, xtol=1E-12, args=(Method, Re_almost, main))


def time_v_terminal_Stokes(D, rhop, rho, mu, V0, tol=1e-14):
    r'''Calculates the time required for a particle in Stoke's regime only to
    reach terminal velocity (approximately). An infinitely long period is
    required theoretically, but with floating points, it is possible to
    calculate the time required to come within a specified `tol` of that
    terminal velocity.

    .. math::
        t_{term} = -\frac{1}{18\mu}\ln \left(\frac{D^2g\rho - D^2 g \rho_p
        + 18\mu V_{term}}{D^2g\rho - D^2 g \rho_p + 18\mu V_0 } \right) D^2
        \rho_p

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    rhop : float
        Particle density, [kg/m^3]
    rho : float
        Density of the surrounding fluid, [kg/m^3]
    mu : float
        Viscosity of the surrounding fluid [Pa*s]
    V0 : float
        Initial velocity of the particle, [m/s]
    tol : float, optional
        How closely to approach the terminal velocity - the target velocity is
        the terminal velocity multiplied by 1 (+/-) this, depending on if the
        particle is accelerating or decelerating, [-]

    Returns
    -------
    t : float
        Time for the particle to reach the terminal velocity to within the
        specified or an achievable tolerance, [s]

    Notes
    -----
    The symbolic solution was obtained via Wolfram Alpha.

    If a solution cannot be obtained due to floating point error at very high
    tolerance, an exception is raised - but first, the tolerance is doubled,
    up to fifty times in an attempt to obtain the highest possible precision
    while sill giving an answer. If at any point the tolerance is larger than
    1%, an exception is also raised.

    Examples
    --------
    >>> time_v_terminal_Stokes(D=1e-7, rhop=2200., rho=1.2, mu=1.78E-5, V0=1)
    3.1880031137871528e-06
    >>> time_v_terminal_Stokes(D=1e-2, rhop=2200., rho=1.2, mu=1.78E-5, V0=1,
    ... tol=1e-30)
    24800.636391801996
    '''
    if tol < 1e-17:
        tol = 2e-17
    term = D*D*g*rho - D*D*g*rhop
    denominator = term + 18.*mu*V0
    v_term_base = g*D*D*(rhop-rho)/(18.*mu)

    const = D*D*rhop/mu*-1.0/18.
    for i in range(50):
        try:
            if v_term_base < V0:
                v_term = v_term_base*(1.0 + tol)
            else:
                v_term = v_term_base*(1.0 - tol)
            numerator = term + 18.*mu*v_term
            return log(numerator/denominator)*const
        except:
            tol = tol + tol
            if tol > 0.01:
                raise ValueError('Could not find a solution')
    raise ValueError('Could not find a solution')


def integrate_drag_sphere(D, rhop, rho, mu, t, V=0, Method=None,
                          distance=False):
    r'''Integrates the velocity and distance traveled by a particle moving
    at a speed which will converge to its terminal velocity.

    Performs an integration of the following expression for acceleration:

    .. math::
        a = \frac{g(\rho_p-\rho_f)}{\rho_p} - \frac{3C_D \rho_f u^2}{4D \rho_p}

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    rhop : float
        Particle density, [kg/m^3]
    rho : float
        Density of the surrounding fluid, [kg/m^3]
    mu : float
        Viscosity of the surrounding fluid [Pa*s]
    t : float
        Time to integrate the particle to, [s]
    V : float
        Initial velocity of the particle, [m/s]
    Method : string, optional
        A string of the function name to use, as in the dictionary
        drag_sphere_correlations
    distance : bool, optional
        Whether or not to calculate the distance traveled and return it as
        well

    Returns
    -------
    v : float
        Velocity of falling sphere after time `t` [m/s]
    x : float, returned only if `distance` == True
        Distance traveled by the falling sphere in time `t`, [m]

    Notes
    -----
    This can be relatively slow as drag correlations can be complex.

    There are analytical solutions available for the Stokes law regime (Re <
    0.3). They were obtained from Wolfram Alpha. [1]_ was not used in the
    derivation, but also describes the derivation fully.

    .. math::
        V(t) = \frac{\exp(-at) (V_0 a + b(\exp(at) - 1))}{a}

    .. math::
        x(t) = \frac{\exp(-a t)\left[V_0 a(\exp(a t) - 1) + b\exp(a t)(a t-1)
        + b\right]}{a^2}

    .. math::
        a = \frac{18\mu_f}{D^2\rho_p}

    .. math::
        b = \frac{g(\rho_p-\rho_f)}{\rho_p}

    The analytical solution will automatically be used if the initial and
    terminal velocity is show the particle's behavior to be laminar. Note
    that this behavior requires that the terminal velocity of the particle be
    solved for - this adds slight (1%) overhead for the cases where particles
    are not laminar.

    Examples
    --------
    >>> integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.5,
    ... V=30, distance=True)
    (9.686465044053, 7.8294546436299)

    References
    ----------
    .. [1] Timmerman, Peter, and Jacobus P. van der Weele. "On the Rise and
       Fall of a Ball with Linear or Quadratic Drag." American Journal of
       Physics 67, no. 6 (June 1999): 538-46. https://doi.org/10.1119/1.19320.
    '''
    # Delayed import of necessaray functions
    from scipy.integrate import odeint, cumtrapz
    import numpy as np
    laminar_initial = Reynolds(V=V, rho=rho, D=D, mu=mu) < 0.01
    v_laminar_end_assumed = v_terminal(D=D, rhop=rhop, rho=rho, mu=mu, Method=Method)
    laminar_end = Reynolds(V=v_laminar_end_assumed, rho=rho, D=D, mu=mu) < 0.01
    if Method == 'Stokes' or (laminar_initial and laminar_end and Method is None):
        try:
            t1 = 18.0*mu/(D*D*rhop)
            t2 = g*(rhop-rho)/rhop
            V_end = exp(-t1*t)*(t1*V + t2*(exp(t1*t) - 1.0))/t1
            x_end = exp(-t1*t)*(V*t1*(exp(t1*t) - 1.0) + t2*exp(t1*t)*(t1*t - 1.0) + t2)/(t1*t1)
            if distance:
                return V_end, x_end
            else:
                return V_end
        except OverflowError:
            # It is only necessary to integrate to terminal velocity
            t_to_terminal = time_v_terminal_Stokes(D, rhop, rho, mu, V0=V, tol=1e-9)
            if t_to_terminal > t:
                raise ValueError('Should never happen')
            V_end, x_end = integrate_drag_sphere(D=D, rhop=rhop, rho=rho, mu=mu, t=t_to_terminal, V=V, Method='Stokes', distance=True)
            # terminal velocity has been reached - V does not change, but x does
            # No reason to believe this isn't working even though it isn't
            # matching the ode solver
            if distance:
                return V_end, x_end + V_end*(t - t_to_terminal)
            else:
                return V_end

            # This is a serious problem for small diameters
            # It would be possible to step slowly, using smaller increments
            # of time to avlid overflows. However, this unfortunately quickly
            # gets much, exponentially, slower than just using odeint because
            # for example solving 10000 seconds might require steps of .0001
            # seconds at a diameter of 1e-7 meters.
#            x = 0.0
#            subdivisions = 10
#            dt = t/subdivisions
#            for i in range(subdivisions):
#                V, dx = integrate_drag_sphere(D=D, rhop=rhop, rho=rho, mu=mu,
#                                              t=dt, V=V, distance=True,
#                                              Method=Method)
#                x += dx
#            if distance:
#                return V, x
#            else:
#                return V

    Re_ish = rho*D/mu
    c1 = g*(rhop-rho)/rhop
    c2 = -0.75*rho/(D*rhop)

    def dv_dt(V, t):
        if V == 0:
            # 64/Re goes to infinity, but gets multiplied by 0 squared.
            t2 = 0.0
        else:
#            t2 = c2*V*V*Stokes(Re_ish*V)
            t2 = c2*V*V*drag_sphere(float(Re_ish*V), Method=Method)
        return c1 + t2

    # Number of intervals for the solution to be solved for; the integrator
    # doesn't care what we give it, but a large number of intervals are needed
    # For an accurate integration of the particle's distance traveled
    pts = 1000 if distance else 2
    ts = np.linspace(0, t, pts)


    # Perform the integration
    Vs = odeint(dv_dt, [V], ts)
    #
    V_end = float(Vs[-1])
    if distance:
        # Calculate the distance traveled
        x = float(cumtrapz(np.ravel(Vs), ts)[-1])
        return V_end, x
    else:
        return V_end
