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
from math import exp, log, log10, tanh
from scipy.constants import g

__all__ = ['Stokes', 'Barati', 'Barati_high', 'Rouse', 'Engelund_Hansen',
'Clift_Gauvin', 'Morsi_Alexander', 'Graf', 'Flemmer_Banks', 'Khan_Richardson',
'Swamee_Ojha', 'Yen', 'Haider_Levenspiel', 'Cheng', 'Terfous',
'Mikhailov_Freire', 'Clift', 'Ceylan', 'Almedeij', 'Morrison']

def Stokes(Re):
    r'''Calculates drag coefficient of a smooth sphere using Stoke's law.

    .. math::
        C_D = 24/Re

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re
    return Cd


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
        Reynolds number of the sphere, [-]

    Returns
    -------
    Cd : float
        Drag coefficient [-]

    Notes
    -----
    Range is Re <= 2E5

    Examples
    --------
    Maching example in [1]_, in a table of calculated values.

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
    Cd = (5.4856E9*tanh(4.3774E-9/Re) + 0.0709*tanh(700.6574/Re)
    + 0.3894*tanh(74.1539/Re) - 0.1198*tanh(7429.0843/Re)
    + 1.7174*tanh(9.9851/(Re+2.3384)) + 0.4744)
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
        Reynolds number of the sphere, [-]

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
    Maching example in [1]_, in a table of calculated values.

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
    Cd = (8E-6*((Re/6530.)**2 + tanh(Re) - 8*log(Re)/log(10.))
    - 0.4119*exp(-2.08E43/(Re+Re**2)**4)
    - 2.1344*exp(-((log(Re**2 + 10.7563)/log(10))**2 + 9.9867)/Re)
    + 0.1357*exp(-((Re/1620.)**2 + 10370.)/Re)
    - 8.5E-3*(2*log(tanh(tanh(Re)))/log(10) - 2825.7162)/Re + 2.4795)
    return Cd


def Rouse(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + \frac{3}{Re^{0.5}} + 0.34

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re + 3/Re**0.5 + 0.34
    return Cd


def Engelund_Hansen(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + 1.5

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re + 1.5
    return Cd


def Clift_Gauvin(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re}(1 + 0.152Re^{0.677}) + \frac{0.417}
        {1 + 5070Re^{-0.94}}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re*(1 + 0.152*Re**0.677) + 0.417/(1 + 5070*Re**-0.94)
    return Cd


def Morsi_Alexander(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \left\{ \begin{array}{ll}
        \frac{24}{Re} & \mbox{if $Re < 0.1$}\\
        \frac{22.73}{Re}+\frac{0.0903}{Re^2} + 3.69 & \mbox{if $0.1 < Re < 1$}\\
        \frac{29.1667}{Re}-\frac{3.8889}{Re^2} + 1.2220 & \mbox{if $1 < Re < 10$}\\
        \frac{46.5}{Re}-\frac{116.67}{Re^2} + 0.6167 & \mbox{if $10 < Re < 100$}\\
        \frac{98.33}{Re}-\frac{2778}{Re^2} + 0.3644 & \mbox{if $100 < Re < 1000$}\\
        \frac{148.62}{Re}-\frac{4.75\times10^4}{Re^2} + 0.3570 & \mbox{if $1000 < Re < 5000$}\\
        \frac{-490.5460}{Re}+\frac{57.87\times10^4}{Re^2} + 0.46 & \mbox{if $5000 < Re < 10000$}\\
        \frac{-1662.5}{Re}+\frac{5.4167\times10^6}{Re^2} + 0.5191 & \mbox{if $10000 < Re < 50000$}\end{array} \right.

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
        Cd = 24./Re
    elif Re < 1:
        Cd = 22.73/Re + 0.0903/Re**2 + 3.69
    elif Re < 10:
        Cd = 29.1667/Re - 3.8889/Re**2 + 1.222
    elif Re < 100:
        Cd = 46.5/Re - 116.67/Re**2 + 0.6167
    elif Re < 1000:
        Cd = 98.33/Re - 2778./Re**2 + 0.3644
    elif Re < 5000:
        Cd = 148.62/Re - 4.75E4/Re**2 + 0.357
    elif Re < 10000:
        Cd = -490.546/Re + 57.87E4/Re**2 + 0.46
    else:
        Cd = -1662.5/Re + 5.4167E6/Re**2 + 0.5191
    return Cd


def Graf(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re} + \frac{7.3}{1+Re^{0.5}} + 0.25

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re + 7.3/(1 + Re**0.5) + 0.25
    return Cd


def Flemmer_Banks(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{24}{Re}10^E

        E = 0.383Re^{0.356}-0.207Re^{0.396} - \frac{0.143}{1+(\log_{10} Re)^2}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re*10**E
    return Cd


def Khan_Richardson(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = (2.49Re^{-0.328} + 0.34Re^{0.067})^{3.18}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = (2.49*Re**-0.328 + 0.34*Re**0.067)**3.18
    return Cd


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
        Reynolds number of the sphere, [-]

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
    Cd = 0.5*(16*((24./Re)**1.6 + (130./Re)**0.72)**2.5 + ((40000./Re)**2 + 1)**-0.25)**0.25
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
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re*(1 + 0.15*Re**0.5 + 0.017*Re) - 0.208/(1 + 1E4*Re**-0.5)
    return Cd


def Haider_Levenspiel(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D=\frac{24}{Re}(1+0.1806Re^{0.6459})+\left(\frac{0.4251}{1
        +\frac{6880.95}{Re}}\right)

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re*(1 + 0.1806*Re**0.6459) + (0.4251/(1 + 6880.95/Re))
    return Cd


def Cheng(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D=\frac{24}{Re}(1+0.27Re)^{0.43}+0.47[1-\exp(-0.04Re^{0.38})]

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 24./Re*(1 + 0.27*Re)**0.43 + 0.47*(1 - exp(-0.04*Re**0.38))
    return Cd


def Terfous(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = 2.689 + \frac{21.683}{Re} + \frac{0.131}{Re^2}
        - \frac{10.616}{Re^{0.1}} + \frac{12.216}{Re^{0.2}}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = 2.689 + 21.683/Re + 0.131/Re**2 - 10.616/Re**0.1 + 12.216/Re**0.2
    return Cd


def Mikhailov_Freire(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \frac{3808[(1617933/2030) + (178861/1063)Re + (1219/1084)Re^2]}
        {681Re[(77531/422) + (13529/976)Re - (1/71154)Re^2]}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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

    .. math::
        C_D = \left\{ \begin{array}{ll}
        \frac{24}{Re} + \frac{3}{16} & \mbox{if $Re < 0.01$}\\
        \frac{24}{Re}(1 + 0.1315Re^{0.82 - 0.05\log Re}) & \mbox{if $0.01 < Re < 20$}\\
        \frac{24}{Re}(1 + 0.1935Re^{0.6305}) & \mbox{if $20 < Re < 260$}\\
        10^{[1.6435 - 1.1242\log Re + 0.1558[\log Re]^2} & \mbox{if $260 < Re < 1500$}\\
        10^{[-2.4571 + 2.5558\log Re - 0.9295[\log Re]^2 + 0.1049[\log Re]^3} & \mbox{if $1500 < Re < 12000$}\\
        10^{[-1.9181 + 0.6370\log Re - 0.0636[\log Re]^2} & \mbox{if $12000 < Re < 44000$}\\
        10^{[-4.3390 + 1.5809\log Re - 0.1546[\log Re]^2} & \mbox{if $44000 < Re < 338000$}\\
        9.78 - 5.3\log Re & \mbox{if $338000 < Re < 400000$}\\
        0.19\log Re - 0.49 & \mbox{if $400000 < Re < 1000000$}\end{array} \right.

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
        Cd = 24./Re + 3/16.
    elif Re < 20:
        Cd = 24./Re*(1 + 0.1315*Re**(0.82 - 0.05*log10(Re)))
    elif Re < 260:
        Cd = 24./Re*(1 + 0.1935*Re**(0.6305))
    elif Re < 1500:
        Cd = 10**(1.6435 - 1.1242*log10(Re) + 0.1558*(log10(Re))**2)
    elif Re < 12000:
        Cd = 10**(-2.4571 + 2.5558*log10(Re) - 0.9295*(log10(Re))**2 + 0.1049*log10(Re)**3)
    elif Re < 44000:
        Cd = 10**(-1.9181 + 0.6370*log10(Re) - 0.0636*(log10(Re))**2)
    elif Re < 338000:
        Cd = 10**(-4.3390 + 1.5809*log10(Re) - 0.1546*(log10(Re))**2)
    elif Re < 400000:
        Cd = 29.78 - 5.3*log10(Re)
    else:
        Cd = 0.19*log10(Re) - 0.49
    return Cd


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
        Reynolds number of the sphere, [-]

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
    Cd = (1 - 0.5*exp(0.182) + 10.11*Re**(-2/3.)*exp(0.952*Re**-0.25)
    - 0.03859*Re**(-4/3.)*exp(1.30*Re**-0.5) + 0.037E-4*Re*exp(-0.125E-4*Re)
    - 0.116E-10*Re**2*exp(-0.444E-5*Re))
    return Cd


def Almedeij(Re):
    r'''Calculates drag coefficient of a smooth sphere using the method in
    [1]_ as described in [2]_.

    .. math::
        C_D = \left[\frac{1}{(\phi_1 + \phi_2)^{-1} + (\phi_3)^{-1}} + \phi_4\right]^{0.1}

        \phi_1 = (24Re^{-1})^{10} + (21Re^{-0.67})^{10} + (4Re^{-0.33})^{10} + 0.4^{10}

        \phi_2 = \left[(0.148Re^{0.11})^{-10} + (0.5)^{-10}\right]^{-1}

        \phi_3 = (1.57\times10^8Re^{-1.625})^{10}

        \phi_4 = \left[(6\times10^{-17}Re^{2.63})^{-10} + (0.2)^{-10}\right]^{-1}

    Parameters
    ----------
    Re : float
        Reynolds number of the sphere, [-]

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
    Cd = (1/((phi1 + phi2)**-1 + phi3**-1) + phi4)**0.1
    return Cd


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
        Reynolds number of the sphere, [-]

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

