# -*- coding: utf-8 -*-
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

This module contains correlations for calculating the saltation velocity of
entrained particles.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Correlations
------------
.. autofunction :: Rizk
.. autofunction :: Matsumoto_1974
.. autofunction :: Matsumoto_1975
.. autofunction :: Matsumoto_1977
.. autofunction :: Schade
.. autofunction :: Weber_saltation
.. autofunction :: Geldart_Ling

"""

from __future__ import division
from fluids.constants import g, pi
from math import sqrt

__all__ = ['Rizk', 'Matsumoto_1974', 'Matsumoto_1975', 'Matsumoto_1977',
'Schade', 'Weber_saltation', 'Geldart_Ling']


def Rizk(mp, dp, rhog, D):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_ as described in [2]_ and many others.

    .. math::
        \mu=\left(\frac{1}{10^{1440d_p+1.96}}\right)\left(Fr_s\right)^{1100d_p+2.5}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.

    Examples
    --------
    Example is from [3]_.

    >>> Rizk(mp=0.25, dp=100E-6, rhog=1.2, D=.078)
    9.8833092829357

    References
    ----------
    .. [1] Rizk, F. "Pneumatic conveying at optimal operation conditions and a
       solution of Bath's equation." Proceedings of Pneumotransport 3,
       paper D4. BHRA Fluid Engineering, Cranfield, England (1973)
    .. [2] Klinzing, G. E., F. Rizk, R. Marcus, and L. S. Leung. Pneumatic
       Conveying of Solids: A Theoretical and Practical Approach.
       Springer, 2013.
    .. [3] Rhodes, Martin J. Introduction to Particle Technology. Wiley, 2013.
    '''
    alpha = 1440.0*dp + 1.96
    beta = 1100.0*dp + 2.5
    term1 = 0.1**alpha
    Frs_sorta = 1.0/sqrt(g*D)
    expression1 = term1*Frs_sorta**beta
    expression2 = mp/rhog/(pi/4*D*D)
    return (expression2/expression1)**(1./(1. + beta))


def Matsumoto_1974(mp, rhop, dp, rhog, D, Vterminal=1):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_. Also described in [2]_.

    .. math::
        \mu = 0.448 \left(\frac{\rho_p}{\rho_f}\right)^{0.50}\left(\frac{Fr_p}
        {10}\right)^{-1.75}\left(\frac{Fr_s}{10}\right)^{3}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        Fr_p = \frac{V_{terminal}}{\sqrt{gd_p}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhop : float
        Particle density, [kg/m^3]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    Vterminal : float
        Terminal velocity of particle settling in gas, [m/s]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.
    Result looks high, something may be wrong.
    For particles > 0.3 mm.

    Examples
    --------
    >>> Matsumoto_1974(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    19.583617317317895

    References
    ----------
    .. [1] Matsumoto, Shigeru, Michio Kara, Shozaburo Saito, and Siro Maeda.
       "Minimum Transport Velocity for Horizontal Pneumatic Conveying."
       Journal of Chemical Engineering of Japan 7, no. 6 (1974): 425-30.
       doi:10.1252/jcej.7.425.
    .. [2] Jones, Peter J., and L. S. Leung. "A Comparison of Correlations for
       Saltation Velocity in Horizontal Pneumatic Conveying." Industrial &
       Engineering Chemistry Process Design and Development 17, no. 4
       (October 1, 1978): 571-75. doi:10.1021/i260068a031
    '''
    A = pi/4*D**2
    Frp = Vterminal/sqrt(g*dp)
    Frs_sorta = 1./sqrt(g*D)
    expression1 = 0.448*sqrt(rhop/rhog)*(Frp/10.)**-1.75*(Frs_sorta/10.)**3
    expression2 = mp/rhog/A
    return (expression2/expression1)**(1/4.)


def Matsumoto_1975(mp, rhop, dp, rhog, D, Vterminal=1):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_. Also described in [2]_.

    .. math::
        \mu = 1.11 \left(\frac{\rho_p}{\rho_f}\right)^{0.55}\left(\frac{Fr_p}
        {10}\right)^{-2.3}\left(\frac{Fr_s}{10}\right)^{3}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        Fr_p = \frac{V_{terminal}}{\sqrt{gd_p}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhop : float
        Particle density, [kg/m^3]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    Vterminal : float
        Terminal velocity of particle settling in gas, [m/s]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.
    Result looks high, something may be wrong.
    For particles > 0.3 mm.

    Examples
    --------
    >>> Matsumoto_1975(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    18.04523091703009

    References
    ----------
    .. [1] Matsumoto, Shigeru, Shundo Harada, Shozaburo Saito, and Siro Maeda.
       "Saltation Velocity for Horizontal Pneumatic Conveying." Journal of
       Chemical Engineering of Japan 8, no. 4 (1975): 331-33.
       doi:10.1252/jcej.8.331.
    .. [2] Jones, Peter J., and L. S. Leung. "A Comparison of Correlations for
       Saltation Velocity in Horizontal Pneumatic Conveying." Industrial &
       Engineering Chemistry Process Design and Development 17, no. 4
       (October 1, 1978): 571-75. doi:10.1021/i260068a031
    '''
    A = pi/4*D**2
    Frp = Vterminal/sqrt(g*dp)
    Frs_sorta = 1./sqrt(g*D)
    expression1 = 1.11*(rhop/rhog)**0.55*(Frp/10.)**-2.3*(Frs_sorta/10.)**3
    expression2 = mp/rhog/A
    return (expression2/expression1)**(1/4.)


def Matsumoto_1977(mp, rhop, dp, rhog, D, Vterminal=1):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_ and reproduced in [2]_, [3]_, and [4]_.

    First equation is used if third equation yields d* higher than dp.
    Otherwise, use equation 2.

    .. math::
        \mu = 5560\left(\frac{d_p}{D}\right)^{1.43}\left(\frac{Fr_s}{10}\right)^4

    .. math::
        \mu = 0.373 \left(\frac{\rho_p}{\rho_f}\right)^{1.06}\left(\frac{Fr_p}
        {10}\right)^{-3.7}\left(\frac{Fr_s}{10}\right)^{3.61}

    .. math::
        \frac{d_p^*}{D} = 1.39\left(\frac{\rho_p}{\rho_f}\right)^{-0.74}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        Fr_p = \frac{V_{terminal}}{\sqrt{gd_p}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhop : float
        Particle density, [kg/m^3]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    Vterminal : float
        Terminal velocity of particle settling in gas, [m/s]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearanged to be explicit in terms of saltation velocity
    internally.r

    Examples
    --------
    Example is only a self-test.

    Course routine, terminal velocity input is from example in [2].

    >>> Matsumoto_1977(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    16.64284834446686

    References
    ----------
    .. [1] Matsumoto, Shigeru, Makoto Kikuta, and Siro Maeda. "Effect of
       Particle Size on the Minimum Transport Velocity for Horizontal Pneumatic
       Conveying of Solids." Journal of Chemical Engineering of Japan 10,
       no. 4 (1977): 273-79. doi:10.1252/jcej.10.273.
    .. [2] Klinzing, G. E., F. Rizk, R. Marcus, and L. S. Leung. Pneumatic
       Conveying of Solids: A Theoretical and Practical Approach.
       Springer, 2013.
    .. [3] Gomes, L. M., and A. L. Amarante Mesquita. "On the Prediction of
       Pickup and Saltation Velocities in Pneumatic Conveying." Brazilian
       Journal of Chemical Engineering 31, no. 1 (March 2014): 35-46.
       doi:10.1590/S0104-66322014000100005
    .. [4] Rabinovich, Evgeny, and Haim Kalman. "Threshold Velocities of
       Particle-Fluid Flows in Horizontal Pipes and Ducts: Literature Review."
       Reviews in Chemical Engineering 27, no. 5-6 (January 1, 2011).
       doi:10.1515/REVCE.2011.011.
    '''
    limit = 1.39*D*(rhop/rhog)**-0.74
    A = pi/4*D**2
    if limit < dp:
        # Coarse routine
        Frp = Vterminal/sqrt(g*dp)
        Frs_sorta = 1./sqrt(g*D)
        expression1 = 0.373*(rhop/rhog)**1.06*(Frp/10.)**-3.7*(Frs_sorta/10.)**3.61
        expression2 = mp/rhog/A
        return (expression2/expression1)**(1/4.61)
    else:
        Frs_sorta = 1./sqrt(g*D)
        expression1 = 5560*(dp/D)**1.43*(Frs_sorta/10.)**4
        expression2 = mp/rhog/A
        return (expression2/expression1)**(0.2)


def Schade(mp, rhop, dp, rhog, D):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_ as described in [2]_, [3]_, [4]_, and [5]_.

    .. math::
        Fr_s = \mu^{0.11}\left(\frac{D}{d_p}\right)^{0.025}\left(\frac{\rho_p}
        {\rho_f}\right)^{0.34}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhop : float
        Particle density, [kg/m^3]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.

    Examples
    --------
    >>> Schade(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1)
    13.697415809497912

    References
    ----------
    .. [1] Schade, B., Zum Ubergang Sprung-Strahnen-forderung bei der
       Horizontalen Pneumatischen Feststoffordrung. Dissertation, University of
       Karlsruche (1987)
    .. [2] Rabinovich, Evgeny, and Haim Kalman. "Threshold Velocities of
       Particle-Fluid Flows in Horizontal Pipes and Ducts: Literature Review."
       Reviews in Chemical Engineering 27, no. 5-6 (January 1, 2011).
       doi:10.1515/REVCE.2011.011.
    .. [3] Setia, G., S. S. Mallick, R. Pan, and P. W. Wypych. "Modeling
       Minimum Transport Boundary for Fluidized Dense-Phase Pneumatic Conveying
       Systems." Powder Technology 277 (June 2015): 244-51.
       doi:10.1016/j.powtec.2015.02.050.
    .. [4] Bansal, A., S. S. Mallick, and P. W. Wypych. "Investigating
       Straight-Pipe Pneumatic Conveying Characteristics for Fluidized
       Dense-Phase Pneumatic Conveying." Particulate Science and Technology
       31, no. 4 (July 4, 2013): 348-56. doi:10.1080/02726351.2012.732677.
    .. [5] Gomes, L. M., and A. L. Amarante Mesquita. "On the Prediction of
       Pickup and Saltation Velocities in Pneumatic Conveying." Brazilian
       Journal of Chemical Engineering 31, no. 1 (March 2014): 35-46.
       doi:10.1590/S0104-66322014000100005
    '''
    B = (D/dp)**0.025*(rhop/rhog)**0.34
    A = sqrt(g*D)
    C = mp/(rhog*pi/4*D**2)
    return (C**0.11*B*A)**(1/1.11)


def Weber_saltation(mp, rhop, dp, rhog, D, Vterminal=4):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_ as described in [2]_, [3]_, [4]_, and [5]_.

    If Vterminal is under 3 m/s, use equation 1; otherwise, equation 2.

    .. math::
        Fr_s = \left(7 + \frac{8}{3}V_{terminal}\right)\mu^{0.25}
        \left(\frac{d_p}{D}\right)^{0.1}

    .. math::
        Fr_s = 15\mu^{0.25}\left(\frac{d_p}{D}\right)^{0.1}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhop : float
        Particle density, [kg/m^3]
    dp : float
        Particle diameter, [m]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    Vterminal : float
        Terminal velocity of particle settling in gas, [m/s]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.

    Examples
    --------
    Examples are only a self-test.

    >>> Weber_saltation(mp=1, rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=4)
    15.227445436331474

    References
    ----------
    .. [1] Weber, M. 1981. Principles of hydraulic and pneumatic conveying in
       pipes. Bulk Solids Handling 1: 57-63.
    .. [2] Rabinovich, Evgeny, and Haim Kalman. "Threshold Velocities of
       Particle-Fluid Flows in Horizontal Pipes and Ducts: Literature Review."
       Reviews in Chemical Engineering 27, no. 5-6 (January 1, 2011).
       doi:10.1515/REVCE.2011.011.
    .. [3] Setia, G., S. S. Mallick, R. Pan, and P. W. Wypych. "Modeling
       Minimum Transport Boundary for Fluidized Dense-Phase Pneumatic Conveying
       Systems." Powder Technology 277 (June 2015): 244-51.
       doi:10.1016/j.powtec.2015.02.050.
    .. [4] Bansal, A., S. S. Mallick, and P. W. Wypych. "Investigating
       Straight-Pipe Pneumatic Conveying Characteristics for Fluidized
       Dense-Phase Pneumatic Conveying." Particulate Science and Technology
       31, no. 4 (July 4, 2013): 348-56. doi:10.1080/02726351.2012.732677.
    .. [5] Gomes, L. M., and A. L. Amarante Mesquita. "On the Prediction of
       Pickup and Saltation Velocities in Pneumatic Conveying." Brazilian
       Journal of Chemical Engineering 31, no. 1 (March 2014): 35-46.
       doi:10.1590/S0104-66322014000100005
    '''
    if Vterminal <= 3:
        term1 = (7 + 8/3.*Vterminal)*(dp/D)**0.1
    else:
        term1 = 15.*(dp/D)**0.1
    term2 = 1./sqrt(g*D)
    term3 = mp/rhog/(pi/4*D**2)
    return (term1/term2*sqrt(sqrt(term3)))**(1/1.25)


def Geldart_Ling(mp, rhog, D, mug):
    r'''Calculates saltation velocity of the gas for pneumatic conveying,
    according to [1]_ as described in [2]_ and [3]_.

    if Gs/D < 47000, use equation 1, otherwise use equation 2.

    .. math::
        V_{salt} = 1.5G_s^{0.465}D^{-0.01} \mu^{0.055}\rho_f^{-0.42}

    .. math::
        V_{salt} = 8.7G_s^{0.302}D^{0.153} \mu^{0.055}\rho_f^{-0.42}

    .. math::
        Fr_s = 15\mu^{0.25}\left(\frac{d_p}{D}\right)^{0.1}

    .. math::
        Fr_s = \frac{V_{salt}}{\sqrt{gD}}

    .. math::
        \mu = \frac{m_p}{\frac{\pi}{4}D^2V \rho_f}

    .. math::
        G_s = \frac{m_p}{A}

    Parameters
    ----------
    mp : float
        Solid mass flow rate, [kg/s]
    rhog : float
        Gas density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    mug : float
        Gas viscosity, [Pa*s]

    Returns
    -------
    V : float
        Saltation velocity of gas, [m/s]

    Notes
    -----
    Model is rearranged to be explicit in terms of saltation velocity
    internally.

    Examples
    --------
    >>> Geldart_Ling(1., 1.2, 0.1, 2E-5)
    7.467495862402707

    References
    ----------
    .. [1] Weber, M. 1981. Principles of hydraulic and pneumatic conveying in
       pipes. Bulk Solids Handling 1: 57-63.
    .. [2] Rabinovich, Evgeny, and Haim Kalman. "Threshold Velocities of
       Particle-Fluid Flows in Horizontal Pipes and Ducts: Literature Review."
       Reviews in Chemical Engineering 27, no. 5-6 (January 1, 2011).
       doi:10.1515/REVCE.2011.011.
    .. [3] Gomes, L. M., and A. L. Amarante Mesquita. "On the Prediction of
       Pickup and Saltation Velocities in Pneumatic Conveying." Brazilian
       Journal of Chemical Engineering 31, no. 1 (March 2014): 35-46.
       doi:10.1590/S0104-66322014000100005
    '''
    Gs = mp/(0.25*pi*D*D)
    if Gs/D <= 47000.0:
        return 1.5*Gs**0.465*D**-0.01*mug**0.055*rhog**-0.42
    else:
        return 8.7*Gs**0.302*D**0.153*mug**0.055*rhog**-0.42

