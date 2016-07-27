# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

#from __future__ import division
from math import exp

__all__ = ['Thom', 'Zivi', 'Smith', 'Fauske', 'Chisholm', 'Turner_Wallis',
           'homogeneous']

### Models based on slip ratio

def Thom(x, rhol, rhog, mul, mug):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_ as given in [2]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)\left(\frac{\rho_g}
        {\rho_l}\right)^{0.89}\left(\frac{\mu_l}{\mu_g}\right)^{0.18}
        \right]^{-1}
        
    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    Based on experimental data for boiling of water. [3]_ presents a slightly
    different model. However, its results are quite similar, as may be 
    compared as follows. Neither expression was aparent from a brief review
    of [1]_.
    
    >>> from sympy import *
    >>> x, rhol, rhog, mug, mul = symbols('x, rhol, rhog, mug, mul')
    >>> Z = (rhol/rhog)**Rational(555,1000)*(mug/mul)**Rational(111,1000)
    >>> gamma = Z**1.6
    >>> alpha = (gamma*x/(1 + x*(gamma-1)))
    >>> alpha
    x*((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6/(x*(((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6 - 1) + 1)
    >>> alpha.subs([(x, .4), (rhol, 800), (rhog, 2.5), (mul, 1E-3), (mug, 1E-5)])
    0.980138792146901
    
    Examples
    --------
    >>> Thom(.4, 800, 2.5, 1E-3, 1E-5)
    0.9801482164042417

    References
    ----------
    .. [1] Thom, J. R. S. "Prediction of Pressure Drop during Forced 
       Circulation Boiling of Water." International Journal of Heat and Mass 
       Transfer 7, no. 7 (July 1, 1964): 709-24. 
       doi:10.1016/0017-9310(64)90002-X.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    return (1 + (1-x)/x * (rhog/rhol)**0.89 * (mul/mug)**0.18)**-1
#    return x*((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6/(x*(((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6 - 1) + 1)


def Zivi(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)
        \left(\frac{\rho_g}{\rho_l}\right)^{2/3}\right]^{-1}
        
    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    Based on experimental data for boiling of water.
    More complicated variants of this are also in [1]_.
    
    Examples
    --------
    >>> Zivi(.4, 800, 2.5)
    0.9689339909056356

    References
    ----------
    .. [1] Zivi, S. M. "Estimation of Steady-State Steam Void-Fraction by Means
       of the Principle of Minimum Entropy Production." Journal of Heat 
       Transfer 86, no. 2 (May 1, 1964): 247-51. doi:10.1115/1.3687113.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    return (1 + (1-x)/x * (rhog/rhol)**(2/3.))**-1


def Smith(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_, also given in [2]_ and [3]_.

    .. math::
        \alpha = \left\{1 + \left(\frac{1-x}{x}\right)
        \left(\frac{\rho_g}{\rho_l}\right)\left[K+(1-K)
        \sqrt{\frac{\frac{\rho_l}{\rho_g} + K\left(\frac{1-x}{x}\right)}
        {1 + K\left(\frac{1-x}{x}\right)}}\right] \right\}^{-1}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ is an easy to read paper and has been reviewed.
    The form of the expression here is rearanged somewhat differently
    than in [1]_ but has been verified to be numerically equivalent. The form
    of this in [3]_ is missing a square root on a bracketed term; this appears
    in multiple papers by the authors.

    Examples
    --------
    >>> Smith(.4, 800, 2.5)
    0.959981235534199

    References
    ----------
    .. [1] Smith, S. L. "Void Fractions in Two-Phase Flow: A Correlation Based 
       upon an Equal Velocity Head Model." Proceedings of the Institution of 
       Mechanical Engineers 184, no. 1 (June 1, 1969): 647-64. 
       doi:10.1243/PIME_PROC_1969_184_051_02.  
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    K = 0.4
    x_ratio = (1-x)/x
    root = ((rhol/rhog + K*x_ratio) / (1 + K*x_ratio))**0.5
    alpha = (1 + (x_ratio) * (rhog/rhol) * (K + (1-K)*root))**-1
    return alpha


def Fauske(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_, as given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)
        \left(\frac{\rho_g}{\rho_l}\right)^{0.5}\right]^{-1}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has not been reviewed. However, both [2]_ and [3]_ present it the
    same way.

    Examples
    --------
    >>> Fauske(.4, 800, 2.5)
    0.9226347262627932

    References
    ----------
    .. [1] Fauske, H., Critical two-phase, steam-water flows, in: Heat Transfer
       and Fluid Mechanics Institute 1961: Proceedings. Stanford University 
       Press, 1961, p. 79-89.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    return (1 + (1-x)/x*(rhog/rhol)**0.5)**-1


def Chisholm(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_, as given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)\left(\frac{\rho_g}
        {\rho_l}\right)\sqrt{1 - x\left(1-\frac{\rho_l}{\rho_g}\right)}
        \right]^{-1}
        
    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has not been reviewed. However, both [2]_ and [3]_ present it the
    same way.

    Examples
    --------
    >>> Chisholm(.4, 800, 2.5)
    0.949525900374774

    References
    ----------
    .. [1] Chisholm, D. "Pressure Gradients due to Friction during the Flow of 
       Evaporating Two-Phase Mixtures in Smooth Tubes and Channels." 
       International Journal of Heat and Mass Transfer 16, no. 2 (February 1, 
       1973): 347-58. doi:10.1016/0017-9310(73)90063-X.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    S = (1 - x*(1-rhol/rhog))**0.5
    alpha = (1 + (1-x)/x*rhog/rhol*S)**-1
    return alpha


def Turner_Wallis(x, rhol, rhog, mul, mug):
    r'''Calculates void fraction in two-phase flow according to the model of 
    [1]_, as given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)^{0.72}\left(\frac{\rho_g}
        {\rho_l}\right)^{0.4}\left(\frac{\mu_l}{\mu_g}\right)^{0.08}
        \right]^{-1}
        
    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has not been reviewed. However, both [2]_ and [3]_ present it the
    same way, if slightly differently rearranged.

    Examples
    --------
    >>> Turner_Wallis(.4, 800, 2.5, 1E-3, 1E-5)
    0.8384824581634625

    References
    ----------
    .. [1] J.M. Turner, G.B. Wallis, The Separate-cylinders Model of Two-phase 
       Flow, NYO-3114-6, Thayer's School Eng., Dartmouth College, Hanover, New 
       Hampshire, USA, 1965.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    return (1 + ((1-x)/x)**0.72 * (rhog/rhol)**0.4 * (mul/mug)**0.08)**-1
 
 
### Models using the Homogeneous flow model
 
 
def homogeneous(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the homogeneous
    flow model, refiewed in [1]_, [2]_, and [3]_. 

    .. math::
        \alpha = \frac{1}{1 + \left(\frac{1-x}{x}\right)\frac{\rho_g}{\rho_l}}
        
    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----

    Examples
    --------
    >>> homogeneous(.4, 800, 2.5)
    0.995334370139969

    References
    ----------
    .. [1] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no. 
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032. 
    .. [2] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of 
       Void Fraction Models on the Two-Phase Friction Factor of R134a during 
       Condensation in Vertical Downward Flow in a Smooth Tube." International 
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008): 
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void 
       Fraction Correlations for Different Flow Patterns in Horizontal and 
       Upward Inclined Pipes." International Journal of Multiphase Flow 33, 
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    return 1./(1 + (1-x)/x*(rhog/rhol))
