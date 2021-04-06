# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains functions for calculating void fraction/holdup in
two-phase flow. This is an important parameter for predicting pressure drop.
Also included are empirical "two phase viscosity" definitions which do not
have a physical meaning but are often used in pressure drop correlations.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Interfaces
----------
.. autofunction:: liquid_gas_voidage
.. autofunction:: liquid_gas_voidage_methods
.. autofunction:: density_two_phase
.. autofunction:: gas_liquid_viscosity
.. autofunction:: gas_liquid_viscosity_methods

Void Fraction/Holdup Correlations
---------------------------------
.. autofunction:: Thom
.. autofunction:: Zivi
.. autofunction:: Smith
.. autofunction:: Fauske
.. autofunction:: Chisholm_voidage
.. autofunction:: Turner_Wallis
.. autofunction:: homogeneous
.. autofunction:: Chisholm_Armand
.. autofunction:: Armand
.. autofunction:: Nishino_Yamazaki
.. autofunction:: Guzhov
.. autofunction:: Kawahara
.. autofunction:: Baroczy
.. autofunction:: Tandon_Varma_Gupta
.. autofunction:: Harms
.. autofunction:: Domanski_Didion
.. autofunction:: Graham
.. autofunction:: Yashar
.. autofunction:: Huq_Loth
.. autofunction:: Kopte_Newell_Chato
.. autofunction:: Steiner
.. autofunction:: Rouhani_1
.. autofunction:: Rouhani_2
.. autofunction:: Nicklin_Wilkes_Davidson
.. autofunction:: Gregory_Scott
.. autofunction:: Dix
.. autofunction:: Sun_Duffey_Peng
.. autofunction:: Xu_Fang_voidage
.. autofunction:: Woldesemayat_Ghajar

Utilities
---------
.. autofunction:: Lockhart_Martinelli_Xtt
.. autofunction:: two_phase_voidage_experimental

Gas/Liquid Viscosity
--------------------
.. autofunction:: Beattie_Whalley
.. autofunction:: McAdams
.. autofunction:: Cicchitti
.. autofunction:: Lin_Kwok
.. autofunction:: Fourar_Bories
.. autofunction:: Duckler

"""

from __future__ import division
from math import exp, log, pi, sin, cos, radians, sqrt
from fluids.constants import g
from fluids.core import Froude

__all__ = ['Thom', 'Zivi', 'Smith', 'Fauske', 'Chisholm_voidage', 'Turner_Wallis',
           'homogeneous', 'Chisholm_Armand', 'Armand', 'Nishino_Yamazaki',
           'Guzhov', 'Kawahara', 'Baroczy', 'Tandon_Varma_Gupta', 'Harms',
           'Domanski_Didion', 'Graham', 'Yashar', 'Huq_Loth',
           'Kopte_Newell_Chato', 'Steiner', 'Rouhani_1', 'Rouhani_2',
           'Nicklin_Wilkes_Davidson', 'Gregory_Scott', 'Dix',
           'Sun_Duffey_Peng', 'Xu_Fang_voidage', 'Woldesemayat_Ghajar',
           'Lockhart_Martinelli_Xtt', 'two_phase_voidage_experimental',
           'density_two_phase', 'Beattie_Whalley', 'McAdams', 'Cicchitti',
           'Lin_Kwok', 'Fourar_Bories','Duckler', 'liquid_gas_voidage',
           'liquid_gas_voidage_methods', 'gas_liquid_viscosity',
           'gas_liquid_viscosity_methods',
           'two_phase_voidage_correlations', 'liquid_gas_viscosity_correlations']

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
    different model. However, its results are almost identical. A comparison can
    be found in the unit tests. Neither expression was found in [1]_ in a brief
    review.

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
    The form of the expression here is rearranged somewhat differently
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
    root = sqrt((rhol/rhog + K*x_ratio) / (1 + K*x_ratio))
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
    return (1 + (1-x)/x*sqrt(rhog/rhol))**-1


def Chisholm_voidage(x, rhol, rhog):
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
    >>> Chisholm_voidage(.4, 800, 2.5)
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
    S = sqrt(1 - x*(1-rhol/rhog))
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
    flow model, reviewed in [1]_, [2]_, and [3]_.

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
    return 1.0/(1.0 + (1.0 - x)/x*(rhog/rhol))


def Chisholm_Armand(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model
    presented in [1]_ based on that of [2]_ as shown in [3]_, [4]_, and [5]_.

    .. math::
        \alpha = \frac{\alpha_h}{\alpha_h + (1-\alpha_h)^{0.5}}

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
    >>> Chisholm_Armand(.4, 800, 2.5)
    0.9357814394262114

    References
    ----------
    .. [1] Chisholm, Duncan. Two-Phase Flow in Pipelines and Heat Exchangers.
       Institution of Chemical Engineers, 1983.
    .. [2] Armand, Aleksandr Aleksandrovich. The Resistance During the Movement
       of a Two-Phase System in Horizontal Pipes. Atomic Energy Research
       Establishment, 1959.
    .. [3] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [4] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [5] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    alpha_h = homogeneous(x, rhol, rhog)
    return alpha_h/(alpha_h + sqrt(1-alpha_h))


def Armand(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model
    presented in [1]_  as shown in [2]_, [3]_, and [4]_.

    .. math::
        \alpha = 0.833\alpha_h

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
    >>> Armand(.4, 800, 2.5)
    0.8291135303265941

    References
    ----------
    .. [1] Armand, Aleksandr Aleksandrovich. The Resistance During the Movement
       of a Two-Phase System in Horizontal Pipes. Atomic Energy Research
       Establishment, 1959.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [4] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    return 0.833*homogeneous(x, rhol, rhog)


def Nishino_Yamazaki(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model
    presented in [1]_ as shown in [2]_.

    .. math::
        \alpha = 1 - \left(\frac{1-x}{x}\frac{\rho_g}{\rho_l}\right)^{0.5}
        \alpha_h^{0.5}

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
    [1]_ is in Japanese.

    [3]_ either shows this model as iterative in terms of voidage, or forgot
    to add a H subscript to its second voidage term; the second is believed
    more likely.

    Examples
    --------
    >>> Nishino_Yamazaki(.4, 800, 2.5)
    0.931694583962682

    References
    ----------
    .. [1] Nishino, Haruo, and Yasaburo Yamazaki. "A New Method of Evaluating
       Steam Volume Fractions in Boiling Systems." Journal of the Atomic Energy
       Society of Japan / Atomic Energy Society of Japan 5, no. 1 (1963):
       39-46. doi:10.3327/jaesj.5.39.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    alpha_h = homogeneous(x, rhol, rhog)
    return 1 - sqrt((1-x)*rhog/x/rhol)*sqrt(alpha_h)


def Guzhov(x, rhol, rhog, m, D):
    r'''Calculates void fraction in two-phase flow according to the model
    in [1]_ as shown in [2]_ and [3]_.

    .. math::
        \alpha = 0.81[1 - \exp(-2.2\sqrt{Fr_{tp}})]\alpha_h

        Fr_{tp} = \frac{G_{tp}^2}{gD\rho_{tp}^2}

        \rho_{tp} = \left(\frac{1-x}{\rho_l} + \frac{x}{\rho_g}\right)^{-1}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----

    Examples
    --------
    >>> Guzhov(.4, 800, 2.5, 1, .3)
    0.7626030108534588

    References
    ----------
    .. [1] Guzhov, A. I, Vasiliĭ Andreevich Mamaev, and G. E Odisharii︠a︡. A
       Study of Transportation in Gas-Liquid Systems. Une Étude Sur Le
       Transport Des Systèmes Gaz-Liquides. Bruxelles: International Gas Union,
       1967.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    rho_tp = ((1-x)/rhol + x/rhog)**-1
    G = m/(pi/4*D**2)
    V_tp = G/rho_tp
    Fr = Froude(V=V_tp, L=D, squared=True) # squaring in undone later; Fr**0.5
    alpha_h = homogeneous(x, rhol, rhog)
    return 0.81*(1 - exp(-2.2*sqrt(Fr)))*alpha_h


def Kawahara(x, rhol, rhog, D):
    r'''Calculates void fraction in two-phase flow according to the model
    presented in [1]_, also reviewed in [2]_ and [3]_. This expression is for
    microchannels.

    .. math::
        \alpha = \frac{C_1 \alpha_h^{0.5}}{1 - C_2\alpha_h^{0.5}}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    D : float
        Diameter of the channel, [m]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    C1 and C2 were constants for different diameters. Only diameters of
    100 and 50 mircometers were studied in [1]_. Here, the coefficients are
    distributed for three ranges, > 250 micrometers, 250-75 micrometers, and
    < 75 micrometers.

    The `Armand` model is used for the first, C1 and C2 are 0.03 and
    0.97 for the second, and C1 and C2 are 0.02 and 0.98 for the third.

    Examples
    --------
    >>> Kawahara(.4, 800, 2.5, 100E-6)
    0.9276148194410238

    References
    ----------
    .. [1] Kawahara, A., M. Sadatomi, K. Okayama, M. Kawaji, and P. M.-Y.
       Chung. "Effects of Channel Diameter and Liquid Properties on Void
       Fraction in Adiabatic Two-Phase Flow Through Microchannels." Heat
       Transfer Engineering 26, no. 3 (February 16, 2005): 13-19.
       doi:10.1080/01457630590907158.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    if D > 250E-6:
        return Armand(x, rhol, rhog)
    elif D > 75E-6:
        C1, C2 = 0.03, 0.97
    else:
        C1, C2 = 0.02, 0.98
    alpha_h = homogeneous(x, rhol, rhog)
    return C1*sqrt(alpha_h)/(1. - C2*sqrt(alpha_h))

### Miscellaneous correlations

def Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug, pow_x=0.9, pow_rho=0.5,
                            pow_mu=0.1, n=None):
    r'''Calculates the Lockhart-Martinelli Xtt two-phase flow parameter in a
    general way according to [2]_. [1]_ is said to describe this. However,
    very different definitions of this parameter have been used elsewhere.
    Accordingly, the powers of each of the terms can be set. Alternatively, if
    the parameter `n` is provided, the powers for viscosity and phase fraction
    will be calculated from it as shown below.

    .. math::
        X_{tt} = \left(\frac{1-x}{x}\right)^{0.9} \left(\frac{\rho_g}{\rho_l}
        \right)^{0.5}\left(\frac{\mu_l}{\mu_g}\right)^{0.1}

    .. math::
        X_{tt} = \left(\frac{1-x}{x}\right)^{(2-n)/2} \left(\frac{\rho_g}
        {\rho_l}\right)^{0.5}\left(\frac{\mu_l}{\mu_g}\right)^{n/2}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval [-]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]
    pow_x : float, optional
        Power for the phase ratio (1-x)/x, [-]
    pow_rho : float, optional
        Power for the density ratio rhog/rhol, [-]
    pow_mu : float, optional
        Power for the viscosity ratio mul/mug, [-]
    n : float, optional
        Number to be used for calculating pow_x and pow_mu if provided, [-]

    Returns
    -------
    Xtt : float
        Xtt Lockhart-Martinelli two-phase flow parameter [-]

    Notes
    -----
    Xtt is best regarded as an empirical parameter.
    If used, n is often 0.2 or 0.25.

    Examples
    --------
    >>> Lockhart_Martinelli_Xtt(0.4, 800, 2.5, 1E-3, 1E-5)
    0.12761659240532292

    References
    ----------
    .. [1] Lockhart, R. W. & Martinelli, R. C. (1949), "Proposed correlation of
       data for isothermal two-phase, two-component flow in pipes", Chemical
       Engineering Progress 45 (1), 39-48.
    .. [2] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    if n is not None:
        pow_x = (2-n)/2.
        pow_mu = n/2.
    return ((1-x)/x)**pow_x * (rhog/rhol)**pow_rho * (mul/mug)**pow_mu


def Baroczy(x, rhol, rhog, mul, mug):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_, [3]_, and [4]_.

    .. math::
        \alpha = \left[1 + \left(\frac{1-x}{x}\right)^{0.74}\left(\frac{\rho_g}
        {\rho_l}\right)^{0.65}\left(\frac{\mu_l}{\mu_g}\right)^{0.13}
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

    Examples
    --------
    >>> Baroczy(.4, 800, 2.5, 1E-3, 1E-5)
    0.9453544598460807

    References
    ----------
    .. [1] Baroczy, C. Correlation of liquid fraction in two-phase flow with
       applications to liquid metals, Chem. Eng. Prog. Symp. Ser. 61 (1965)
       179-191.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [4] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug,
                                  pow_x=0.74, pow_rho=0.65, pow_mu=0.13)
    return (1 + Xtt)**-1


def Tandon_Varma_Gupta(x, rhol, rhog, mul, mug, m, D):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_, [3]_, and [4]_.

    For 50 < Rel < 1125:

    .. math::
        \alpha = 1- 1.928Re_l^{-0.315}[F(X_{tt})]^{-1} + 0.9293Re_l^{-0.63}
        [F(X_{tt})]^{-2}

    For Rel > 1125:

    .. math::
        \alpha = 1- 0.38 Re_l^{-0.088}[F(X_{tt})]^{-1} + 0.0361 Re_l^{-0.176}
        [F(X_{tt})]^{-2}

    .. math::
        F(X_{tt}) = 0.15[X_{tt}^{-1} + 2.85X_{tt}^{-0.476}]

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
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ does not specify how it defines the liquid Reynolds number.
    [2]_ disagrees with [3]_ and [4]_; the later variant was selected, with:

    .. math::
        Re_l = \frac{G_{tp}D}{\mu_l}

    The lower limit on Reynolds number is not enforced.

    Examples
    --------
    >>> Tandon_Varma_Gupta(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3)
    0.9228265670341428

    References
    ----------
    .. [1] Tandon, T. N., H. K. Varma, and C. P. Gupta. "A Void Fraction Model
       for Annular Two-Phase Flow." International Journal of Heat and Mass
       Transfer 28, no. 1 (January 1, 1985): 191-198.
       doi:10.1016/0017-9310(85)90021-3.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [4] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    G = m/(pi/4*D**2)
    Rel = G*D/mul
    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug)
    Fxtt = 0.15*(Xtt**-1 + 2.85*Xtt**-0.476)
    if Rel < 1125:
        alpha = 1 - 1.928*Rel**-0.315/Fxtt + 0.9293*Rel**-0.63/Fxtt**2
    else:
        alpha = 1 - 0.38*Rel**-0.088/Fxtt + 0.0361*Rel**-0.176/Fxtt**2
    return alpha


def Harms(x, rhol, rhog, mul, mug, m, D):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 - 10.06Re_l^{-0.875}(1.74 + 0.104Re_l^{0.5})^2
        \left(1.376 + \frac{7.242}{X_{tt}^{1.655}}\right)^{-0.5}\right]^2

    .. math::
        Re_l = \frac{G_{tp}(1-x)D}{\mu_l}

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
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has been reviewed.

    Examples
    --------
    >>> Harms(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3)
    0.9653289762907554

    References
    ----------
    .. [1] Tandon, T. N., H. K. Varma, and C. P. Gupta. "A Void Fraction Model
       for Annular Two-Phase Flow." International Journal of Heat and Mass
       Transfer 28, no. 1 (January 1, 1985): 191-198.
       doi:10.1016/0017-9310(85)90021-3.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    G = m/(pi/4*D**2)
    Rel = G*D*(1-x)/mul
    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug)
    return (1 - 10.06*Rel**-0.875*(1.74 + 0.104*sqrt(Rel))**2
            *1.0/sqrt(1.376 + 7.242/Xtt**1.655))


def Domanski_Didion(x, rhol, rhog, mul, mug):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_ and [3]_.

    if Xtt < 10:

    .. math::
        \alpha = (1 + X_{tt}^{0.8})^{-0.378}

    Otherwise:

    .. math::
        \alpha = 0.823- 0.157\ln(X_{tt})

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
    [1]_ has been reviewed. [2]_ gives an exponent of -0.38 instead of -0.378
    as is in [1]_. [3]_ describes only the novel half of the correlation.
    The portion for Xtt > 10 is novel; the other is said to be from their 31st
    reference, Wallis.

    There is a discontinuity at Xtt = 10.

    Examples
    --------
    >>> Domanski_Didion(.4, 800, 2.5, 1E-3, 1E-5)
    0.9355795597059169

    References
    ----------
    .. [1] Domanski, Piotr, and David A. Didion. "Computer Modeling of the
       Vapor Compression Cycle with Constant Flow Area Expansion Device."
       Report. UNT Digital Library, May 1983.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug)
    if Xtt < 10:
        return (1 + Xtt**0.8)**-0.378
    else:
        return 0.823 - 0.157*log(Xtt)


def Graham(x, rhol, rhog, mul, mug, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_ and [3]_.

    .. math::
        \alpha = 1 - \exp\{-1 - 0.3\ln(Ft) - 0.0328[\ln(Ft)]^2\}

    .. math::
        Ft = \left[\frac{G_{tp}^2 x^3}{(1-x)\rho_g^2gD}\right]^{0.5}

    .. math::
        \alpha = 0 \text{ for } F_t \le 0.01032

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
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has been reviewed. [2]_ does not list that the expression is not
    real below a certain value of Ft.

    Examples
    --------
    >>> Graham(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3)
    0.6403336287530644

    References
    ----------
    .. [1] Graham, D. M. "Experimental Investigation of Void Fraction During
       Refrigerant Condensation." ACRC Technical Report 135. Air Conditioning
       and Refrigeration Center. College of Engineering. University of Illinois
       at Urbana-Champaign., December 1997.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    G = m/(pi/4*D**2)
    Ft = sqrt(G**2*x**3/((1-x)*rhog**2*g*D))
    if Ft < 0.01032:
        return 0
    else:
        return 1 - exp(-1 - 0.3*log(Ft) - 0.0328*log(Ft)**2)


def Yashar(x, rhol, rhog, mul, mug, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_ and [3]_.

    .. math::
        \alpha = \left[1 + \frac{1}{Ft} + X_{tt}\right]^{-0.321}

    .. math::
        Ft = \left[\frac{G_{tp}^2 x^3}{(1-x)\rho_g^2gD}\right]^{0.5}

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
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has been reviewed; both [2]_ and [3]_ give it correctly.

    Examples
    --------
    >>> Yashar(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3)
    0.7934893185789146

    References
    ----------
    .. [1] Yashar, D. A., M. J. Wilson, H. R. Kopke, D. M. Graham, J. C. Chato,
       and T. A. Newell. "An Investigation of Refrigerant Void Fraction in
       Horizontal, Microfin Tubes." HVAC&R Research 7, no. 1 (January 1, 2001):
       67-82. doi:10.1080/10789669.2001.10391430.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    G = m/(pi/4*D**2)
    Ft = sqrt(G**2*x**3/((1-x)*rhog**2*g*D))
    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug)
    return (1 + 1./Ft + Xtt)**-0.321


def Huq_Loth(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_, also given in [2]_, [3]_, and [4]_.

    .. math::
        \alpha = 1 - \frac{2(1-x)^2}{1 - 2x + \left[1 + 4x(1-x)\left(\frac
        {\rho_l}{\rho_g}-1\right)\right]^{0.5}}

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
    [1]_ has been reviewed, and matches the expressions given in the reviews
    [2]_, [3]_, and [4]_; the form of the expression is rearranged somewhat
    differently.

    Examples
    --------
    >>> Huq_Loth(.4, 800, 2.5)
    0.9593868838476147

    References
    ----------
    .. [1] Huq, Reazul, and John L. Loth. "Analytical Two-Phase Flow Void
       Prediction Method." Journal of Thermophysics and Heat Transfer 6, no. 1
       (January 1, 1992): 139-44. doi:10.2514/3.329.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    .. [4] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    B = 2*x*(1-x)
    D = sqrt(1 + 2*B*(rhol/rhog -1))
    return 1 - 2*(1-x)**2/(1 - 2*x + D)


def Kopte_Newell_Chato(x, rhol, rhog, mul, mug, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_.

    .. math::
        \alpha = 1.045 - \exp\{-1 - 0.342\ln(Ft) - 0.0268[\ln(Ft)]^2
        + 0.00597[\ln(Ft)]^3\}

    .. math::
        Ft = \left[\frac{G_{tp}^2 x^3}{(1-x)\rho_g^2gD}\right]^{0.5}

    .. math::
        \alpha = \alpha_h \text{ for } F_t \le 0.044

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
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has been reviewed. If is recommended this expression not be used above
    Ft values of 454.

    Examples
    --------
    >>> Kopte_Newell_Chato(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3)
    0.6864466770087425

    References
    ----------
    .. [1] Kopke, H. R. "Experimental Investigation of Void Fraction During
       Refrigerant Condensation in Horizontal Tubes." ACRC Technical Report
       142. Air Conditioning and Refrigeration Center. College of Engineering.
       University of Illinois at Urbana-Champaign., August 1998.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    '''
    G = m/(pi/4*D**2)
    Ft = sqrt(G**2*x**3/((1-x)*rhog**2*g*D))
    if Ft < 0.044:
        return homogeneous(x, rhol, rhog)
    else:
        return 1.045 - exp(-1 - 0.342*log(Ft) - 0.0268*log(Ft)**2 + 0.00597*log(Ft)**3)

### Drift flux models


def Steiner(x, rhol, rhog, sigma, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ also given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = \frac{1.18(1-x)}{\rho_l^{0.5}}[g\sigma(\rho_l-\rho_g)]^{0.25}

    .. math::
        C_0 = 1 + 0.12(1-x)

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    [1]_ has been reviewed.

    Examples
    --------
    >>> Steiner(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3)
    0.895950181381335

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Dalkilic, A. S., S. Laohalertdecha, and S. Wongwises. "Effect of
       Void Fraction Models on the Two-Phase Friction Factor of R134a during
       Condensation in Vertical Downward Flow in a Smooth Tube." International
       Communications in Heat and Mass Transfer 35, no. 8 (October 2008):
       921-27. doi:10.1016/j.icheatmasstransfer.2008.04.001.
    '''
    G = m/(pi/4*D**2)
    C0 = 1 + 0.12*(1-x)
    vgm = 1.18*(1-x)/sqrt(rhol)*sqrt(sqrt(g*sigma*(rhol-rhog)))
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


def Rouhani_1(x, rhol, rhog, sigma, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = \frac{1.18(1-x)}{\rho_l^{0.5}}[g\sigma(\rho_l-\rho_g)]^{0.25}

    .. math::
        C_0 = 1 + 0.2(1-x)

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    The expression as quoted in [2]_ and [3]_ could not be found in [1]_.

    Examples
    --------
    >>> Rouhani_1(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3)
    0.8588420244136714

    References
    ----------
    .. [1] Rouhani, S. Z, and E Axelsson. "Calculation of Void Volume Fraction
       in the Subcooled and Quality Boiling Regions." International Journal of
       Heat and Mass Transfer 13, no. 2 (February 1, 1970): 383-93.
       doi:10.1016/0017-9310(70)90114-6.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    G = m/(pi/4*D**2)
    C0 = 1 + 0.2*(1-x)
    vgm = 1.18*(1-x)/sqrt(rhol)*sqrt(sqrt(g*sigma*(rhol-rhog)))
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


def Rouhani_2(x, rhol, rhog, sigma, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = \frac{1.18(1-x)}{\rho_l^{0.5}}[g\sigma(\rho_l-\rho_g)]^{0.25}

    .. math::
        C_0 = 1 + 0.2(1-x)(gD)^{0.25}\left(\frac{\rho_l}{G_{tp}}\right)^{0.5}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    The expression as quoted in [2]_ and [3]_ could not be found in [1]_.

    Examples
    --------
    >>> Rouhani_2(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3)
    0.44819733138968865

    References
    ----------
    .. [1] Rouhani, S. Z, and E Axelsson. "Calculation of Void Volume Fraction
       in the Subcooled and Quality Boiling Regions." International Journal of
       Heat and Mass Transfer 13, no. 2 (February 1, 1970): 383-93.
       doi:10.1016/0017-9310(70)90114-6.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    G = m/(pi/4*D**2)
    C0 = 1 + 0.2*(1-x)*sqrt(sqrt(g*D))*sqrt(rhol/G)
    vgm = 1.18*(1-x)/sqrt(rhol)*sqrt(sqrt(g*sigma*(rhol-rhog)))
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


def Nicklin_Wilkes_Davidson(x, rhol, rhog, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = 0.35\sqrt{gD}

    .. math::
        C_0 = 1.2

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----

    Examples
    --------
    >>> Nicklin_Wilkes_Davidson(0.4, 800., 2.5, m=1, D=0.3)
    0.6798826626721431

    References
    ----------
    .. [1] D. Nicklin, J. Wilkes, J. Davidson, "Two-phase flow in vertical
       tubes", Trans. Inst. Chem. Eng. 40 (1962) 61-68.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    G = m/(pi/4*D**2)
    C0 = 1.2
    vgm = 0.35*sqrt(g*D)
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


def Gregory_Scott(x, rhol, rhog):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = 0

    .. math::
        C_0 = 1.19

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
    >>> Gregory_Scott(0.4, 800., 2.5)
    0.8364154370924108

    References
    ----------
    .. [1] Gregory, G. A., and D. S. Scott. "Correlation of Liquid Slug
       Velocity and Frequency in Horizontal Cocurrent Gas-Liquid Slug Flow."
       AIChE Journal 15, no. 6 (November 1, 1969): 933-35.
       doi:10.1002/aic.690150623.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    C0 = 1.19
    return x/rhog*(C0*(x/rhog + (1-x)/rhol))**-1


def Dix(x, rhol, rhog, sigma, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = 2.9\left(g\sigma\frac{\rho_l-\rho_g}{\rho_l^2}\right)^{0.25}

    .. math::
        C_0 = \frac{v_{sg}}{v_m}\left[1 + \left(\frac{v_{sl}}{v_{sg}}\right)
        ^{\left(\left(\frac{\rho_g}{\rho_l}\right)^{0.1}\right)}\right]

    .. math::
        v_{gs} = \frac{mx}{\rho_g \frac{\pi}{4}D^2}

    .. math::
        v_{ls} = \frac{m(1-x)}{\rho_l \frac{\pi}{4}D^2}

    .. math::
        v_m = v_{gs} + v_{ls}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    Has formed the basis for several other correlations.

    Examples
    --------
    >>> Dix(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3)
    0.8268737961156514

    References
    ----------
    .. [1] Gary Errol. Dix. "Vapor Void Fractions for Forced Convection with
       Subcooled Boiling at Low Flow Rates." Thesis. University of California,
       Berkeley, 1971.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    vgs = m*x/(rhog*pi/4*D**2)
    vls = m*(1-x)/(rhol*pi/4*D**2)
    G = m/(pi/4*D**2)
    C0 = vgs/(vls+vgs)*(1 + (vls/vgs)**((rhog/rhol)**0.1))
    vgm = 2.9*sqrt(sqrt(g*sigma*(rhol-rhog)/rhol**2))
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


def Sun_Duffey_Peng(x, rhol, rhog, sigma, m, D, P, Pc, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_ as given in [2]_ and [3]_.

    .. math::
        \alpha = \frac{x}{\rho_g}\left[C_0\left(\frac{x}{\rho_g} + \frac{1-x}
        {\rho_l}\right) +\frac{v_{gm}}{G} \right]^{-1}

    .. math::
        v_{gm} = 1.41\left[\frac{g\sigma(\rho_l-\rho_g)}{\rho_l^2}\right]^{0.25}

    .. math::
        C_0 = \left(0.82 + 0.18\frac{P}{P_c}\right)^{-1}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    P : float
        Pressure of the fluid, [Pa]
    Pc : float
        Critical pressure of the fluid, [Pa]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----

    Examples
    --------
    >>> Sun_Duffey_Peng(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3, P=1E5, Pc=7E6)
    0.7696546506515833

    References
    ----------
    .. [1] K.H. Sun, R.B. Duffey, C.M. Peng, A thermal-hydraulic analysis of
       core uncover, in: Proceedings of the 19th National Heat Transfer
       Conference, Experimental and Analytical Modeling of LWR Safety
       Experiments, 1980, pp. 1-10. Orlando, Florida, USA.
    .. [2] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    .. [3] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    G = m/(pi/4*D**2)
    Pr = P/Pc if Pc is not None else 0.5
    C0 = (0.82 + 0.18*Pr)**-1
    vgm = 1.41*sqrt(sqrt(g*sigma*(rhol-rhog)/rhol**2))
    return x/rhog*(C0*(x/rhog + (1-x)/rhol) + vgm/G)**-1


# Correlations developed in reviews


def Xu_Fang_voidage(x, rhol, rhog, m, D, g=g):
    r'''Calculates void fraction in two-phase flow according to the model
    developed in the review of [1]_.

    .. math::
        \alpha = \left[1 + \left(1 + 2Fr_{lo}^{-0.2}\alpha_h^{3.5}\right)\left(
        \frac{1-x}{x}\right)\left(\frac{\rho_g}{\rho_l}\right)\right]^{-1}

    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    Claims an AARD of 5.0%, and suitability for any flow regime,
    mini and micro channels, adiabatic, evaporating, or condensing flow,
    and for Frlo from 0.02 to 145, rhog/rhol from 0.004-0.153, and x from 0 to
    1.

    Examples
    --------
    >>> Xu_Fang_voidage(0.4, 800., 2.5, m=1, D=0.3)
    0.9414660089942093

    References
    ----------
    .. [1] Xu, Yu, and Xiande Fang. "Correlations of Void Fraction for Two-
       Phase Refrigerant Flow in Pipes." Applied Thermal Engineering 64, no.
       1-2 (March 2014): 242–51. doi:10.1016/j.applthermaleng.2013.12.032.
    '''
    G = m/(pi/4*D**2)
    alpha_h = homogeneous(x, rhol, rhog)
    Frlo = G**2/(g*D*rhol**2)
    return (1 + (1 + 2*Frlo**-0.2*alpha_h**3.5)*((1-x)/x)*(rhog/rhol))**-1


def Woldesemayat_Ghajar(x, rhol, rhog, sigma, m, D, P, angle=0, g=g):
    r'''Calculates void fraction in two-phase flow according to the model of
    [1]_.

    .. math::
        \alpha = \frac{v_{gs}}{v_{gs}\left(1 + \left(\frac{v_{ls}}{v_{gs}}
        \right)^{\left(\frac{\rho_g}{\rho_l}\right)^{0.1}}\right)
        + 2.9\left[\frac{gD\sigma(1+\cos\theta)(\rho_l-\rho_g)}
        {\rho_l^2}\right]^{0.25}(1.22 + 1.22\sin\theta)^{\frac{P}{P_{atm}}}}

    .. math::
        v_{gs} = \frac{mx}{\rho_g \frac{\pi}{4}D^2}

    .. math::
        v_{ls} = \frac{m(1-x)}{\rho_l \frac{\pi}{4}D^2}


    Parameters
    ----------
    x : float
        Quality at the specific tube interval []
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    sigma : float
        Surface tension of liquid [N/m]
    m : float
        Mass flow rate of both phases, [kg/s]
    D : float
        Diameter of the channel, [m]
    P : float
        Pressure of the fluid, [Pa]
    angle : float
        Angle of the channel with respect to the horizontal (vertical = 90),
        [degrees]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Notes
    -----
    Strongly recommended.

    Examples
    --------
    >>> Woldesemayat_Ghajar(0.4, 800., 2.5, sigma=0.2, m=1, D=0.3, P=1E6, angle=45)
    0.7640815513429202

    References
    ----------
    .. [1] Woldesemayat, Melkamu A., and Afshin J. Ghajar. "Comparison of Void
       Fraction Correlations for Different Flow Patterns in Horizontal and
       Upward Inclined Pipes." International Journal of Multiphase Flow 33,
       no. 4 (April 2007): 347-370. doi:10.1016/j.ijmultiphaseflow.2006.09.004.
    '''
    vgs = m*x/(rhog*pi/4*D**2)
    vls = m*(1-x)/(rhol*pi/4*D**2)
    first = vgs*(1 + (vls/vgs)**((rhog/rhol)**0.1))
    second = 2.9*sqrt(sqrt((g*D*sigma*(1 + cos(radians(angle)))*(rhol-rhog))/rhol**2))
    if P is None: P = 101325.0
    third = (1.22 + 1.22*sin(radians(angle)))**(101325./P)
    return vgs/(first + second*third)

# x, rhol, rhog 2ill be the minimum inputs

two_phase_voidage_correlations = {'Thom' : (Thom, ('x', 'rhol', 'rhog', 'mul', 'mug')),
'Zivi' : (Zivi, ('x', 'rhol', 'rhog')),
'Smith' : (Smith, ('x', 'rhol', 'rhog')),
'Fauske' : (Fauske, ('x', 'rhol', 'rhog')),
'Chisholm_voidage' : (Chisholm_voidage, ('x', 'rhol', 'rhog')),
'Turner Wallis' : (Turner_Wallis, ('x', 'rhol', 'rhog', 'mul', 'mug')),
'homogeneous' : (homogeneous, ('x', 'rhol', 'rhog')),
'Chisholm Armand' : (Chisholm_Armand, ('x', 'rhol', 'rhog')),
'Armand' : (Armand, ('x', 'rhol', 'rhog')),
'Nishino Yamazaki' : (Nishino_Yamazaki, ('x', 'rhol', 'rhog')),
'Guzhov' : (Guzhov, ('x', 'rhol', 'rhog', 'm', 'D')),
'Kawahara' : (Kawahara, ('x', 'rhol', 'rhog', 'D')),
'Baroczy' : (Baroczy, ('x', 'rhol', 'rhog', 'mul', 'mug')),
'Tandon Varma Gupta' : (Tandon_Varma_Gupta, ('x', 'rhol', 'rhog', 'mul', 'mug', 'm', 'D')),
'Harms' : (Harms, ('x', 'rhol', 'rhog', 'mul', 'mug', 'm', 'D')),
'Domanski Didion' : (Domanski_Didion, ('x', 'rhol', 'rhog', 'mul', 'mug')),
'Graham' : (Graham, ('x', 'rhol', 'rhog', 'mul', 'mug', 'm', 'D', 'g')),
'Yashar' : (Yashar, ('x', 'rhol', 'rhog', 'mul', 'mug', 'm', 'D', 'g')),
'Huq_Loth' : (Huq_Loth, ('x', 'rhol', 'rhog')),
'Kopte_Newell_Chato' : (Kopte_Newell_Chato, ('x', 'rhol', 'rhog', 'mul', 'mug', 'm', 'D', 'g')),
'Steiner' : (Steiner, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'g')),
'Rouhani 1' : (Rouhani_1, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'g')),
'Rouhani 2' : (Rouhani_2, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'g')),
'Nicklin Wilkes Davidson' : (Nicklin_Wilkes_Davidson, ('x', 'rhol', 'rhog', 'm', 'D', 'g')),
'Gregory_Scott' : (Gregory_Scott, ('x', 'rhol', 'rhog')),
'Dix' : (Dix, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'g')),
'Sun Duffey Peng' : (Sun_Duffey_Peng, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'P', 'Pc', 'g')),
'Xu Fang voidage' : (Xu_Fang_voidage, ('x', 'rhol', 'rhog', 'm', 'D', 'g')),
'Woldesemayat Ghajar' : (Woldesemayat_Ghajar, ('x', 'rhol', 'rhog', 'sigma', 'm', 'D', 'P', 'angle', 'g'))}

_unknown_two_phase_voidage_corr = 'Method not recognized; available methods are %s' %list(two_phase_voidage_correlations.keys())
# All the available arguments are:
#{'rhol', 'angle=0', 'x', 'P', 'mug', 'rhog', 'D', 'g', 'Pc', 'sigma', 'mul', 'm'}
def liquid_gas_voidage_methods(x, rhol, rhog, D=None, m=None, mul=None, mug=None,
                               sigma=None, P=None, Pc=None, angle=0.0, g=g,
                               check_ranges=False):
    r'''This function returns a list of liquid-gas voidage correlation names
    which can perform the calculation with the provided inputs. The holdup is
    for two-phase liquid-gas flow inside channels. 29 calculation methods are
    available, with varying input requirements.

    Parameters
    ----------
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    D : float, optional
        Diameter of pipe, [m]
    m : float, optional
        Mass flow rate of fluid, [kg/s]
    mul : float, optional
        Viscosity of liquid, [Pa*s]
    mug : float, optional
        Viscosity of gas, [Pa*s]
    sigma : float, optional
        Surface tension, [N/m]
    P : float, optional
        Pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    angle : float, optional
        Angle of the channel with respect to the horizontal (vertical = 90),
        [degrees]
    g : float, optional
        Acceleration due to gravity, [m/s^2]
    check_ranges : bool, optional
        Added for future use only

    Returns
    -------
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate two-phase liquid-gas
        voidage with the given inputs.

    Examples
    --------
    >>> len(liquid_gas_voidage_methods(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05))
    27
    '''
    vals = {'x': x, 'rhol': rhol, 'rhog': rhog, 'D': D, 'm': m, 'mul': mul,
            'mug': mug, 'sigma': sigma, 'P': P, 'Pc': Pc, 'angle': angle,
            'g': g, 'check_ranges': check_ranges}
    usable_methods = []
    for method, value in two_phase_voidage_correlations.items():
        f, args = value
        if all(vals[i] is not None for i in args):
            usable_methods.append(method)
    return usable_methods

def liquid_gas_voidage(x, rhol, rhog, D=None, m=None, mul=None, mug=None,
                       sigma=None, P=None, Pc=None, angle=0, g=g, Method=None):
    r'''This function handles calculation of two-phase liquid-gas voidage
    for flow inside channels. 29 calculation methods are available, with
    varying input requirements. A correlation will be automatically selected if
    none is specified.

    This function is used to calculate the (liquid) holdup as well, as:

    .. math::
        \text{holdup} = 1 - \text{voidage}

    If no correlation is selected, the following rules are used, with the
    earlier options attempted first:

        * TODO: defaults

    Parameters
    ----------
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    D : float, optional
        Diameter of pipe, [m]
    m : float, optional
        Mass flow rate of fluid, [kg/s]
    mul : float, optional
        Viscosity of liquid, [Pa*s]
    mug : float, optional
        Viscosity of gas, [Pa*s]
    sigma : float, optional
        Surface tension, [N/m]
    P : float, optional
        Pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    angle : float, optional
        Angle of the channel with respect to the horizontal (vertical = 90),
        [degrees]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        two_phase_voidage_correlations.

    Notes
    -----

    Examples
    --------
    >>> liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05)
    0.9744097632663492
    '''
    if Method is None:
        Method2 = 'homogeneous'
    else:
        Method2 = Method

    if Method2 == "Thom":
        return Thom(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    elif Method2 == "Zivi":
        return Zivi(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Smith":
        return Smith(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Fauske":
        return Fauske(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Chisholm_voidage":
        return Chisholm_voidage(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Turner Wallis":
        return Turner_Wallis(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    elif Method2 == "homogeneous":
        return homogeneous(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Chisholm Armand":
        return Chisholm_Armand(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Armand":
        return Armand(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Nishino Yamazaki":
        return Nishino_Yamazaki(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Guzhov":
        return Guzhov(x=x, rhol=rhol, rhog=rhog, m=m, D=D)
    elif Method2 == "Kawahara":
        return Kawahara(x=x, rhol=rhol, rhog=rhog, D=D)
    elif Method2 == "Baroczy":
        return Baroczy(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    elif Method2 == "Tandon Varma Gupta":
        return Tandon_Varma_Gupta(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, m=m, D=D)
    elif Method2 == "Harms":
        return Harms(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, m=m, D=D)
    elif Method2 == "Domanski Didion":
        return Domanski_Didion(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    elif Method2 == "Graham":
        return Graham(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, m=m, D=D, g=g)
    elif Method2 == "Yashar":
        return Yashar(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, m=m, D=D, g=g)
    elif Method2 == "Huq_Loth":
        return Huq_Loth(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Kopte_Newell_Chato":
        return Kopte_Newell_Chato(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, m=m, D=D, g=g)
    elif Method2 == "Steiner":
        return Steiner(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, g=g)
    elif Method2 == "Rouhani 1":
        return Rouhani_1(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, g=g)
    elif Method2 == "Rouhani 2":
        return Rouhani_2(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, g=g)
    elif Method2 == "Nicklin Wilkes Davidson":
        return Nicklin_Wilkes_Davidson(x=x, rhol=rhol, rhog=rhog, m=m, D=D, g=g)
    elif Method2 == "Gregory_Scott":
        return Gregory_Scott(x=x, rhol=rhol, rhog=rhog)
    elif Method2 == "Dix":
        return Dix(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, g=g)
    elif Method2 == "Sun Duffey Peng":
        return Sun_Duffey_Peng(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, P=P, Pc=Pc, g=g)
    elif Method2 == "Xu Fang voidage":
        return Xu_Fang_voidage(x=x, rhol=rhol, rhog=rhog, m=m, D=D, g=g)
    elif Method2 == "Woldesemayat Ghajar":
        return Woldesemayat_Ghajar(x=x, rhol=rhol, rhog=rhog, sigma=sigma, m=m, D=D, P=P, angle=angle, g=g)
    else:
        raise ValueError(_unknown_two_phase_voidage_corr)


def density_two_phase(alpha, rhol, rhog):
    r'''Calculates the "effective" density of fluid in a liquid-gas flow. If
    the weight of fluid in a pipe pipe could be measured and the volume of
    the pipe were known, an effective density of the two-phase mixture could be
    calculated. This is directly relatable to the void fraction of the pipe,
    a parameter used to predict the pressure drop. This function converts
    void fraction to effective two-phase density.

    .. math::
        \rho_m = \alpha \rho_g + (1-\alpha)\rho_l

    Parameters
    ----------
    alpha : float
        Void fraction (area of gas / total area of channel), [-]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    rho_lg : float
        Two-phase effective density [kg/m^3]

    Notes
    -----
    **THERE IS NO THERMODYNAMIC DEFINITION FOR THIS QUANTITY. DO NOT USE THIS
    VALUE IN SINGLE-PHASE CORRELATIONS.**

    Examples
    --------
    >>> density_two_phase(.4, 800, 2.5)
    481.0

    References
    ----------
    .. [1] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    '''
    return alpha*rhog + (1. - alpha)*rhol


def two_phase_voidage_experimental(rho_lg, rhol, rhog):
    r'''Calculates the void fraction for two-phase liquid-gas pipeflow. If
    the weight of fluid in a pipe pipe could be measured and the volume of
    the pipe were known, an effective density of the two-phase mixture could be
    calculated. This is directly relatable to the void fraction of the pipe,
    a parameter used to predict the pressure drop. This function converts
    that measured effective two-phase density to void fraction for use in
    developing correlations.

    .. math::
        \alpha = \frac{\rho_m - \rho_l}{\rho_g - \rho_l}

    Parameters
    ----------
    rho_lg : float
        Two-phase effective density [kg/m^3]
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
    >>> two_phase_voidage_experimental(481.0, 800, 2.5)
    0.4

    References
    ----------
    .. [1] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    '''
    return (rho_lg - rhol)/(rhog - rhol)


### two-phase viscosity models


def Beattie_Whalley(x, mul, mug, rhol, rhog):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_ and [3]_.

    .. math::
        \mu_m = \mu_l(1-\alpha_m)(1 + 2.5\alpha_m) + \mu_g\alpha_m

    .. math::
        \alpha_m = \frac{1}{1 + \left(\frac{1-x}{x}\right)\frac{\rho_g}{\rho_l}}
        \text{(homogeneous model)}

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    Examples
    --------
    >>> Beattie_Whalley(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2)
    1.7363806909512365e-05

    References
    ----------
    .. [1] Beattie, D. R. H., and P. B. Whalley. "A Simple Two-Phase Frictional
       Pressure Drop Calculation Method." International Journal of Multiphase
       Flow 8, no. 1 (February 1, 1982): 83-87.
       doi:10.1016/0301-9322(82)90009-X.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    alpha = homogeneous(x, rhol, rhog)
    return mul*(1. - alpha)*(1. + 2.5*alpha) + mug*alpha


def McAdams(x, mul, mug):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_ and [3]_.

    .. math::
        \mu_m = \left(\frac{x}{\mu_g} + \frac{1-x}{\mu_l}\right)^{-1}

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    [3]_ states this is the most common definition of two-phase liquid-gas
    viscosity.

    Examples
    --------
    >>> McAdams(x=0.4, mul=1E-3, mug=1E-5)
    2.4630541871921184e-05

    References
    ----------
    .. [1] McAdams, W. H. "Vaporization inside Horizontal Tubes-II Benzene-Oil
       Mixtures." Trans. ASME 39 (1949): 39-48.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    return 1./(x/mug + (1. - x)/mul)


def Cicchitti(x, mul, mug):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_ and [3]_.

    .. math::
        \mu_m = x\mu_g + (1-x)\mu_l

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    Examples
    --------
    >>> Cicchitti(x=0.4, mul=1E-3, mug=1E-5)
    0.0006039999999999999

    References
    ----------
    .. [1] Cicchitti, A., C. Lombardi, M. Silvestri, G. Soldaini, and R.
       Zavattarelli. "Two-Phase Cooling Experiments: Pressure Drop, Heat
       Transfer and Burnout Measurements." Centro Informazioni Studi
       Esperienze, Milan, January 1, 1959.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    return x*mug + (1. - x)*mul


def Lin_Kwok(x, mul, mug):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_.

    .. math::
        \mu_m = \frac{\mu_l \mu_g}{\mu_g + x^{1.4}(\mu_l - \mu_g)}

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    Examples
    --------
    >>> Lin_Kwok(x=0.4, mul=1E-3, mug=1E-5)
    3.515119398126066e-05

    References
    ----------
    .. [1] Lin, S., C. C. K. Kwok, R. -Y. Li, Z. -H. Chen, and Z. -Y. Chen.
       "Local Frictional Pressure Drop during Vaporization of R-12 through
       Capillary Tubes." International Journal of Multiphase Flow 17, no. 1
       (January 1, 1991): 95-102. doi:10.1016/0301-9322(91)90072-B.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    '''
    return mul*mug/(mug + x**1.4*(mul - mug))


def Fourar_Bories(x, mul, mug, rhol, rhog):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_ and [3]_.

    .. math::
        \mu_m = \rho_m\left(\sqrt{x\nu_g} + \sqrt{(1-x)\nu_l}\right)^2

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    rhol : float
        Density of the liquid, [kg/m^3]
    rhog : float
        Density of the gas, [kg/m^3]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    This was first expressed in the equalivalent form as follows:

    .. math::
        \mu_m = \rho_m\left(x\nu_g + (1-x)\nu_l + 2\sqrt{x(1-x)\nu_g\nu_l}
        \right)

    Examples
    --------
    >>> Fourar_Bories(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2)
    2.127617150298565e-05

    References
    ----------
    .. [1] Fourar, M., and S. Bories. "Experimental Study of Air-Water
       Two-Phase Flow through a Fracture (Narrow Channel)." International
       Journal of Multiphase Flow 21, no. 4 (August 1, 1995): 621-37.
       doi:10.1016/0301-9322(95)00005-I.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    .. [3] Aung, NZ, and T. Yuwono. "Evaluation of Mixture Viscosity Models in
       the Prediction of Two-Phase Flow Pressure Drops." ASEAN Journal on
       Science and Technology for Development 29, no. 2 (2012).
    '''
    rhom = 1./(x/rhog + (1. - x)/rhol)
    nul = mul/rhol # = nu_mu_converter(rho=rhol, mu=mul)
    nug = mug/rhog # = nu_mu_converter(rho=rhog, mu=mug)
    return rhom*(sqrt(x*nug) + sqrt((1. - x)*nul))**2


def Duckler(x, mul, mug, rhol, rhog):
    r'''Calculates a suggested definition for liquid-gas two-phase flow
    viscosity in internal pipe flow according to the form in [1]_ and shown
    in [2]_, [3]_, and [4]_.

    .. math::
        \mu_m = \frac{\frac{x\mu_g}{\rho_g} + \frac{(1-x)\mu_l}{\rho_l} }
        {\frac{x}{\rho_g} + \frac{(1-x)}{\rho_l}   }

    Parameters
    ----------
    x : float
        Quality of the gas-liquid flow, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    rhol : float
        Density of the liquid, [kg/m^3]
    rhog : float
        Density of the gas, [kg/m^3]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Notes
    -----
    This model converges to the liquid or gas viscosity as the quality
    approaches either limits.

    This has also been expressed in the following form:

    .. math::
        \mu_m = \rho_m \left[x\left(\frac{\mu_g}{\rho_g}\right)
        + (1 - x)\left(\frac{\mu_l}{\rho_l}\right)\right]

    According to the homogeneous definition of two-phase density.

    Examples
    --------
    >>> Duckler(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2)
    1.2092040385066917e-05

    References
    ----------
    .. [1] Fourar, M., and S. Bories. "Experimental Study of Air-Water
       Two-Phase Flow through a Fracture (Narrow Channel)." International
       Journal of Multiphase Flow 21, no. 4 (August 1, 1995): 621-37.
       doi:10.1016/0301-9322(95)00005-I.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    .. [4] Aung, NZ, and T. Yuwono. "Evaluation of Mixture Viscosity Models in
       the Prediction of Two-Phase Flow Pressure Drops." ASEAN Journal on
       Science and Technology for Development 29, no. 2 (2012).
    '''
    return (x*mug/rhog + (1. - x)*mul/rhol)/(x/rhog + (1. - x)/rhol)


liquid_gas_viscosity_correlations = {'Beattie Whalley': (Beattie_Whalley, 1),
                                     'Fourar Bories': (Fourar_Bories, 1),
                                     'Duckler': (Duckler, 1),
                                     'McAdams': (McAdams, 0),
                                     'Cicchitti': (Cicchitti, 0),
                                     'Lin Kwok': (Lin_Kwok, 0)}
liquid_gas_viscosity_correlations_list = ['Beattie Whalley', 'Fourar Bories', 'Duckler', 'McAdams', 'Cicchitti', 'Lin Kwok']

def gas_liquid_viscosity_methods(rhol=None, rhog=None, check_ranges=False):
    r'''This function returns a list of methods which can be used for calculating
    two-phase liquid-gas viscosity.
    Six calculation methods are available; three of them require only `x`,
    `mul`, and `mug`; the other three require `rhol` and `rhog` as well.

    Parameters
    ----------
    rhol : float, optional
        Liquid density, [kg/m^3]
    rhog : float, optional
        Gas density, [kg/m^3]
    check_ranges : bool, optional
        Added for compatibility only, never used

    Returns
    -------
    methods : list
        List of methods which can be used to calculate two-phase liquid-gas
        viscosity with the given inputs.

    Examples
    --------
    >>> gas_liquid_viscosity_methods()
    ['McAdams', 'Cicchitti', 'Lin Kwok']
    >>> gas_liquid_viscosity_methods(rhol=1000, rhog=2)
    ['Beattie Whalley', 'Fourar Bories', 'Duckler', 'McAdams', 'Cicchitti', 'Lin Kwok']
    '''
    methods = ['McAdams', 'Cicchitti', 'Lin Kwok']
    if rhol is not None and rhog is not None:
        methods = liquid_gas_viscosity_correlations_list
    return methods
_gas_liquid_viscosity_method_unknown = 'Method not recognized; available methods are %s' %list(liquid_gas_viscosity_correlations.keys())


def gas_liquid_viscosity(x, mul, mug, rhol=None, rhog=None, Method=None):
    r'''This function handles the calculation of two-phase liquid-gas viscosity.
    Six calculation methods are available; three of them require only `x`,
    `mul`, and `mug`; the other three require `rhol` and `rhog` as well.

    The 'McAdams' method will be used if no method is specified.
    The full list of correlation can be obtained with the `AvailableMethods`
    flag.

    **ALL OF THESE METHODS ARE ONLY SUGGESTED DEFINITIONS, POTENTIALLY
    USEFUL FOR EMPIRICAL WORK ONLY!**

    Parameters
    ----------
    x : float
        Quality of fluid, [-]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    rhol : float, optional
        Liquid density, [kg/m^3]
    rhog : float, optional
        Gas density, [kg/m^3]

    Returns
    -------
    mu_lg : float
        Liquid-gas viscosity (**a suggested definition, potentially useful
        for empirical work only!**) [Pa*s]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        liquid_gas_viscosity_correlations.

    Notes
    -----
    All of these models converge to the liquid or gas viscosity as the quality
    approaches either limits. Other definitions have been proposed, such as
    using only liquid viscosity.

    These values cannot just be plugged into single phase correlations!

    Examples
    --------
    >>> gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2, Method='Duckler')
    1.2092040385066917e-05
    >>> gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5)
    2.4630541871921184e-05
    '''
    if Method is None:
        Method = 'McAdams'

    if Method == 'Beattie Whalley':
        return Beattie_Whalley(x, mul, mug, rhol=rhol, rhog=rhog)
    elif Method == 'Fourar Bories':
        return Fourar_Bories(x, mul, mug, rhol=rhol, rhog=rhog)
    elif Method == 'Duckler':
        return Duckler(x, mul, mug, rhol=rhol, rhog=rhog)
    elif Method == 'McAdams':
        return McAdams(x, mul, mug)
    elif Method == 'Cicchitti':
        return Cicchitti(x, mul, mug)
    elif Method == 'Lin Kwok':
        return Lin_Kwok(x, mul, mug)
    else:
        raise ValueError(_gas_liquid_viscosity_method_unknown)
