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

This module contains functionality for calculating rating and designing
vapor-liquid separators.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Functions
---------
.. autofunction :: v_Sounders_Brown
.. autofunction :: K_separator_Watkins
.. autofunction :: K_separator_demister_York
.. autofunction :: K_Sounders_Brown_theoretical
"""

from __future__ import division
from math import log, exp, pi, sqrt
from fluids.constants import g, foot, psi
from fluids.numerics import splev, implementation_optimize_tck

__all__ = ['v_Sounders_Brown', 'K_separator_Watkins',
           'K_separator_demister_York', 'K_Sounders_Brown_theoretical']


# 92 points taken from a 2172x3212 page scan, after dewarping the scan,
# digitization with Engauge Digitizer, and extensive checking; every 5th point
# it produced was selected plus the last point. The initial value is adjusted
# to be the lower limit of the graph.


tck_Watkins = implementation_optimize_tck([[-5.115995809754082, -5.115995809754082, -5.115995809754082,
                                            -5.115995809754082, -4.160106231099973, -3.209113630523477,
                                            -1.2175106961204154, 0.4587657198189318, 1.1197669427405068,
                                            1.6925908552310418, 1.6925908552310418, 1.6925908552310418,
                                            1.6925908552310418],
                                        [-1.4404286048266364, -1.2375168139385286, -0.9072614905522024,
                                         -0.7662335745829165, -0.944537665617708, -1.957339717378027,
                                         -3.002614318094637, -3.5936804378352956, -3.8779153181940553,
                                         0.0, 0.0, 0.0, 0.0],
                                         3])

def K_separator_Watkins(x, rhol, rhog, horizontal=False, method='spline'):
    r'''Calculates the Sounders-Brown `K` factor as used in determining maximum
    allowable gas velocity in a two-phase separator in either a horizontal or
    vertical orientation. This function approximates a graph published in [1]_
    to determine `K` as used in the following equation:

    .. math::
        v_{max} =  K_{SB}\sqrt{\frac{\rho_l-\rho_g}{\rho_g}}

    The graph has `K_{SB}` on its y-axis, and the following as its x-axis:

    .. math::
        \frac{m_l}{m_g}\sqrt{\rho_g/\rho_l}
        = \frac{(1-x)}{x}\sqrt{\rho_g/\rho_l}

    Cubic spline interpolation is the default method of retrieving a value
    from the graph, which was digitized with Engauge-Digitizer.

    Also supported are two published curve fits to the graph. The first is that
    of Blackwell (1984) [2]_, as follows:

    .. math::
        K_{SB} = \exp(-1.942936 -0.814894X -0.179390 X^2 -0.0123790 X^3
        + 0.000386235 X^4 + 0.000259550 X^5)

        X = \ln\left[\frac{(1-x)}{x}\sqrt{\rho_g/\rho_l}\right]

    The second is that of Branan (1999), as follows:

    .. math::
        K_{SB} = \exp(-1.877478097 -0.81145804597X -0.1870744085 X^2
        -0.0145228667 X^3 -0.00101148518 X^4)

        X = \ln\left[\frac{(1-x)}{x}\sqrt{\rho_g/\rho_l}\right]

    Parameters
    ----------
    x : float
        Quality of fluid entering separator, [-]
    rhol : float
        Density of liquid phase [kg/m^3]
    rhog : float
        Density of gas phase [kg/m^3]
    horizontal : bool, optional
        Whether to use the vertical or horizontal value; horizontal is 1.25
        higher
    method : str
        One of 'spline, 'blackwell', or 'branan'

    Returns
    -------
    K : float
        Sounders Brown horizontal or vertical `K` factor for two-phase
        separator design only, [m/s]

    Notes
    -----
    Both the 'branan' and 'blackwell' models are used frequently. However,
    the spline is much more accurate.

    No limits checking is enforced. However, the x-axis spans only 0.006 to
    5.4, and the function should not be used outside those limits.

    Examples
    --------
    >>> K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True)
    0.07951613600476297

    References
    ----------
    .. [1] Watkins (1967). Sizing Separators and Accumulators, Hydrocarbon
       Processing, November 1967.
    .. [2] Blackwell, W. Wayne. Chemical Process Design on a Programmable
       Calculator. New York: Mcgraw-Hill, 1984.
    .. [3] Branan, Carl R. Pocket Guide to Chemical Engineering. 1st edition.
       Houston, Tex: Gulf Professional Publishing, 1999.
    '''
    factor = (1. - x)/x*sqrt(rhog/rhol)
    if method == 'spline':
        K = exp(float(splev(log(factor), tck_Watkins)))
    elif method == 'blackwell':
        X = log(factor)
        A = -1.877478097
        B = -0.81145804597
        C = -0.1870744085
        D = -0.0145228667
        E = -0.00101148518
        K = exp(A + X*(B + X*(C + X*(D + E*X))))
    elif method == 'branan':
        X = log(factor)
        A = -1.942936
        B = -0.814894
        C = -0.179390
        D = -0.0123790
        E = 0.000386235
        F = 0.000259550
        K = exp(A + X*(B + X*(C + X*(D + X*(E + F*X)))))
    else:
        raise ValueError("Only methods 'spline', 'branan', and 'blackwell' are supported.")
    K *= foot # Converts units of ft/s to m/s; the graph and all fits are in ft/s
    if horizontal:
        K *= 1.25 # Watkins recommends a factor of 1.25 for horizontal separators over vertical separators
    return K


def K_separator_demister_York(P, horizontal=False):
    r'''Calculates the Sounders Brown `K` factor as used in determining maximum
    permissible gas velocity in a two-phase separator in either a horizontal or
    vertical orientation, *with a demister*.
    This function is a curve fit to [1]_ published in [2]_ and is widely used.

    For 1 < P < 15 psia:

    .. math::
        K = 0.1821 + 0.0029P + 0.0460\ln P

    For 15 <= P <= 40 psia:

    .. math::
        K = 0.35

    For P < 5500 psia:

    .. math::
        K = 0.430 - 0.023\ln P

    In the above equations, P is in units of psia.

    Parameters
    ----------
    P : float
        Pressure of separator, [Pa]
    horizontal : bool, optional
        Whether to use the vertical or horizontal value; horizontal is 1.25
        times higher, [-]

    Returns
    -------
    K : float
        Sounders Brown Horizontal or vertical `K` factor for two-phase
        separator design with a demister, [m/s]

    Notes
    -----
    If the input pressure is under 1 psia, 1 psia is used. If the
    input pressure is over 5500 psia, 5500 psia is used.

    Examples
    --------
    >>> K_separator_demister_York(975*psi)
    0.08281536035331669

    References
    ----------
    .. [2] Otto H. York Company, "Mist Elimination in Gas Treatment Plants and
       Refineries," Engineering, Parsippany, NJ.
    .. [1] Svrcek, W. Y., and W. D. Monnery. "Design Two-Phase Separators
       within the Right Limits" Chemical Engineering Progress, (October 1,
       1993): 53-60.
    '''
    P = P/psi # Correlation in terms of psia
    if P < 15:
        if P < 1:
            P = 1 # Prevent negative K values, but as a consequence be
            # optimistic for K values; limit is 0.185 ft/s but real values
            # should probably be lower
        K = 0.1821 + 0.0029*P + 0.0460*log(P)
    elif P < 40:
        K = 0.35
    else:
        if P > 5500:
            P = 5500 # Do not allow for lower K values above 5500 psia, as
            # the limit is stated to be 5500
        K = 0.430 - 0.023*log(P)
    K *= foot # Converts units of ft/s to m/s; the graph and all fits are in ft/s
    if horizontal:
        # Watkins recommends a factor of 1.25 for horizontal separators over
        # vertical separators as well
        K *= 1.25
    return K


def v_Sounders_Brown(K, rhol, rhog):
    r'''Calculates the maximum allowable vapor velocity in a two-phase
    separator to permit separation between entrained droplets and the gas
    using an empirical `K` factor, named after Sounders and Brown [1]_.
    This is a simplifying expression for terminal velocity and drag on
    particles.

    .. math::
        v_{max} =  K_{SB} \sqrt{\frac{\rho_l-\rho_g}{\rho_g}}

    Parameters
    ----------
    K : float
        Sounders Brown `K` factor for two-phase separator design, [m/s]
    rhol : float
        Density of liquid phase [kg/m^3]
    rhog : float
        Density of gas phase [kg/m^3]

    Returns
    -------
    v_max : float
        Maximum allowable vapor velocity in a two-phase separator to permit
        separation between entrained droplets and the gas, [m/s]

    Notes
    -----
    The Sounders Brown K factor is related to the terminal velocity as shown in
    the following expression.

    .. math::
        v_{term} = v_{max} = \sqrt{\frac{4 g d_p (\rho_p-\rho_f)}{3 C_D \rho_f }}

        v_{term} = \sqrt{\frac{(\rho_p-\rho_f)}{\rho_f}} \sqrt{\frac{4 g d_p}{3 C_D}}

        v_{term} = K_{SB} \sqrt{\frac{4 g d_p}{3 C_D}}

    Note this form corresponds to the Newton's law range (Re > 500), but in
    reality droplets are normally in the intermediate or Stoke's law region
    [2]_. For this reason using the drag coefficient expression directly is
    cleaner, but identical results can be found with the Sounders Brown
    equation.

    Examples
    --------
    >>> v_Sounders_Brown(K=0.08, rhol=985.4, rhog=1.3)
    2.2010906387516167

    References
    ----------
    .. [1] Souders, Mott., and George Granger. Brown. "Design of Fractionating
       Columns I. Entrainment and Capacity." Industrial & Engineering Chemistry
       26, no. 1 (January 1, 1934): 98-103. https://doi.org/10.1021/ie50289a025.
    .. [2] Vasude, Gael D. Ulrich and Palligarnai T. Chemical Engineering
       Process Design and Economics : A Practical Guide. 2nd edition. Durham,
       N.H: Process Publishing, 2004.
    '''
    return K*sqrt((rhol - rhog)/rhog)


def K_Sounders_Brown_theoretical(D, Cd, g=g):
    r'''Converts a known drag coefficient into a Sounders-Brown `K` factor
    for two-phase separator design. This factor is the traditional way for
    separator diameters to be obtained although it is unnecessary and the
    theoretical drag coefficient method can be used instead.

    .. math::
        K_{SB} = \sqrt{\frac{(\rho_p-\rho_f)}{\rho_f}}
        = \sqrt{\frac{4 g d_p}{3 C_D}}

    Parameters
    ----------
    D : float
        Design diameter of the droplets, [m]
    Cd : float
        Drag coefficient [-]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    K : float
        Sounders Brown `K` factor for two-phase separator design, [m/s]

    Notes
    -----
    Drag coefficient is a function of velocity; so iteration is needed to
    obtain the most correct answer. The following example shows the use of
    iteration to obtain the final velocity:

    >>> from fluids import *
    >>> V = 2.0
    >>> D = 150E-6
    >>> rho = 1.3
    >>> rhol = 700.
    >>> mu = 1E-5
    >>> for i in range(10):
    ...     Re = Reynolds(V=V, rho=rho, mu=mu, D=D)
    ...     Cd = drag_sphere(Re)
    ...     K = K_Sounders_Brown_theoretical(D=D, Cd=Cd)
    ...     V = v_Sounders_Brown(K, rhol=rhol, rhog=rho)
    ...     print('%.14f' %V)
    0.76093307417658
    0.56242939340131
    0.50732895050696
    0.48957142095508
    0.48356021946899
    0.48149076033622
    0.48077414934614
    0.48052549959141
    0.48043916249756
    0.48040917690193

    The use of Sounders-Brown constants can be replaced as follows (the
    v_terminal method includes its own solver for terminal velocity):

    >>> from fluids.drag import v_terminal
    >>> v_terminal(D=D, rhop=rhol, rho=rho, mu=mu)
    0.4803932186998

    Examples
    --------
    >>> K_Sounders_Brown_theoretical(D=150E-6, Cd=0.5)
    0.06263114241333939

    References
    ----------
    .. [1] Svrcek, W. Y., and W. D. Monnery. "Design Two-Phase Separators
       within the Right Limits" Chemical Engineering Progress, (October 1,
       1993): 53-60.
    '''
    return sqrt((4.0/3.0)*g*D/(Cd))
