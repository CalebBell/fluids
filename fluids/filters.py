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

This module contains correlations for the loss coefficient of various types
of filters in a pipe or channel.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Loss Coefficients for Screens
-----------------------------
.. autofunction:: round_edge_screen
.. autofunction:: round_edge_open_mesh
.. autofunction:: square_edge_screen

Loss Coefficients for Grills
----------------------------
.. autofunction:: square_edge_grill
.. autofunction:: round_edge_grill

"""

from __future__ import division
from math import radians, cos
from fluids.numerics import interp, implementation_optimize_tck, splev

__all__ = ['round_edge_screen', 'round_edge_open_mesh', 'square_edge_screen',
'square_edge_grill', 'round_edge_grill']

round_Res = [20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0]
round_betas = [1.3, 1.1, 0.95, 0.83, 0.75, 0.7, 0.6, 0.52]
'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits'''


round_thetas = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0]
round_gammas = [1.0, 0.97, 0.88, 0.75, 0.59, 0.45, 0.3, 0.23, 0.15, 0.09]
'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits'''

'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits. Last actual value in the original table is
K=1000 at alpha=0.05; the rest are extrapolated.'''
square_alphas = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.]
square_Ks = [1024000., 256000, 64000, 16000, 4000, 1000., 250., 85., 52., 30., 17., 11., 7.7, 5.5, 3.8, 2.8, 2, 1.5, 1.1, 0.78, 0.53, 0.35, 0.08, 0.]


grills_rounded_alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
grills_rounded_Ks = [2.0, 1.0, 0.6, 0.4, 0.2]
'''Cubic interpolation with no smoothing, constant value extremities
returned when outside table limits'''
grills_rounded_tck = implementation_optimize_tck([[0.3, 0.3, 0.3, 0.45, 0.55, 0.7, 0.7, 0.7],
                                                  [2.0, 1.0014285714285716, 0.5799999999999998,
                                                   0.3585714285714287, 0.2, 0.0, 0.0, 0.0],
                                                   2])


def round_edge_screen(alpha, Re, angle=0.0):
    r'''Returns the loss coefficient for a round edged wire screen or bar
    screen, as shown in [1]_. Angle of inclination may be specified as well.

    Parameters
    ----------
    alpha : float
        Fraction of screen open to flow [-]
    Re : float
        Reynolds number of flow through screen with D = space between rods, []
    angle : float, optional
        Angle of inclination, with 0 being straight and 90 being parallel to
        flow [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Linear interpolation between a table of values. Re table extends
    from 20 to 400, with constant values outside of the table. This behavior
    should be adequate.
    alpha should be between 0.05 and 0.8.
    If angle is over 85 degrees, the value at 85 degrees is used.

    The velocity the loss coefficient relates to is the approach velocity
    before the screen.

    Examples
    --------
    >>> round_edge_screen(0.5, 100)
    2.0999999999999996
    >>> round_edge_screen(0.5, 100, 45)
    1.05

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    beta = interp(Re, round_Res, round_betas)
    alpha2 = alpha*alpha
    K = beta*(1.0 - alpha2)/alpha2
    if angle is not None:
        if angle <= 45.0:
            v = cos(radians(angle))
            K *= v*v
        else:
            K *= interp(angle, round_thetas, round_gammas)
    return K


def round_edge_open_mesh(alpha, subtype='diamond pattern wire', angle=0.0):
    r'''Returns the loss coefficient for a round edged open net/screen
    made of one of the following patterns, according to [1]_:

    'round bar screen':

    .. math::
        K = 0.95(1-\alpha) + 0.2(1-\alpha)^2

    'diamond pattern wire':

    .. math::
        K = 0.67(1-\alpha) + 1.3(1-\alpha)^2

    'knotted net':

    .. math::
        K = 0.70(1-\alpha) + 4.9(1-\alpha)^2

    'knotless net':

    .. math::
        K = 0.72(1-\alpha) + 2.1(1-\alpha)^2

    Parameters
    ----------
    alpha : float
        Fraction of net/screen open to flow [-]
    subtype : str
        One of 'round bar screen', 'diamond pattern wire', 'knotted net' or
        'knotless net'.
    angle : float, optional
        Angle of inclination, with 0 being straight and 90 being parallel to
        flow [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    `alpha` should be between 0.85 and 1 for these correlations.
    Flow should be turbulent, with Re > 500.

    The velocity the loss coefficient relates to is the approach velocity
    before the mesh.

    Examples
    --------
    >>> round_edge_open_mesh(0.96, angle=33.)
    0.02031327712601458

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    one_m_alpha = (1.0-alpha)
    if subtype == 'round bar screen':
        K = 0.95 + 0.2*one_m_alpha
    elif subtype == 'diamond pattern wire':
        K = 0.67 + 1.3*one_m_alpha
    elif subtype == 'knotted net':
        K = 0.70 + 4.9*one_m_alpha
    elif subtype == 'knotless net':
        K = 0.72 + 2.1*one_m_alpha
    else:
        raise ValueError('Subtype not recognized')
    K *= one_m_alpha
    if angle is not None:
        if angle < 45.0:
            K *= cos(radians(angle))**2.0
        else:
            K *= interp(angle, round_thetas, round_gammas)
    return K


def square_edge_screen(alpha):
    r'''Returns the loss coefficient for a square wire screen or square bar
    screen or perforated plate with squared edges, as shown in [1]_.

    Parameters
    ----------
    alpha : float
        Fraction of screen open to flow [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Linear interpolation between a table of values.
    The velocity the loss coefficient relates to is the approach velocity
    before the screen.

    Examples
    --------
    >>> square_edge_screen(0.99)
    0.008000000000000007

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    return interp(alpha, square_alphas, square_Ks)


def square_edge_grill(alpha, l=None, Dh=None, fd=None):
    r'''Returns the loss coefficient for a square grill or square bar
    screen or perforated plate with squared edges of thickness l, as shown in
    [1]_.

    for Dh < l < 50D

    .. math::
        K = \frac{0.5(1-\alpha) + (1-\alpha^2)}{\alpha^2}

    else:

    .. math::
        K = \frac{0.5(1-\alpha) + (1-\alpha^2) + f{l}/D}{\alpha^2}

    Parameters
    ----------
    alpha : float
        Fraction of grill open to flow [-]
    l : float, optional
        Thickness of the grill or plate [m]
    Dh : float, optional
        Hydraulic diameter of gap in grill, [m]
    fd : float, optional
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    If l, Dh, or fd is not provided, the first expression is used instead.
    The alteration of the expression to include friction factor is there
    if the grill is long enough to have considerable friction along the
    surface of the grill.

    The velocity the loss coefficient relates to is the approach velocity
    before the grill.

    Examples
    --------
    >>> square_edge_grill(.45)
    5.296296296296296
    >>> square_edge_grill(.45, l=.15, Dh=.002, fd=.0185)
    12.148148148148147

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    x0 = 0.5*(1.0 - alpha)
    alpha2 = alpha*alpha
    x0 += (1.0 - alpha2)
    if Dh is not None and l is not None and fd is not None and l > 50.0*Dh:
        x0 += fd*l/Dh
    return x0/alpha2


def round_edge_grill(alpha, l=None, Dh=None, fd=None):
    r'''Returns the loss coefficient for a rounded square grill or square bar
    screen or perforated plate with rounded edges of thickness l, as shown in
    [1]_.

    for Dh < l < 50D

    .. math::
        K = lookup(alpha)

    else:

    .. math::
        K = lookup(alpha) + \frac{fl}{\alpha^2D}

    Parameters
    ----------
    alpha : float
        Fraction of grill open to flow [-]
    l : float, optional
        Thickness of the grill or plate [m]
    Dh : float, optional
        Hydraulic diameter of gap in grill, [m]
    fd : float, optional
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    If l, Dh, or fd is not provided, the first expression is used instead.
    The alteration of the expression to include friction factor is there
    if the grill is long enough to have considerable friction along the
    surface of the grill.
    alpha must be between 0.3 and 0.7.

    The velocity the loss coefficient relates to is the approach velocity
    before the grill.

    Examples
    --------
    >>> round_edge_grill(.4)
    1.0
    >>> round_edge_grill(.4, l=.15, Dh=.002, fd=.0185)
    2.3874999999999997

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    t1 = float(splev(alpha, grills_rounded_tck))
    if Dh and l and fd and l > 50.0*Dh:
        return t1 + fd*l/Dh
    else:
        return t1
