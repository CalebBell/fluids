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

from __future__ import division
from scipy.interpolate import interp1d, UnivariateSpline
from math import radians, cos

__all__ = ['round_edge_screen', 'round_edge_open_mesh', 'square_edge_screen',
'square_edge_grill', 'round_edge_grill']

round_Res = [20, 30, 40, 60, 80, 100, 200, 400]
round_betas = [1.3, 1.1, 0.95, 0.83, 0.75, 0.7, 0.6, 0.52]
#round_interp = interp1d(round_Res, round_betas, kind='linear')
'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits'''
round_interp = UnivariateSpline(round_Res, round_betas, s=0, k=1)


round_thetas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85]
round_gammas = [1, 0.97, 0.88, 0.75, 0.59, 0.45, 0.3, 0.23, 0.15, 0.09]
#inclined_round_interp = interp1d(round_thetas, round_gammas, kind='linear')
'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits'''
inclined_round_interp = UnivariateSpline(round_thetas, round_gammas, s=0, k=1)

#square_alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.]
#square_Ks = [100000., 1000., 250., 85., 52., 30., 17., 11., 7.7, 5.5, 3.8, 2.8, 2, 1.5, 1.1, 0.78, 0.53, 0.35, 0.08, 0.]
#square_interp = interp1d(square_alphas, square_Ks, kind='linear')
'''Quadratic interpolation with no smoothing, constant value extremities
returned when outside table limits. Last actual value in the original table is
K=1000 at alpha=0.05; the rest are extrapolated.'''
square_alphas = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.]
square_Ks = [1024000.,256000, 64000, 16000, 4000, 1000., 250., 85., 52., 30., 17., 11., 7.7, 5.5, 3.8, 2.8, 2, 1.5, 1.1, 0.78, 0.53, 0.35, 0.08, 0.]
square_interp = UnivariateSpline(square_alphas, square_Ks, s=0, k=1)


grills_rounded_alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
grills_rounded_Ks = [2, 1, 0.6, 0.4, 0.2]
#grills_rounded_interp = interp1d(grills_rounded_alphas, grills_rounded_Ks, kind='linear')
'''Cubic interpolation with no smoothing, constant value extremities
returned when outside table limits'''
grills_rounded_interp = UnivariateSpline(grills_rounded_alphas, grills_rounded_Ks, s=0, k=2)

def round_edge_screen(alpha, Re, angle=0):
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
    beta = float(round_interp(Re))
    K = beta*(1 - alpha**2)/alpha**2
    if angle:
        if angle <= 45:
            K *= cos(radians(angle))**2
        else:
            K *= float(inclined_round_interp(angle))
    return K


def round_edge_open_mesh(alpha, subtype='diamond pattern wire', angle=0):
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

    Examples
    --------
    >>> round_edge_open_mesh(0.96, angle=33.)
    0.02031327712601458

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    if subtype == 'round bar screen':
        K = 0.95*(1-alpha) + 0.2*(1-alpha)**2
    elif subtype == 'diamond pattern wire':
        K = 0.67*(1-alpha) + 1.3*(1-alpha)**2
    elif subtype == 'knotted net':
        K = 0.70*(1-alpha) + 4.9*(1-alpha)**2
    elif subtype == 'knotless net':
        K = 0.72*(1-alpha) + 2.1*(1-alpha)**2
    else:
        raise Exception('Subtype not recognized')
    if angle:
        if angle < 45:
            K *= cos(radians(angle))**2
        else:
            K *= float(inclined_round_interp(angle))
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

    Examples
    --------
    >>> square_edge_screen(0.99)
    0.008000000000000009

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    K = float(square_interp(alpha))
    return K


def square_edge_grill(alpha=None, l=None, Dh=None, fd=None):
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
    l : float
        Thickness of the grill or plate [m]
    Dh : float
        Hydraulic diameter of gap in grill, [m]
    fd : float
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
    if Dh and l and fd and l > 50*Dh:
        K = (0.5*(1-alpha) + (1-alpha**2) + fd*l/Dh)/alpha**2
    else:
        K = (0.5*(1-alpha) + (1-alpha**2))/alpha**2
    return K


def round_edge_grill(alpha=None, l=None, Dh=None, fd=None):
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
    if Dh and l and fd and l > 50*Dh:
        K = float(grills_rounded_interp(alpha)) + fd*l/Dh
    else:
        K = float(grills_rounded_interp(alpha))
    return K

