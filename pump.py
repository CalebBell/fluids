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

from math import log

def Corripio_pump_efficiency(Q):
    r'''Estimates pump efficiency using the method in Corripio (1982)
    as shown in [1]_ and originally in [2]_. Estimation only

    .. math::
        \eta_P = -0.316 + 0.24015\ln(Q) - 0.01199\ln(Q)^2

    Parameters
    ----------
    Q : float
        Volumetric flow rate, [m^3/s]

    Returns
    -------
    effciency : float
        Pump efficiency, [-]

    Notes
    -----
    For Centrifugal pumps only.
    Range is 50 to 5000 GPM, but input variable is in metric.
    Values above this range and below this range will go negative,
    although small deviations are acceptable.
    Example 16.5 in [1]_.

    Examples
    --------
    >>> Corripio_pump_efficiency(461./15850.323)
    0.7058888670951621

    References
    ----------
    .. [1] Seider, Warren D., J. D. Seader, and Daniel R. Lewin. Product and
       Process Design Principles: Synthesis, Analysis, and Evaluation.
       2 edition. New York: Wiley, 2003.
    .. [2] Corripio, A.B., K.S. Chrien, and L.B. Evans, "Estimate Costs of
       Centrifugal Pumps and Electric Motors," Chem. Eng., 89, 115-118,
       February 22 (1982).
    '''
    Q *= 15850.323
    eta = -0.316 + 0.24015*log(Q) - 0.01199*log(Q)**2
    return eta

#print [Corripio_pump_efficiency(461./15850.323)]


def Corripio_motor_efficiency(P):
    r'''Estimates motor efficiency using the method in Corripio (1982)
    as shown in [1]_ and originally in [2]_. Estimation only.

    .. math::
        \eta_M = 0.8  + 0.0319\ln(P_B) - 0.00182\ln(P_B)^2

    Parameters
    ----------
    P : float
        Power, [W]

    Returns
    -------
    effciency : float
        Motor efficiency, [-]

    Notes
    -----
    Example 16.5 in [1]_.

    Examples
    --------
    >>> Corripio_motor_efficiency(137*745.7)
    0.9128920875679222

    References
    ----------
    .. [1] Seider, Warren D., J. D. Seader, and Daniel R. Lewin. Product and
       Process Design Principles: Synthesis, Analysis, and Evaluation.
       2 edition. New York: Wiley, 2003.
    .. [2] Corripio, A.B., K.S. Chrien, and L.B. Evans, "Estimate Costs of
       Centrifugal Pumps and Electric Motors," Chem. Eng., 89, 115-118,
       February 22 (1982).
    '''
    P = P/745.69987
    eta = 0.8 + 0.0319*log(P) - 0.00182*log(P)**2
    return eta

#print [Corripio_motor_efficiency(137*745.7)]