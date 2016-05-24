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
from math import log
from collections import namedtuple
from scipy.interpolate import interp1d, interp2d
from scipy.constants import hp
import os

__all__ = ['VFD_efficiency', 'CSA_motor_efficiency', 'motor_efficiency_underloaded',
'Corripio_pump_efficiency', 'Corripio_motor_efficiency',
'specific_speed', 'specific_diameter', 'speed_synchronous', 'nema_sizes',
'nema_sizes_hp', 'motor_round_size', 'nema_min_P', 'nema_high_P', 'plug_types',
'voltages_1_phase_residential', 'voltages_3_phase', 'frequencies',
'residential_power', 'industrial_power', 'current_ideal']

folder = os.path.join(os.path.dirname(__file__), 'data')


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


VFD_efficiencies = [[0.31, 0.77, 0.86, 0.9, 0.91, 0.93, 0.94],
                    [0.35, 0.8, 0.88, 0.91, 0.92, 0.94, 0.95],
                    [0.41, 0.83, 0.9, 0.93, 0.94, 0.95, 0.96],
                    [0.47, 0.86, 0.93, 0.94, 0.95, 0.96, 0.97],
                    [0.5, 0.88, 0.93, 0.95, 0.95, 0.96, 0.97],
                    [0.46, 0.86, 0.92, 0.95, 0.95, 0.96, 0.97],
                    [0.51, 0.87, 0.92, 0.95, 0.95, 0.96, 0.97],
                    [0.47, 0.86, 0.93, 0.95, 0.96, 0.97, 0.97],
                    [0.55, 0.89, 0.94, 0.95, 0.96, 0.97, 0.97],
                    [0.61, 0.91, 0.95, 0.96, 0.96, 0.97, 0.97],
                    [0.61, 0.91, 0.95, 0.96, 0.96, 0.97, 0.97]]
VFD_efficiency_interp = interp2d([0.016, 0.125, 0.25, 0.42, 0.5, 0.75, 1],
                                 [3, 5, 10, 20, 30, 50, 60, 75, 100, 200, 400],
                                 VFD_efficiencies)


def VFD_efficiency(P, load=1):
    r'''Returns the efficiency of a Variable Frequency Drive according to [1]_.
    These values are generic, and not standardized as minimum values.
    Older VFDs often have much worse performance.

    Parameters
    ----------
    P : float
        Power, [W]
    load : float, optional
        Fraction of motor's rated electrical capacity being used

    Returns
    -------
    effciency : float
        VFD efficiency, [-]

    Notes
    -----
    The use of a VFD does change the characteristics of a pump curve's
    efficiency, but this has yet to be quantified. The effect is small.
    This value should be multiplied by the product of the pump and motor
    efficiency to determine the overall efficiency.

    Efficiency table is in units of hp, so a conversion is performed internally.
    If load not specified, assumed 1 - where maximum efficiency occurs.
    Table extends down to 3 hp and up to 400 hp; values outside these limits
    are rounded to the nearest known value. Values between standardized sizes
    are interpolated linearly. Load values extend down to 0.016.

    Examples
    --------
    >>> VFD_efficiency(10*hp)
    0.96
    >>> VFD_efficiency(100*hp, load=0.5)
    0.96

    References
    ----------
    .. [1] GoHz.com. Variable Frequency Drive Efficiency.
       http://www.variablefrequencydrive.org/vfd-efficiency
    '''
    P = P/hp
    if P < 3:
        P = 3
    elif P > 400:
        P = 400
    if load < 0.016:
        load = 0.016
    efficiency = round(float(VFD_efficiency_interp(load, P)), 4)
    return efficiency


nema_sizes_hp = [.25, 1/3., .5, .75, 1, 1.5, 2, 3, 4, 5, 5.5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
'''list: all NEMA motor sizes in increasing order, in horsepower.
'''
nema_sizes = [i*hp for i in nema_sizes_hp]
'''list: all NEMA motor sizes in increasing order, in Watts.
'''


def motor_round_size(P):
    r'''Rounds up the power for a motor to the nearest NEMA standard power.
    The returned power is always larger or equal to the input power.

    Parameters
    ----------
    P : float
        Power, [W]

    Returns
    -------
    P_actual : float
        Actual power, equal to or larger than input [W]

    Notes
    -----
    An exception is raised if the power required is larger than any of
    the NEMA sizes. Larger motors are available, but are unstandardized.

    Examples
    --------
    >>> motor_round_size(1E5)
    111854.98073734052

    References
    ----------
    .. [1] Natural Resources Canada. Electric Motors (1 to 500 HP/0.746 to
       375 kW). As modified 2015-12-17.
       https://www.nrcan.gc.ca/energy/regulations-codes-standards/products/6885
    '''
    for P_actual in nema_sizes:
        if P_actual >= P:
            return P_actual
    raise Exception('Required power is larger than can be provided with one valve')


nema_high_P = [1, 1.5, 2, 3, 4, 5, 5.5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]
nema_high_full_open_2p = [0.77, 0.84, 0.855, 0.855, 0.865, 0.865, 0.865, 0.885, 0.895, 0.902, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95]
nema_high_full_open_4p = [0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.91, 0.917, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.954, 0.954, 0.958, 0.958, 0.958]
nema_high_full_open_6p = [0.825, 0.865, 0.875, 0.885, 0.895, 0.895, 0.895, 0.902, 0.917, 0.917, 0.924, 0.93, 0.936, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954]
nema_high_full_closed_2p = [0.77, 0.84, 0.855, 0.865, 0.885, 0.885, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.95, 0.95, 0.954, 0.954]
nema_high_full_closed_4p = [0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.945, 0.95, 0.954, 0.954, 0.954, 0.958, 0.962, 0.962]
nema_high_full_closed_6p = [0.825, 0.875, 0.885, 0.895, 0.895, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.958, 0.958, 0.958]

nema_high_full_open_2p_i = interp1d(nema_high_P, nema_high_full_open_2p)
nema_high_full_open_4p_i = interp1d(nema_high_P, nema_high_full_open_4p)
nema_high_full_open_6p_i = interp1d(nema_high_P, nema_high_full_open_6p)

nema_high_full_closed_2p_i = interp1d(nema_high_P, nema_high_full_closed_2p)
nema_high_full_closed_4p_i = interp1d(nema_high_P, nema_high_full_closed_4p)
nema_high_full_closed_6p_i = interp1d(nema_high_P, nema_high_full_closed_6p)

nema_min_P = [1, 1.5, 2, 3, 4, 5, 5.5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
nema_min_full_open_2p  = [0.755, 0.825, 0.84, 0.84, 0.84, 0.855, 0.855, 0.875, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.93, 0.936, 0.936, 0.945, 0.945, 0.945, 0.95, 0.95, 0.954, 0.958, 0.958]
nema_min_full_open_4p = [0.825, 0.84, 0.84, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.958]
nema_min_full_open_6p = [0.8, 0.84, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.902, 0.902, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954]
nema_min_full_open_8p = [0.74, 0.755, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]
nema_min_full_closed_2p = [0.755, 0.825, 0.84, 0.855, 0.855, 0.875, 0.875, 0.885, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954]
nema_min_full_closed_4p = [0.825, 0.84, 0.84, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.91, 0.91, 0.924, 0.924, 0.93, 0.93, 0.936, 0.941, 0.945, 0.945, 0.95, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958]
nema_min_full_closed_6p = [0.8, 0.855, 0.865, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.902, 0.902, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
nema_min_full_closed_8p = [0.74, 0.77, 0.825, 0.84, 0.84, 0.855, 0.855, 0.855, 0.885, 0.885, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]

nema_min_full_open_2p_i = interp1d(nema_min_P, nema_min_full_open_2p)
nema_min_full_open_4p_i = interp1d(nema_min_P, nema_min_full_open_4p)
nema_min_full_open_6p_i = interp1d(nema_min_P, nema_min_full_open_6p)
nema_min_full_open_8p_i = interp1d(nema_min_P, nema_min_full_open_8p)

nema_min_full_closed_2p_i = interp1d(nema_min_P, nema_min_full_closed_2p)
nema_min_full_closed_4p_i = interp1d(nema_min_P, nema_min_full_closed_4p)
nema_min_full_closed_6p_i = interp1d(nema_min_P, nema_min_full_closed_6p)
nema_min_full_closed_8p_i = interp1d(nema_min_P, nema_min_full_closed_8p)

#print nema_min_full_closed_8p_i(345)

def CSA_motor_efficiency(P, closed=False, poles=2, high_efficiency=False):
    r'''Returns the efficiency of a NEMA motor according to [1]_.
    These values are standards, but are only for full-load operation.

    Parameters
    ----------
    P : float
        Power, [W]
    closed : bool, optional
        Whether or not the motor is enclosed
    poles : int, optional
        The number of poles of the motor
    high_efficiency : bool, optional
        Whether or not to look up the high-efficiency value

    Returns
    -------
    effciency : float
        Guaranteed full-load motor efficiency, [-]

    Notes
    -----
    Criteria for being required to meet the high-efficiency standard is:

    * Designed for continuous operation
    * Operates by three-phase induction
    * Is a squirrel-cage or cage design
    * Is NEMA type A, B, or C with T or U frame; or IEC design N or H
    * Is designed for single-speed operation
    * Has a nominal voltage of less than 600 V AC
    * Has a nominal frequency of 60 Hz or 50/60 Hz
    * Has 2, 4, or 6 pole construction
    * Is either open or closed

    Pretty much every motor is required to meet the low-standard efficiency
    table, however.

    Several low-efficiency standard high power values were added to allow for
    easy programming; values are the last listed efficiency in the table.

    Examples
    --------
    >>> CSA_motor_efficiency(100*hp)
    0.93
    >>> CSA_motor_efficiency(100*hp, closed=True, poles=6, high_efficiency=True)
    0.95

    References
    ----------
    .. [1] Natural Resources Canada. Electric Motors (1 to 500 HP/0.746 to
       375 kW). As modified 2015-12-17.
       https://www.nrcan.gc.ca/energy/regulations-codes-standards/products/6885
    '''
    P = P/hp
    if high_efficiency:
        if closed:
            if poles == 2:
                efficiency = nema_high_full_closed_2p_i(P)
            elif poles == 4:
                efficiency = nema_high_full_closed_4p_i(P)
            elif poles == 6:
                efficiency = nema_high_full_closed_6p_i(P)
        else:
            if poles == 2:
                efficiency = nema_high_full_open_2p_i(P)
            elif poles == 4:
                efficiency = nema_high_full_open_4p_i(P)
            elif poles == 6:
                efficiency = nema_high_full_open_6p_i(P)
    else:
        if closed:
            if poles == 2:
                efficiency = nema_min_full_closed_2p_i(P)
            elif poles == 4:
                efficiency = nema_min_full_closed_4p_i(P)
            elif poles == 6:
                efficiency = nema_min_full_closed_6p_i(P)
            elif poles == 8:
                efficiency = nema_min_full_closed_8p_i(P)
        else:
            if poles == 2:
                efficiency = nema_min_full_open_2p_i(P)
            elif poles == 4:
                efficiency = nema_min_full_open_4p_i(P)
            elif poles == 6:
                efficiency = nema_min_full_open_6p_i(P)
            elif poles == 8:
                efficiency = nema_min_full_open_8p_i(P)
    efficiency = round(float(efficiency), 4)
    return efficiency

# Test high efficiency:
#print([CSA_motor_efficiency(k*hp, high_efficiency=False, closed=i, poles=j) for i in [True, False] for j in [2, 4, 6, 8] for k in nema_min_P])



_to_1 = [0.015807118828266818, 4.3158627514876216, -8.5612097969025438, 8.2040355039147386, -3.0147603718043068]
_to_5 = [0.015560190519232379, 4.5699731811493152, -7.6800154569463883, 5.4701698738380813, -1.3630071852989643]
_to_10 = [0.059917274403963446, 6.356781885851186, -17.099192527703369, 20.707077651470666, -9.2215133149377841]
_to_25 = [0.29536141765389839, 4.9918188632064329, -13.785081664656504, 16.908273659093812, -7.5816775136809609]
_to_60 = [0.46934299949154384, 4.0298663805446004, -11.632822556859477, 14.616967043793032, -6.6284514347522245]
_to_infty = [0.68235730304242914, 2.4402956771025748, -6.8306770996860182, 8.2108432911172713, -3.5629309804411577]
_efficiency_lists = [_to_1, _to_5, _to_10, _to_25, _to_60, _to_infty]
_efficiency_ones = [0.9218102, 0.64307597, 0.61724113, 0.61569791, 0.6172238, 0.40648294]

def motor_efficiency_underloaded(P, load=0.5):
    r'''Returns the efficiency of a motor opperating under its design power
    according to [1]_.These values are generic; manufacturers usually list 4
    points on their product information, but full-scale data is hard to find
    and not regulated.

    Parameters
    ----------
    P : float
        Power, [W]
    load : float, optional
        Fraction of motor's rated electrical capacity being used

    Returns
    -------
    effciency : float
        Motor efficiency, [-]

    Notes
    -----
    If the efficiency returned by this function is unattractive, use a VFD.
    The curves used here are polynomial fits to [1]_'s graph, and curves were
    available for the following motor power ranges:
    0-1 hp, 1.5-5 hp, 10 hp, 15-25 hp, 30-60 hp, 75-100 hp
    If above the upper limit of one range, the next value is returned.

    Examples
    --------
    >>> motor_efficiency_underloaded(1*hp)
    0.8705179600980149
    >>> motor_efficiency_underloaded(10.1*hp,  .1)
    0.6728425932357025

    References
    ----------
    .. [1] Washington State Energy Office. Energy-Efficient Electric Motor
       Selection Handbook. 1993.
    '''
    P = P/hp
    if P <=1:
        i = 0
    elif P <= 5:
        i = 1
    elif P <= 10:
        i = 2
    elif P <= 25:
        i = 3
    elif P <= 60:
        i = 4
    else:
        i = 5
    if load > _efficiency_ones[i]:
        efficiency = 1
    else:
        cs = _efficiency_lists[i]
        efficiency = cs[0] + cs[1]*load + cs[2]*load**2 + cs[3]*load**3 + cs[4]*load**4
    return efficiency


def specific_speed(Q, H, n=3600.):
    r'''Returns the specific speed of a pump operating at a specified Q, H,
    and n.

    .. math::
        n_S = \frac{n\sqrt{Q}}{H^{0.75}}

    Parameters
    ----------
    Q : float
        Flow rate, [m^3/s]
    H : float
        Head generated by the pump, [m]
    n : float, optional
        Speed of pump [rpm]

    Returns
    -------
    nS : float
        Specific Speed, [rpm*m^0.75/s^0.5]

    Notes
    -----
    Defined at the BEP, with maximum fitting diameter impeller, at a given
    rotational speed.

    Examples
    --------
    Example from [1]_.

    >>> specific_speed(0.0402, 100, 3550)
    22.50823182748925

    References
    ----------
    .. [1] HI 1.3 Rotodynamic Centrifugal Pumps for Design and Applications
    '''
    nS = n*Q**0.5/H**0.75
    return nS


def specific_diameter(Q, H, D):
    r'''Returns the specific diameter of a pump operating at a specified Q, H,
    and D.

    .. math::
        D_s = \frac{DH^{1/4}}{\sqrt{Q}}

    Parameters
    ----------
    Q : float
        Flow rate, [m^3/s]
    H : float
        Head generated by the pump, [m]
    D : float
        Pump impeller diameter [m]

    Returns
    -------
    Ds : float
        Specific diameter, [m^0.25/s^0.5]

    Notes
    -----
    Used in certain pump sizing calculations.

    Examples
    --------
    >>> specific_diameter(Q=0.1, H=10., D=0.1)
    0.5623413251903491

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    Ds = D*H**0.25/Q**0.5
    return Ds


def speed_synchronous(f, poles=2, phase=3):
    r'''Returns the synchronous speed of a synchronous motor according to [1]_.

    .. math::
        N_s = \frac{120 f \cdot\text{phase}}{\text{poles}}

    Parameters
    ----------
    f : float
        Line frequency, [Hz]
    poles : int, optional
        The number of poles of the motor
    phase : int, optional
        Line AC phase

    Returns
    -------
    Ns : float
        Speed of synchronous motor, [rpm]

    Notes
    -----
    Synchronous motors have no slip. Large synchronous motors are not
    self-starting.

    Examples
    --------
    >>> speed_synchronous(50, poles=12)
    1500.0
    >>> speed_synchronous(60, phase=1)
    3600.0

    References
    ----------
    .. [1] All About Circuits. Synchronous Motors. Chapter 13 - AC Motors
       http://www.allaboutcircuits.com/textbook/alternating-current/chpt-13/synchronous-motors/
    '''
    Ns = 120.*f*phase/poles
    return Ns


def current_ideal(P, V, phase=3, PF=1):
    r'''Returns the current drawn by a motor of power `P` operating at voltage
    `V`, with line AC of phase `phase` and power factor `PF` according to [1]_.

    Single-phase power:

    .. math::
        I = \frac{P}{V \cdot \text{PF}}

    3-phase power:

    .. math::
        I = \frac{P}{V \cdot \text{PF} \sqrt{3}}


    Parameters
    ----------
    P : float
        Power, [W]
    V : float
        Voltage, [V]
    phase : int, optional
        Line AC phase, either 1 or 3
    PF : float, optional
        Power factor of motor

    Returns
    -------
    I : float
        Power drawn by motor, [A]

    Notes
    -----
    Does not include power used by the motor's fan, or startor, or internal
    losses. These are all significant.

    Examples
    --------
    >>> current_ideal(V=120, P=1E4, PF=1, phase=1)
    83.33333333333333

    References
    ----------
    .. [1] Electrical Construction, and Maintenance. "Calculating Single- and
       3-Phase Parameters." April 1, 2008.
       http://ecmweb.com/basics/calculating-single-and-3-phase-parameters.
    '''
    if phase not in [1, 3]:
        raise Exception('Only 1 and 3 phase power supported')
    if phase == 3:
        I = P/(V*3**0.5*PF)
    else:
        I = P/(V*PF)
    return I


with open(os.path.join(folder, 'residential power.csv')) as f:
    residential_power_raw = f.read()

with open(os.path.join(folder, '3 phase power.csv')) as f:
    industrial_power_raw = f.read()

residential_power = {}
industrial_power = {}
residential_power_data = namedtuple('residential_power_data', ['plugs', 'voltage', 'freq', 'country'])
industrial_power_data = namedtuple('industrial_power_data', ['voltage', 'freq', 'country'])
for line in residential_power_raw.split('\n')[1:]:
    country, code, plugs, voltage, freq = line.split('\t')
    plugs = plugs.replace(' ', '').split(',')
    residential_power[code] = residential_power_data(plugs, int(voltage), int(freq), country)
for line in industrial_power_raw.split('\n')[1:]:
    code, country, voltage, freq = line.split('\t')
    voltage = [int(i) for i in voltage.replace(' ', '').split(',')]
    industrial_power[code] = industrial_power_data(voltage, int(freq), country)


plug_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
voltages_1_phase_residential = [100, 110, 115, 120, 127, 220, 230, 240]
voltages_3_phase = [120, 190, 200, 208, 220, 230, 240, 277, 380, 400, 415, 440, 480]
frequencies = [50, 60]

