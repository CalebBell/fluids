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

This module contains correlations for calculating the efficiency of a pump,
motor, or VFD. It also contains some functions for modeling the performance of
a pump, and has been adapted to contain electrical information relevant to
chemical engineering design.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Pump Efficiency
---------------
.. autofunction :: Corripio_pump_efficiency

Motor Efficiency
----------------
.. autofunction :: CSA_motor_efficiency
.. autofunction :: motor_efficiency_underloaded
.. autofunction :: Corripio_motor_efficiency

VFD Efficiency
--------------
.. autofunction :: VFD_efficiency

Pump Utilities
--------------
.. autofunction :: specific_speed
.. autofunction :: specific_diameter
.. autofunction :: speed_synchronous

Motor Utilities
---------------
.. autofunction :: motor_round_size
.. autodata :: nema_sizes
.. autodata :: nema_sizes_hp

Electrical Utilities
--------------------
.. autofunction :: current_ideal
.. autoclass :: CountryPower
.. autodata :: electrical_plug_types
.. autodata :: voltages_1_phase_residential
.. autodata :: voltages_3_phase
.. autodata :: residential_power_frequencies
.. autodata :: industrial_power
.. autodata :: residential_power

"""

from __future__ import division
from math import log, sqrt
from fluids.constants import hp
from fluids.numerics import interp, tck_interp2d_linear, bisplev

__all__ = ['VFD_efficiency', 'CSA_motor_efficiency', 'motor_efficiency_underloaded',
'Corripio_pump_efficiency', 'Corripio_motor_efficiency',
'specific_speed', 'specific_diameter', 'speed_synchronous', 'nema_sizes',
'nema_sizes_hp', 'motor_round_size', 'nema_min_P', 'nema_high_P', 'electrical_plug_types',
'voltages_1_phase_residential', 'voltages_3_phase', 'residential_power_frequencies',
'residential_power', 'industrial_power', 'current_ideal',
'CountryPower']



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
    efficiency : float
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
    0.705888867095162

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
    logQ = log(Q)
    return -0.316 + 0.24015*logQ - 0.01199*logQ*logQ


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
    efficiency : float
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
    logP = log(P)
    return 0.8 + 0.0319*logP - 0.00182*logP*logP

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
VFD_efficiency_loads = [0.016, 0.125, 0.25, 0.42, 0.5, 0.75, 1.0]
VFD_efficiency_powers = [3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 75.0,
                         100.0, 200.0, 400.0]
VFD_efficiency_tck = tck_interp2d_linear(VFD_efficiency_loads,
                                         VFD_efficiency_powers,
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
    efficiency : float
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

    The table used is for Pulse Width Modulation (PWM) VFDs.

    Examples
    --------
    >>> VFD_efficiency(10*hp)
    0.96
    >>> VFD_efficiency(100*hp, load=0.2)
    0.92

    References
    ----------
    .. [1] GoHz.com. Variable Frequency Drive Efficiency.
       http://www.variablefrequencydrive.org/vfd-efficiency
    '''
    P = P/hp # convert to hp
    if P < 3.0:
        P = 3.0
    elif P > 400.0:
        P = 400.0
    if load < 0.016:
        load = 0.016
    return round(float(bisplev(load, P, VFD_efficiency_tck)), 4)


nema_sizes_hp = [0.25, 0.3333333333333333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                 5.0, 5.5, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0,
                 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0,
                 400.0, 450.0, 500.0]
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
    raise ValueError('Required power is larger than can be provided with one motor')


nema_high_P = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.5, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]
nema_high_full_open_2p = [0.77, 0.84, 0.855, 0.855, 0.865, 0.865, 0.865, 0.885, 0.895, 0.902, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95]
nema_high_full_open_4p = [0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.91, 0.917, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.954, 0.954, 0.958, 0.958, 0.958]
nema_high_full_open_6p = [0.825, 0.865, 0.875, 0.885, 0.895, 0.895, 0.895, 0.902, 0.917, 0.917, 0.924, 0.93, 0.936, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954]
nema_high_full_closed_2p = [0.77, 0.84, 0.855, 0.865, 0.885, 0.885, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.95, 0.95, 0.954, 0.954]
nema_high_full_closed_4p = [0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.945, 0.95, 0.954, 0.954, 0.954, 0.958, 0.962, 0.962]
nema_high_full_closed_6p = [0.825, 0.875, 0.885, 0.895, 0.895, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.958, 0.958, 0.958]


nema_min_P = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.5, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
nema_min_full_open_2p  = [0.755, 0.825, 0.84, 0.84, 0.84, 0.855, 0.855, 0.875, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.93, 0.936, 0.936, 0.945, 0.945, 0.945, 0.95, 0.95, 0.954, 0.958, 0.958]
nema_min_full_open_4p = [0.825, 0.84, 0.84, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.958]
nema_min_full_open_6p = [0.8, 0.84, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.902, 0.902, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954]
nema_min_full_open_8p = [0.74, 0.755, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]
nema_min_full_closed_2p = [0.755, 0.825, 0.84, 0.855, 0.855, 0.875, 0.875, 0.885, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954]
nema_min_full_closed_4p = [0.825, 0.84, 0.84, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.91, 0.91, 0.924, 0.924, 0.93, 0.93, 0.936, 0.941, 0.945, 0.945, 0.95, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958]
nema_min_full_closed_6p = [0.8, 0.855, 0.865, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.902, 0.902, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
nema_min_full_closed_8p = [0.74, 0.77, 0.825, 0.84, 0.84, 0.855, 0.855, 0.855, 0.885, 0.885, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]

nema_min_full_open_2p_i = (nema_min_P, nema_min_full_open_2p)
nema_min_full_open_4p_i = (nema_min_P, nema_min_full_open_4p)
nema_min_full_open_6p_i = (nema_min_P, nema_min_full_open_6p)
nema_min_full_open_8p_i = (nema_min_P, nema_min_full_open_8p)

nema_min_full_closed_2p_i = (nema_min_P, nema_min_full_closed_2p)
nema_min_full_closed_4p_i = (nema_min_P, nema_min_full_closed_4p)
nema_min_full_closed_6p_i = (nema_min_P, nema_min_full_closed_6p)
nema_min_full_closed_8p_i = (nema_min_P, nema_min_full_closed_8p)


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
    efficiency : float
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
    # This could be replaced by a dict and a jump list
    if high_efficiency:
        if closed:
            if poles == 2:
                efficiency = interp(P, nema_high_P, nema_high_full_closed_2p)
            elif poles == 4:
                efficiency = interp(P, nema_high_P, nema_high_full_closed_4p)
            elif poles == 6:
                efficiency = interp(P, nema_high_P, nema_high_full_closed_6p)
        else:
            if poles == 2:
                efficiency = interp(P, nema_high_P, nema_high_full_open_2p)
            elif poles == 4:
                efficiency = interp(P, nema_high_P, nema_high_full_open_4p)
            elif poles == 6:
                efficiency = interp(P, nema_high_P, nema_high_full_open_6p)
    else:
        if closed:
            if poles == 2:
                efficiency = interp(P, nema_min_P, nema_min_full_closed_2p)
            elif poles == 4:
                efficiency = interp(P, nema_min_P, nema_min_full_closed_4p)
            elif poles == 6:
                efficiency = interp(P, nema_min_P, nema_min_full_closed_6p)
            elif poles == 8:
                efficiency = interp(P, nema_min_P, nema_min_full_closed_8p)
        else:
            if poles == 2:
                efficiency = interp(P, nema_min_P, nema_min_full_open_2p)
            elif poles == 4:
                efficiency = interp(P, nema_min_P, nema_min_full_open_4p)
            elif poles == 6:
                efficiency = interp(P, nema_min_P, nema_min_full_open_6p)
            elif poles == 8:
                efficiency = interp(P, nema_min_P, nema_min_full_open_8p)

    return round(efficiency, 4)


_to_1 = [0.015807118828266818, 4.3158627514876216, -8.5612097969025438, 8.2040355039147386, -3.0147603718043068]
_to_5 = [0.015560190519232379, 4.5699731811493152, -7.6800154569463883, 5.4701698738380813, -1.3630071852989643]
_to_10 = [0.059917274403963446, 6.356781885851186, -17.099192527703369, 20.707077651470666, -9.2215133149377841]
_to_25 = [0.29536141765389839, 4.9918188632064329, -13.785081664656504, 16.908273659093812, -7.5816775136809609]
_to_60 = [0.46934299949154384, 4.0298663805446004, -11.632822556859477, 14.616967043793032, -6.6284514347522245]
_to_infty = [0.68235730304242914, 2.4402956771025748, -6.8306770996860182, 8.2108432911172713, -3.5629309804411577]
_efficiency_lists = [_to_1, _to_5, _to_10, _to_25, _to_60, _to_infty]
_efficiency_ones = [0.9218102, 0.64307597, 0.61724113, 0.61569791, 0.6172238, 0.40648294]

def motor_efficiency_underloaded(P, load=0.5):
    r'''Returns the efficiency of a motor operating under its design power
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
    efficiency : float
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
    if P <= 1.0:
        i = 0
    elif P <= 5.0:
        i = 1
    elif P <= 10.0:
        i = 2
    elif P <= 25.0:
        i = 3
    elif P <= 60:
        i = 4
    else:
        i = 5
    if load > _efficiency_ones[i]:
        return 1
    else:
        cs = _efficiency_lists[i]
        return cs[0] + cs[1]*load + cs[2]*load**2 + cs[3]*load**3 + cs[4]*load**4


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
    return n*sqrt(Q)/H**0.75


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
    0.5623413251903492

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    return D*sqrt(sqrt(H)/Q)


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
    return 120.*f*phase/poles


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
        raise ValueError('Only 1 and 3 phase power supported')
    if phase == 3:
        return P/(V*sqrt(3)*PF)
    else:
        return P/(V*PF)


class CountryPower(object):
    """Class to hold information on the residential or electrical data of a
    country. Data from Wikipedia, obtained in 2017.

    Parameters
    ----------
    plugs : tuple(str)
        Tuple of residential plug letter codes in use in the country, [-]
    voltage : float or tuple(float)
        Voltage or voltages in common use of the country (residential data
        has one voltage; industrial data has multiple often), [V]
    freq : float
        The electrical frequency in use in the country, [Hz]
    country : str
        The name of the country, [-]
    """
    __slots__ = ('plugs', 'voltage', 'freq', 'country')

    def __repr__(self):
        return ('CountryPower(country="%s", voltage=%s, freq=%d, plugs=%s)'
                %(self.plugs, self.voltage, self.freq, self.country))
    def __init__(self, country, voltage, freq, plugs=None):
        self.plugs = plugs
        self.voltage = voltage
        self.freq = freq
        self.country = country

residential_power = {
    "at": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Austria"),
    "bj": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Benin"),
    "gh": CountryPower(plugs=('D', 'G'), voltage=230, freq=50, country="Ghana"),
    "sc": CountryPower(plugs=('G',), voltage=240, freq=50, country="Seychelles"),
    "bg": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Bulgaria"),
    "me": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Montenegro"),
    "fo": CountryPower(plugs=('C', 'E', 'F', 'K'), voltage=230, freq=50, country="Faroe Islands"),
    "ne": CountryPower(plugs=('A', 'B', 'C', 'D', 'E', 'F'), voltage=220, freq=50, country="Niger"),
    "za": CountryPower(plugs=('C', 'F', 'M', 'N'), voltage=230, freq=50, country="South Africa"),
    "az": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Azerbaijan"),
    "so": CountryPower(plugs=('C',), voltage=220, freq=50, country="Somalia"),
    "sn": CountryPower(plugs=('C', 'D', 'E', 'K'), voltage=230, freq=50, country="Senegal"),
    "np": CountryPower(plugs=('C', 'D', 'M'), voltage=230, freq=50, country="Nepal"),
    "sl": CountryPower(plugs=('D', 'G'), voltage=230, freq=50, country="Sierra Leone"),
    "be": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Belgium"),
    "vg": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="British Virgin Islands"),
    "bz": CountryPower(plugs=('A', 'B', 'G'), voltage=110, freq=60, country="Belize"),
    "tw": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Taiwan"),
    "bf": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Burkina Faso"),
    "ao": CountryPower(plugs=('C',), voltage=220, freq=50, country="Angola"),
    "gi": CountryPower(plugs=('C', 'G'), voltage=240, freq=50, country="Gibraltar"),
    "ee": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Estonia"),
    "bs": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Bahamas"),
    "ir": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Iran"),
    "sv": CountryPower(plugs=('A', 'B'), voltage=115, freq=60, country="El Salvador"),
    "am": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Armenia"),
    "is": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Iceland"),
    "uy": CountryPower(plugs=('C', 'F', 'I', 'L'), voltage=230, freq=50, country="Uruguay"),
    "mc": CountryPower(plugs=('C', 'D', 'E', 'F'), voltage=230, freq=50, country="Monaco"),
    "jm": CountryPower(plugs=('A', 'B'), voltage=110, freq=50, country="Jamaica"),
    "im": CountryPower(plugs=('G',), voltage=240, freq=50, country="Isle of Man"),
    "dm": CountryPower(plugs=('D', 'G'), voltage=230, freq=50, country="Dominica"),
    "mu": CountryPower(plugs=('C', 'G'), voltage=230, freq=50, country="Mauritius"),
    "cz": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Czech Republic"),
    "kh": CountryPower(plugs=('A', 'C', 'G'), voltage=230, freq=50, country="Cambodia"),
    "cf": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Central African Republic"),
    "se": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Sweden"),
    "uz": CountryPower(plugs=('C', 'I'), voltage=220, freq=50, country="Uzbekistan"),
    "sk": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Slovakia"),
    "ky": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Cayman Islands"),
    "tn": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Tunisia"),
    "do": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Dominican Republic"),
    "hu": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Hungary"),
    "af": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Afghanistan"),
    "et": CountryPower(plugs=('C', 'E', 'F', 'L'), voltage=220, freq=50, country="Ethiopia"),
    "tv": CountryPower(plugs=('I',), voltage=220, freq=50, country="Tuvalu"),
    "ad": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Andorra"),
    "hn": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Honduras"),
    "ls": CountryPower(plugs=('M',), voltage=220, freq=50, country="Lesotho"),
    "na": CountryPower(plugs=('D', 'M'), voltage=220, freq=50, country="Namibia"),
    "jo": CountryPower(plugs=('B', 'C', 'D', 'F', 'G', 'J'), voltage=230, freq=50, country="Jordan"),
    "pl": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Poland"),
    "bt": CountryPower(plugs=('C', 'D', 'F', 'G', 'M'), voltage=230, freq=50, country="Bhutan"),
    "fm": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Micronesia"),
    "no": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Norway"),
    "fk": CountryPower(plugs=('G',), voltage=240, freq=50, country="Falkland Islands"),
    "je": CountryPower(plugs=('G',), voltage=230, freq=50, country="Jersey"),
    "ye": CountryPower(plugs=('A', 'D', 'G'), voltage=230, freq=50, country="Yemen"),
    "cm": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Cameroon"),
    "md": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Moldova"),
    "cn": CountryPower(plugs=('A', 'I', 'C'), voltage=220, freq=50, country="China"),
    "gm": CountryPower(plugs=('G',), voltage=230, freq=50, country="Gambia"),
    "sg": CountryPower(plugs=('C', 'G', 'M'), voltage=230, freq=50, country="Singapore"),
    "tj": CountryPower(plugs=('C', 'F', 'I'), voltage=220, freq=50, country="Tajikistan"),
    "gt": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Guatemala"),
    "ma": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Morocco"),
    "mv": CountryPower(plugs=('D', 'G', 'J', 'K', 'L'), voltage=230, freq=50, country="Maldives"),
    "ga": CountryPower(plugs=('C',), voltage=220, freq=50, country="Gabon"),
    "bo": CountryPower(plugs=('A', 'C'), voltage=115, freq=50, country="Bolivia"),
    "ly": CountryPower(plugs=('C', 'D', 'F', 'L'), voltage=127, freq=50, country="Libya"),
    "rw": CountryPower(plugs=('C', 'J'), voltage=230, freq=50, country="Rwanda"),
    "cg": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Congo, Republic of the"),
    "kz": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Kazakhstan"),
    "jp": CountryPower(plugs=('A', 'B'), voltage=100, freq=50, country="Japan"),
    "co": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Colombia"),
    "sm": CountryPower(plugs=('C', 'F', 'L'), voltage=230, freq=50, country="San Marino"),
    "rs": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Serbia"),
    "gw": CountryPower(plugs=('C',), voltage=220, freq=50, country="Guinea-Bissau"),
    "kr": CountryPower(plugs=('C', 'F'), voltage=220, freq=60, country="South Korea"),
    "py": CountryPower(plugs=('C',), voltage=220, freq=50, country="Paraguay"),
    "lt": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Lithuania"),
    "tr": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Turkey"),
    "pa": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Panama"),
    "ba": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Bosnia and Herzegovina"),
    "vn": CountryPower(plugs=('A', 'C', 'G'), voltage=220, freq=50, country="Vietnam"),
    "iq": CountryPower(plugs=('C', 'D', 'G'), voltage=230, freq=50, country="Iraq"),
    "pk": CountryPower(plugs=('C', 'D', 'G', 'M'), voltage=230, freq=50, country="Pakistan"),
    "li": CountryPower(plugs=('C', 'J'), voltage=230, freq=50, country="Liechtenstein"),
    "mz": CountryPower(plugs=('C', 'F', 'M'), voltage=220, freq=50, country="Mozambique"),
    "au": CountryPower(plugs=('I',), voltage=230, freq=50, country="Australia"),
    "ws": CountryPower(plugs=('I',), voltage=230, freq=50, country="Samoa"),
    "sr": CountryPower(plugs=('C', 'F'), voltage=127, freq=60, country="Suriname"),
    "mn": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Mongolia"),
    "bw": CountryPower(plugs=('D', 'G', 'M'), voltage=230, freq=50, country="Botswana"),
    "gb": CountryPower(plugs=('G',), voltage=230, freq=50, country="United Kingdom"),
    "pg": CountryPower(plugs=('I',), voltage=240, freq=50, country="Papua New Guinea"),
    "dj": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Djibouti"),
    "th": CountryPower(plugs=('A', 'B', 'C', 'F'), voltage=220, freq=50, country="Thailand"),
    "us": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="United States"),
    "gr": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Greece"),
    "kn": CountryPower(plugs=('A', 'B', 'D', 'G'), voltage=110, freq=60, country="St. Kitts and Nevis"),
    "ug": CountryPower(plugs=('G',), voltage=240, freq=50, country="Uganda"),
    "ie": CountryPower(plugs=('G',), voltage=230, freq=50, country="Ireland"),
    "tg": CountryPower(plugs=('C',), voltage=220, freq=50, country="Togo"),
    "td": CountryPower(plugs=('C', 'D', 'E', 'F'), voltage=220, freq=50, country="Chad"),
    "la": CountryPower(plugs=('C', 'E', 'F'), voltage=230, freq=50, country="Laos"),
    "sy": CountryPower(plugs=('C', 'E', 'L'), voltage=220, freq=50, country="Syria"),
    "bm": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Bermuda"),
    "il": CountryPower(plugs=('C', 'H', 'M'), voltage=230, freq=50, country="Israel"),
    "nz": CountryPower(plugs=('I',), voltage=230, freq=50, country="New Zealand"),
    "mg": CountryPower(plugs=('C', 'D', 'E', 'J', 'K'), voltage=220, freq=50, country="Madagascar"),
    "ve": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Venezuela"),
    "dk": CountryPower(plugs=('C', 'E', 'F', 'K'), voltage=230, freq=50, country="Denmark"),
    "lb": CountryPower(plugs=('A', 'B', 'C', 'D', 'G'), voltage=220, freq=50, country="Lebanon"),
    "kp": CountryPower(plugs=('A', 'C', 'F'), voltage=110, freq=60, country="North Korea"),
    "vu": CountryPower(plugs=('C', 'G', 'I'), voltage=220, freq=50, country="Vanuatu"),
    "cu": CountryPower(plugs=('A', 'B', 'C'), voltage=110, freq=60, country="Cuba"),
    "pt": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Portugal"),
    "kw": CountryPower(plugs=('C', 'G'), voltage=240, freq=50, country="Kuwait"),
    "cd": CountryPower(plugs=('C', 'D', 'E'), voltage=220, freq=50, country="Congo, Democratic Republic of the"),
    "nr": CountryPower(plugs=('I',), voltage=240, freq=50, country="Nauru"),
    "si": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Slovenia"),
    "bd": CountryPower(plugs=('C', 'D', 'G', 'K'), voltage=220, freq=50, country="Bangladesh"),
    "al": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Albania"),
    "ec": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Ecuador"),
    "gy": CountryPower(plugs=('A', 'B', 'D', 'G'), voltage=110, freq=60, country="Guyana"),
    "bb": CountryPower(plugs=('A', 'B'), voltage=115, freq=50, country="Barbados"),
    "ke": CountryPower(plugs=('G',), voltage=240, freq=50, country="Kenya"),
    "mx": CountryPower(plugs=('A', 'B'), voltage=127, freq=60, country="Mexico"),
    "gq": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Equatorial Guinea"),
    "gn": CountryPower(plugs=('C', 'F', 'K'), voltage=220, freq=50, country="Guinea"),
    "bi": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Burundi"),
    "lv": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Latvia"),
    "fj": CountryPower(plugs=('I',), voltage=240, freq=50, country="Fiji"),
    "ci": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="Côte d'Ivoire"),
    "ai": CountryPower(plugs=('A',), voltage=110, freq=60, country="Anguilla"),
    "gu": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Guam"),
    "lr": CountryPower(plugs=('A', 'B', 'C', 'E', 'F'), voltage=120, freq=60, country="Liberia"),
    "br": CountryPower(plugs=('C', 'N'), voltage=220, freq=60, country="Brazil"),
    "cv": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Cape Verde"),
    "cl": CountryPower(plugs=('L',), voltage=220, freq=50, country="Chile"),
    "in": CountryPower(plugs=('C', 'D', 'M'), voltage=230, freq=50, country="India"),
    "gg": CountryPower(plugs=('G',), voltage=230, freq=50, country="Guernsey"),
    "tt": CountryPower(plugs=('A', 'B'), voltage=115, freq=60, country="Trinidad & Tobago"),
    "de": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Germany"),
    "qa": CountryPower(plugs=('D', 'G'), voltage=240, freq=50, country="Qatar"),
    "ph": CountryPower(plugs=('A', 'B'), voltage=220, freq=60, country="Philippines"),
    "sd": CountryPower(plugs=('C', 'D'), voltage=230, freq=50, country="Sudan"),
    "mm": CountryPower(plugs=('C', 'D', 'F', 'G'), voltage=230, freq=50, country="Myanmar"),
    "gd": CountryPower(plugs=('G',), voltage=230, freq=50, country="Grenada"),
    "st": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="São Tomé and Príncipe"),
    "sz": CountryPower(plugs=('M',), voltage=230, freq=50, country="Swaziland"),
    "ro": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Romania"),
    "xk": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Kosovo"),
    "cy": CountryPower(plugs=('G',), voltage=240, freq=50, country="Cyprus"),
    "dz": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Algeria"),
    "zm": CountryPower(plugs=('C', 'D', 'G'), voltage=230, freq=50, country="Zambia"),
    "by": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Belarus"),
    "hr": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Croatia"),
    "lu": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Luxembourg"),
    "fi": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Finland"),
    "zw": CountryPower(plugs=('D', 'G'), voltage=220, freq=50, country="Zimbabwe"),
    "km": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Comoros"),
    "tl": CountryPower(plugs=('C', 'E', 'F', 'I'), voltage=220, freq=50, country="Timor-Leste "),
    "tz": CountryPower(plugs=('D', 'G'), voltage=230, freq=50, country="Tanzania"),
    "ht": CountryPower(plugs=('A', 'B'), voltage=110, freq=60, country="Haiti"),
    "vc": CountryPower(plugs=('C', 'E', 'G', 'I', 'K'), voltage=230, freq=50, country="St. Vincent and the Grenadines"),
    "es": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Spain"),
    "my": CountryPower(plugs=('C', 'G', 'M'), voltage=230, freq=50, country="Malaysia"),
    "lc": CountryPower(plugs=('G',), voltage=240, freq=50, country="St. Lucia"),
    "tm": CountryPower(plugs=('B', 'C', 'F'), voltage=220, freq=50, country="Turkmenistan"),
    "pe": CountryPower(plugs=('A', 'B', 'C'), voltage=220, freq=60, country="Peru"),
    "ua": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Ukraine"),
    "eg": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Egypt"),
    "sb": CountryPower(plugs=('I', 'G'), voltage=220, freq=50, country="Solomon Islands"),
    "to": CountryPower(plugs=('I',), voltage=240, freq=50, country="Tonga"),
    "fr": CountryPower(plugs=('C', 'E'), voltage=230, freq=50, country="France"),
    "ng": CountryPower(plugs=('D', 'G'), voltage=240, freq=50, country="Nigeria"),
    "sh": CountryPower(plugs=('G',), voltage=240, freq=50, country="Saint Helena, Ascension and Tristan da Cunha"),
    "mw": CountryPower(plugs=('G',), voltage=230, freq=50, country="Malawi"),
    "ms": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Montserrat"),
    "ae": CountryPower(plugs=('C', 'D', 'G'), voltage=220, freq=50, country="United Arab Emirates"),
    "nl": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Netherlands"),
    "id": CountryPower(plugs=('C', 'F', 'G'), voltage=230, freq=50, country="Indonesia"),
    "ru": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Russia"),
    "ar": CountryPower(plugs=('C', 'I'), voltage=220, freq=50, country="Argentina"),
    "bn": CountryPower(plugs=('G',), voltage=240, freq=50, country="Brunei"),
    "pw": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Palau"),
    "kg": CountryPower(plugs=('C', 'F'), voltage=220, freq=50, country="Kyrgyzstan"),
    "bh": CountryPower(plugs=('G',), voltage=230, freq=50, country="Bahrain"),
    "ml": CountryPower(plugs=('C', 'E'), voltage=220, freq=50, country="Mali"),
    "it": CountryPower(plugs=('C', 'F', 'L'), voltage=230, freq=50, country="Italy"),
    "sa": CountryPower(plugs=('A', 'B', 'G'), voltage=220, freq=60, country="Saudi Arabia"),
    "ag": CountryPower(plugs=('A', 'B'), voltage=230, freq=60, country="Antigua and Barbuda"),
    "mr": CountryPower(plugs=('C',), voltage=220, freq=50, country="Mauritania"),
    "om": CountryPower(plugs=('C', 'G'), voltage=240, freq=50, country="Oman"),
    "lk": CountryPower(plugs=('D', 'G', 'M'), voltage=230, freq=50, country="Sri Lanka"),
    "er": CountryPower(plugs=('C', 'L'), voltage=230, freq=50, country="Eritrea"),
    "mk": CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Macedonia"),
    "ni": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Nicaragua"),
    "ch": CountryPower(plugs=('C', 'J'), voltage=230, freq=50, country="Switzerland"),
    "ca": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Canada"),
    "cr": CountryPower(plugs=('A', 'B'), voltage=120, freq=60, country="Costa Rica")
}
'''Dictionary of country-code to CountryPower instances for residential use.'''

CONST_380 = 380
CONST_400 = 400
CONST_415 = 415
CONST_440 = 440
CONST_480 = 480
TUP_190 = (190,)
TUP_208 = (208,)
TUP_240 = (240,)
TUP_380 = (CONST_380,)
TUP_400 = (CONST_400,)
TUP_415 = (CONST_415,)

industrial_power = {
    "at": CountryPower(voltage=TUP_400, freq=50, country='Austria'),
    "bj": CountryPower(voltage=TUP_380, freq=50, country='Benin'),
    "gh": CountryPower(voltage=TUP_400, freq=50, country='Ghana'),
    "sc": CountryPower(voltage=TUP_240, freq=50, country='Seychelles'),
    "bg": CountryPower(voltage=TUP_400, freq=50, country='Bulgaria'),
    "me": CountryPower(voltage=TUP_400, freq=50, country='Montenegro'),
    "fo": CountryPower(voltage=TUP_400, freq=50, country='Faeroe Islands'),
    "iq": CountryPower(voltage=TUP_400, freq=50, country='Iraq'),
    "ne": CountryPower(voltage=TUP_380, freq=50, country='Niger'),
    "za": CountryPower(voltage=TUP_400, freq=50, country='South Africa'),
    "az": CountryPower(voltage=TUP_380, freq=50, country='Azerbaijan'),
    "so": CountryPower(voltage=TUP_380, freq=50, country='Somalia'),
    "sn": CountryPower(voltage=TUP_400, freq=50, country='Senegal'),
    "np": CountryPower(voltage=TUP_400, freq=50, country='Nepal'),
    "sl": CountryPower(voltage=TUP_400, freq=50, country='Sierra Leone'),
    "be": CountryPower(voltage=TUP_400, freq=50, country='Belgium'),
    "vg": CountryPower(voltage=TUP_190, freq=60, country='British Virgin Islands'),
    "bz": CountryPower(voltage=(190, CONST_380), freq=60, country='Belize'),
    "tw": CountryPower(voltage=(220,), freq=60, country='Taiwan'),
    "bf": CountryPower(voltage=TUP_380, freq=50, country='Burkina Faso'),
    "ao": CountryPower(voltage=TUP_380, freq=50, country='Angola'),
    "ee": CountryPower(voltage=TUP_400, freq=50, country='Estonia'),
    "bs": CountryPower(voltage=TUP_208, freq=60, country='Bahamas'),
    "ir": CountryPower(voltage=TUP_400, freq=50, country='Iran'),
    "sv": CountryPower(voltage=(200,), freq=60, country='El Salvador'),
    "am": CountryPower(voltage=TUP_400, freq=50, country='Armenia'),
    "is": CountryPower(voltage=TUP_400, freq=50, country='Iceland'),
    "uy": CountryPower(voltage=TUP_380, freq=50, country='Uruguay'),
    "mc": CountryPower(voltage=TUP_400, freq=50, country='Monaco'),
    "jm": CountryPower(voltage=TUP_190, freq=50, country='Jamaica'),
    "im": CountryPower(voltage=TUP_415, freq=50, country='Isle of Man'),
    "dm": CountryPower(voltage=TUP_400, freq=50, country='Dominica'),
    "mu": CountryPower(voltage=TUP_400, freq=50, country='Mauritius'),
    "cz": CountryPower(voltage=TUP_400, freq=50, country='Czech Republic'),
    "kh": CountryPower(voltage=TUP_400, freq=50, country='Cambodia'),
    "cf": CountryPower(voltage=TUP_380, freq=50, country='Central African Republic'),
    "se": CountryPower(voltage=TUP_400, freq=50, country='Sweden'),
    "uz": CountryPower(voltage=TUP_380, freq=50, country='Uzbekistan'),
    "sk": CountryPower(voltage=TUP_400, freq=50, country='Slovakia'),
    "ky": CountryPower(voltage=TUP_240, freq=60, country='Cayman Islands'),
    "tn": CountryPower(voltage=TUP_400, freq=50, country='Tunisia'),
    "hu": CountryPower(voltage=TUP_400, freq=50, country='Hungary'),
    "af": CountryPower(voltage=TUP_380, freq=50, country='Afghanistan'),
    "tc": CountryPower(voltage=TUP_240, freq=60, country='Turks and Caicos Islands'),
    "et": CountryPower(voltage=TUP_380, freq=50, country='Ethiopia'),
    "sd": CountryPower(voltage=TUP_400, freq=50, country='Sudan'),
    "ad": CountryPower(voltage=TUP_400, freq=50, country='Andorra'),
    "hn": CountryPower(voltage=(208, 230, 240, 460, CONST_480), freq=60, country='Honduras'),
    "ls": CountryPower(voltage=TUP_380, freq=50, country='Lesotho'),
    "na": CountryPower(voltage=TUP_380, freq=50, country='Namibia'),
    "pl": CountryPower(voltage=TUP_400, freq=50, country='Poland'),
    "bt": CountryPower(voltage=TUP_400, freq=50, country='Bhutan'),
    "sa": CountryPower(voltage=TUP_400, freq=60, country='Saudi Arabia'),
    "no": CountryPower(voltage=(230, 400), freq=50, country='Norway'),
    "fk": CountryPower(voltage=TUP_415, freq=50, country='Falkland Islands'),
    "ye": CountryPower(voltage=TUP_400, freq=50, country='Yemen'),
    "gi": CountryPower(voltage=TUP_400, freq=50, country='Gibraltar'),
    "md": CountryPower(voltage=TUP_400, freq=50, country='Moldova'),
    "cn": CountryPower(voltage=TUP_380, freq=50, country='China'),
    "gm": CountryPower(voltage=TUP_400, freq=50, country='Gambia'),
    "sg": CountryPower(voltage=TUP_400, freq=50, country='Singapore'),
    "tj": CountryPower(voltage=TUP_380, freq=50, country='Tajikistan'),
    "gt": CountryPower(voltage=TUP_208, freq=60, country='Guatemala'),
    "ma": CountryPower(voltage=TUP_380, freq=50, country='Morocco'),
    "mv": CountryPower(voltage=TUP_400, freq=50, country='Maldives'),
    "ga": CountryPower(voltage=TUP_380, freq=50, country='Gabon'),
    "bo": CountryPower(voltage=TUP_400, freq=50, country='Bolivia'),
    "ly": CountryPower(voltage=TUP_400, freq=50, country='Libya'),
    "rw": CountryPower(voltage=TUP_400, freq=50, country='Rwanda'),
    "cg": CountryPower(voltage=TUP_400, freq=50, country="People's Republic of Congo"),
    "kz": CountryPower(voltage=TUP_380, freq=50, country='Kazakhstan'),
    "jp": CountryPower(voltage=(200,), freq=50, country='Japan'),
    "co": CountryPower(voltage=(220, 440), freq=60, country='Colombia'),
    "sm": CountryPower(voltage=TUP_400, freq=50, country='San Marino'),
    "rs": CountryPower(voltage=TUP_400, freq=50, country='Serbia'),
    "gw": CountryPower(voltage=TUP_380, freq=50, country='Guinea-Bissau'),
    "kr": CountryPower(voltage=TUP_380, freq=60, country='South Korea'),
    "py": CountryPower(voltage=TUP_380, freq=50, country='Paraguay'),
    "lt": CountryPower(voltage=TUP_400, freq=50, country='Lithuania'),
    "tr": CountryPower(voltage=TUP_400, freq=50, country='Turkey'),
    "ss": CountryPower(voltage=TUP_400, freq=50, country='South Sudan'),
    "ba": CountryPower(voltage=TUP_400, freq=50, country='Bosnia & Herzegovina'),
    "vn": CountryPower(voltage=TUP_380, freq=50, country='Vietnam'),
    "do": CountryPower(voltage=(120, 208, 277, 480), freq=60, country='Dominican Republic'),
    "pk": CountryPower(voltage=TUP_400, freq=50, country='Pakistan'),
    "li": CountryPower(voltage=TUP_400, freq=50, country='Liechtenstein'),
    "mz": CountryPower(voltage=TUP_380, freq=50, country='Mozambique'),
    "au": CountryPower(voltage=TUP_400, freq=50, country='Australia'),
    "ws": CountryPower(voltage=TUP_400, freq=50, country='Samoa'),
    "sr": CountryPower(voltage=(220, 400,), freq=60, country='Suriname'),
    "mn": CountryPower(voltage=TUP_400, freq=50, country='Mongolia'),
    "bw": CountryPower(voltage=TUP_400, freq=50, country='Botswana'),
    "gb": CountryPower(voltage=TUP_415, freq=50, country='United Kingdom'),
    "pg": CountryPower(voltage=TUP_415, freq=50, country='Papua New Guinea'),
    "dj": CountryPower(voltage=TUP_380, freq=50, country='Djibouti'),
    "th": CountryPower(voltage=TUP_400, freq=50, country='Thailand'),
    "us": CountryPower(voltage=(120, 208, 277, 480, 120, 240, 240, CONST_480), freq=60, country='United States of America'),
    "gr": CountryPower(voltage=TUP_400, freq=50, country='Greece'),
    "ug": CountryPower(voltage=TUP_415, freq=50, country='Uganda'),
    "ie": CountryPower(voltage=TUP_415, freq=50, country='Ireland'),
    "tg": CountryPower(voltage=TUP_380, freq=50, country='Togo'),
    "td": CountryPower(voltage=TUP_380, freq=50, country='Chad'),
    "la": CountryPower(voltage=TUP_400, freq=50, country='Laos'),
    "sy": CountryPower(voltage=TUP_380, freq=50, country='Syria'),
    "bm": CountryPower(voltage=TUP_208, freq=60, country='Bermuda'),
    "il": CountryPower(voltage=TUP_400, freq=50, country='Israel'),
    "nz": CountryPower(voltage=TUP_400, freq=50, country='New Zealand'),
    "mg": CountryPower(voltage=TUP_380, freq=50, country='Madagascar'),
    "ve": CountryPower(voltage=(120,), freq=60, country='Venezuela'),
    "dk": CountryPower(voltage=TUP_400, freq=50, country='Denmark'),
    "lb": CountryPower(voltage=TUP_400, freq=50, country='Lebanon'),
    "kp": CountryPower(voltage=TUP_380, freq=50, country='North Korea'),
    "vu": CountryPower(voltage=TUP_400, freq=50, country='Vanuatu'),
    "cu": CountryPower(voltage=(190, 440), freq=60, country='Cuba'),
    "kw": CountryPower(voltage=TUP_415, freq=50, country='Kuwait'),
    "cd": CountryPower(voltage=TUP_380, freq=50, country='Democratic Republic of Congo'),
    "nr": CountryPower(voltage=TUP_415, freq=50, country='Nauru'),
    "si": CountryPower(voltage=TUP_400, freq=50, country='Slovenia'),
    "mt": CountryPower(voltage=TUP_400, freq=50, country='Malta'),
    "bd": CountryPower(voltage=TUP_380, freq=50, country='Bangladesh'),
    "al": CountryPower(voltage=TUP_400, freq=50, country='Albania'),
    "ec": CountryPower(voltage=TUP_208, freq=60, country='Ecuador'),
    "gy": CountryPower(voltage=TUP_190, freq=60, country='Guyana'),
    "bb": CountryPower(voltage=(200,), freq=50, country='Barbados'),
    "ke": CountryPower(voltage=TUP_415, freq=50, country='Kenya'),
    "mx": CountryPower(voltage=(220, CONST_480), freq=60, country='Mexico'),
    "gn": CountryPower(voltage=TUP_380, freq=50, country='Guinea'),
    "bi": CountryPower(voltage=TUP_380, freq=50, country='Burundi'),
    "lv": CountryPower(voltage=TUP_400, freq=50, country='Latvia'),
    "fj": CountryPower(voltage=TUP_415, freq=50, country='Fiji'),
    "ci": CountryPower(voltage=TUP_380, freq=50, country='Côte d’Ivoire'),
    "ai": CountryPower(voltage=(120, 208, 127, 220, 240, 415), freq=60, country='Anguilla'),
    "gu": CountryPower(voltage=TUP_190, freq=60, country='Guam'),
    "lr": CountryPower(voltage=TUP_208, freq=60, country='Liberia'),
    "br": CountryPower(voltage=(220, 380), freq=60, country='Brazil'),
    "cv": CountryPower(voltage=TUP_400, freq=50, country='Cape Verde'),
    "cl": CountryPower(voltage=TUP_380, freq=50, country='Chile'),
    "in": CountryPower(voltage=TUP_400, freq=50, country='India'),
    "tt": CountryPower(voltage=(115, 230, 230, 400), freq=60, country='Trinidad & Tobago'),
    "de": CountryPower(voltage=TUP_400, freq=50, country='Germany'),
    "pa": CountryPower(voltage=TUP_240, freq=60, country='Panama'),
    "qa": CountryPower(voltage=TUP_415, freq=50, country='Qatar'),
    "ph": CountryPower(voltage=TUP_380, freq=60, country='Philippines'),
    "jo": CountryPower(voltage=TUP_400, freq=50, country='Jordan'),
    "mm": CountryPower(voltage=TUP_400, freq=50, country='Myanmar'),
    "gd": CountryPower(voltage=TUP_400, freq=50, country='Grenada'),
    "st": CountryPower(voltage=TUP_400, freq=50, country='São Tomé and Príncipe'),
    "sz": CountryPower(voltage=TUP_400, freq=50, country='Swaziland'),
    "ro": CountryPower(voltage=TUP_400, freq=50, country='Romania'),
    "xk": CountryPower(voltage=(230, 400), freq=50, country='Kosovo'),
    "cy": CountryPower(voltage=TUP_400, freq=50, country='Cyprus'),
    "dz": CountryPower(voltage=TUP_400, freq=50, country='Algeria'),
    "zm": CountryPower(voltage=TUP_400, freq=50, country='Zambia'),
    "by": CountryPower(voltage=TUP_380, freq=50, country='Belarus'),
    "hr": CountryPower(voltage=TUP_400, freq=50, country='Croatia'),
    "lu": CountryPower(voltage=TUP_400, freq=50, country='Luxembourg'),
    "fi": CountryPower(voltage=TUP_400, freq=50, country='Finland'),
    "zw": CountryPower(voltage=TUP_415, freq=50, country='Zimbabwe'),
    "km": CountryPower(voltage=TUP_380, freq=50, country='Comoros'),
    "tl": CountryPower(voltage=TUP_380, freq=50, country='East Timor'),
    "tz": CountryPower(voltage=TUP_415, freq=50, country='Tanzania'),
    "ht": CountryPower(voltage=TUP_190, freq=60, country='Haiti'),
    "vc": CountryPower(voltage=TUP_400, freq=50, country='Saint Vincent and the Grenadines'),
    "es": CountryPower(voltage=TUP_400, freq=50, country='Spain'),
    "my": CountryPower(voltage=TUP_415, freq=50, country='Malaysia'),
    "lc": CountryPower(voltage=TUP_400, freq=50, country='Saint Lucia'),
    "tm": CountryPower(voltage=TUP_380, freq=50, country='Turkmenistan'),
    "pe": CountryPower(voltage=(220,), freq=60, country='Peru'),
    "ua": CountryPower(voltage=TUP_400, freq=50, country='Ukraine'),
    "eg": CountryPower(voltage=TUP_380, freq=50, country='Egypt'),
    "to": CountryPower(voltage=TUP_415, freq=50, country='Tonga'),
    "fr": CountryPower(voltage=TUP_400, freq=50, country='France'),
    "ng": CountryPower(voltage=TUP_415, freq=50, country='Nigeria'),
    "mw": CountryPower(voltage=TUP_400, freq=50, country='Malawi'),
    "ms": CountryPower(voltage=TUP_400, freq=60, country='Montserrat'),
    "ae": CountryPower(voltage=TUP_400, freq=50, country='United Arab Emirates'),
    "nl": CountryPower(voltage=TUP_400, freq=50, country='Netherlands'),
    "id": CountryPower(voltage=TUP_400, freq=50, country='Indonesia'),
    "ru": CountryPower(voltage=TUP_380, freq=50, country='Russia'),
    "ar": CountryPower(voltage=TUP_380, freq=50, country='Argentina'),
    "bn": CountryPower(voltage=TUP_415, freq=50, country='Brunei'),
    "pw": CountryPower(voltage=TUP_208, freq=60, country='Palau'),
    "kg": CountryPower(voltage=TUP_380, freq=50, country='Kyrgyzstan'),
    "bh": CountryPower(voltage=TUP_400, freq=50, country='Bahrain'),
    "ml": CountryPower(voltage=TUP_380, freq=50, country='Mali'),
    "it": CountryPower(voltage=TUP_400, freq=50, country='Italy'),
    "cm": CountryPower(voltage=TUP_380, freq=50, country='Cameroon'),
    "ag": CountryPower(voltage=TUP_400, freq=60, country='Antigua and Barbuda'),
    "mr": CountryPower(voltage=(220,), freq=50, country='Mauritania'),
    "om": CountryPower(voltage=TUP_415, freq=50, country='Oman'),
    "lk": CountryPower(voltage=TUP_400, freq=50, country='Sri Lanka'),
    "er": CountryPower(voltage=TUP_400, freq=50, country='Eritrea'),
    "mk": CountryPower(voltage=TUP_400, freq=50, country='Macedonia, Republic of'),
    "ni": CountryPower(voltage=TUP_208, freq=60, country='Nicaragua'),
    "ch": CountryPower(voltage=TUP_400, freq=50, country='Switzerland'),
    "ca": CountryPower(voltage=(120, 208, 240, CONST_480, 347, 600), freq=60, country='Canada'),
    "cr": CountryPower(voltage=TUP_240, freq=60, country='Costa Rica')
}
'''Dictionary of country-code to CountryPower instances for industrial use.'''

electrical_plug_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
'''List of all electrical plug types in use around the world.'''
voltages_1_phase_residential = [100, 110, 115, 120, 127, 220, 230, 240]
'''List of all AC 1-phase voltages used in residential settings around the world.'''
voltages_3_phase = [120, 190, 200, 208, 220, 230, 240, 277, 380, 400, 415, 440, 480]
'''List of all AC 3-phase voltages used in industrial settings around the world.'''
residential_power_frequencies = [50, 60]
'''List of all AC 1-phase frequencies used in residential settings around the world.'''


# https://www.grainger.com/content/supplylink-v-belt-maintenance-key-to-electric-motor-efficiency
# Source of values for v belt, notched, and synchronous
# Technology assessment: energy-efficient belt transmissions
# Source of cogged value, their range is 95-98

V_BELT = 'V'
COGGED_V_BELT = 'cogged'

NOTCHED_BELT = 'notched'
SYNCHRONOUS_BELT = 'synchronous'

belt_efficiencies = {V_BELT: 0.95,
                     NOTCHED_BELT: 0.97,
                     COGGED_V_BELT: 0.965,
                     SYNCHRONOUS_BELT: 0.98}


DEEP_GROOVE_BALL = "Deep groove ball"
ANGULAR_CONTACT_BALL_SINGLE_ROW = "Angular contact ball Single row"
ANGULAR_CONTACT_BALL_DOUBLE_ROW = "Angular contact ball Double row"
FOUR_POINT_CONTACT_BALL = "Four point contact ball"
SELF_ALIGNING_BALL = "Self aligning ball"
CYLINDRICAL_ROLLER_WITH_CAGE = "Cylindrical roller with cage"
CYLINDRICAL_ROLLER_FULL_COMPLEMENT = "Cylindrical roller full complement"
NEEDLE_ROLLER = "Needle roller"
TAPER_ROLLER = "Taper roller"
SPHERICAL_ROLLER = "Spherical roller"
THRUST_BALL = "Thrust ball"
CYLINDRICAL_ROLLER_THRUST = "Cylindrical roller thrust"
NEEDLE_ROLLER_THRUST = "Needle roller thrust"
SPHERICAL_ROLLER_THRUST = "Spherical roller thrust"


bearing_friction_factors = {DEEP_GROOVE_BALL: 0.0015,
ANGULAR_CONTACT_BALL_SINGLE_ROW: 0.002,
ANGULAR_CONTACT_BALL_DOUBLE_ROW: 0.0024,
FOUR_POINT_CONTACT_BALL: 0.0024,
SELF_ALIGNING_BALL: 0.001,
CYLINDRICAL_ROLLER_WITH_CAGE: 0.0011,
CYLINDRICAL_ROLLER_FULL_COMPLEMENT: 0.002,
NEEDLE_ROLLER: 0.0025,
TAPER_ROLLER: 0.0018,
SPHERICAL_ROLLER: 0.0018,
THRUST_BALL: 0.0013,
CYLINDRICAL_ROLLER_THRUST: 0.005,
NEEDLE_ROLLER_THRUST: 0.005,
SPHERICAL_ROLLER_THRUST: 0.0018}

# In m, diameter of fans
fan_diameters = [0.125, 0.132, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.212,
                 0.224, 0.236, 0.25, 0.265, 0.28, 0.3, 0.315, 0.335, 0.355,
                 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.53, 0.56, 0.6, 0.63,
                 0.67, 0.71, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

FEG90 = [0.425, 0.448, 0.472, 0.501, 0.527, 0.552, 0.574, 0.594, 0.613, 0.633, 0.652, 0.669, 0.686, 0.703, 0.718, 0.735, 0.746, 0.759, 0.77, 0.779, 0.789, 0.797, 0.804, 0.81, 0.815, 0.819, 0.823, 0.827, 0.83, 0.833, 0.835, 0.837, 0.838, 0.84, 0.841, 0.841, 0.841]
FEG85 = [0.401, 0.423, 0.446, 0.473, 0.498, 0.521, 0.542, 0.561, 0.579, 0.598, 0.616, 0.631, 0.648, 0.664, 0.678, 0.694, 0.704, 0.717, 0.727, 0.736, 0.745, 0.753, 0.759, 0.765, 0.769, 0.774, 0.777, 0.781, 0.784, 0.786, 0.788, 0.79, 0.791, 0.793, 0.793, 0.794, 0.794]
FEG80 = [0.378, 0.399, 0.421, 0.447, 0.47, 0.492, 0.511, 0.53, 0.546, 0.565, 0.581, 0.596, 0.612, 0.627, 0.64, 0.655, 0.665, 0.676, 0.686, 0.695, 0.703, 0.711, 0.717, 0.722, 0.726, 0.73, 0.734, 0.738, 0.74, 0.742, 0.744, 0.746, 0.747, 0.748, 0.749, 0.75, 0.75]
FEG75 = [0.357, 0.377, 0.398, 0.422, 0.444, 0.464, 0.483, 0.5, 0.516, 0.533, 0.549, 0.563, 0.578, 0.592, 0.604, 0.618, 0.628, 0.639, 0.648, 0.656, 0.664, 0.671, 0.677, 0.681, 0.685, 0.689, 0.693, 0.696, 0.698, 0.701, 0.703, 0.704, 0.705, 0.706, 0.707, 0.708, 0.708]
FEG71 = [0.337, 0.356, 0.375, 0.398, 0.419, 0.438, 0.456, 0.472, 0.487, 0.503, 0.518, 0.531, 0.545, 0.559, 0.57, 0.584, 0.593, 0.603, 0.612, 0.619, 0.627, 0.633, 0.639, 0.643, 0.647, 0.651, 0.654, 0.657, 0.659, 0.661, 0.663, 0.665, 0.666, 0.667, 0.668, 0.668, 0.668]
FEG67 = [0.318, 0.336, 0.354, 0.376, 0.395, 0.414, 0.43, 0.446, 0.46, 0.475, 0.489, 0.502, 0.515, 0.527, 0.538, 0.551, 0.559, 0.569, 0.577, 0.584, 0.592, 0.598, 0.603, 0.607, 0.611, 0.614, 0.617, 0.621, 0.622, 0.624, 0.626, 0.627, 0.629, 0.63, 0.63, 0.631, 0.631]
FEG63 = [0.301, 0.317, 0.334, 0.355, 0.373, 0.39, 0.406, 0.421, 0.434, 0.448, 0.462, 0.473, 0.486, 0.498, 0.508, 0.52, 0.528, 0.537, 0.545, 0.552, 0.559, 0.565, 0.569, 0.573, 0.577, 0.58, 0.583, 0.586, 0.588, 0.59, 0.591, 0.592, 0.594, 0.594, 0.595, 0.595, 0.596]
FEG60 = [0.284, 0.299, 0.316, 0.335, 0.352, 0.369, 0.383, 0.397, 0.41, 0.423, 0.436, 0.447, 0.459, 0.47, 0.48, 0.491, 0.499, 0.507, 0.515, 0.521, 0.528, 0.533, 0.538, 0.541, 0.545, 0.548, 0.55, 0.553, 0.555, 0.557, 0.558, 0.559, 0.56, 0.561, 0.562, 0.562, 0.562]
FEG56 = [0.268, 0.282, 0.298, 0.316, 0.333, 0.348, 0.362, 0.375, 0.387, 0.4, 0.411, 0.422, 0.433, 0.444, 0.453, 0.464, 0.471, 0.479, 0.486, 0.492, 0.498, 0.503, 0.507, 0.511, 0.514, 0.517, 0.519, 0.522, 0.524, 0.525, 0.527, 0.528, 0.529, 0.53, 0.53, 0.531, 0.531]
FEG53 = [0.253, 0.267, 0.281, 0.298, 0.314, 0.329, 0.342, 0.354, 0.365, 0.377, 0.388, 0.398, 0.409, 0.419, 0.428, 0.438, 0.444, 0.452, 0.459, 0.464, 0.47, 0.475, 0.479, 0.482, 0.485, 0.488, 0.49, 0.493, 0.494, 0.496, 0.497, 0.498, 0.499, 0.5, 0.501, 0.501, 0.501]
FEG50 = [0.239, 0.252, 0.266, 0.282, 0.297, 0.31, 0.323, 0.334, 0.345, 0.356, 0.367, 0.376, 0.386, 0.395, 0.404, 0.413, 0.42, 0.427, 0.433, 0.438, 0.444, 0.448, 0.452, 0.455, 0.458, 0.461, 0.463, 0.465, 0.467, 0.468, 0.47, 0.47, 0.471, 0.472, 0.473, 0.473, 0.473]

fan_bare_shaft_efficiencies = {'FEG90': FEG90,
                               'FEG85': FEG85,
                               'FEG80': FEG80,
                               'FEG75': FEG75,
                               'FEG71': FEG71,
                               'FEG67': FEG67,
                               'FEG63': FEG63,
                               'FEG60': FEG60,
                               'FEG56': FEG56,
                               'FEG53': FEG53,
                               'FEG50': FEG50}

'''for key, values in fan_bare_shaft_efficiencies.items():
    plt.plot(fan_diameters, values, label=key)

plt.legend()
plt.show()'''



FMEG_axial_powers = [125.0, 300.0, 1000.0, 2500.0, 5000.0, 8000.0, 10000.0, 20000.0, 60000.0, 160000.0, 300000.0, 375000.0, 500000.0]

FMEG27 = [0.15, 0.174, 0.207, 0.232, 0.251, 0.264, 0.27, 0.275, 0.283, 0.291, 0.296, 0.297, 0.3]
FMEG31 = [0.19, 0.214, 0.247, 0.272, 0.291, 0.304, 0.31, 0.315, 0.323, 0.331, 0.336, 0.337, 0.34]
FMEG35 = [0.23, 0.254, 0.287, 0.312, 0.331, 0.344, 0.35, 0.355, 0.363, 0.371, 0.376, 0.377, 0.38]
FMEG39 = [0.27, 0.294, 0.327, 0.352, 0.371, 0.384, 0.39, 0.395, 0.403, 0.411, 0.416, 0.417, 0.42]
FMEG42 = [0.3, 0.324, 0.357, 0.382, 0.401, 0.414, 0.42, 0.425, 0.433, 0.441, 0.446, 0.447, 0.45]
FMEG46 = [0.34, 0.364, 0.397, 0.422, 0.441, 0.454, 0.46, 0.465, 0.473, 0.481, 0.486, 0.487, 0.49]
FMEG50 = [0.38, 0.404, 0.437, 0.462, 0.481, 0.494, 0.5, 0.505, 0.513, 0.521, 0.526, 0.527, 0.53]
FMEG53 = [0.41, 0.434, 0.467, 0.492, 0.511, 0.524, 0.53, 0.535, 0.543, 0.551, 0.556, 0.557, 0.56]
FMEG55 = [0.43, 0.454, 0.487, 0.512, 0.531, 0.544, 0.55, 0.555, 0.563, 0.571, 0.576, 0.577, 0.58]
FMEG58 = [0.46, 0.484, 0.517, 0.542, 0.561, 0.574, 0.58, 0.585, 0.593, 0.601, 0.606, 0.607, 0.61]
FMEG60 = [0.48, 0.504, 0.537, 0.562, 0.581, 0.594, 0.6, 0.605, 0.613, 0.621, 0.626, 0.627, 0.63]
FMEG62 = [0.5, 0.524, 0.557, 0.582, 0.601, 0.614, 0.62, 0.625, 0.633, 0.641, 0.646, 0.647, 0.65]
FMEG64 = [0.52, 0.544, 0.577, 0.602, 0.621, 0.634, 0.64, 0.645, 0.653, 0.661, 0.666, 0.667, 0.67]
FMEG66 = [0.54, 0.564, 0.597, 0.622, 0.641, 0.654, 0.66, 0.665, 0.673, 0.681, 0.686, 0.687, 0.69]

fan_driven_axial_efficiencies = {'FMEG27': FMEG27,
                                 'FMEG31': FMEG31,
                                 'FMEG35': FMEG35,
                                 'FMEG39': FMEG39,
                                 'FMEG42': FMEG42,
                                 'FMEG46': FMEG46,
                                 'FMEG50': FMEG50,
                                 'FMEG53': FMEG53,
                                 'FMEG55': FMEG55,
                                 'FMEG58': FMEG58,
                                 'FMEG60': FMEG60,
                                 'FMEG62': FMEG62,
                                 'FMEG64': FMEG64,
                                 'FMEG66': FMEG66}

FMEG_centrifugal_backward_powers = FMEG_axial_powers
FMEG35 = [0.15, 0.19, 0.245, 0.287, 0.318, 0.34, 0.35, 0.357, 0.369, 0.38, 0.387, 0.389, 0.392]
FMEG39 = [0.19, 0.23, 0.285, 0.327, 0.358, 0.38, 0.39, 0.397, 0.409, 0.42, 0.427, 0.429, 0.432]
FMEG42 = [0.22, 0.26, 0.315, 0.357, 0.388, 0.41, 0.42, 0.427, 0.439, 0.45, 0.457, 0.459, 0.462]
FMEG46 = [0.26, 0.3, 0.355, 0.397, 0.428, 0.45, 0.46, 0.467, 0.479, 0.49, 0.497, 0.499, 0.502]
FMEG50 = [0.3, 0.34, 0.395, 0.437, 0.468, 0.49, 0.5, 0.507, 0.519, 0.53, 0.537, 0.539, 0.542]
FMEG53 = [0.33, 0.37, 0.425, 0.467, 0.498, 0.52, 0.53, 0.537, 0.549, 0.56, 0.567, 0.569, 0.572]
FMEG55 = [0.35, 0.39, 0.445, 0.487, 0.518, 0.54, 0.55, 0.557, 0.569, 0.58, 0.587, 0.589, 0.592]
FMEG58 = [0.38, 0.42, 0.475, 0.517, 0.548, 0.57, 0.58, 0.587, 0.599, 0.61, 0.617, 0.619, 0.622]
FMEG60 = [0.4, 0.44, 0.495, 0.537, 0.568, 0.59, 0.6, 0.607, 0.619, 0.63, 0.637, 0.639, 0.642]
FMEG62 = [0.42, 0.46, 0.515, 0.557, 0.588, 0.61, 0.62, 0.627, 0.639, 0.65, 0.657, 0.659, 0.662]
FMEG64 = [0.44, 0.48, 0.535, 0.577, 0.608, 0.63, 0.64, 0.647, 0.659, 0.67, 0.677, 0.679, 0.682]
FMEG66 = [0.46, 0.5, 0.555, 0.597, 0.628, 0.65, 0.66, 0.667, 0.679, 0.69, 0.697, 0.699, 0.702]
FMEG68 = [0.48, 0.52, 0.575, 0.617, 0.648, 0.67, 0.68, 0.687, 0.699, 0.71, 0.717, 0.719, 0.722]
FMEG70 = [0.5, 0.54, 0.595, 0.637, 0.668, 0.69, 0.7, 0.707, 0.719, 0.73, 0.737, 0.739, 0.742]
FMEG72 = [0.52, 0.56, 0.615, 0.657, 0.688, 0.71, 0.72, 0.727, 0.739, 0.75, 0.757, 0.759, 0.762]
FMEG74 = [0.54, 0.58, 0.635, 0.677, 0.708, 0.73, 0.74, 0.747, 0.759, 0.77, 0.777, 0.779, 0.782]
FMEG76 = [0.56, 0.6, 0.655, 0.697, 0.728, 0.75, 0.76, 0.767, 0.779, 0.79, 0.797, 0.799, 0.802]

fan_centrifugal_backward_efficiencies = {'FMEG35': FMEG35,
                                         'FMEG39': FMEG39,
                                         'FMEG42': FMEG42,
                                         'FMEG46': FMEG46,
                                         'FMEG50': FMEG50,
                                         'FMEG53': FMEG53,
                                         'FMEG55': FMEG55,
                                         'FMEG58': FMEG58,
                                         'FMEG60': FMEG60,
                                         'FMEG62': FMEG62,
                                         'FMEG64': FMEG64,
                                         'FMEG66': FMEG66,
                                         'FMEG68': FMEG68,
                                         'FMEG70': FMEG70,
                                         'FMEG72': FMEG72,
                                         'FMEG74': FMEG74,
                                         'FMEG76': FMEG76}

FMEG_cross_flow_powers = [130.0, 300.0, 500.0, 800.0, 1000.0, 2000.0, 3000.0,
                          4000.0, 5000.0, 8000.0, 10000.0, 16000.0, 22000.0]

FMEG08 = [0.03, 0.04, 0.046, 0.051, 0.054, 0.062, 0.067, 0.07, 0.072, 0.078, 0.08, 0.08, 0.08]
FMEG11 = [0.06, 0.07, 0.076, 0.081, 0.084, 0.092, 0.097, 0.1, 0.102, 0.108, 0.11, 0.11, 0.11]
FMEG14 = [0.09, 0.1, 0.106, 0.111, 0.114, 0.122, 0.127, 0.13, 0.132, 0.138, 0.14, 0.14, 0.14]
FMEG19 = [0.14, 0.15, 0.156, 0.161, 0.164, 0.172, 0.177, 0.18, 0.182, 0.188, 0.19, 0.19, 0.19]
FMEG23 = [0.18, 0.19, 0.196, 0.201, 0.204, 0.212, 0.217, 0.22, 0.222, 0.228, 0.23, 0.23, 0.23]
FMEG28 = [0.23, 0.24, 0.246, 0.251, 0.254, 0.262, 0.267, 0.27, 0.272, 0.278, 0.28, 0.28, 0.28]
FMEG32 = [0.27, 0.28, 0.286, 0.291, 0.294, 0.302, 0.307, 0.31, 0.312, 0.318, 0.32, 0.32, 0.32]

fan_crossflow_efficiencies = {'FMEG08': FMEG08,
                              'FMEG11': FMEG11,
                              'FMEG14': FMEG14,
                              'FMEG19': FMEG19,
                              'FMEG23': FMEG23,
                              'FMEG28': FMEG28,
                              'FMEG32': FMEG32}