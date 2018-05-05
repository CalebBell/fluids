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
from math import log
from collections import namedtuple
from scipy.interpolate import interp1d, interp2d
from scipy.constants import hp
import os
from io import open
from pprint import pprint


__all__ = ['VFD_efficiency', 'CSA_motor_efficiency', 'motor_efficiency_underloaded',
'Corripio_pump_efficiency', 'Corripio_motor_efficiency',
'specific_speed', 'specific_diameter', 'speed_synchronous', 'nema_sizes',
'nema_sizes_hp', 'motor_round_size', 'nema_min_P', 'nema_high_P', 'plug_types',
'voltages_1_phase_residential', 'voltages_3_phase', 'frequencies',
'residential_power', 'industrial_power', 'current_ideal', 
'liquid_jet_pump']

folder = os.path.join(os.path.dirname(__file__), 'data')


def liquid_jet_pump(rhop, rhos, Km=.15, Kd=0.1, Ks=0.1, Kp=0.0, 
                     d_nozzle=None, d_mixing=None, d_diffuser=None,
                     Qp=None, Qs=None, P1=None, P2=None, P5=None,
                     nozzle_retracted=True):
    r'''Calculate the remaining variable in a liquid jet pump, using a model
    presented in [1]_ as well as [2]_, [3]_, and [4]_.
    
    .. math::
        N = \frac{2R + \frac{2 C M^2 R^2}{1-R} - R^2 (1+CM) (1+M) (1 + K_m 
        + K_d + \alpha^2) - \frac{CM^2R^2}{(1-R)^2} (1+K_s)}
        {(1+K_p) - 2R - \frac{2CM^2R^2}{1-R} + R^2(1+CM)(1+M)(1+K_m + K_d 
        + \alpha^2) + (1-j)\left(\frac{CM^2R^2}{({1-R})^2} \right)(1+K_s)}
        
        \text{Pressure ratio} = N = \frac{P_5 - P_2}{P_1 - P_5}
        
        \text{Volume flow ratio} = M =  \frac{Q_s}{Q_p}
        
        \text{Jet pump efficiency} = \eta = M\cdot N = 
        \frac{Q_s(P_5-P_2)}{Q_p(P_1 - P_5)}
        
        R = \frac{A_n}{A_m}
        
        C = \frac{\rho_s}{\rho_p}
    
    Parameters
    ----------
    rhop : float
        The density of the primary (motive) fluid, [kg/m^3]
    rhos : float
        The density of the secondary fluid (drawn from the vacuum chamber),
        [kg/m^3]
    Km : float, optional
        The mixing chamber loss coefficient, [-]
    Kd : float, optional
        The diffuser loss coefficient, [-]
    Ks : float, optional
        The secondary inlet loss coefficient, [-]
    Kp : float, optional
        The primary nozzle loss coefficient, [-]
    d_nozzle : float, optional
        The inside diameter of the primary fluid's nozle, [m]
    d_mixing : float, optional
        The diameter of the mixing chamber, [m]
    d_diffuser : float, optional
        The diameter of the diffuser at its exit, [m]
    Qp : float, optional
        The volumetric flow rate of the primary fluid, [m^3/s]
    Qs : float, optional
        The volumetric flow rate of the secondary fluid, [m^3/s]
    P1 : float, optional
        The pressure of the primary fluid entering its nozzle, [Pa]
    P2 : float, optional
        The pressure of the secondary fluid at the entry of the ejector, [Pa]
    P5 : float, optional
        The pressure at the exit of the diffuser, [Pa]
    nozzle_retracted : bool, optional
        Whether or not the primary nozzle's exit is before the mixing chamber,
        or somewhat inside it, [-]
        
    Returns
    -------
    solution : dict
        Dictionary of calculated parameters, [-]

    Notes
    -----
    
    The assumptions of the model are:
        
    * The flows are one dimensional except in the mixing chamber.
    * The mixing chamber has constant cross-sectional area.
    * The mixing happens entirely in the mixing chamber, prior to entry into
      the diffuser.
    * The primary nozzle is in a straight line with the middle of the mixing 
      chamber.
    * Both fluids are incompressible, and have no excess volume on mixing.
    * Primary and secondary flows both enter the mixing throat with their
      own uniform velocity distribution; the mixed stream leaves with a uniform
      velocity profile.
    * If the secondary fluid is a gas, it undergoes isothermal compression in
      the throat and diffuser. 
    * If the secondary fluid is a gas or contains a bubbly gas, it is
      homogeneously distributed in a continuous liquid phase.
    * Heat transfer between the fluids is negligible - there is no change in
      density due to temperature changes
    * The change in the solubility of a disolved gas, if there is one, is
      negigibly changed by temperature or pressure changes.

    Examples
    --------
    >>> pprint(liquid_jet_pump(rhop=998., rhos=1098., Km=.186, Kd=0.12, Ks=0.11, 
    ... Kp=0.04, d_nozzle=0.0223, d_mixing=0.045, Qs=0.01, Qp=.01, P2=133600, 
    ... P5=200E3, nozzle_retracted=False))
    {'M': 1.0,
     'N': 0.2938665390183575,
     'P1': 425952.91121542786,
     'P2': 133600,
     'P5': 200000.0,
     'Qp': 0.01,
     'Qs': 0.01,
     'R': 0.24557530864197535,
     'alpha': 1e-06,
     'd_diffuser': 45.0,
     'd_mixing': 0.045,
     'd_nozzle': 0.0223,
     'efficiency': 0.2938665390183575}

    References
    ----------
    .. [1] Karassik, Igor J., Joseph P. Messina, Paul Cooper, and Charles C. 
       Heald. Pump Handbook. 4th edition. New York: McGraw-Hill Education, 2007.
    .. [2] Winoto S. H., Li H., and Shah D. A. "Efficiency of Jet Pumps." 
       Journal of Hydraulic Engineering 126, no. 2 (February 1, 2000): 150-56. 
       https://doi.org/10.1061/(ASCE)0733-9429(2000)126:2(150).
    .. [3] Elmore, Emily, Khalid Al-Mutairi, Bilal Hussain, and A. Sherif
       El-Gizawy. "Development of Analytical Model for Predicting Dual-Phase
       Ejector Performance," November 11, 2016, V007T09A013.
    .. [4] Ejectors and Jet Pumps. Design and Performance for Incompressible 
       Liquid Flow. 85032. ESDU International PLC, 1985.
    '''
    C = rhos/rhop
    if nozzle_retracted:
        j = 0.0
    else:
        j = 1.0
    # The diffuser diameter has little effect - it can be specified, and the 
    # results will be a little more accurate - or it can be omitted and it 
    # will be set to a large value so it is ignored in the calculations.
    if d_diffuser is None:
        if d_mixing is not None:
            d_diffuser = d_mixing*1E3
        elif d_nozzle is not None:
            d_diffuser = d_nozzle*1E3
            
    unknowns = sum(i is not None for i in (d_nozzle, d_mixing, Qs, Qp, P1, P2, P5))
    if unknowns < 1:
        raise Exception('Too many unknowns')
    elif unknowns < 1:
        raise Exception('Overspecified')
    if P1 is None or P2 is None or P5 is None:
        # Direct solution - easy
        R = d_nozzle**2/d_mixing**2
        alpha = d_mixing**2/d_diffuser**2
        M = Qs/Qp
        
        M2, R2, alpha2 = M*M, R*R, alpha*alpha
        num = 2.0*R + 2*C*M2*R2/(1.0 - R)
        num -= R2*(1.0 + C*M)*(1.0 + M)*(1.0 + Km + Kd + alpha2)
        num -= C*M2*R2/(1.0 - R)**2*(1.0 + Ks)
        
        den = (1.0 + Kp) - 2.0*R - 2.0*C*M2*R2/(1.0 - R)
        den += R2*(1.0 + C*M)*(1.0 + M)*(1.0 + Km + Kd + alpha2)
        den += (1.0 - j)*(C*M2/((1.0 - R)/R)**2)*(1.0 - Ks)
        N = num/den
        if P1 is None:
            P1 = (-P2 + P5*N + P5)/N
        elif P2 is None:
            P2 = -P1*N + P5*N + P5
        elif P5 is None:
            P5 = (P1*N + P2)/(N + 1.0)
            
    solution = {}
    solution['P1'] = P1
    solution['P2'] = P2
    solution['P5'] = P5
    solution['d_nozzle'] = d_nozzle
    solution['d_mixing'] = d_mixing
    solution['d_diffuser'] = d_diffuser
    solution['Qs'] = Qs
    solution['Qp'] = Qp
    solution['N'] = N
    solution['M'] = M
    solution['R'] = R
    solution['alpha'] = alpha
    solution['efficiency'] = M*N
    return solution


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
    return -0.316 + 0.24015*log(Q) - 0.01199*log(Q)**2


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
    return 0.8 + 0.0319*log(P) - 0.00182*log(P)**2

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
    P = P/hp
    if P < 3:
        P = 3
    elif P > 400:
        P = 400
    if load < 0.016:
        load = 0.016
    return round(float(VFD_efficiency_interp(load, P)), 4)


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
    return round(float(efficiency), 4)

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
    return n*Q**0.5/H**0.75


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
    return D*H**0.25/Q**0.5


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
        raise Exception('Only 1 and 3 phase power supported')
    if phase == 3:
        return P/(V*3**0.5*PF)
    else:
        return P/(V*PF)


with open(os.path.join(folder, 'residential power.csv'), encoding='utf-8') as f:
    residential_power_raw = f.read()

with open(os.path.join(folder, '3 phase power.csv'), encoding='utf-8') as f:
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

# In mm, diameter of fans -> convert to SI
fan_diameters = [125, 132, 140, 150, 160, 170, 180, 190, 200, 212, 224, 236, 250, 265, 280, 300, 315, 335, 355, 375, 400, 425, 450, 475, 500, 530, 560, 600, 630, 670, 710, 750, 800, 850, 900, 950, 1000]
fan_diameters = [i*1E-3 for i in fan_diameters]

FEG90 = [42.5, 44.8, 47.2, 50.1, 52.7, 55.2, 57.4, 59.4, 61.3, 63.3, 65.2, 66.9, 68.6, 70.3, 71.8, 73.5, 74.6, 75.9, 77, 77.9, 78.9, 79.7, 80.4, 81, 81.5, 81.9, 82.3, 82.7, 83, 83.3, 83.5, 83.7, 83.8, 84, 84.1, 84.1, 84.1]
FEG85 = [40.1, 42.3, 44.6, 47.3, 49.8, 52.1, 54.2, 56.1, 57.9, 59.8, 61.6, 63.1, 64.8, 66.4, 67.8, 69.4, 70.4, 71.7, 72.7, 73.6, 74.5, 75.3, 75.9, 76.5, 76.9, 77.4, 77.7, 78.1, 78.4, 78.6, 78.8, 79, 79.1, 79.3, 79.3, 79.4, 79.4]
FEG80 = [37.8, 39.9, 42.1, 44.7, 47, 49.2, 51.1, 53, 54.6, 56.5, 58.1, 59.6, 61.2, 62.7, 64, 65.5, 66.5, 67.6, 68.6, 69.5, 70.3, 71.1, 71.7, 72.2, 72.6, 73, 73.4, 73.8, 74, 74.2, 74.4, 74.6, 74.7, 74.8, 74.9, 75, 75]
FEG75 = [35.7, 37.7, 39.8, 42.2, 44.4, 46.4, 48.3, 50, 51.6, 53.3, 54.9, 56.3, 57.8, 59.2, 60.4, 61.8, 62.8, 63.9, 64.8, 65.6, 66.4, 67.1, 67.7, 68.1, 68.5, 68.9, 69.3, 69.6, 69.8, 70.1, 70.3, 70.4, 70.5, 70.6, 70.7, 70.8, 70.8]
FEG71 = [33.7, 35.6, 37.5, 39.8, 41.9, 43.8, 45.6, 47.2, 48.7, 50.3, 51.8, 53.1, 54.5, 55.9, 57, 58.4, 59.3, 60.3, 61.2, 61.9, 62.7, 63.3, 63.9, 64.3, 64.7, 65.1, 65.4, 65.7, 65.9, 66.1, 66.3, 66.5, 66.6, 66.7, 66.8, 66.8, 66.8]
FEG67 = [31.8, 33.6, 35.4, 37.6, 39.5, 41.4, 43, 44.6, 46, 47.5, 48.9, 50.2, 51.5, 52.7, 53.8, 55.1, 55.9, 56.9, 57.7, 58.4, 59.2, 59.8, 60.3, 60.7, 61.1, 61.4, 61.7, 62.1, 62.2, 62.4, 62.6, 62.7, 62.9, 63, 63, 63.1, 63.1]
FEG63 = [30.1, 31.7, 33.4, 35.5, 37.3, 39, 40.6, 42.1, 43.4, 44.8, 46.2, 47.3, 48.6, 49.8, 50.8, 52, 52.8, 53.7, 54.5, 55.2, 55.9, 56.5, 56.9, 57.3, 57.7, 58, 58.3, 58.6, 58.8, 59, 59.1, 59.2, 59.4, 59.4, 59.5, 59.5, 59.6]
FEG60 = [28.4, 29.9, 31.6, 33.5, 35.2, 36.9, 38.3, 39.7, 41, 42.3, 43.6, 44.7, 45.9, 47, 48, 49.1, 49.9, 50.7, 51.5, 52.1, 52.8, 53.3, 53.8, 54.1, 54.5, 54.8, 55, 55.3, 55.5, 55.7, 55.8, 55.9, 56, 56.1, 56.2, 56.2, 56.2]
FEG56 = [26.8, 28.2, 29.8, 31.6, 33.3, 34.8, 36.2, 37.5, 38.7, 40, 41.1, 42.2, 43.3, 44.4, 45.3, 46.4, 47.1, 47.9, 48.6, 49.2, 49.8, 50.3, 50.7, 51.1, 51.4, 51.7, 51.9, 52.2, 52.4, 52.5, 52.7, 52.8, 52.9, 53, 53, 53.1, 53.1]
FEG53 = [25.3, 26.7, 28.1, 29.8, 31.4, 32.9, 34.2, 35.4, 36.5, 37.7, 38.8, 39.8, 40.9, 41.9, 42.8, 43.8, 44.4, 45.2, 45.9, 46.4, 47, 47.5, 47.9, 48.2, 48.5, 48.8, 49, 49.3, 49.4, 49.6, 49.7, 49.8, 49.9, 50, 50.1, 50.1, 50.1]
FEG50 = [23.9, 25.2, 26.6, 28.2, 29.7, 31, 32.3, 33.4, 34.5, 35.6, 36.7, 37.6, 38.6, 39.5, 40.4, 41.3, 42, 42.7, 43.3, 43.8, 44.4, 44.8, 45.2, 45.5, 45.8, 46.1, 46.3, 46.5, 46.7, 46.8, 47, 47, 47.1, 47.2, 47.3, 47.3, 47.3]

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

# TODO convert efficiencies to fractions
        
'''for key, values in fan_bare_shaft_efficiencies.items():
    plt.plot(fan_diameters, values, label=key)

plt.legend()
plt.show()'''



FMEG_axial_powers = [125.0, 300.0, 1000.0, 2500.0, 5000.0, 8000.0, 10000.0, 20000.0, 60000.0, 160000.0, 300000.0, 375000.0, 500000.0]

FMEG27 = [15, 17.4, 20.7, 23.2, 25.1, 26.4, 27, 27.5, 28.3, 29.1, 29.6, 29.7, 30]
FMEG31 = [19, 21.4, 24.7, 27.2, 29.1, 30.4, 31, 31.5, 32.3, 33.1, 33.6, 33.7, 34]
FMEG35 = [23, 25.4, 28.7, 31.2, 33.1, 34.4, 35, 35.5, 36.3, 37.1, 37.6, 37.7, 38]
FMEG39 = [27, 29.4, 32.7, 35.2, 37.1, 38.4, 39, 39.5, 40.3, 41.1, 41.6, 41.7, 42]
FMEG42 = [30, 32.4, 35.7, 38.2, 40.1, 41.4, 42, 42.5, 43.3, 44.1, 44.6, 44.7, 45]
FMEG46 = [34, 36.4, 39.7, 42.2, 44.1, 45.4, 46, 46.5, 47.3, 48.1, 48.6, 48.7, 49]
FMEG50 = [38, 40.4, 43.7, 46.2, 48.1, 49.4, 50, 50.5, 51.3, 52.1, 52.6, 52.7, 53]
FMEG53 = [41, 43.4, 46.7, 49.2, 51.1, 52.4, 53, 53.5, 54.3, 55.1, 55.6, 55.7, 56]
FMEG55 = [43, 45.4, 48.7, 51.2, 53.1, 54.4, 55, 55.5, 56.3, 57.1, 57.6, 57.7, 58]
FMEG58 = [46, 48.4, 51.7, 54.2, 56.1, 57.4, 58, 58.5, 59.3, 60.1, 60.6, 60.7, 61]
FMEG60 = [48, 50.4, 53.7, 56.2, 58.1, 59.4, 60, 60.5, 61.3, 62.1, 62.6, 62.7, 63]
FMEG62 = [50, 52.4, 55.7, 58.2, 60.1, 61.4, 62, 62.5, 63.3, 64.1, 64.6, 64.7, 65]
FMEG64 = [52, 54.4, 57.7, 60.2, 62.1, 63.4, 64, 64.5, 65.3, 66.1, 66.6, 66.7, 67]
FMEG66 = [54, 56.4, 59.7, 62.2, 64.1, 65.4, 66, 66.5, 67.3, 68.1, 68.6, 68.7, 69]

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
FMEG35 = [15, 19, 24.5, 28.7, 31.8, 34, 35, 35.7, 36.9, 38, 38.7, 38.9, 39.2]
FMEG39 = [19, 23, 28.5, 32.7, 35.8, 38, 39, 39.7, 40.9, 42, 42.7, 42.9, 43.2]
FMEG42 = [22, 26, 31.5, 35.7, 38.8, 41, 42, 42.7, 43.9, 45, 45.7, 45.9, 46.2]
FMEG46 = [26, 30, 35.5, 39.7, 42.8, 45, 46, 46.7, 47.9, 49, 49.7, 49.9, 50.2]
FMEG50 = [30, 34, 39.5, 43.7, 46.8, 49, 50, 50.7, 51.9, 53, 53.7, 53.9, 54.2]
FMEG53 = [33, 37, 42.5, 46.7, 49.8, 52, 53, 53.7, 54.9, 56, 56.7, 56.9, 57.2]
FMEG55 = [35, 39, 44.5, 48.7, 51.8, 54, 55, 55.7, 56.9, 58, 58.7, 58.9, 59.2]
FMEG58 = [38, 42, 47.5, 51.7, 54.8, 57, 58, 58.7, 59.9, 61, 61.7, 61.9, 62.2]
FMEG60 = [40, 44, 49.5, 53.7, 56.8, 59, 60, 60.7, 61.9, 63, 63.7, 63.9, 64.2]
FMEG62 = [42, 46, 51.5, 55.7, 58.8, 61, 62, 62.7, 63.9, 65, 65.7, 65.9, 66.2]
FMEG64 = [44, 48, 53.5, 57.7, 60.8, 63, 64, 64.7, 65.9, 67, 67.7, 67.9, 68.2]
FMEG66 = [46, 50, 55.5, 59.7, 62.8, 65, 66, 66.7, 67.9, 69, 69.7, 69.9, 70.2]
FMEG68 = [48, 52, 57.5, 61.7, 64.8, 67, 68, 68.7, 69.9, 71, 71.7, 71.9, 72.2]
FMEG70 = [50, 54, 59.5, 63.7, 66.8, 69, 70, 70.7, 71.9, 73, 73.7, 73.9, 74.2]
FMEG72 = [52, 56, 61.5, 65.7, 68.8, 71, 72, 72.7, 73.9, 75, 75.7, 75.9, 76.2]
FMEG74 = [54, 58, 63.5, 67.7, 70.8, 73, 74, 74.7, 75.9, 77, 77.7, 77.9, 78.2]
FMEG76 = [56, 60, 65.5, 69.7, 72.8, 75, 76, 76.7, 77.9, 79, 79.7, 79.9, 80.2]

fan_centrifugal_backward_efficiencies = {'FMEG35': FMEG35,
                                         'FMEG39': FMEG39,
                                         'FMEG42': FMEG42,
                                         'FMEG46': FMEG46,
                                         'FMEG50': FMEG50,
                                         'FMEG53': FMEG53,
                                         'FMEG55': FMEG55,
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

FMEG_cross_flow_powers = [130.0, 300.0, 500.0, 800.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 8000.0, 10000.0, 16000.0, 22000.0]

FMEG08 = [3, 4, 4.6, 5.1, 5.4, 6.2, 6.7, 7, 7.2, 7.8, 8, 8, 8]
FMEG11 = [6, 7, 7.6, 8.1, 8.4, 9.2, 9.7, 10, 10.2, 10.8, 11, 11, 11]
FMEG14 = [9, 10, 10.6, 11.1, 11.4, 12.2, 12.7, 13, 13.2, 13.8, 14, 14, 14]
FMEG19 = [14, 15, 15.6, 16.1, 16.4, 17.2, 17.7, 18, 18.2, 18.8, 19, 19, 19]
FMEG23 = [18, 19, 19.6, 20.1, 20.4, 21.2, 21.7, 22, 22.2, 22.8, 23, 23, 23]
FMEG28 = [23, 24, 24.6, 25.1, 25.4, 26.2, 26.7, 27, 27.2, 27.8, 28, 28, 28]
FMEG32 = [27, 28, 28.6, 29.1, 29.4, 30.2, 30.7, 31, 31.2, 31.8, 32, 32, 32]

fan_crossflow_efficiencies = {'FMEG08': FMEG08,
                              'FMEG11': FMEG11,
                              'FMEG14': FMEG14,
                              'FMEG19': FMEG19,
                              'FMEG23': FMEG23,
                              'FMEG28': FMEG28,
                              'FMEG32': FMEG32}

'''Convert the efficiencies of: 
    * Bare shafts
    * Centrifugal backward bladed mixed flow fans
    * Cross flow driven fans
    * Driven forward curved radial centrifugal fans
to fractions, instead of percents.
'''
for d in (fan_bare_shaft_efficiencies, fan_driven_axial_efficiencies, fan_centrifugal_backward_efficiencies, fan_crossflow_efficiencies):
    for values in d.values():
        for i in range(len(values)):
            values[i] = values[i]*1E-2
