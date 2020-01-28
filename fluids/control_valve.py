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
from math import log10, exp, pi
from fluids.constants import R, psi, gallon, minute
from fluids.numerics import interp, implementation_optimize_tck, splev
from fluids.fittings import Cv_to_Kv, Kv_to_Cv

__all__ = ['size_control_valve_l', 'size_control_valve_g', 'cavitation_index',
           'FF_critical_pressure_ratio_l', 'is_choked_turbulent_l', 
           'is_choked_turbulent_g', 'Reynolds_valve', 
           'loss_coefficient_piping', 'Reynolds_factor',
           'Cv_char_quick_opening', 'Cv_char_linear', 
           'Cv_char_equal_percentage',
           'convert_flow_coefficient', 'control_valve_choke_P_l',
           'control_valve_choke_P_g', 'control_valve_noise_l_2015',
           'control_valve_noise_g_2011']

N1 = 0.1 # m^3/hr, kPa
N2 = 1.6E-3 # mm
N4 = 7.07E-2 # m^3/hr, m^2/s
N5 = 1.8E-3 # mm
N6 = 3.16 # kg/hr, kPa, kg/m^3
N7 = 4.82 # m^3/hr kPa K
N8 = 1.10 # kPa kg/hr K
#N9 = 2.60E1 # m^3/hr kPa K at 15 deg C
N9 = 2.46E1 # m^3/hr kPa K at 0 deg C
N18 = 8.65E-1 # mm
N19 = 2.5 # mm
#N22 = 1.84E1 # m^3/hr kPa K at 15 deg C
N27 = 7.75E-1 # kg/hr kPa K at 0 deg C
N32 = 1.4E2 # mm


rho0 = 999.10329075702327 # Water at 288.15 K


def cavitation_index(P1, P2, Psat):
    r'''Calculates the cavitation index of a valve with upstream and downstream
    absolute pressures `P1` and `P2` for a fluid with a vapor pressure `Psat`.

    .. math::
        \sigma = \frac{P_1 - P_{sat}}{P_1 - P_2}

    Parameters
    ----------
    P1 : float
        Absolute pressure upstream of the valve [Pa]
    P2 : float
        Absolute pressure downstream of the valve [Pa]
    Psat : float
        Saturation pressure of the liquid at inlet temperature [Pa]

    Returns
    -------
    sigma : float
        Cavitation index of the valve [-]

    Notes
    -----
    Larger values are safer. Models for adjusting cavitation indexes provided
    by the manufacturer to the user's conditions are available, making use
    of scaling the pressure differences and size differences.

    Values can be calculated for incipient cavitation, constant cavitation,
    maximum vibration cavitation, incipient damage, and choking cavitation.

    Has also been defined as:

    .. math::
            \sigma = \frac{P_2 - P_{sat}}{P_1 - P_2}

    Another definition and notation series is:

    .. math::
        K = xF = \frac{1}{\sigma} = \frac{P_1 - P_2}{P_1 - P_{sat}}

    Examples
    --------
    >>> cavitation_index(1E6, 8E5, 2E5)
    4.0

    References
    ----------
    .. [1] ISA. "RP75.23 Considerations for Evaluating Control Valve
       Cavitation." 1995.
    '''
    return (P1 - Psat)/(P1 - P2)


def FF_critical_pressure_ratio_l(Psat, Pc):
    r'''Calculates FF, the liquid critical pressure ratio factor,
    for use in IEC 60534 liquid valve sizing calculations.

    .. math::
        F_F = 0.96 - 0.28\sqrt{\frac{P_{sat}}{P_c}}

    Parameters
    ----------
    Psat : float
        Saturation pressure of the liquid at inlet temperature [Pa]
    Pc : float
        Critical pressure of the liquid [Pa]

    Returns
    -------
    FF : float
        Liquid critical pressure ratio factor [-]

    Examples
    --------
    From [1]_, matching example.

    >>> FF_critical_pressure_ratio_l(70100.0, 22120000.0)
    0.9442375225233299

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    return 0.96 - 0.28*(Psat/Pc)**0.5


def control_valve_choke_P_l(Psat, Pc, FL, P1=None, P2=None, disp=True):
    r'''Calculates either the upstream or downstream pressure at which choked
    flow though a liquid control valve occurs, given either a set upstream or 
    downstream pressure. Implements an analytical solution of 
    the needed equations from the full function
    :py:func:`~.size_control_valve_l`. For some pressures, no choked flow 
    is possible; for choked flow to occur the direction if flow must be 
    reversed. If `disp` is True, an exception will be raised for these
    conditions.
    
    .. math::
        P_1 = \frac{F_{F} F_{L}^{2} P_{sat} - P_{2}}{F_{L}^{2} - 1}
        
    .. math::
        P_2 = F_{F} F_{L}^{2} P_{sat} - F_{L}^{2} P_{1} + P_{1}
    
    Parameters
    ----------
    Psat : float
        Saturation pressure of the liquid at inlet temperature [Pa]
    Pc : float
        Critical pressure of the liquid [Pa]
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached 
        fittings [-]
    P1 : float, optional
        Absolute pressure upstream of the valve [Pa]
    P2 : float, optional
        Absolute pressure downstream of the valve [Pa]
    disp : bool, optional
        Whether or not to raise an exception on flow reversal, [-]

    Returns
    -------
    P_choke : float
        Pressure at which a choke occurs in the liquid valve [Pa]

    Notes
    -----
    Extremely cheap to compute.
    
    Examples
    --------
    >>> control_valve_choke_P_l(69682.89291024722, 22048320.0, 0.6, 680000.0)
    458887.5306077305
    >>> control_valve_choke_P_l(69682.89291024722, 22048320.0, 0.6, P2=458887.5306077305)
    680000.0
    '''
    FF = FF_critical_pressure_ratio_l(Psat=Psat, Pc=Pc)
    Pmin_absolute = FF*Psat
    if P2 is None:
        ans = P2 = FF*FL*FL*Psat - FL*FL*P1 + P1
    elif P1 is None:
        ans = P1 = (FF*FL*FL*Psat - P2)/(FL*FL - 1.0)
    else:
        raise Exception('Either P1 or P2 needs to be specified')
    if P2 > P1 and disp:
        raise Exception('Specified P1 is too low for choking to occur '
                        'at any downstream pressure; minimum '
                        'upstream pressure for choking to be possible '
                        'is %g Pa.' %Pmin_absolute)
    return ans


def control_valve_choke_P_g(xT, gamma, P1=None, P2=None):
    r'''Calculates either the upstream or downstream pressure at which choked
    flow though a gas control valve occurs, given either a set upstream or 
    downstream pressure. Implements an analytical solution of 
    the needed equations from the full function
    :py:func:`~.size_control_valve_g`. A singularity arises as `xT` goes to 1
    and `gamma` goes to 1.4.
    
    .. math::
        P_1 = - \frac{7 P_{2}}{5 \gamma x_T - 7}
        
    .. math::
        P_2 = \frac{P_{1}}{7} \left(- 5 \gamma x_T + 7\right)
    
    Parameters
    ----------
    xT : float, optional
        Pressure difference ratio factor of a valve without fittings at choked
        flow [-]
    gamma : float
        Specific heat capacity ratio [-]
    P1 : float, optional
        Absolute pressure upstream of the valve [Pa]
    P2 : float, optional
        Absolute pressure downstream of the valve [Pa]

    Returns
    -------
    P_choke : float
        Pressure at which a choke occurs in the gas valve [Pa]

    Notes
    -----
    Extremely cheap to compute.
    
    Examples
    --------
    >>> control_valve_choke_P_g(1, 1.3, 1E5)
    7142.857142857143
    >>> control_valve_choke_P_g(1, 1.3, P2=7142.857142857143)
    100000.0
    '''
    if P2 is None:
        ans = P2 = P1*(-5.0*gamma*xT + 7.0)/7.0
    elif P1 is None:
        ans = P1 = -7.0*P2/(5.0*gamma*xT - 7.0)
    else:
        raise Exception('Either P1 or P2 needs to be specified')
    return ans


def is_choked_turbulent_l(dP, P1, Psat, FF, FL=None, FLP=None, FP=None):
    r'''Calculates if a liquid flow in IEC 60534 calculations is critical or
    not, for use in IEC 60534 liquid valve sizing calculations.
    Either FL may be provided or FLP and FP, depending on the calculation
    process.

    .. math::
        \Delta P > F_L^2(P_1 - F_F P_{sat})

    .. math::
        \Delta P >= \left(\frac{F_{LP}}{F_P}\right)^2(P_1 - F_F P_{sat})

    Parameters
    ----------
    dP : float
        Differential pressure across the valve, with reducer/expanders [Pa]
    P1 : float
        Pressure of the fluid before the valve and reducers/expanders [Pa]
    Psat : float
        Saturation pressure of the fluid at inlet temperature [Pa]
    FF : float
        Liquid critical pressure ratio factor [-]
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached fittings [-]
    FLP : float, optional
        Combined liquid pressure recovery factor with piping geometry factor,
        for a control valve with attached fittings [-]
    FP : float, optional
        Piping geometry factor [-]

    Returns
    -------
    choked : bool
        Whether or not the flow is choked [-]

    Examples
    --------
    >>> is_choked_turbulent_l(460.0, 680.0, 70.1, 0.94, 0.9)
    False
    >>> is_choked_turbulent_l(460.0, 680.0, 70.1, 0.94, 0.6)
    True

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    if FLP and FP:
        return dP >= (FLP/FP)**2*(P1-FF*Psat)
    elif FL:
        return dP >= FL**2*(P1-FF*Psat)
    else:
        raise Exception('Either (FLP and FP) or FL is needed')


def is_choked_turbulent_g(x, Fgamma, xT=None, xTP=None):
    r'''Calculates if a gas flow in IEC 60534 calculations is critical or
    not, for use in IEC 60534 gas valve sizing calculations.
    Either xT or xTP must be provided, depending on the calculation process.

    .. math::
        x \ge F_\gamma x_T

    .. math::
        x \ge F_\gamma x_{TP}

    Parameters
    ----------
    x : float
        Differential pressure over inlet pressure, [-]
    Fgamma : float
        Specific heat ratio factor [-]
    xT : float, optional
        Pressure difference ratio factor of a valve without fittings at choked
        flow [-]
    xTP : float
        Pressure difference ratio factor of a valve with fittings at choked
        flow [-]

    Returns
    -------
    choked : bool
        Whether or not the flow is choked [-]

    Examples
    --------
    Example 3, compressible flow, non-choked with attached fittings:

    >>> is_choked_turbulent_g(0.544, 0.929, 0.6)
    False
    >>> is_choked_turbulent_g(0.544, 0.929, xTP=0.625)
    False

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    if xT:
        return x >= Fgamma*xT
    elif xTP:
        return x >= Fgamma*xTP
    else:
        raise Exception('Either xT or xTP is needed')


def Reynolds_valve(nu, Q, D1, FL, Fd, C):
    r'''Calculates Reynolds number of a control valve for a liquid or gas
    flowing through it at a specified Q, for a specified D1, FL, Fd, C, and
    with kinematic viscosity `nu` according to IEC 60534 calculations.

    .. math::
        Re_v = \frac{N_4 F_d Q}{\nu \sqrt{C F_L}}\left(\frac{F_L^2 C^2}
        {N_2D^4} +1\right)^{1/4}

    Parameters
    ----------
    nu : float
        Kinematic viscosity, [m^2/s]
    Q : float
        Volumetric flow rate of the fluid [m^3/s]
    D1 : float
        Diameter of the pipe before the valve [m]
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached 
        fittings []
    Fd : float
        Valve style modifier [-]
    C : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]

    Returns
    -------
    Rev : float
        Valve reynolds number [-]

    Examples
    --------
    >>> Reynolds_valve(3.26e-07, 360, 150.0, 0.9, 0.46, 165)
    2966984.7525455453

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    return N4*Fd*Q/nu/(C*FL)**0.5*(FL**2*C**2/(N2*D1**4) + 1)**0.25


def loss_coefficient_piping(d, D1=None, D2=None):
    r'''Calculates the sum of loss coefficients from possible
    inlet/outlet reducers/expanders around a control valve according to
    IEC 60534 calculations.

    .. math::
        \Sigma \xi = \xi_1 + \xi_2 + \xi_{B1} - \xi_{B2}

    .. math::
        \xi_1 = 0.5\left[1 -\left(\frac{d}{D_1}\right)^2\right]^2

    .. math::
        \xi_2 = 1.0\left[1 -\left(\frac{d}{D_2}\right)^2\right]^2

    .. math::
        \xi_{B1} = 1 - \left(\frac{d}{D_1}\right)^4

    .. math::
        \xi_{B2} = 1 - \left(\frac{d}{D_2}\right)^4

    Parameters
    ----------
    d : float
        Diameter of the valve [m]
    D1 : float
        Diameter of the pipe before the valve [m]
    D2 : float
        Diameter of the pipe after the valve [m]

    Returns
    -------
    loss : float
        Sum of the four loss coefficients [-]

    Examples
    --------
    In example 3, non-choked compressible flow with fittings:

    >>> loss_coefficient_piping(0.05, 0.08, 0.1)
    0.6580810546875

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    loss = 0.
    if D1:
        loss += 1. - (d/D1)**4 # Inlet flow energy
        loss += 0.5*(1. - (d/D1)**2)**2 # Inlet reducer
    if D2:
        loss += 1.0*(1. - (d/D2)**2)**2 # Outlet reducer (expander)
        loss -= 1. - (d/D2)**4 # Outlet flow energy
    return loss


def Reynolds_factor(FL, C, d, Rev, full_trim=True):
    r'''Calculates the Reynolds number factor `FR` for a valve with a Reynolds
    number `Rev`, diameter `d`, flow coefficient `C`, liquid pressure recovery
    factor `FL`, and with either full or reduced trim, all according to
    IEC 60534 calculations.


    If full trim:

    .. math::
        F_{R,1a} = 1 + \left(\frac{0.33F_L^{0.5}}{n_1^{0.25}}\right)\log_{10}
        \left(\frac{Re_v}{10000}\right)

    .. math::
        F_{R,2} = \min(\frac{0.026}{F_L}\sqrt{n_1 Re_v},\; 1)

    .. math::
        n_1 = \frac{N_2}{\left(\frac{C}{d^2}\right)^2}

    .. math::
        F_R = F_{R,2} \text{ if Rev < 10 else } \min(F_{R,1a}, F_{R,2})

    Otherwise :

    .. math::
        F_{R,3a} = 1 + \left(\frac{0.33F_L^{0.5}}{n_2^{0.25}}\right)\log_{10}
        \left(\frac{Re_v}{10000}\right)

    .. math::
        F_{R,4} = \frac{0.026}{F_L}\sqrt{n_2 Re_v}

    .. math::
        n_2 = 1 + N_{32}\left(\frac{C}{d}\right)^{2/3}

    .. math::
        F_R = F_{R,4} \text{ if Rev < 10 else } \min(F_{R,3a}, F_{R,4})

    Parameters
    ----------
    FL : float
        Liquid pressure recovery factor of a control valve without attached
        fittings []
    C : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]
    d : float
        Diameter of the valve [m]
    Rev : float
        Valve reynolds number [-]
    full_trim : bool
        Whether or not the valve has full trim

    Returns
    -------
    FR : float
        Reynolds number factor for laminar or transitional flow []

    Examples
    --------
    In Example 4, compressible flow with small flow trim sized for gas flow
    (Cv in the problem was converted to Kv here to make FR match with N32, N2):

    >>> Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=False)
    0.7148753122302025


    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    if full_trim:
        n1 = N2/(min(C/d**2, 0.04))**2 # C/d**2 must not exceed 0.04
        FR_1a = 1 + (0.33*FL**0.5)/n1**0.25*log10(Rev/10000.)
        FR_2 = 0.026/FL*(n1*Rev)**0.5
        if Rev < 10:
            FR = FR_2
        else:
            FR = min(FR_2, FR_1a)
    else:
        n2 = 1 + N32*(C/d**2)**(2/3.)
        FR_3a = 1 + (0.33*FL**0.5)/n2**0.25*log10(Rev/10000.)
        FR_4 = min(0.026/FL*(n2*Rev)**0.5, 1)
        if Rev < 10:
            FR = FR_4
        else:
            FR = min(FR_3a, FR_4)
    return FR


def size_control_valve_l(rho, Psat, Pc, mu, P1, P2, Q, D1=None, D2=None,
                         d=None, FL=0.9, Fd=1, allow_choked=True, 
                         allow_laminar=True, full_output=False):
    r'''Calculates flow coefficient of a control valve passing a liquid
    according to IEC 60534. Uses a large number of inputs in SI units. Note the
    return value is not standard SI. All parameters are required.
    This sizing model does not officially apply to liquid mixtures, slurries,
    non-Newtonian fluids, or liquid-solid conveyance systems. For details
    of the calculations, consult [1]_.

    Parameters
    ----------
    rho : float
        Density of the liquid at the inlet [kg/m^3]
    Psat : float
        Saturation pressure of the fluid at inlet temperature [Pa]
    Pc : float
        Critical pressure of the fluid [Pa]
    mu : float
        Viscosity of the fluid [Pa*s]
    P1 : float
        Inlet pressure of the fluid before valves and reducers [Pa]
    P2 : float
        Outlet pressure of the fluid after valves and reducers [Pa]
    Q : float
        Volumetric flow rate of the fluid [m^3/s]
    D1 : float, optional
        Diameter of the pipe before the valve [m]
    D2 : float, optional
        Diameter of the pipe after the valve [m]
    d : float, optional
        Diameter of the valve [m]
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached 
        fittings (normally 0.8-0.9 at full open and decreasing as opened 
        further to below 0.5; use default very cautiously!) []
    Fd : float, optional
        Valve style modifier (0.1 to 1; varies tremendously depending on the
        type of valve and position; do not use the default at all!) []
    allow_choked : bool, optional
        Overrides the automatic transition into the choked regime if this is
        False and returns as if choked flow does not exist
    allow_laminar : bool, optional
        Overrides the automatic transition into the laminar regime if this is
        False and returns as if laminar flow does not exist
    full_output : bool, optional
        If True, returns intermediate calculation values as
        well as Kv in the form of a dictionary containing 'Kv', 'Rev', 'choked',
        'FL', 'FLP', 'FR', 'FP', and 'laminar'. Some may be None if they are 
        not used in the calculation.

    Returns
    -------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]
        
    Notes
    -----
    It is possible to use this model without any diameters specified; in that
    case, turbulent flow is assumed. Choked flow can still be modeled. This is
    not recommended. All three diameters need to be None for this to work. 
    `FL` and `Fd` are not used by the models when the diameters are not 
    specified.

    Examples
    --------
    From [1]_, matching example 1 for a globe, parabolic plug,
    flow-to-open valve.

    >>> size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4,
    ... P1=680E3, P2=220E3, Q=0.1, D1=0.15, D2=0.15, d=0.15,
    ... FL=0.9, Fd=0.46)
    164.9954763704956

    From [1]_, matching example 2 for a ball, segmented ball,
    flow-to-open valve.

    >>> size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4,
    ... P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.1,
    ... FL=0.6, Fd=0.98)
    238.05817216710483
    
    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    if full_output:
        ans = {'FLP': None, 'FP': None, 'FR': None}
    # Pa to kPa, according to constants in standard
    P1, P2, Psat, Pc = P1/1000., P2/1000., Psat/1000., Pc/1000.
    Q = Q*3600. # m^3/s to m^3/hr, according to constants in standard
    nu = mu/rho # kinematic viscosity used in standard

    dP = P1 - P2
    FF = FF_critical_pressure_ratio_l(Psat=Psat, Pc=Pc)
    choked = is_choked_turbulent_l(dP=dP, P1=P1, Psat=Psat, FF=FF, FL=FL)
    if choked and allow_choked:
        # Choked flow, equation 3
        C = Q/N1/FL*(rho/rho0/(P1 - FF*Psat))**0.5
    else:
        # non-choked flow, eq 1
        C = Q/N1*(rho/rho0/dP)**0.5
    if D1 is None and D2 is None and d is None:
        # Assume turbulent if no diameters are provided, no other calculations
        Rev = 1e5
    else:
        # m to mm, according to constants in standard
        D1, D2, d = D1*1000., D2*1000., d*1000.
        Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=C)
        # normal calculation path
        if (Rev > 10000 or not allow_laminar) and (D1 != d or D2 != d):
            # liquid, using Fp and FLP
            FP = 1
            Ci = C
            def iterate_piping_turbulent(Ci):
                loss = loss_coefficient_piping(d, D1, D2)
                FP = (1 + loss/N2*(Ci/d**2)**2)**-0.5
                loss_upstream = loss_coefficient_piping(d, D1)
                FLP = FL*(1 + FL**2/N2*loss_upstream*(Ci/d**2)**2)**-0.5
                choked = is_choked_turbulent_l(dP, P1, Psat, FF, FLP=FLP, FP=FP)
                if choked:
                    # Choked flow with piping, equation 4
                    C = Q/N1/FLP*(rho/rho0/(P1-FF*Psat))**0.5
                else:
                    # Non-Choked flow with piping, equation 5
                    C = Q/N1/FP*(rho/rho0/dP)**0.5
                if Ci/C < 0.99:
                    C = iterate_piping_turbulent(C)
                    
                if full_output:
                    ans['FLP'] = FLP
                    ans['FP'] = FP
                return C
    
            C = iterate_piping_turbulent(Ci)
        elif Rev <= 10000 and allow_laminar:
            # Laminar
            def iterate_piping_laminar(C):
                Ci = 1.3*C
                Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=Ci)                
                if Ci/d**2 > 0.016*N18:
                    FR = Reynolds_factor(FL=FL, C=Ci, d=d, Rev=Rev, full_trim=False)
                else:
                    FR = Reynolds_factor(FL=FL, C=Ci, d=d, Rev=Rev, full_trim=True)
                if C/FR >= Ci:
                    Ci = iterate_piping_laminar(Ci) # pragma: no cover
                    
                if full_output:
                    ans['Rev'] = Rev
                    ans['FR'] = FR
                return Ci
            C = iterate_piping_laminar(C)
    if full_output:
        ans['FF'] = FF
        ans['choked'] = choked
        ans['Kv'] = C
        ans['laminar'] = Rev <= 10000
        
        # For the laminar case this is already set and needs to not be overwritten
        if 'Rev' not in ans:
            ans['Rev'] = Rev
        return ans
    else:
        return C


def size_control_valve_g(T, MW, mu, gamma, Z, P1, P2, Q, D1=None, D2=None, 
                         d=None, FL=0.9, Fd=1, xT=0.7, allow_choked=True, 
                         allow_laminar=True, full_output=False):
    r'''Calculates flow coefficient of a control valve passing a gas
    according to IEC 60534. Uses a large number of inputs in SI units. Note the
    return value is not standard SI. All parameters are required. For details
    of the calculations, consult [1]_. Note the inlet gas flow conditions.

    Parameters
    ----------
    T : float
        Temperature of the gas at the inlet [K]
    MW : float
        Molecular weight of the gas [g/mol]
    mu : float
        Viscosity of the fluid at inlet conditions [Pa*s]
    gamma : float
        Specific heat capacity ratio [-]
    Z : float
        Compressibility factor at inlet conditions, [-]
    P1 : float
        Inlet pressure of the gas before valves and reducers [Pa]
    P2 : float
        Outlet pressure of the gas after valves and reducers [Pa]
    Q : float
        Volumetric flow rate of the gas at *273.15 K* and 1 atm specifically
        [m^3/s]
    D1 : float, optional
        Diameter of the pipe before the valve [m]
    D2 : float, optional
        Diameter of the pipe after the valve [m]
    d : float, optional
        Diameter of the valve [m]        
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached 
        fittings (normally 0.8-0.9 at full open and decreasing as opened 
        further to below 0.5; use default very cautiously!) []
    Fd : float, optional
        Valve style modifier (0.1 to 1; varies tremendously depending on the
        type of valve and position; do not use the default at all!) []
    xT : float, optional
        Pressure difference ratio factor of a valve without fittings at choked
        flow (increasing to 0.9 or higher as the valve is closed further and 
        decreasing to 0.1 or lower as the valve is opened further; use default
        very cautiously!) [-]
    allow_choked : bool, optional
        Overrides the automatic transition into the choked regime if this is
        False and returns as if choked flow does not exist
    allow_laminar : bool, optional
        Overrides the automatic transition into the laminar regime if this is
        False and returns as if laminar flow does not exist
    full_output : bool, optional
        If True, returns intermediate calculation values as
        well as Kv in the form of a dictionary containing 'Kv', 'Rev', 'choked',
        'Y', 'FR', 'FP', 'xTP', and 'laminar'. Some may be None if they are 
        not used in the calculation.
        
    Returns
    -------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]

    Notes
    -----
    It is possible to use this model without any diameters specified; in that
    case, turbulent flow is assumed. Choked flow can still be modeled. This is
    not recommended. All three diameters need to be None for this to work.
    `FL` and `Fd` are not used by the models when the diameters are not 
    specified, but `xT` definitely is used by the model.
    
    Examples
    --------
    From [1]_, matching example 3 for non-choked gas flow with attached
    fittings  and a rotary, eccentric plug, flow-to-open control valve:

    >>> size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,
    ... Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05,
    ... FL=0.85, Fd=0.42, xT=0.60)
    72.58664545391052

    From [1]_, roughly matching example 4 for a small flow trim sized tapered
    needle plug valve. Difference is 3% and explained by the difference in
    algorithms used.

    >>> size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0,
    ... P1=2.8E5, P2=1.3E5, Q=0.46/3600., D1=0.015, D2=0.015, d=0.015, FL=0.98,
    ... Fd=0.07, xT=0.8)
    0.016498765335995726

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    MAX_C_POSSIBLE = 1E40 # Quit iterations if C reaches this high
    # Pa to kPa, according to constants in standard
    P1, P2 = P1/1000., P2/1000.
    Q = Q*3600. # m^3/s to m^3/hr, according to constants in standard
    # Convert dynamic viscosity to kinematic viscosity
    Vm = Z*R*T/(P1*1000)
    rho = (Vm)**-1*MW/1000.
    nu = mu/rho # kinematic viscosity used in standard

    dP = P1 - P2
    Fgamma = gamma/1.40
    x = dP/P1
    Y = max(1 - x/(3*Fgamma*xT), 2/3.)

    choked = is_choked_turbulent_g(x, Fgamma, xT)
    if choked and allow_choked:
        # Choked, and flow coefficient from eq 14a
        C = Q/(N9*P1*Y)*(MW*T*Z/xT/Fgamma)**0.5
    else:
        # Non-choked, and flow coefficient from eq 8a
        C = Q/(N9*P1*Y)*(MW*T*Z/x)**0.5


    if full_output:
        ans = {'FP': None, 'xTP': None, 'FR': None, 
               'choked': choked, 'Y': Y}

    if D1 is None and D2 is None and d is None:
        # Assume turbulent if no diameters are provided, no other calculations
        Rev = 1e5
        if full_output:
            ans['Rev'] = None
    else:
        # m to mm, according to constants in standard
        D1, D2, d = D1*1000., D2*1000., d*1000. # Convert diameters to mm which is used in the standard
        Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=C)
        if full_output:
            ans['Rev'] = Rev

        if (Rev > 10000 or not allow_laminar) and (D1 != d or D2 != d):
            # gas, using xTP and FLP
            FP = 1.
            MAX_ITER = 20
            def iterate_piping_coef(Ci, iterations):
                loss = loss_coefficient_piping(d, D1, D2)
                FP = (1. + loss/N2*(Ci/d**2)**2)**-0.5
                loss_upstream = loss_coefficient_piping(d, D1)
                xTP = xT/FP**2/(1 + xT*loss_upstream/N5*(Ci/d**2)**2)
                choked = is_choked_turbulent_g(x, Fgamma, xTP=xTP)
                if choked:
                    # Choked flow with piping, equation 17a
                    C = Q/(N9*FP*P1*Y)*(MW*T*Z/xTP/Fgamma)**0.5
                else:
                    # Non-choked flow with piping, equation 11a
                    C = Q/(N9*FP*P1*Y)*(MW*T*Z/x)**0.5
                if Ci/C < 0.99 and iterations < MAX_ITER and Ci < MAX_C_POSSIBLE:
                    C = iterate_piping_coef(C, iterations+1)
                if full_output:
                    ans['xTP'] = xTP
                    ans['FP'] = FP
                    ans['choked'] = choked
                    if MAX_ITER == iterations or Ci >= MAX_C_POSSIBLE:
                        ans['warning'] = 'Not converged in inner loop'
                return C
            C = iterate_piping_coef(C, 0)
        elif Rev <= 10000 and allow_laminar:
            # Laminar;
            def iterate_piping_laminar(C):
                Ci = 1.3*C
                Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=Ci)
                if Ci/d**2 > 0.016*N18:
                    FR = Reynolds_factor(FL=FL, C=Ci, d=d, Rev=Rev, full_trim=False)
                else:
                    FR = Reynolds_factor(FL=FL, C=Ci, d=d, Rev=Rev, full_trim=True)
                if C/FR >= Ci:
                    Ci = iterate_piping_laminar(Ci)
                if full_output:
                    ans['FR'] = FR
                    ans['Rev'] = Rev
                return Ci
            C = iterate_piping_laminar(C)
    if full_output:
        ans['Kv'] = C
        ans['laminar'] = Rev <= 10000
        return ans
    else:
        return C


# Valve data from Emerson Valve Handbook 5E
# Quick opening valve data, spline fit, and interpolating function
opening_quick = [0.0, 0.0136, 0.02184, 0.03256, 0.04575, 0.06221, 0.07459, 0.0878, 0.10757, 0.12654, 0.14301, 0.16032,
    0.18009, 0.18999, 0.20233, 0.23105, 0.25483, 0.28925, 0.32365, 0.36541, 0.42188, 0.46608, 0.53319, 0.61501,
    0.7034, 0.78033, 0.84415, 0.91944, 1.000]
frac_CV_quick = [0.0, 0.04984, 0.07582, 0.12044, 0.16614, 0.21707, 0.26998, 0.32808, 0.39353, 0.46516, 0.52125, 0.58356,
    0.64798, 0.68845, 0.72277, 0.76565, 0.79399, 0.82459, 0.84589, 0.86732, 0.88078, 0.89399, 0.90867, 0.92053,
    0.93973, 0.95872, 0.96817, 0.98611, 1.0]
opening_quick_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.02184, 0.03256, 0.04575, 0.06221, 0.07459,
    0.0878, 0.10757, 0.12654, 0.14301, 0.16032, 0.18009, 0.18999, 0.20233, 0.23105, 0.25483, 0.28925,
    0.32365, 0.36541, 0.42188, 0.46608, 0.53319, 0.61501, 0.7034, 0.78033, 0.84415, 1.0, 1.0, 1.0, 1.0], 
    [-3.2479258181113327e-19, 0.037650956835178835, 0.054616164261637117, 0.12657862552611354,
    0.17115105822542115, 0.2075233903194021, 0.27084055195333684, 0.34208963001568016, 0.38730839943796663,
    0.4656002247400036, 0.5196995880922897, 0.5907033063634928, 0.6304293931726886, 0.6953064258075168,
    0.7382935002453699, 0.7631579537132379, 0.7997961180795559, 0.8262370617883222, 0.8471954722933543,
    0.873096858463145, 0.8776128736976467, 0.897647305294458, 0.9105672165523071, 0.9192771703370824,
    0.9377349743236904, 0.9603716623033031, 0.9688863605959851, 0.9980062718267431, 1.0, 0.0, 0.0, 0.0, 0.0],
    3])
Cv_char_quick_opening = lambda opening: float(splev(opening, opening_quick_tck))

opening_linear = [0., 1.0]
frac_CV_linear = [0, 1]
Cv_char_linear = lambda opening: interp(opening, opening_linear, frac_CV_linear)

# Equal opening valve data, spline fit, and interpolating function
opening_equal = [0.0, 0.05523, 0.09287, 0.15341, 0.18942, 0.22379, 0.25816, 0.29582, 0.33348, 0.34985, 0.3826, 0.45794,
    0.49235, 0.51365, 0.54479, 0.57594, 0.60218, 0.62843, 0.77628, 0.796, 0.83298, 0.86995, 0.90936, 0.95368, 1.00]
frac_CV_equal = [0.0, 0.00845, 0.01339, 0.01877, 0.02579, 0.0349, 0.04189, 0.05528, 0.07079, 0.07533, 0.09074, 0.13444,
        0.15833, 0.17353, 0.20159, 0.23388, 0.26819, 0.30461, 0.60113, 0.64588, 0.72583, 0.80788, 0.87519, 0.94999, 1.]
opening_equal_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.09287, 0.15341, 0.18942, 0.22379, 0.25816,
        0.29582, 0.33348, 0.34985, 0.3826, 0.45794, 0.49235, 0.51365, 0.54479, 0.57594, 0.60218, 0.62843,
        0.77628, 0.796, 0.83298, 0.86995, 0.90936, 1.0, 1.0, 1.0, 1.0], 
      [1.3522591106779132e-19, 0.004087873896711868, 0.014374150571122216, 0.016455484312674015, 0.024946845435605228,
        0.03592972456181881, 0.040710119644626126, 0.054518468768197687, 0.06976905178508139,
        0.07587146190282387, 0.0985485829020452, 0.1238160142641967, 0.15558350087382017, 0.17487348629353283,
        0.20157507610951217, 0.22995771158118564, 0.2683886931491415, 0.3574766835730407, 0.5027678906008036,
        0.659729970241158, 0.7233389559355903, 0.8155475382785987, 0.8983628328699896, 0.9871204658597236, 1.0,
        0.0, 0.0, 0.0, 0.0],
        3])
Cv_char_equal_percentage = lambda opening: float(splev(opening, opening_equal_tck))


def convert_flow_coefficient(flow_coefficient, old_scale, new_scale):
    '''Convert from one flow coefficient scale to another; supports the `Kv`
    `Cv`, and `Av` scales.
    
    Other scales are `Qn` and `Cg`, but clear definitions have yet to be
    found.
    
    Parameters
    ----------
    flow_coefficient : float
        Value of the flow coefficient to be converted, expressed in the 
        original scale.
    old_scale : str
        String specifying the original scale; one of 'Av', 'Cv', or 'Kv', [-]
    new_scale : str
        String specifying the new scale; one of 'Av', 'Cv', or 'Kv', [-]
    
    Returns
    -------
    converted_flow_coefficient : float 
        Flow coefficient converted to the specified scale.
    
    Notes
    -----
    `Qn` is a scale based on a flow of air in units of L/minute as air travels
    through a valve and loses one bar of pressure (initially 7 bar absolute,
    to 6 bar absolute). No consistent conversion factors have been found and 
    those from theory do not match what have been found. Some uses of `Qn` use
    its flow rate as in normal (STP reference conditions) flow rate of air;
    others use something like the 7 bar absolute condition.

    Examples
    --------
    >>> convert_flow_coefficient(10, 'Kv', 'Av')
    0.0002776532068951358
    '''
    # Convert from `old_scale` to Kv
    if old_scale == 'Cv':
        Kv = Cv_to_Kv(flow_coefficient)
    elif old_scale == 'Kv':
        Kv = flow_coefficient
    elif old_scale == 'Av':
        Cv = flow_coefficient/((rho0/psi)**0.5*gallon/minute)
        Kv = Cv_to_Kv(Cv)
    else:
        raise NotImplementedError("%s scale is unsupported" %old_scale)

    if new_scale == 'Cv':
        ans = Kv_to_Cv(Kv)
    elif new_scale == 'Kv':
        ans = Kv
    elif new_scale == 'Av':
        Cv = Kv_to_Cv(Kv)
        ans = Cv*((rho0/psi)**0.5*gallon/minute)
    else:
        raise NotImplementedError("%s scale is unsupported" %old_scale)

    return ans


# Third octave center frequency fi Hz
fis_l_2015 = [12.5, 16, 20, 25, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 
              160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 
              1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 
              8000.0, 10000.0, 12500.0, 16000.0, 20000.0]
#fis_l_2015_inv = [1.0/fi for fi in fis_l_2015]
#fis_l_2015_1_5 = [fi**1.5 for fi in fis_l_2015]
#fis_l_2015_n1_5 = [fi**-1.5 for fi in fis_l_2015]

fis_l_2015_inv, fis_l_2015_1_5, fis_l_2015_n1_5 = [], [], []
for fi in fis_l_2015:
    fi_rt_inv = fi**-0.5
    fis_l_2015_inv.append(fi_rt_inv*fi_rt_inv)
    fis_l_2015_1_5.append(fi*fi*fi_rt_inv)
    fis_l_2015_n1_5.append(fi_rt_inv*fi_rt_inv*fi_rt_inv)


fis_length = 33


# dLa(fi), dB
A_weights_l_2015 = [-63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, 
                    -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, 
                    -1.9, -0.8, 0.0, 0.6, 1, 1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1, 
                    -2.5, -4.3, -6.6, -9.3]


def control_valve_noise_l_2015(m, P1, P2, Psat, rho, c, Kv, d, Di, FL, Fd,
                               t_pipe, rho_pipe=7800.0, c_pipe=5000.0, 
                               rho_air=1.2, c_air=343.0, xFz=None, An=-4.6):
    r'''Calculates the sound made by a liquid flowing through a control valve
    according to the standard IEC 60534-8-4 (2015) [1]_.

    Parameters
    ----------
    m : float
        Mass flow rate of liquid through the control valve, [kg/s]
    P1 : float
        Inlet pressure of the fluid before valves and reducers [Pa]
    P2 : float
        Outlet pressure of the fluid after valves and reducers [Pa]
    Psat : float
        Saturation pressure of the fluid at inlet temperature [Pa]
    rho : float
        Density of the liquid at the inlet [kg/m^3]
    c : float
        Speed of sound of the liquid at the inlet conditions [m/s]
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]
    d : float
        Diameter of the valve [m]
    Di : float
        Internal diameter of the pipe before and after the valve [m]
    FL : float, optional
        Liquid pressure recovery factor of a control valve without attached 
        fittings (normally 0.8-0.9 at full open and decreasing as opened 
        further to below 0.5) [-]
    Fd : float, optional
        Valve style modifier [-]
    t_pipe : float
        Wall thickness of the pipe after the valve, [m]
    rho_pipe : float, optional
        Density of the pipe wall material at flowing conditions, [kg/m^3]
    c_pipe : float, optional
        Speed of sound of the pipe wall material at flowing conditions, [m/s]
    rho_air : float, optional
        Density of the air surrounding the valve and pipe wall, [kg/m^3]
    c_air : float, optional
        Speed of sound of the air surrounding the valve and pipe wall, [m/s]
    xFz : float, optional
        If specified, this value `xFz` is used instead of estimated; the 
        calculation is sensitive to this value, [-]
    An : float, optional
        Valve correction factor for acoustic efficiency

    Returns
    -------
    LpAe1m : float
        A weighted sound pressure level 1 m from the pipe wall, 1 m distance
        dowstream of the valve (at reference sound pressure level 2E-5), [dBA]
        
    Notes
    -----
    For formulas see [1]_. This takes on the order of 100 us to compute.
    This model can also tell if noise is being produced in a valve just due to
    turbulent flow, or cavitation. For values of `An`, see [1]_; it is
    normally -4.6 for globel valves, -4.3 for butterfly valves, and -4.0 for
    expanders.
    
    This model was checked against three examples in [1]_; they match to all
    given decimals.
    
    A formula is given in [1]_ for multihole trim valves to estimate `xFz`
    as well; this is not implemented here and `xFz` must be calculated by the
    user separately. The formula is
    
    .. math::
        x_{Fz} = \left(4.5 + 1650\frac{N_0d_H^2}{F_L}\right)^{-1/2}
        
    Where `N0` is the number of open channels and `dH` is the multihole trim
    hole diameter.

    Examples
    --------
    >>> control_valve_noise_l_2015(m=40, P1=1E6, P2=6.5E5, Psat=2.32E3, 
    ... rho=997, c=1400, Kv=77.848, d=0.1, Di=0.1071, FL=0.92, Fd=0.42, 
    ... t_pipe=0.0036, rho_pipe=7800.0, c_pipe=5000.0,rho_air=1.293, 
    ... c_air=343.0, An=-4.6)
    81.58200097996539

    References
    ----------
    .. [1] IEC 60534-8-4 : Industrial-Process Control Valves - Part 8-4: Noise 
       Considerations - Prediction of Noise Generated by Hydrodynamic Flow.
       (2015)
    '''
    # Convert Kv to Cv as C
    N34 = 1.17 # for Cv - conversion constant but not to many decimals
    N14 = 0.0046
    
    C = Kv_to_Cv(Kv)
    xF = (P1-P2)/(P1-Psat)
    dPc = min(P1-P2, FL*FL*(P1 - Psat))
    
    if xFz is None:
        xFz = 0.9*(1.0 + 3.0*Fd*(C/(N34*FL))**0.5)**-0.5
    xFzp1 = xFz*(6E5/P1)**0.125
    
    Dj = N14*Fd*(C*FL)**0.5
    
    Uvc = 1.0/FL*(2.0*dPc/rho)**0.5
    Wm = 0.5*m*Uvc*Uvc*FL*FL
    cavitating = False if xF <= xFzp1 else True
    
    eta_turb = 10.0**An*Uvc/c
    
    if cavitating:
    	eta_cav = 0.32*eta_turb*((P1 - P2)/(dPc*xFzp1))**0.5*exp(5.0*xFzp1)*((1.0 
                 - xFzp1)/(1.0 - xF))**0.5*(xF/xFzp1)**5*(xF - xFzp1)**1.5
    	Wa = (eta_turb+eta_cav)*Wm
    else:
    	Wa = eta_turb*Wm
    
    Lpi = 10.0*log10(3.2E9*Wa*rho*c/(Di*Di))
    Stp = 0.036*FL*FL*C*Fd**0.75/(N34*xFzp1**1.5*d*d)*(1.0/(P1 - Psat))**0.57
    f_p_turb = Stp*Uvc/Dj
    
    if cavitating:
        f_p_cav = 6.0*f_p_turb*((1.0 - xF)/(1.0 - xFzp1))**2*(xFzp1/xF)**2.5
        f_p_cav_inv = 1.0/f_p_cav
        f_p_cav_inv_1_5 = f_p_cav_inv**1.5
        f_p_cav_inv_1_5_1_4 = 0.25*f_p_cav_inv_1_5
        f_p_cav_1_5 = 1.0/f_p_cav_inv_1_5
        eta_denom = 1.0/(eta_turb + eta_cav)
        t1 = eta_turb*eta_denom
        t2 = eta_cav*eta_denom
        log10_t1 = log10(t1)
        

    fr = c_pipe/(pi*Di)    
    fr_inv = 1.0/fr
    TL_fr = -10.0 - 10.0*log10(c_pipe*rho_pipe*t_pipe/(c_air*rho_air*Di))
    
    t3 = - 10.0*log10((Di + 2.0*t_pipe + 2.0)/(Di + 2.0*t_pipe))

#    F_cavs = []
#    F_turbs = []
#    LPis = []
#    TL_fis = []
#    L_pe1m_fis = []
    LpAe1m_sum = 0.0
    
    f_p_turb_inv = 1.0/f_p_turb
    
    fr_inv_1_5 = fr_inv**1.5
    
    
    for i in range(fis_length):
#    for fi, fi_inv, fi_1_5, fi_1_5_inv, A in zip(fis_l_2015, fis_l_2015_inv, fis_l_2015_1_5, fis_l_2015_n1_5, A_weights_l_2015):
#        fi_inv = 1.0/fi
        fi_turb_ratio = fis_l_2015[i]*f_p_turb_inv
#        fi_turb_ratio = fi*f_p_turb_inv
        F_turb = -8.0 - 10.0*log10(0.25*fi_turb_ratio*fi_turb_ratio*fi_turb_ratio
                                   + fis_l_2015_inv[i]*f_p_turb) 
#        F_turbs.append(F_turb)
        if cavitating:
#            fi_cav_ratio = fi_1_5*f_p_cav_inv_1_5#   (fi*f_p_cav_inv)**1.5
            F_cav = -9.0 - 10.0*log10(f_p_cav_inv_1_5_1_4*fis_l_2015_1_5[i] + fis_l_2015_n1_5[i]*f_p_cav_1_5) # 1.0/fi_cav_ratio, fi_1_5_inv*f_p_cav_1_5
            LPif = (Lpi + 10.0*log10(t1*10.0**(0.1*F_turb)  + t2*10.0**(0.1*F_cav)))
            # Shoule be able to save 1 power in the above function somehow, combine the tow terms in exponent
        else:
            LPif = Lpi + F_turb
#        LPis.append(LPif)
        TL_fi = TL_fr - 20.0*log10(fr*fis_l_2015_inv[i] + fis_l_2015_1_5[i]*fr_inv_1_5) #  (fi*fr_inv)**1.5
#        TL_fis.append(TL_fi)
        L_pe1m_fi = LPif + TL_fi + t3
#        L_pe1m_fis.append(L_pe1m_fi)
        LpAe1m_sum += 10.0**(0.1*(L_pe1m_fi + A_weights_l_2015[i]))
    LpAe1m = 10.0*log10(LpAe1m_sum)
    return LpAe1m


def control_valve_noise_g_2011(m, P1, P2, T1, rho, gamma, MW, Kv, 
                               d, Di, t_pipe, Fd, FL, FLP=None, FP=None,
                               rho_pipe=7800.0, c_pipe=5000.0, 
                               P_air=101325.0, rho_air=1.2, c_air=343.0, 
                               An=-3.8, Stp=0.2, T2=None, beta=0.93):
    r'''Calculates the sound made by a gas flowing through a control valve
    according to the standard IEC 60534-8-3 (2011) [1]_.

    Parameters
    ----------
    m : float
        Mass flow rate of gas through the control valve, [kg/s]
    P1 : float
        Inlet pressure of the gas before valves and reducers [Pa]
    P2 : float
        Outlet pressure of the gas after valves and reducers [Pa]
    T1 : float
        Inlet gas temperature, [K]
    rho : float
        Density of the gas at the inlet [kg/m^3]
    gamma : float
        Specific heat capacity ratio [-]
    MW : float
        Molecular weight of the gas [g/mol]
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop  
        of 1 bar) [m^3/hr]
    d : float
        Diameter of the valve [m]
    Di : float
        Internal diameter of the pipe before and after the valve [m]
    t_pipe : float
        Wall thickness of the pipe after the valve, [m]
    Fd : float
        Valve style modifier (0.1 to 1; varies tremendously depending on the
        type of valve and position; do not use the default at all!) [-]
    FL : float
        Liquid pressure recovery factor of a control valve without attached 
        fittings (normally 0.8-0.9 at full open and decreasing as opened 
        further to below 0.5; use default very cautiously!) [-]
    FLP : float, optional
        Combined liquid pressure recovery factor with piping geometry factor,
        for a control valve with attached fittings [-]
    FP : float, optional
        Piping geometry factor [-]
    rho_pipe : float, optional
        Density of the pipe wall material at flowing conditions, [kg/m^3]
    c_pipe : float, optional
        Speed of sound of the pipe wall material at flowing conditions, [m/s]
    P_air : float, optional
        Pressure of the air surrounding the valve and pipe wall, [Pa]
    rho_air : float, optional
        Density of the air surrounding the valve and pipe wall, [kg/m^3]
    c_air : float, optional
        Speed of sound of the air surrounding the valve and pipe wall, [m/s]
    An : float, optional
        Valve correction factor for acoustic efficiency
    Stp : float, optional
        Strouhal number at the peak `fp`; between 0.1 and 0.3 typically, [-]
    T2 : float, optional
        Outlet gas temperature; assumed `T1` if not provided (a PH flash 
        should be used to obtain this if possible), [K]
    beta : float, optional
        Valve outlet / expander inlet contraction coefficient, [-]

    Returns
    -------
    LpAe1m : float
        A weighted sound pressure level 1 m from the pipe wall, 1 m distance
        dowstream of the valve (at reference sound pressure level 2E-5), [dBA]
        
    Notes
    -----
    For formulas see [1]_. This takes on the order of 100 us to compute.
    For values of `An`, see [1]_.
    
    This model was checked against six examples in [1]_; they match to all
    given decimals.
    
    Several additional formulas are given for multihole trim valves,
    control valves with two or more fixed area stages, and multipath,
    multistage trim valves. 
    
    Examples
    --------
    >>> control_valve_noise_g_2011(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3, 
    ... gamma=1.22, MW=19.8, Kv=77.85,  d=0.1, Di=0.2031, FL=None, FLP=0.792, 
    ... FP=0.98, Fd=0.296, t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0, 
    ... rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    91.67702674629604

    References
    ----------
    .. [1] IEC 60534-8-3 : Industrial-Process Control Valves - Part 8-3: Noise 
       Considerations - Control Valve Aerodynamic Noise Prediction Method."
    '''
    k = gamma # alias
    C = Kv_to_Cv(Kv)
    N14 = 4.6E-3
    N16 = 4.89E4
    fs = 1.0 # structural loss factor reference frequency, Hz
    P_air_std = 101325.0
    if T2 is None:
        T2 = T1
    x = (P1 - P2)/P1
    

    # FLP/FP when fittings attached
    FL_term = FLP/FP if FP is not None else FL

    P_vc = P1*(1.0 - x/FL_term**2)

    x_vcc = 1.0 - (2.0/(k + 1.0))**(k/(k - 1.0)) # mostly matches
    xc = FL_term**2*x_vcc
    alpha = (1.0 - x_vcc)/(1.0 - xc)
    xB = 1.0 - 1.0/alpha*(1.0/k)**((k/(k - 1.0)))
    xCE = 1.0 - 1.0/(22.0*alpha)

    # Regime determination check - should be ordered or won't work
    assert xc < x_vcc
    assert x_vcc < xB
    assert xB < xCE
    regime = None
    if x <= xc:
        regime = 1
    elif xc < x <= x_vcc:
        regime = 2
    elif x_vcc < x <= xB:
        regime = 3
    elif xB < x <= xCE:
        regime = 4
    else:
        regime = 5
#     print('regime', regime)

    Dj = N14*Fd*(C*(FL_term))**0.5

    Mj5 = (2.0/(k - 1.0)*( 22.0**((k-1.0)/k) - 1.0  ))**0.5
    if regime == 1:
        Mvc = ( (2.0/(k-1.0)) *((1.0 - x/FL_term**2)**((1.0 - k)/k)   - 1.0)   )**0.5 # Not match
    elif regime in (2, 3, 4):
        Mj = ( (2.0/(k-1.0))*((1.0/(alpha*(1.0-x)))**((k - 1.0)/k) - 1.0)   )**0.5 # Not match
        Mj = min(Mj, Mj5)
    elif regime == 5:
        pass

    if regime == 1:
        Tvc = T1*(1.0 - x/(FL_term)**2)**((k - 1.0)/k)
        cvc = (k*P1/rho*(1 - x/(FL_term)**2)**((k-1.0)/k))**0.5
        Wm = 0.5*m*(Mvc*cvc)**2
    else:
        Tvcc = 2.0*T1/(k + 1.0)
        cvcc = (2.0*k*P1/(k+1.0)/rho)**0.5
        Wm = 0.5*m*cvcc*cvcc
#     print('Wm', Wm)

    if regime == 1:
        fp = Stp*Mvc*cvc/Dj
    elif regime in (2, 3):
        fp = Stp*Mj*cvcc/Dj
    elif regime == 4:
        fp = 1.4*Stp*cvcc/Dj/(Mj*Mj - 1.0)**0.5
    elif regime == 5:
        fp = 1.4*Stp*cvcc/Dj/(Mj5*Mj5 - 1.0)**0.5
#     print('fp', fp)

    if regime == 1:
        eta = 10.0**An*FL_term**2*(Mvc)**3
    elif regime == 2:
        eta = 10.0**An*x/x_vcc*Mj**(6.6*FL_term*FL_term)
    elif regime == 3:
        eta = 10.0**An*Mj**(6.6*FL_term*FL_term)
    elif regime == 4:
        eta = 0.5*10.0**An*Mj*Mj*(2.0**0.5)**(6.6*FL_term*FL_term)
    elif regime == 5:
        eta = 0.5*10.0**An*Mj5*Mj5*(2.0**0.5)**(6.6*FL_term*FL_term)
#     print('eta', eta)

    Wa = eta*Wm

    rho2 = rho*(P2/P1)
    # Speed of sound
    c2 = (k*R*T2/(MW/1000.))**0.5

    Mo = 4.0*m/(pi*d*d*rho2*c2)

    M2 = 4.0*m/(pi*Di*Di*rho2*c2)
#     print('M2', M2)

    Lg = 16.0*log10(1.0/(1.0 - min(M2, 0.3))) # dB
    
    if M2 > 0.3:
        Up = 4.0*m/(pi*rho2*Di*Di)
        UR = Up*Di*Di/(beta*d*d)
        WmR = 0.5*m*UR*UR*( (1.0 - d*d/(Di*Di))**2 + 0.2)
        fpR = Stp*UR/d
        MR = UR/c2
        # Value listed in appendix here is wrong, "based on another
        # earlier standard. Calculation thereon is wrong". Assumed
        # correct, matches spreadsheet to three decimals.
        eta_R = 10**An*MR**3
        WaR = eta_R*WmR
        L_piR = 10.0*log10((3.2E9)*WaR*rho2*c2/(Di*Di)) + Lg
#         print('Up', Up)
#         print('UR', UR)
#         print('WmR', WmR)
#         print('fpR', fpR)
#         print('MR', MR)
#         print('eta_R', eta_R, eta_R/8.8E-4)
#         print('WaR', WaR)
#         print('L_piR', L_piR)

    L_pi = 10.0*log10((3.2E9)*Wa*rho2*c2/(Di*Di)) + Lg
#     print('L_pi', L_pi)

    fr = c_pipe/(pi*Di) 
    fo = 0.25*fr*(c2/c_air)
    fg = 3**0.5*c_air**2/(pi*t_pipe*c_pipe)

    if d > 0.15:
        dTL = 0.0
    elif 0.05 <= d <= 0.15:
        dTL = -16660.0*d**3 + 6370.0*d**2 - 813.0*d + 35.8
    else:
        dTL = 9.0
#     print(dTL, 'dTL')

    P_air_ratio = P_air/P_air_std

    LpAe1m_sum = 0.0
    LPis = []
    LPIRs = []
    L_pe1m_fis = []
    for fi, A_weight in zip(fis_l_2015, A_weights_l_2015):
        # This gets adjusted when Ma > 0.3
        fi_turb_ratio = fi/fp

        t1 = 1.0 + (0.5*fi_turb_ratio)**2.5
        t2 = 1.0 + (0.5/fi_turb_ratio)**1.7

        # Formula forgot to use log10, but log10 is needed for the numbers
        Lpif = L_pi - 8.0 - 10.0*log10(t1*t2)
#         print(Lpif, 'Lpif')
        LPis.append(Lpif)
    
        if M2 > 0.3:
            fiR_turb_ratio = fi/fpR
            t1 = 1.0 + (0.5*fiR_turb_ratio)**2.5
            t2 = 1.0 + (0.5/fiR_turb_ratio)**1.7
            # Again, log10 is missing
            LpiRf = L_piR - 8.0 - 10.0*log10(t1*t2)
            LPIRs.append(LpiRf)
            
            LpiSf = 10.0*log10( 10**(0.1*Lpif) + 10.0**(0.1*LpiRf) )
            
        if fi < fo:
            Gx = (fo/fr)**(2.0/3.0)*(fi/fo)**4.0
            if fo < fg:
                Gy = (fo/fg)
            else:
                Gy = 1.0
        else:
            if fi < fr:
                Gx = (fi/fr)**0.5
            else:
                Gx = 1.0
            if fi < fg:
                Gy = fi/fg
            else:
                Gy = 1.0

        eta_s = (0.01/fi)**0.5
#         print('eta_s', eta_s)
        # up to eta_s is good

        den = (rho2*c2 + 2.0*pi*t_pipe*fi*rho_pipe*eta_s)/(415.0*Gy) + 1.0
        TL_fi = 10.0*log10(8.25E-7*(c2/(t_pipe*fi))**2*Gx/den*P_air_ratio) - dTL

        # Formula forgot to use log10, but log10 is needed for the numbers
        if M2 > 0.3:
            term = LpiSf
        else:
            term = Lpif
        
        L_pe1m_fi = term + TL_fi - 10.0*log10((Di + 2.0*t_pipe + 2.0)/(Di + 2.0*t_pipe))
        L_pe1m_fis.append(L_pe1m_fi)
#         print(L_pe1m_fi)

        LpAe1m_sum += 10.0**(0.1*(L_pe1m_fi + A_weight))
    LpAe1m = 10.0*log10(LpAe1m_sum)
    return LpAe1m

