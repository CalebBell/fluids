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
from math import log10
from scipy.constants import R

__all__ = ['cavitation_index', 'size_control_valve_l', 'size_control_valve_g']

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
    sigma = (P1 - Psat)/(P1 - P2)
    return sigma


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
    FF = 0.96 - 0.28*(Psat/Pc)**0.5
    return FF


def is_choked_turbulent_l(dP, P1, Psat, FF, FL=None, FLP=None, FP=None):
    r'''Calculates if a liquid flow in IEC 60534 calculations is critical or
    not, for use in IEC 60534 liquid valve sizing calculations.
    Either FL may be provided or FLP and FP, depending on the calculation
    process.

    .. math::
        \Delta P > F_L^2(P_1 - F_F P_{sat})

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
        Liquid pressure recovery factor of a control valve without attached fittings []
    FLP : float, optional
        Combined liquid pressure recovery factor with piping geometry factor,
        for a control valve with attached fittings []
    FP : float, optional
        Piping geometry factor []

    Returns
    -------
    choked : bool
        Whether or not the flow is choked [-]

    Examples
    --------
    >>> is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.9)
    False
    >>> is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.6)
    True

    References
    ----------
    .. [1] IEC 60534-2-1 / ISA-75.01.01-2007
    '''
    if FLP and FP:
        choked = dP >= (FLP/FP)**2*(P1-FF*Psat)
    elif FL:
        choked = dP >= FL**2*(P1-FF*Psat)
    else:
        raise Exception('Either (FLP and FP) or FL is needed')
    return choked


def is_choked_turbulent_g(x, Fgamma, xT=None, xTP=None):
    r'''Calculates if a gas flow in IEC 60534 calculations is critical or
    not, for use in IEC 60534 gas valve sizing calculations.
    Either xT or xTP must be provided, depending on the calculation process.

    .. math::
        x \ge F_\gamma x_T

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
        choked = x >= Fgamma*xT
    elif xTP:
        choked = x >= Fgamma*xTP
    else:
        raise Exception('Either xT or xTP is needed')
    return choked


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
        Liquid pressure recovery factor of a control valve without attached fittings []
    Fd : float
        Valve style modifier []
    C : float
        Kv flow coefficient [m^3/hr at a dP of 1 bar]

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
    Rev = N4*Fd*Q/nu/(C*FL)**0.5*(FL**2*C**2/(N2*D1**4) + 1)**0.25
    return Rev


def loss_coefficient_piping(d, D1=None, D2=None):
    r'''Calculates the sum of loss coefficients from possible
    inlet/outlet reducers/expanders around a control valve according to
    IEC 60534 calculations.

    .. math::
        \Sigma \xi = \xi_1 + \xi_2 + \xi_{B1} - \xi_{B2}

        \xi_1 = 0.5\left[1 -\left(\frac{d}{D_1}\right)^2\right]^2

        \xi_2 = 1.0\left[1 -\left(\frac{d}{D_2}\right)^2\right]^2

        \xi_{B1} = 1 - \left(\frac{d}{D_1}\right)^4

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
    loss = 0
    if D1:
        loss += 1 - (d/D1)**4 # Inlet flow energy
        loss += 0.5*(1 - (d/D1)**2)**2 # Inlet reducer
    if D2:
        loss += 1.0*(1 - (d/D2)**2)**2 # Outlet reducer (expander)
        loss -= 1 - (d/D2)**4 # Outlet flow energy
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

        F_{R,2} = \min(\frac{0.026}{F_L}\sqrt{n_1 Re_v},\; 1)

        n_1 = \frac{N_2}{\left(\frac{C}{d^2}\right)^2}

        F_R = F_{R,2} \text{ if Rev < 10 else } \min(F_{R,1a}, F_{R,2})

    Otherwise :

    .. math::
        F_{R,3a} = 1 + \left(\frac{0.33F_L^{0.5}}{n_2^{0.25}}\right)\log_{10}
        \left(\frac{Re_v}{10000}\right)

        F_{R,4} = \frac{0.026}{F_L}\sqrt{n_2 Re_v}

        n_2 = 1 + N_{32}\left(\frac{C}{d}\right)^{2/3}

        F_R = F_{R,4} \text{ if Rev < 10 else } \min(F_{R,3a}, F_{R,4})

    Parameters
    ----------
    FL : float
        Liquid pressure recovery factor of a control valve without attached
        fittings []
    C : float
        Kv flow coefficient [m^3/hr at a dP of 1 bar]
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


def size_control_valve_l(rho, Psat, Pc, mu, P1, P2, Q, D1, D2, d, FL, Fd):
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
    D1 : float
        Diameter of the pipe before the valve [m]
    D2 : float
        Diameter of the pipe after the valve [m]
    d : float
        Diameter of the valve [m]
    FL : float
        Liquid pressure recovery factor of a control valve without attached fittings []
    Fd : float
        Valve style modifier []

    Returns
    -------
    C : float
        Kv flow coefficient [m^3/hr at a dP of 1 bar]

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
    # Pa to kPa, according to constants in standard
    P1, P2, Psat, Pc = P1/1000., P2/1000., Psat/1000., Pc/1000.
    # m to mm, according to constants in standard
    D1, D2, d = D1*1000., D2*1000., d*1000.
    Q = Q*3600. # m^3/s to m^3/hr, according to constants in standard
    nu = mu/rho # kinematic viscosity used in standard

    dP = P1 - P2
    FF = FF_critical_pressure_ratio_l(Psat=Psat, Pc=Pc)
    choked = is_choked_turbulent_l(dP=dP, P1=P1, Psat=Psat, FF=FF, FL=FL)
    if choked:
        # Choked flow, equation 3
        C = Q/N1/FL*(rho/rho0/(P1-FF*Psat))**0.5
    else:
        # non-choked flow, eq 1
        C = Q/N1*(rho/rho0/dP)**0.5
    Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=C)
    if Rev > 10000 and (D1 != d or D2 != d):
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
                # Non-Choked flow with piping, equation 4
                C = Q/N1/FP*(rho/rho0/dP)**0.5
            if Ci/C < 0.99:
                C = iterate_piping_turbulent(C)
            return C

        C = iterate_piping_turbulent(Ci)
    elif Rev <= 10000:
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
            return Ci
        C = iterate_piping_laminar(C)
    return C


#print [size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.09, d=0.08, FL=0.9, Fd=0.46)]
#print [size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.1, FL=0.6, Fd=0.98)]
#print [size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.95, FL=0.6, Fd=0.98)]
#print [size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-2, P1=680E3, P2=220E3, Q=0.001, D1=0.01, D2=0.01, d=0.01, FL=0.6, Fd=0.98)]



def size_control_valve_g(T, MW, mu, gamma, Z, P1, P2, Q, D1, D2, d, FL, Fd, xT):
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
    D1 : float
        Diameter of the pipe before the valve [m]
    D2 : float
        Diameter of the pipe after the valve [m]
    d : float
        Diameter of the valve [m]
    FL : float
        Liquid pressure recovery factor of a control valve without attached
        fittings []
    Fd : float
        Valve style modifier []
    xT : float
        Pressure difference ratio factor of a valve without fittings at choked
        flow [-]

    Returns
    -------
    C : float
        Kv flow coefficient [m^3/hr at a dP of 1 bar]

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
    # Pa to kPa, according to constants in standard
    P1, P2 = P1/1000., P2/1000.
    # m to mm, according to constants in standard
    D1, D2, d = D1*1000., D2*1000., d*1000.
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
    if choked:
        # Choked, and flow coefficient from eq 14a
        C = Q/(N9*P1*Y)*(MW*T*Z/xT/Fgamma)**0.5
    else:
        # Non-choked, and flow coefficient from eq 8a
        C = Q/(N9*P1*Y)*(MW*T*Z/x)**0.5

    Rev = Reynolds_valve(nu=nu, Q=Q, D1=D1, FL=FL, Fd=Fd, C=C)
    if Rev > 10000 and (D1 != d or D2 != d):
        # gas, using xTP and FLP
        FP = 1
        def iterate_piping_coef(Ci):
            loss = loss_coefficient_piping(d, D1, D2)
            FP = (1 + loss/N2*(Ci/d**2)**2)**-0.5
            loss_upstream = loss_coefficient_piping(d, D1)
            xTP = xT/FP**2/(1 + xT*loss_upstream/N5*(Ci/d**2)**2)
            choked = is_choked_turbulent_g(x, Fgamma, xTP=xTP)
            if choked:
                # Choked flow with piping, equation 17a
                C = Q/(N9*FP*P1*Y)*(MW*T*Z/xTP/Fgamma)**0.5
            else:
                # Non-choked flow with piping, equation 11a
                C = Q/(N9*FP*P1*Y)*(MW*T*Z/x)**0.5
            if Ci/C < 0.99:
                C = iterate_piping_coef(C)
            return C
        C = iterate_piping_coef(C)
    elif Rev <= 10000:
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
            return Ci
        C = iterate_piping_laminar(C)
    return C

#print [size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30, Z=0.988, P1=680E3,
#             P2=30E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)]
#print [size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5,
#           P2=2.7E5, Q=0.1/3600., D1=0.015, D2=0.015, d=0.001, FL=0.98, Fd=0.07, xT=0.8)]
