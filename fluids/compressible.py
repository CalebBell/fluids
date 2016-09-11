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
__all__ = ['Panhandle_A', 'Panhandle_B', 'Weymouth', 'Spitzglass_high', 
           'Spitzglass_low', 'Oliphant', 'Fritzsche',
           'T_critical_flow', 'P_critical_flow', 'is_critical_flow',
           'stagnation_energy', 'P_stagnation', 'T_stagnation',
           'T_stagnation_ideal']

from scipy.optimize import newton 

def T_critical_flow(T, k):
    r'''Calculates critical flow temperature `Tcf` for a fluid with the
    given isentropic coefficient. `Tcf` is in a flow (with Ma=1) whose
    stagnation conditions are known. Normally used with converging/diverging
    nozzles.

    .. math::
        \frac{T^*}{T_0} = \frac{2}{k+1}

    Parameters
    ----------
    T : float
        Stagnation temperature of a fluid with Ma=1 [K]
    k : float
        Isentropic coefficient []

    Returns
    -------
    Tcf : float
        Critical flow temperature at Ma=1 [K]

    Notes
    -----
    Assumes isentropic flow.

    Examples
    --------
    Example 12.4 in [1]_:

    >>> T_critical_flow(473, 1.289)
    413.2809086937528

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Tcf = T*2/(k+1.)
    return Tcf


def P_critical_flow(P, k):
    r'''Calculates critical flow pressure `Pcf` for a fluid with the
    given isentropic coefficient. `Pcf` is in a flow (with Ma=1) whose
    stagnation conditions are known. Normally used with converging/diverging
    nozzles.

    .. math::
        \frac{P^*}{P_0} = \left(\frac{2}{k+1}\right)^{k/(k-1)}

    Parameters
    ----------
    P : float
        Stagnation pressure of a fluid with Ma=1 [Pa]
    k : float
        Isentropic coefficient []

    Returns
    -------
    Pcf : float
        Critical flow pressure at Ma=1 [Pa]

    Notes
    -----
    Assumes isentropic flow.

    Examples
    --------
    Example 12.4 in [1]_:

    >>> P_critical_flow(1400000, 1.289)
    766812.9022792266

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Pcf = P*(2/(k+1.))**(k/(k-1))
    return Pcf


def is_critical_flow(P1, P2, k):
    r'''Determines if a flow of a fluid driven by pressure gradient
    P1 - P2 is critical, for a fluid with the given isentropic coefficient.
    This function calculates critical flow pressure, and checks if this is
    larger than P2. If so, the flow is critical and choked.

    Parameters
    ----------
    P1 : float
        Higher, source pressure [Pa]
    P2 : float
        Lower, downstream pressure [Pa]
    k : float
        Isentropic coefficient []

    Returns
    -------
    flowtype : bool
        True if the flow is choked; otherwise False

    Notes
    -----
    Assumes isentropic flow. Uses P_critical_flow function.

    Examples
    --------
    Examples 1-2 from API 520.

    >>> is_critical_flow(670E3, 532E3, 1.11)
    False
    >>> is_critical_flow(670E3, 101E3, 1.11)
    True

    References
    ----------
    .. [1] API. 2014. API 520 - Part 1 Sizing, Selection, and Installation of
       Pressure-relieving Devices, Part I - Sizing and Selection, 9E.
    '''
    Pcf = P_critical_flow(P1, k)
    flowtype = Pcf > P2
    return flowtype


def stagnation_energy(V):
    r'''Calculates the increase in enthalpy `dH` which is provided by a fluid's
    velocity `V`.

    .. math::
        \Delta H = \frac{V^2}{2}

    Parameters
    ----------
    V : float
        Velocity [m/s]

    Returns
    -------
    dH : float
        Incease in enthalpy [J/kg]

    Notes
    -----
    The units work out. This term is pretty small, but not trivial.

    Examples
    --------
    >>> stagnation_energy(125)
    7812.5

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    dH = V**2/2.
    return dH


def P_stagnation(P, T, Tst, k):
    r'''Calculates stagnation flow pressure `Pst` for a fluid with the
    given isentropic coefficient and specified stagnation temperature and
    normal temperature. Normally used with converging/diverging nozzles.

    .. math::
        \frac{P_0}{P}=\left(\frac{T_0}{T}\right)^{\frac{k}{k-1}}

    Parameters
    ----------
    P : float
        Normal pressure of a fluid [Pa]
    T : float
        Normal temperature of a fluid [K]
    Tst : float
        Stagnation temperature of a fluid moving at a certain velocity [K]
    k : float
        Isentropic coefficient []

    Returns
    -------
    Pst : float
        Stagnation pressure of a fluid moving at a certain velocity [Pa]

    Notes
    -----
    Assumes isentropic flow.

    Examples
    --------
    Example 12-1 in [1]_.

    >>> P_stagnation(54050., 255.7, 286.8, 1.4)
    80772.80495900588

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Pst = P*(Tst/T)**(k/(k-1))
    return Pst


def T_stagnation(T, P, Pst, k):
    r'''Calculates stagnation flow temperature `Tst` for a fluid with the
    given isentropic coefficient and specified stagnation pressure and
    normal pressure. Normally used with converging/diverging nozzles.

    .. math::
        T=T_0\left(\frac{P}{P_0}\right)^{\frac{k-1}{k}}

    Parameters
    ----------
    T : float
        Normal temperature of a fluid [K]
    P : float
        Normal pressure of a fluid [Pa]
    Pst : float
        Stagnation pressure of a fluid moving at a certain velocity [Pa]
    k : float
        Isentropic coefficient []

    Returns
    -------
    Tst : float
        Stagnation temperature of a fluid moving at a certain velocity [K]

    Notes
    -----
    Assumes isentropic flow.

    Examples
    --------
    Example 12-1 in [1]_.

    >>> T_stagnation(286.8, 54050, 54050*8, 1.4)
    519.5230938217768

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Tst = T*(Pst/P)**((k - 1)/k)
    return Tst


def T_stagnation_ideal(T, V, Cp):
    r'''Calculates the ideal stagnation temperature `Tst` calculated assuming
    the fluid has a constant heat capacity `Cp` and with a specified
    velocity `V` and tempeature `T`.

    .. math::
        T^* = T + \frac{V^2}{2C_p}

    Parameters
    ----------
    T : float
        Tempearture [K]
    V : float
        Velocity [m/s]
    Cp : float
        Ideal heat capacity [J/kg/K]

    Returns
    -------
    Tst : float
        Stagnation temperature [J/kg]

    Examples
    --------
    Example 12-1 in [1]_.

    >>> T_stagnation_ideal(255.7, 250, 1005.)
    286.79452736318405

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Tst = T + V**2/2./Cp
    return Tst


def Panhandle_A(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=0.92):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Panhandle A formula. Can calculate any of the following, 
    given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Panhandle A equation. Here, a new form is developed with all units
    in base SI, based on the work of [1]_.
    
    .. math::
        Q = 158.02053 E \left(\frac{T_s}{P_s}\right)^{1.0788}\left[\frac{P_1^2
        -P_2^2}{L \cdot {SG}^{0.8539} T_{avg}Z_{avg}}\right]^{0.5394}D^{2.6182}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----
    [1]_'s original constant was 4.5965E-3, and it has units of km (length), 
    kPa, mm (diameter), and flowrate in m^3/day.
    
    The form in [2]_ has the same exponents as used here, units of mm 
    (diameter), kPa, km (length), and flow in m^3/hour; its leading constant is
    1.9152E-4.  
    
    The GPSA [3]_ has a leading constant of 0.191, a bracketed power of 0.5392,
    a specific gravity power of 0.853, and otherwise the same constants.
    It is in units of mm (diameter) and kPa and m^3/day; length is stated to be
    in km, but according to the errata is in m.
    
    [4]_ has a leading constant of 1.198E7, a specific gravity of power of 0.8541,
    and a power of diameter which is under the root of 4.854 and is otherwise
    the same. It has units of kPa and m^3/day, but is otherwise in base SI 
    units.
    
    [5]_ has a leading constant of 99.5211, but its reference correction has no 
    exponent; other exponents are the same as here. It is entirely in base SI 
    units.
    
    [6]_ has pressures in psi, diameter in inches, length in miles, Q in 
    ft^3/day, T in degrees Rankine, and a constant of 435.87.
    Its reference condition power is 1.07881, and it has a specific gravity
    correction outside any other term with a power of 0.4604.

    Examples
    --------
    >>> Panhandle_A(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    42.56082051195928

    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    .. [2] Co, Crane. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [3] GPSA. GPSA Engineering Data Book. 13th edition. Gas Processors
       Suppliers Association, Tulsa, OK, 2012.
    .. [4] Campbell, John M. Gas Conditioning and Processing, Vol. 2: The 
       Equipment Modules. 7th edition. Campbell Petroleum Series, 1992.
    .. [5] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    .. [6] Ikoku, Chi U. Natural Gas Production Engineering. Malabar, Fla: 
       Krieger Pub Co, 1991.
    '''
    c1 = 1.0788
    c2 = 0.8539
    c3 = 0.5394
    c4 = 2.6182
    c5 = 158.0205328706957220332831680508433862787 # 45965*10**(591/1250)/864
    if Q is None and (None not in [L, D, P1, P2]):
        return c5*E*(Ts/Ps)**c1*((P1**2 - P2**2)/(L*SG**c2*Tavg*Zavg))**c3*D**c4
    elif D is None and (None not in [L, Q, P1, P2]):
        return (Q*(Ts/Ps)**(-c1)*(SG**(-c2)*(P1**2 - P2**2)/(L*Tavg*Zavg))**(-c3)/(E*c5))**(1./c4)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return (L*SG**c2*Tavg*Zavg*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(1./c3) + P2**2)**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return (-L*SG**c2*Tavg*Zavg*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(1./c3) + P1**2)**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return SG**(-c2)*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(-1./c3)*(P1**2 - P2**2)/(Tavg*Zavg)
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Panhandle_B(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=0.92):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Panhandle B formula. Can calculate any of the following, 
    given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Panhandle B equation. Here, a new form is developed with all units
    in base SI, based on the work of [1]_.
    
    .. math::
        Q = 152.88116 E \left(\frac{T_s}{P_s}\right)^{1.02}\left[\frac{P_1^2
        -P_2^2}{L \cdot {SG}^{0.961} T_{avg}Z_{avg}}\right]^{0.51}D^{2.53}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----
    [1]_'s original constant was 1.002E-2, and it has units of km (length), 
    kPa, mm (diameter), and flowrate in m^3/day.
    
    The form in [2]_ has the same exponents as used here, units of mm 
    (diameter), kPa, km (length), and flow in m^3/hour; its leading constant is
    4.1749E-4.  
    
    The GPSA [3]_ has a leading constant of 0.339, and otherwise the same constants.
    It is in units of mm (diameter) and kPa and m^3/day; length is stated to be
    in km, but according to the errata is in m.
    
    [4]_ has a leading constant of 1.264E7, a diameter power of 4.961 which is
    also under the 0.51 power, and is otherwise the same. It has units of kPa 
    and m^3/day, but is otherwise in base SI units.
    
    [5]_ has a leading constant of 135.8699, but its reference correction has  
    no exponent and its specific gravity has a power of 0.9608; the other 
    exponents are the same as here. It is entirely in base SI units.
    
    [6]_ has pressures in psi, diameter in inches, length in miles, Q in 
    ft^3/day, T in degrees Rankine, and a constant of 737 with the exponents 
    the same as here.

    Examples
    --------
    >>> Panhandle_B(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    42.35366178004172

    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    .. [2] Co, Crane. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [3] GPSA. GPSA Engineering Data Book. 13th edition. Gas Processors
       Suppliers Association, Tulsa, OK, 2012.
    .. [4] Campbell, John M. Gas Conditioning and Processing, Vol. 2: The 
       Equipment Modules. 7th edition. Campbell Petroleum Series, 1992.
    .. [5] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    .. [6] Ikoku, Chi U. Natural Gas Production Engineering. Malabar, Fla: 
       Krieger Pub Co, 1991.
    '''
    c1 = 1.02 # reference condition power
    c2 = 0.961 # sg power
    c3 = 0.51 # main power
    c4 = 2.53 # diameter power
    c5 = 152.8811634298055458624385985866624419060 # 4175*10**(3/25)/36
    if Q is None and (None not in [L, D, P1, P2]):
        return c5*E*(Ts/Ps)**c1*((P1**2 - P2**2)/(L*SG**c2*Tavg*Zavg))**c3*D**c4
    elif D is None and (None not in [L, Q, P1, P2]):
        return (Q*(Ts/Ps)**(-c1)*(SG**(-c2)*(P1**2 - P2**2)/(L*Tavg*Zavg))**(-c3)/(E*c5))**(1./c4)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return (L*SG**c2*Tavg*Zavg*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(1./c3) + P2**2)**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return (-L*SG**c2*Tavg*Zavg*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(1./c3) + P1**2)**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return SG**(-c2)*(D**(-c4)*Q*(Ts/Ps)**(-c1)/(E*c5))**(-1./c3)*(P1**2 - P2**2)/(Tavg*Zavg)
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Weymouth(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=0.92):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Weymouth formula. Can calculate any of the following, 
    given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Weymouth equation. Here, a new form is developed with all units
    in base SI, based on the work of [1]_.
    
    .. math::
        Q = 137.32958 E \frac{T_s}{P_s}\left[\frac{P_1^2
        -P_2^2}{L \cdot {SG} \cdot T_{avg}Z_{avg}}\right]^{0.5}D^{2.667}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----
    [1]_'s original constant was 3.7435E-3, and it has units of km (length), 
    kPa, mm (diameter), and flowrate in m^3/day.
    
    The form in [2]_ has the same exponents as used here, units of mm 
    (diameter), kPa, km (length), and flow in m^3/hour; its leading constant is
    1.5598E-4.  
    
    The GPSA [3]_ has a leading constant of 0.1182, and otherwise the same constants.
    It is in units of mm (diameter) and kPa and m^3/day; length is stated to be
    in km, but according to the errata is in m.
    
    [4]_ has a leading constant of 1.162E7, a diameter power of 5.333 which is
    also under the 0.50 power, and is otherwise the same. It has units of kPa 
    and m^3/day, but is otherwise in base SI units.
    
    [5]_ has a leading constant of 137.2364; the other 
    exponents are the same as here. It is entirely in base SI units.
    
    [6]_ has pressures in psi, diameter in inches, length in miles, Q in 
    ft^3/hour, T in degrees Rankine, and a constant of 18.062 with the  
    exponents the same as here.

    Examples
    --------
    >>> Weymouth(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    32.07729055913029

    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    .. [2] Co, Crane. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [3] GPSA. GPSA Engineering Data Book. 13th edition. Gas Processors
       Suppliers Association, Tulsa, OK, 2012.
    .. [4] Campbell, John M. Gas Conditioning and Processing, Vol. 2: The 
       Equipment Modules. 7th edition. Campbell Petroleum Series, 1992.
    .. [5] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    .. [6] Ikoku, Chi U. Natural Gas Production Engineering. Malabar, Fla: 
       Krieger Pub Co, 1991.
    '''
    c3 = 0.5 # main power
    c4 = 2.667 # diameter power
    c5 = 137.3295809942512546732179684618143090992 # 37435*10**(501/1000)/864
    if Q is None and (None not in [L, D, P1, P2]):
        return c5*E*(Ts/Ps)*((P1**2 - P2**2)/(L*SG*Tavg*Zavg))**c3*D**c4
    elif D is None and (None not in [L, Q, P1, P2]):
        return (Ps*Q*((P1**2 - P2**2)/(L*SG*Tavg*Zavg))**(-c3)/(E*Ts*c5))**(1./c4)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return (L*SG*Tavg*Zavg*(D**(-c4)*Ps*Q/(E*Ts*c5))**(1./c3) + P2**2)**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return (-L*SG*Tavg*Zavg*(D**(-c4)*Ps*Q/(E*Ts*c5))**(1./c3) + P1**2)**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return (D**(-c4)*Ps*Q/(E*Ts*c5))**(-1./c3)*(P1**2 - P2**2)/(SG*Tavg*Zavg)
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Spitzglass_high(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=1.):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Spitzglass (high pressure drop) formula. Can calculate  
    any of the following, given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe (numerical solution)
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Spitzglass (high pressure drop) formula. Here, the form as in [1]_
    is used but with a more precise metric conversion from inches to m.
    
    .. math::
        Q = 125.1060 E \left(\frac{T_s}{P_s}\right)\left[\frac{P_1^2
        -P_2^2}{L \cdot {SG} T_{avg}Z_{avg} (1 + 0.09144/D + \frac{150}{127}D)}
        \right]^{0.5}D^{2.5}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----    
    This equation is often presented without any corection for reference
    conditions for specific gravity.
    
    This model is also presented in [2]_ with a leading constant of 1.0815E-2,
    the same exponents as used here, units of mm  (diameter), kPa, km (length),
    and flow in m^3/hour.    

    Examples
    --------
    >>> Spitzglass_high(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    29.42670246281681

    References
    ----------
    .. [1] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    .. [2] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    '''
    c3 = 1.181102362204724409448818897637795275591 # 0.03/inch or 150/127
    c4 = 0.09144
    c5 = 125.1060 
    if Q is None and (None not in [L, D, P1, P2]):
        return (c5*E*Ts/Ps*D**2.5*((P1**2-P2**2)
                /(L*SG*Zavg*Tavg*(1 + c4/D + c3*D)))**0.5)
    elif D is None and (None not in [L, Q, P1, P2]):
        to_solve = lambda D : Q - Spitzglass_high(SG=SG, Tavg=Tavg, L=L, D=D, 
                                                  P1=P1, P2=P2, Ts=Ts, Ps=Ps, 
                                                  Zavg=Zavg, E=E)        
        return newton(to_solve, 0.5)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return ((D**6*E**2*P2**2*Ts**2*c5**2
                 + D**2*L*Ps**2*Q**2*SG*Tavg*Zavg*c3 
                 + D*L*Ps**2*Q**2*SG*Tavg*Zavg 
                 + L*Ps**2*Q**2*SG*Tavg*Zavg*c4)/(D**6*E**2*Ts**2*c5**2))**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return ((D**6*E**2*P1**2*Ts**2*c5**2 
                 - D**2*L*Ps**2*Q**2*SG*Tavg*Zavg*c3 
                 - D*L*Ps**2*Q**2*SG*Tavg*Zavg 
                 - L*Ps**2*Q**2*SG*Tavg*Zavg*c4)/(D**6*E**2*Ts**2*c5**2))**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return (D**6*E**2*Ts**2*c5**2*(P1**2 - P2**2)
                /(Ps**2*Q**2*SG*Tavg*Zavg*(D**2*c3 + D + c4)))
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Spitzglass_low(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=1.):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Spitzglass (low pressure drop) formula. Can calculate  
    any of the following, given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe (numerical solution)
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Spitzglass (low pressure drop) formula. Here, the form as in [1]_
    is used but with a more precise metric conversion from inches to m.
    
    .. math::
        Q = 125.1060 E \left(\frac{T_s}{P_s}\right)\left[\frac{2(P_1
        -P_2)(P_s+1210)}{L \cdot {SG} \cdot T_{avg}Z_{avg} (1 + 0.09144/D 
        + \frac{150}{127}D)}\right]^{0.5}D^{2.5}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----    
    This equation is often presented without any corection for reference
    conditions for specific gravity.
    
    This model is also presented in [2]_ with a leading constant of 5.69E-2,
    the same exponents as used here, units of mm  (diameter), kPa, km (length),
    and flow in m^3/hour. However, it is believed to contain a typo, and gives
    results <1/3 of the correct values. It is also present in [2]_ in imperial
    form; this is believed correct, but makes a slight assumption not done in
    [1]_.

    This model is present in [3]_ without reference corrections. The 1210 
    constant in [1]_ is an approximation necessary for the reference correction
    to function without a square of the pressure difference. The GPSA version
    is as follows, and matches this formulation very closely:
    
    .. math::
        Q = 0.821 \left[\frac{(P_1-P_2)D^5}{L \cdot {SG}
        (1 + 91.44/D + 0.0018D)}\right]^{0.5}
    
    The model is also shown in [4]_, with diameter in inches, length in feet, 
    flow in MMSCFD, pressure drop in inH2O, and a rounded leading constant of 
    0.09; this makes its predictions several percent higher than the model here.

    Examples
    --------
    >>> Spitzglass_low(D=0.154051, P1=6720.3199, P2=0, L=54.864, SG=0.6, Tavg=288.7)
    0.9488775242530617
    
    References
    ----------
    .. [1] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    .. [2] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    .. [3] GPSA. GPSA Engineering Data Book. 13th edition. Gas Processors
       Suppliers Association, Tulsa, OK, 2012.
    .. [4] PetroWiki. "Pressure Drop Evaluation along Pipelines" Accessed 
       September 11, 2016. http://petrowiki.org/Pressure_drop_evaluation_along_pipelines#Spitzglass_equation_2.
    '''
    c3 = 1.181102362204724409448818897637795275591 # 1.1811 ish or 0.03/inch or 150/127
    c4 = 0.09144
    c5 = 125.1060 
    if Q is None and (None not in [L, D, P1, P2]):
        return c5*Ts/Ps*D**2.5*E*(((P1-P2)*2*(Ps+1210.))/(L*SG*Tavg*Zavg*(1 + c4/D + c3*D)))**0.5
    elif D is None and (None not in [L, Q, P1, P2]):
        to_solve = lambda D : Q - Spitzglass_low(SG=SG, Tavg=Tavg, L=L, D=D, P1=P1, P2=P2, Ts=Ts, Ps=Ps, Zavg=Zavg, E=E)        
        return newton(to_solve, 0.5)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return 0.5*(2.0*D**6*E**2*P2*Ts**2*c5**2*(Ps + 1210.0) + D**2*L*Ps**2*Q**2*SG*Tavg*Zavg*c3 + D*L*Ps**2*Q**2*SG*Tavg*Zavg + L*Ps**2*Q**2*SG*Tavg*Zavg*c4)/(D**6*E**2*Ts**2*c5**2*(Ps + 1210.0))
    elif P2 is None and (None not in [L, Q, D, P1]):
        return 0.5*(2.0*D**6*E**2*P1*Ts**2*c5**2*(Ps + 1210.0) - D**2*L*Ps**2*Q**2*SG*Tavg*Zavg*c3 - D*L*Ps**2*Q**2*SG*Tavg*Zavg - L*Ps**2*Q**2*SG*Tavg*Zavg*c4)/(D**6*E**2*Ts**2*c5**2*(Ps + 1210.0))
    elif L is None and (None not in [P2, Q, D, P1]):
        return 2.0*D**6*E**2*Ts**2*c5**2*(P1*Ps + 1210.0*P1 - P2*Ps - 1210.0*P2)/(Ps**2*Q**2*SG*Tavg*Zavg*(D**2*c3 + D + c4))
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Oliphant(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, 
                Ps=101325., Zavg=1, E=0.92):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Oliphant formula. Can calculate any of the following, 
    given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe (numerical solution)
    * Length of pipe
    
    This model is a more complete conversion to metric of the Imperial version
    presented in [1]_.
    
    .. math::
        Q = 84.5872\left(D^{2.5} + 0.20915D^3\right)\frac{T_s}{P_s}\left(\frac
        {P_1^2 - P_2^2}{L\cdot {SG} \cdot T_{avg}}\right)^{0.5}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----
    Recommended in [1]_ for use between vacuum and 100 psi.
    
    The model is simplified by grouping constants here; however, it is presented
    in the imperial unit set inches (diameter), miles (length), psi, Rankine,
    and MMSCFD in [1]_:
    
    .. math::
        Q = 42(24)\left(D^{2.5} + \frac{D^3}{30}\right)\left(\frac{14.4}{P_s}
        \right)\left(\frac{T_s}{520}\right)\left[\left(\frac{0.6}{SG}\right)
        \left(\frac{520}{T_{avg}}\right)\left(\frac{P_1^2 - P_2^2}{L}\right)
        \right]^{0.5}
    
    Examples
    --------
    >>> Oliphant(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    28.851535408143057
    
    References
    ----------
    .. [1] GPSA. GPSA Engineering Data Book. 13th edition. Gas Processors
       Suppliers Association, Tulsa, OK, 2012.
    .. [2] F. N. Oliphant, "Production of Natural Gas," Report. USGS, 1902.
    '''
    # c1 = 42*24*Q*foot**3/day*(mile)**0.5*9/5.*(5/9.)**0.5*psi*(1/psi)*14.4/520.*0.6**0.5*520**0.5/inch**2.5
    c1 = 84.587176139918568651410168968141078948974609375000
    c2 = 0.2091519350460528670065940559652517549694 # 1/(30.*0.0254**0.5)
    if Q is None and (None not in [L, D, P1, P2]):
        return c1*(D**2.5 + c2*D**3)*Ts/Ps*((P1**2-P2**2)/(L*SG*Tavg))**0.5    
    elif D is None and (None not in [L, Q, P1, P2]):
        to_solve = lambda D : Q - Oliphant(SG=SG, Tavg=Tavg, L=L, D=D, P1=P1, P2=P2, Ts=Ts, Ps=Ps, Zavg=Zavg, E=E)        
        return newton(to_solve, 0.5)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return (L*Ps**2*Q**2*SG*Tavg/(Ts**2*c1**2*(D**3*c2 + D**2.5)**2) + P2**2)**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return (-L*Ps**2*Q**2*SG*Tavg/(Ts**2*c1**2*(D**3*c2 + D**2.5)**2) + P1**2)**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return Ts**2*c1**2*(P1**2 - P2**2)*(D**3*c2 + D**2.5)**2/(Ps**2*Q**2*SG*Tavg)
    else:
        raise Exception('This function solves for either flow, upstream \
pressure, downstream pressure, diameter, or length; all other inputs \
must be provided.')


def Fritzsche(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7, Ps=101325., Zavg=1, E=1):
    r'''Calculation function for dealing with flow of a compressible gas in a 
    pipeline with the Fritzsche formula. Can calculate any of the following, 
    given all other inputs:
    
    * Flow rate
    * Upstream pressure
    * Downstream pressure
    * Diameter of pipe
    * Length of pipe
    
    A variety of different constants and expressions have been presented
    for the Fritzsche formula. Here, the form as in [1]_
    is used but with all inputs in base SI units.
    
    .. math::
        Q = 93.500 \frac{T_s}{P_s}\left(\frac{P_1^2 - P_2^2}
        {L\cdot {SG}^{0.8587} \cdot T_{avg}}\right)^{0.538}D^{2.69}

    Parameters
    ----------
    SG : float
        Specific gravity of fluid with respect to air at the reference 
        temperature and pressure `Ts` and `Ps`, [-]
    Tavg : float
        Average temperature of the fluid in the pipeline, [K]
    L : float, optional
        Length of pipe, [m]
    D : float, optional
        Diameter of pipe, [m]
    P1 : float, optional
        Inlet pressure to pipe, [Pa]
    P2 : float, optional
        Outlet pressure from pipe, [Pa]
    Q : float, optional
        Flow rate of gas through pipe, [m^3/s]
    Ts : float, optional
        Reference temperature for the specific gravity of the gas, [K]
    Ps : float, optional
        Reference pressure for the specific gravity of the gas, [Pa]
    Zavg : float, optional
        Average compressibility factor for gas, [-]
    E : float, optional
        Pipeline efficiency, a correction factor between 0 and 1

    Returns
    -------
    Q, P1, P2, D, or L : float
        The missing input which was solved for [base SI]

    Notes
    -----
    This model is also presented in [1]_ with a leading constant of 2.827,
    the same exponents as used here, units of mm  (diameter), kPa, km (length),
    and flow in m^3/hour. 
    
    This model is shown in base SI units in [2]_, and with a leading constant
    of 94.2565, a diameter power of 2.6911, main group power of 0.5382
    and a specific gravity power of 0.858. The difference is very small.

    Examples
    --------
    >>> Fritzsche(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
    39.421535157535565
    
    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    .. [2] Coelho, Paulo M., and Carlos Pinho. "Considerations about Equations 
       for Steady State Flow in Natural Gas Pipelines." Journal of the 
       Brazilian Society of Mechanical Sciences and Engineering 29, no. 3 
       (September 2007): 262-73. doi:10.1590/S1678-58782007000300005.
    '''
    c5 = 93.50009798751128188757518688244137811221 # 14135*10**(57/125)/432
    c2 = 0.8587
    c3 = 0.538
    c4 = 2.69
    if Q is None and (None not in [L, D, P1, P2]):
        return c5*E*(Ts/Ps)*((P1**2 - P2**2)/(SG**c2*Tavg*L*Zavg))**c3*D**c4
    elif D is None and (None not in [L, Q, P1, P2]):
        return (Ps*Q*(SG**(-c2)*(P1**2 - P2**2)/(L*Tavg*Zavg))**(-c3)/(E*Ts*c5))**(1./c4)
    elif P1 is None and (None not in [L, Q, D, P2]):
        return (L*SG**c2*Tavg*Zavg*(D**(-c4)*Ps*Q/(E*Ts*c5))**(1./c3) + P2**2)**0.5
    elif P2 is None and (None not in [L, Q, D, P1]):
        return (-L*SG**c2*Tavg*Zavg*(D**(-c4)*Ps*Q/(E*Ts*c5))**(1./c3) + P1**2)**0.5
    elif L is None and (None not in [P2, Q, D, P1]):
        return SG**(-c2)*(D**(-c4)*Ps*Q/(E*Ts*c5))**(-1./c3)*(P1**2 - P2**2)/(Tavg*Zavg)
    else:
        raise Exception('This function solves for either flow, upstream pressure, downstream pressure, diameter, or length; all other inputs must be provided.')
