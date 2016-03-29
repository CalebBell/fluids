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
__all__ = ['T_critical_flow', 'P_critical_flow', 'is_critical_flow',
           'stagnation_energy', 'P_stagnation', 'T_stagnation',
           'T_stagnation_ideal']


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

#print [P_critical_flow(1400000, 1.289)]

# It would be better to find critical density from an EOS and T and P
#def rho_critical_flow(rho, k):
#    rhocf = rho*(2/(k+1.))**(1/(k-1))
#    return rhocf

def is_critical_flow(P1, P2, k):
    r'''Determines if a flow of a fluid driven by pressure gradient
    P1 - P2 is critical, for a fluid with the given isentropic coefficient.
    This function calculates critical flow pressure, and checks if this is
    larger than P2. If so, the flow is critical and choked.

    Parameters
    ----------
    P1: float
        Higher, source pressure [Pa]
    P2: float
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

