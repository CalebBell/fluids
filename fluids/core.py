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
from math import sin, exp, pi
from scipy.constants import g, R

__all__ = ['Reynolds', 'Prandtl', 'Grashof', 'Nusselt', 'Sherwood', 'Rayleigh',
'Schmidt', 'Peclet_heat', 'Peclet_mass', 'Fourier_heat', 'Fourier_mass',
'Graetz_heat', 'Lewis', 'Weber', 'Mach', 'Knudsen', 'Bond', 'Dean',
'Froude', 'Strouhal', 'Biot', 'Stanton', 'Euler', 'Cavitation', 'Eckert',
'Jakob', 'Power_number', 'Drag', 'Capillary', 'Bejan_L', 'Bejan_p', 'Boiling',
'Confinement', 'Archimedes', 'Ohnesorge', 'Suratman', 'thermal_diffusivity', 'c_ideal_gas',
'relative_roughness', 'nu_mu_converter', 'gravity',
'K_from_f', 'K_from_L_equiv', 'L_equiv_from_K', 'dP_from_K', 'head_from_K', 'head_from_P',
'P_from_head', 'Eotvos']



### Not quite dimensionless groups
def thermal_diffusivity(k, rho, Cp):
    r'''Calculates thermal diffusivity or `alpha` for a fluid with the given
    parameters.

    .. math::
        \alpha = \frac{k}{\rho Cp}

    Parameters
    ----------
    k : float
        Thermal conductivity, [W/m/K]
    Cp : float
        Heat capacity, [J/kg/K]
    rho : float
        Density, [kg/m^3]

    Returns
    -------
    alpha : float
        Thermal diffusivity, [m^2/s]

    Notes
    -----

    Examples
    --------
    >>> thermal_diffusivity(0.02, 1., 1000.)
    2e-05

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    alpha = k/(rho*Cp)
    return alpha


### Ideal gas fluid properties


def c_ideal_gas(T, k, MW):
    r'''Calculates speed of sound `c` in an ideal gas at temperature T.

    .. math::
        c = \sqrt{kR_{specific}T}

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    k : float
        Isentropic exponent of fluid, [-]
    MW : float
        Molecular weight of fluid, [g/mol]

    Returns
    -------
    c : float
        Speed of sound in fluid, [m/s]

    Notes
    -----
    Used in compressible flow calculations.
    Note that the gas constant used is the specific gas constant:

    .. math::
        R_{specific} = R\frac{1000}{MW}

    Examples
    --------
    >>> c_ideal_gas(1.4, 303., 28.96)
    348.9820361755092

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Rspecific = R*1000./MW
    c = (k*Rspecific*T)**0.5
    return c


### Dimensionless groups with documentation

def Reynolds(V, D, rho=None, mu=None, nu=None):
    r'''Calculates Reynolds number or `Re` for a fluid with the given
    properties for the specified velocity and diameter.

    .. math::
        Re = \frac{D \cdot V}{\nu} = \frac{\rho V D}{\mu}

    Inputs either of any of the following sets:

    * V, D, density `rho` and kinematic viscosity `mu`
    * V, D, and dynamic viscosity `nu`

    Parameters
    ----------
    D : float
        Diameter [m]
    V : float
        Velocity [m/s]
    rho : float, optional
        Density, [kg/m^3]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    nu : float, optional
        Kinematic viscosity, [m^2/s]

    Returns
    -------
    Re : float
        Reynolds number []

    Notes
    -----
    .. math::
        Re = \frac{\text{Momentum}}{\text{Viscosity}}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Reynolds(2.5, 0.25, 1.1613, 1.9E-5)
    38200.65789473684
    >>> Reynolds(2.5, 0.25, nu=1.636e-05)
    38202.93398533008

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if rho and mu:
        nu = mu/rho
    elif not nu:
        raise Exception('Either density and viscosity, or dynamic viscosity, \
        is needed')
    Re = V*D/nu
    return Re


def Peclet_heat(V, L, rho=None, Cp=None, k=None, alpha=None):
    r'''Calculates heat transfer Peclet number or `Pe` for a specified velocity
    `V`, characteristic length `L`, and specified properties for the given
    fluid.

    .. math::
        Pe = \frac{VL\rho C_p}{k} = \frac{LV}{\alpha}

    Inputs either of any of the following sets:

    * V, L, density `rho`, heat capcity `Cp`, and thermal conductivity `k`
    * V, L, and thermal diffusivity `alpha`

    Parameters
    ----------
    V : float
        Velocity [m/s]
    L : float
        Characteristic length [m]
    rho : float, optional
        Density, [kg/m^3]
    Cp : float, optional
        Heat capacity, [J/kg/K]
    k : float, optional
        Thermal conductivity, [W/m/K]
    alpha : float, optional
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Pe : float
        Peclet number (heat) []

    Notes
    -----
    .. math::
        Pe = \frac{\text{Bulk heat transfer}}{\text{Conduction heat transfer}}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Peclet_heat(1.5, 2, 1000., 4000., 0.6)
    20000000.0
    >>> Peclet_heat(1.5, 2, alpha=1E-7)
    30000000.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if rho and Cp and k:
        alpha =  k/(rho*Cp)
    elif not alpha:
        raise Exception('Either heat capacity and thermal conductivity and\
        density, or thermal diffusivity is needed')
    Pe = V*L/alpha
    return Pe


def Peclet_mass(V, L, D):
    r'''Calculates mass transfer Peclet number or `Pe` for a specified velocity
    `V`, characteristic length `L`, and diffusion coefficient `D`.

    .. math::
        Pe = \frac{L V}{D}

    Parameters
    ----------
    V : float
        Velocity [m/s]
    L : float
        Characteristic length [m]
    D : float
        Diffusivity of a species, [m^2/s]

    Returns
    -------
    Pe : float
        Peclet number (mass) []

    Notes
    -----
    .. math::
        Pe = \frac{\text{Advective transport rate}}{\text{Diffusive transport rate}}

    Examples
    --------
    >>> Peclet_mass(1.5, 2, 1E-9)
    3000000000.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    Pe = V*L/D
    return Pe


def Fourier_heat(t, L, rho=None, Cp=None, k=None, alpha=None):
    r'''Calculates heat transfer Fourier number or `Fo` for a specified time
    `t`, characteristic length `L`, and specified properties for the given
    fluid.

    .. math::
        Fo = \frac{k t}{C_p \rho L^2} = \frac{\alpha t}{L^2}

    Inputs either of any of the following sets:

    * t, L, density `rho`, heat capcity `Cp`, and thermal conductivity `k`
    * t, L, and thermal diffusivity `alpha`

    Parameters
    ----------
    t : float
        time [s]
    L : float
        Characteristic length [m]
    rho : float, optional
        Density, [kg/m^3]
    Cp : float, optional
        Heat capacity, [J/kg/K]
    k : float, optional
        Thermal conductivity, [W/m/K]
    alpha : float, optional
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Fo : float
        Fourier number (heat) []

    Notes
    -----
    .. math::
        Fo = \frac{\text{Heat conduction rate}}
        {\text{Rate of thermal energy storage in a solid}}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Fourier_heat(1.5, 2, 1000., 4000., 0.6)
    5.625e-08
    >>> Fourier_heat(1.5, 2, alpha=1E-7)
    3.75e-08

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if rho and Cp and k:
        alpha =  k/(rho*Cp)
    elif not alpha:
        raise Exception('Either heat capacity and thermal conductivity and\
        density, or thermal diffusivity is needed')
    Fo = t*alpha/L**2
    return Fo


def Fourier_mass(t, L, D):
    r'''Calculates mass transfer Fourier number or `Fo` for a specified time
    `t`, characteristic length `L`, and diffusion coefficient `D`.

    .. math::
        Fo = \frac{D t}{L^2}

    Parameters
    ----------
    t : float
        time [s]
    L : float
        Characteristic length [m]
    D : float
        Diffusivity of a species, [m^2/s]

    Returns
    -------
    Fo : float
        Fourier number (mass) []

    Notes
    -----
    .. math::
        Fo = \frac{\text{Diffusive transport rate}}{\text{Storage rate}}

    Examples
    --------
    >>> Fourier_mass(1.5, 2, 1E-9)
    3.7500000000000005e-10

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    Fo = t*D/L**2
    return Fo


def Graetz_heat(V, D, x, rho=None, Cp=None, k=None, alpha=None):
    r'''Calculates Graetz number or `Gz` for a specified velocity
    `V`, diameter `D`, axial diatance `x`, and specified properties for the
    given fluid.

    .. math::
        Gz = \frac{VD^2\cdot C_p \rho}{x\cdot k} = \frac{VD^2}{x \alpha}

    Inputs either of any of the following sets:

    * V, D, x, density `rho`, heat capcity `Cp`, and thermal conductivity `k`
    * V, D, x, and thermal diffusivity `alpha`

    Parameters
    ----------
    V : float
        Velocity, [m/s]
    D : float
        Diameter [m]
    x : float
        Axial distance [m]
    rho : float, optional
        Density, [kg/m^3]
    Cp : float, optional
        Heat capacity, [J/kg/K]
    k : float, optional
        Thermal conductivity, [W/m/K]
    alpha : float, optional
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Gz : float
        Graetz number []

    Notes
    -----
    .. math::
        Gz = \frac{\text{Time for radial heat diffusion in a fluid by conduction}}
        {\text{Time taken by fluid to reach distance x}}

        Gz = \frac{D}{x}RePr

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Graetz_heat(1.5, 0.25, 5, 800., 2200., 0.6)
    55000.0
    >>> Graetz_heat(1.5, 0.25, 5, alpha=1E-7)
    187500.0

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    if rho and Cp and k:
        alpha =  k/(rho*Cp)
    elif not alpha:
        raise Exception('Either heat capacity and thermal conductivity and\
        density, or thermal diffusivity is needed')
    Gz = V*D**2/(x*alpha)
    return Gz


def Schmidt(D, mu=None, nu=None, rho=None):
    r'''Calculates Schmidt number or `Sc` for a fluid with the given
    parameters.

    .. math::
        Sc = \frac{\mu}{D\rho} = \frac{\nu}{D}

    Inputs can be any of the following sets:

    * Diffusivity, dynamic viscosity, and density
    * Diffusivity and kinematic viscosity

    Parameters
    ----------
    D : float
        Diffusivity of a species, [m^2/s]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    nu : float, optional
        Kinematic viscosity, [m^2/s]
    rho : float, optional
        Density, [kg/m^3]

    Returns
    -------
    Sc : float
        Schmidt number []

    Notes
    -----
    .. math::
        Sc =\frac{\text{kinematic viscosity}}{\text{molecular diffusivity}}
        = \frac{\text{viscous diffusivity}}{\text{species diffusivity}}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Schmidt(D=2E-6, mu=4.61E-6, rho=800)
    0.00288125
    >>> Schmidt(D=1E-9, nu=6E-7)
    599.9999999999999

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if rho and mu:
        Sc = mu/(rho*D)
    elif nu:
        Sc = nu/D
    else:
        raise Exception('Insufficient information provided for Schmidt number calculation')
    return Sc


def Lewis(D=None, alpha=None, Cp=None, k=None, rho=None):
    r'''Calculates Lewis number or `Le` for a fluid with the given parameters.

    .. math::
        Le = \frac{k}{\rho C_p D} = \frac{\alpha}{D}

    Inputs can be either of the following sets:

    * Diffusivity and Thermal diffusivity
    * Diffusivity, heat capacity, thermal conductivity, and density

    Parameters
    ----------
    D : float
        Diffusivity of a species, [m^2/s]
    alpha : float, optional
        Thermal diffusivity, [m^2/s]
    Cp : float, optional
        Heat capacity, [J/kg/K]
    k : float, optional
        Thermal conductivity, [W/m/K]
    rho : float, optional
        Density, [kg/m^3]

    Returns
    -------
    Le : float
        Lewis number []

    Notes
    -----
    .. math::
        Le=\frac{\text{Thermal diffusivity}}{\text{Mass diffusivity}} =
        \frac{Sc}{Pr}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Lewis(D=22.6E-6, alpha=19.1E-6)
    0.8451327433628318
    >>> Lewis(D=22.6E-6, rho=800., k=.2, Cp=2200)
    0.00502815768302494

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if k and Cp and rho:
        alpha = k/(rho*Cp)
    elif alpha:
        pass
    else:
        raise Exception('Insufficient information provided for Le calculation')
    Le = alpha/D
    return Le


def Weber(V, L, rho, sigma):
    r'''Calculates Weber number, `We`, for a fluid with the given density,
    surface tension, velocity, and geometric parameter (usually diameter
    of bubble).

    .. math::
        We = \frac{V^2 L\rho}{\sigma}

    Parameters
    ----------
    V : float
        Velocity of fluid, [m/s]
    L : float
        Characteristic length, typically bubble diameter [m]
    rho : float
        Density of fluid, [kg/m^3]
    sigma : float
        Surface tension, [N/m]

    Returns
    -------
    We : float
        Weber number []

    Notes
    -----
    Used in bubble calculations.

    .. math::
        We = \frac{\text{inertial force}}{\text{surface tension force}}

    Examples
    --------
    >>> Weber(V=0.18, L=0.001, rho=900., sigma=0.01)
    2.916

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    We = V**2*L*rho/sigma
    return We


def Mach(V, c):
    r'''Calculates Mach number or `Ma` for a fluid of velocity `V` with speed
    of sound `c`.

    .. math::
        Ma = \frac{V}{c}

    Parameters
    ----------
    V : float
        Velocity of fluid, [m/s]
    c : float
        Speed of sound in fluid, [m/s]

    Returns
    -------
    Ma : float
        Mach number []

    Notes
    -----
    Used in compressible flow calculations.

    .. math::
        Ma = \frac{\text{fluid velocity}}{\text{sonic velocity}}

    Examples
    --------
    >>> Mach(33., 330)
    0.1

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    return V/c


def Confinement(D, rhol, rhog, sigma, g=g):
    r'''Calculates Confinement number or `Co` for a fluid in a channel of 
    diameter `D` with liquid and gas densities `rhol` and `rhog` and surface
    tension `sigma`, under the influence of gravitational force `g`.

    .. math::
        \text{Co}=\frac{\left[\frac{\sigma}{g(\rho_l-\rho_g)}\right]^{0.5}}{D}

    Parameters
    ----------
    D : float
        Diameter of channel, [m]
    rhol : float
        Density of liquid phase, [kg/m^3]
    rhog : float
        Density of gas phase, [kg/m^3]
    sigma : float
        Surface tension between liquid-gas phase, [N/m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    Co : float
        Confinement number [-]

    Notes
    -----
    Used in two-phase pressure drop and heat transfer correlations. First used
    in [1]_ according to [3]_.

    .. math::
        \text{Co} = \frac{\frac{\text{surface tension force}}
        {\text{buoyancy force}}}{\text{Channel area}}

    Examples
    --------
    >>> Confinement(0.001, 1077, 76.5, 4.27E-3)
    0.6596978265315191

    References
    ----------
    .. [1] Cornwell, Keith, and Peter A. Kew. "Boiling in Small Parallel 
       Channels." In Energy Efficiency in Process Technology, edited by Dr P. 
       A. Pilavachi, 624-638. Springer Netherlands, 1993. 
       doi:10.1007/978-94-011-1454-7_56.
    .. [2] Kandlikar, Satish G. Heat Transfer and Fluid Flow in Minichannels 
       and Microchannels. Elsevier, 2006.
    .. [3] Tran, T. N, M. -C Chyu, M. W Wambsganss, and D. M France. Two-Phase 
       Pressure Drop of Refrigerants during Flow Boiling in Small Channels: An 
       Experimental Investigation and Correlation Development." International 
       Journal of Multiphase Flow 26, no. 11 (November 1, 2000): 1739-54. 
       doi:10.1016/S0301-9322(99)00119-6.
    '''
    return (sigma/(g*(rhol-rhog)))**0.5/D


def Knudsen(path, L):
    r'''Calculates Knudsen number or `Kn` for a fluid with mean free path
    `path` and for a characteristic length `L`.

    .. math::
        Kn = \frac{\lambda}{L}

    Parameters
    ----------
    path : float
        Mean free path between molecular collisions, [m]
    L : float
        Characteristic length, [m]

    Returns
    -------
    Kn : float
        Knudsen number []

    Notes
    -----
    Used in mass transfer calculations.

    .. math::
        Kn = \frac{\text{Mean free path length}}{\text{Characteristic length}}

    Examples
    --------
    >>> Knudsen(1e-10, .001)
    1e-07

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Kn = path/L
    return Kn


def Prandtl(Cp=None, k=None, mu=None, nu=None, rho=None, alpha=None):
    r'''Calculates Prandtl number or `Pr` for a fluid with the given
    parameters.

    .. math::
        Pr = \frac{C_p \mu}{k} = \frac{\nu}{\alpha} = \frac{C_p \rho \nu}{k}

    Inputs can be any of the following sets:

    * Heat capacity, dynamic viscosity, and thermal conductivity
    * Thermal diffusivity and kinematic viscosity
    * Heat capacity, kinematic viscosity, thermal conductivity, and density

    Parameters
    ----------
    Cp : float
        Heat capacity, [J/kg/K]
    k : float
        Thermal conductivity, [W/m/K]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    nu : float, optional
        Kinematic viscosity, [m^2/s]
    rho : float
        Density, [kg/m^3]
    alpha : float
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Pr : float
        Prandtl number []

    Notes
    -----
    .. math::
        Pr=\frac{\text{kinematic viscosity}}{\text{thermal diffusivity}} = \frac{\text{momentum diffusivity}}{\text{thermal diffusivity}}

    An error is raised if none of the required input sets are provided.

    Examples
    --------
    >>> Prandtl(Cp=1637., k=0.010, mu=4.61E-6)
    0.754657
    >>> Prandtl(Cp=1637., k=0.010, nu=6.4E-7, rho=7.1)
    0.7438528
    >>> Prandtl(nu=6.3E-7, alpha=9E-7)
    0.7000000000000001

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    if k and Cp and mu:
        Pr = Cp*mu/k
    elif nu and rho and Cp and k:
        Pr = nu*rho*Cp/k
    elif nu and alpha:
        Pr = nu/alpha
    else:
        raise Exception('Insufficient information provided for Pr calculation')
    return Pr


def Grashof(L, beta, T1, T2=0, rho=None, mu=None, nu=None, g=g):
    r'''Calculates Grashof number or `Gr` for a fluid with the given
    properties, temperature difference, and characteristic length.

    .. math::
        Gr = \frac{g\beta (T_s-T_\infty)L^3}{\nu^2}
        = \frac{g\beta (T_s-T_\infty)L^3\rho^2}{\mu^2}

    Inputs either of any of the following sets:

    * L, beta, T1 and T2, and density `rho` and kinematic viscosity `mu`
    * L, beta, T1 and T2, and dynamic viscosity `nu`

    Parameters
    ----------
    L : float
        Characteristic length [m]
    beta : float
        Volumetric thermal expansion coefficient [1/K]
    T1 : float
        Temperature 1, usually a film temperature [K]
    T2 : float, optional
        Temperature 2, usually a bulk temperature (or 0 if only a difference
        is provided to the function) [K]
    rho : float, optional
        Density, [kg/m^3]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    nu : float, optional
        Kinematic viscosity, [m^2/s]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    Gr : float
        Grashof number []

    Notes
    -----
    .. math::
        Gr = \frac{\text{Buoyancy forces}}{\text{Viscous forces}}

    An error is raised if none of the required input sets are provided.
    Used in free convection problems only.

    Examples
    --------
    Example 4 of [1]_, p. 1-21 (matches):

    >>> Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613, mu=1.9E-5)
    4656936556.178915
    >>> Grashof(L=0.9144, beta=0.000933, T1=378.2, T2=200, nu=1.636e-05)
    4657491516.530312

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if rho and mu:
        nu = mu/rho
    elif not nu:
        raise Exception('Either density and viscosity, or dynamic viscosity, \
        is needed')
    Gr = g*beta*abs(T2-T1)*L**3/nu**2
    return Gr


def Bond(rhol, rhog, sigma, L):
    r'''Calculates Bond number, `Bo` also known as Eotvos number,
    for a fluid with the given liquid and gas densities, surface tension,
    and geometric parameter (usually length).

    .. math::
        Bo = \frac{g(\rho_l-\rho_g)L^2}{\sigma}

    Parameters
    ----------
    rhol : float
        Density of liquid, [kg/m^3]
    rhog : float
        Density of gas, [kg/m^3]
    sigma : float
        Surface tension, [N/m]
    L : float
        Characteristic length, [m]

    Returns
    -------
    Bo : float
        Bond number []

    Examples
    --------
    >>> Bond(1000., 1.2, .0589, 2)
    665187.2339558573

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    return (g*(rhol-rhog)*L**2/sigma)

Eotvos = Bond


def Rayleigh(Pr, Gr):
    r'''Calculates Rayleigh number or `Ra` using Prandtl number `Pr` and
    Grashof number `Gr` for a fluid with the given
    properties, temperature difference, and characteristic length used
    to calculate `Gr` and `Pr`.

    .. math::
        Ra = PrGr

    Parameters
    ----------
    Pr : float
        Prandtl number []
    Gr : float
        Grashof number []

    Returns
    -------
    Ra : float
        Rayleigh number []

    Notes
    -----
    Used in free convection problems only.

    Examples
    --------
    >>> Rayleigh(1.2, 4.6E9)
    5520000000.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Ra = Pr*Gr
    return Ra


def Froude(V, L, g=g, squared=False):
    r'''Calculates Froude number `Fr` for velocity `V` and geometric length
    `L`. If desired, gravity can be specified as well. Normally the function
    returns the result of the equation below; Froude number is also often
    said to be defined as the square of the equation below.

    .. math::
        Fr = \frac{V}{\sqrt{gL}}

    Parameters
    ----------
    V : float
        Velocity of the particle or fluid, [m/s]
    L : float
        Characteristic length, no typical definition [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]
    squared : bool, optional
        Whether to return the squared form of Frounde number

    Returns
    -------
    Fr : float
        Froude number, [-]

    Notes
    -----
    Many alternate definitions including density ratios have been used.

    .. math::
        Fr = \frac{\text{Inertial Force}}{\text{Gravity Force}}

    Examples
    --------
    >>> Froude(1.83, L=2., g=1.63)
    1.0135432593877318
    >>> Froude(1.83, L=2., squared=True)
    0.17074638128208924

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Fr = V/(L*g)**0.5
    if squared:
        Fr *= Fr
    return Fr


def Strouhal(f, L, V):
    r'''Calculates Strouhal number `St` for a characteristic frequency `f`,
    characteristic length `L`, and velocity `V`.

    .. math::
        St = \frac{fL}{V}

    Parameters
    ----------
    f : float
        Characteristic frequency, usually that of vortex shedding, [Hz]
    L : float
        Characteristic length, [m]
    V : float
        Velocity of the fluid, [m/s]

    Returns
    -------
    St : float
        Strouhal number, [-]

    Notes
    -----
    Sometimes abbreviated to S or Sr.

    .. math::
        St = \frac{\text{Characteristif flow time}}
        {\text{Period of oscillation}}

    Examples
    --------
    >>> Strouhal(8, 2., 4.)
    4.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    St = f*L/V
    return St


def Nusselt(h, L, k):
    r'''Calculates Nusselt number `Nu` for a heat transfer coefficient `h`,
    characteristic length `L`, and thermal conductivity `k`.

    .. math::
        Nu = \frac{hL}{k}

    Parameters
    ----------
    h : float
        Heat transfer coefficient, [W/m^2/K]
    L : float
        Characteristic length, no typical definition [m]
    k : float
        Thermal conductivity of fluid [W/m/K]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Do not confuse k, the thermal conductivity of the fluid, with that
    of within a solid object associated with!

    .. math::
        Nu = \frac{\text{Convective heat transfer}}
        {\text{Conductive heat transfer}}

    Examples
    --------
    >>> Nusselt(1000., 1.2, 300.)
    4.0
    >>> Nusselt(10000., .01, 4000.)
    0.025

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    Nu = h*L/k
    return Nu


def Sherwood(K, L, D):
    r'''Calculates Sherwood number `Sh` for a mass transfer coefficient `K`,
    characteristic length `L`, and diffusivity `D`.

    .. math::
        Sh = \frac{KL}{D}

    Parameters
    ----------
    K : float
        Mass transfer coefficient, [m/s]
    L : float
        Characteristic length, no typical definition [m]
    D : float
        Diffusivity of a species [m/s^2]

    Returns
    -------
    Sh : float
        Sherwood number, [-]

    Notes
    -----

    .. math::
        Sh = \frac{\text{Mass transfer by convection}}
        {\text{Mass transfer by diffusion}} = \frac{K}{D/L}

    Examples
    --------
    >>> Sherwood(1000., 1.2, 300.)
    4.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    Sh = K*L/D
    return Sh


def Biot(h, L, k):
    r'''Calculates Biot number `Br` for heat transfer coefficient `h`,
    geometric length `L`, and thermal conductivity `k`.

    .. math::
        Bi=\frac{hL}{k}

    Parameters
    ----------
    h : float
        Heat transfer coefficient, [W/m^2/K]
    L : float
        Characteristic length, no typical definition [m]
    k : float
        Thermal conductivity, within the object [W/m/K]

    Returns
    -------
    Bi : float
        Biot number, [-]

    Notes
    -----
    Do not confuse k, the thermal conductivity within the object, with that
    of the medium h is calculated with!

    .. math::
        Bi = \frac{\text{Surface thermal resistance}}
        {\text{Internal thermal resistance}}

    Examples
    --------
    >>> Biot(1000., 1.2, 300.)
    4.0
    >>> Biot(10000., .01, 4000.)
    0.025

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Bi = h*L/k
    return Bi


def Stanton(h, V, rho, Cp):
    r'''Calculates Stanton number or `St` for a specified heat transfer
    coefficient `h`, velocity `V`, density `rho`, and heat capacity `Cp`.

    .. math::
        St = \frac{h}{V\rho Cp}

    Parameters
    ----------
    h : float
        Heat transfer coefficient, [W/m^2/K]
    V : float
        Velocity, [m/s]
    rho : float
        Density, [kg/m^3]
    Cp : float
        Heat capacity, [J/kg/K]

    Returns
    -------
    St : float
        Stanton number []

    Notes
    -----
    .. math::
        St = \frac{\text{Heat transfer coefficient}}{\text{Thermal capacity}}

    Examples
    --------
    >>> Stanton(5000, 5, 800, 2000.)
    0.000625

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    St = h/(V*rho*Cp)
    return St


def Euler(dP, rho, V):
    r'''Calculates Euler number or `Eu` for a fluid of velocity `V` and
    density `rho` experiencing a pressure drop `dP`.

    .. math::
        Eu = \frac{\Delta P}{\rho V^2}

    Parameters
    ----------
    dP : float
        Pressure drop experience by the fluid, [Pa]
    rho : float
        Density of the fluid, [kg/m^3]
    V : float
        Velocity of fluid, [m/s]

    Returns
    -------
    Eu : float
        Euler number []

    Notes
    -----
    Used in pressure drop calculations.
    Rarely, this number is divided by two.
    Named after Leonhard Euler applied calculus to fluid dynamics.

    .. math::
        Eu = \frac{\text{Pressure drop}}{2\cdot \text{velocity head}}

    Examples
    --------
    >>> Euler(1E5, 1000., 4)
    6.25

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Eu = dP/(rho*V**2)
    return Eu


def Cavitation(P, Psat, rho, V):
    r'''Calculates Cavitation number or `Ca` for a fluid of velocity `V` with
    a pressure `P`, vapor pressure `Psat`, and density `rho`.

    .. math::
        Ca = \sigma_c = \sigma = \frac{P-P_{sat}}{\frac{1}{2}\rho V^2}

    Parameters
    ----------
    P : float
        Internal pressure of the fluid, [Pa]
    Psat : float
        Vapor pressure of the fluid, [Pa]
    rho : float
        Density of the fluid, [kg/m^3]
    V : float
        Velocity of fluid, [m/s]

    Returns
    -------
    Ca : float
        Cavitation number []

    Notes
    -----
    Used in determining if a flow through a restriction will cavitate.
    Sometimes, the multiplication by 2 will be omitted;

    .. math::
        Ca = \frac{\text{Pressure - Vapor pressure}}
        {\text{Inertial pressure}}

    Examples
    --------
    >>> Cavitation(2E5, 1E4, 1000, 10)
    3.8

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Ca = (P-Psat)/(0.5*rho*V**2)
    return Ca


def Eckert(V, Cp, dT):
    r'''Calculates Eckert number or `Ec` for a fluid of velocity `V` with
    a heat capacity `Cp`, between two temperature given as `dT`.

    .. math::
        Ec = \frac{V^2}{C_p \Delta T}

    Parameters
    ----------
    V : float
        Velocity of fluid, [m/s]
    Cp : float
        Heat capacity of the fluid, [J/kg/K]
    dT : float
        Temperature difference, [K]

    Returns
    -------
    Ec : float
        Eckert number []

    Notes
    -----
    Used in certain heat transfer calculations. Fairly rare.

    .. math::
        Ec = \frac{\text{Kinetic energy} }{ \text{Enthalpy difference}}

    Examples
    --------
    >>> Eckert(10, 2000., 25.)
    0.002

    References
    ----------
    .. [1] Goldstein, Richard J. ECKERT NUMBER. Thermopedia. Hemisphere, 2011.
       10.1615/AtoZ.e.eckert_number
    '''
    Ec = V**2/(Cp*dT)
    return Ec


def Jakob(Cp, Hvap, Te):
    r'''Calculates Jakob number or `Ja` for a boiling fluid with sensible heat
    capacity `Cp`, enthalpy of vaporization `Hvap`, and boiling at `Te` degrees
    above its saturation boiling point.

    .. math::
        Ja = \frac{C_{P}\Delta T_e}{\Delta H_{vap}}

    Parameters
    ----------
    Cp : float
        Heat capacity of the fluid, [J/kg/K]
    Hvap : float
        Enthalpy of vaporization of the fluid at its saturation temperature [J/kg]
    Te : float
        Temperature difference above the fluid's saturation boiling temperature, [K]

    Returns
    -------
    Ja : float
        Jakob number []

    Notes
    -----
    Used in boiling heat transfer analysis. Fairly rare.

    .. math::
        Ja = \frac{\Delta \text{Sensible heat}}{\Delta \text{Latent heat}}

    Examples
    --------
    >>> Jakob(4000., 2E6, 10.)
    0.02

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Ja = Cp*Te/Hvap
    return Ja


def Power_number(P, L, N, rho):
    r'''Calculates power number, `Po`, for an agitator applying a specified
    power `P` with a characteristic length `L`, rotationa speed `N`, to
    a fluid with a specified density `rho`.

    .. math::
        Po = \frac{P}{\rho N^3 D^5}

    Parameters
    ----------
    P : float
        Power applied, [W]
    L : float
        Characteristic length, typically agitator diameter [m]
    N : float
        Speed [revolutions/second]
    rho : float
        Density of fluid, [kg/m^3]

    Returns
    -------
    Po : float
        Power number []

    Notes
    -----
    Used in mixing calculations.

    .. math::
        Po = \frac{\text{Power}}{\text{Rotational inertia}}

    Examples
    --------
    >>> Power_number(P=180, L=0.01, N=2.5, rho=800.)
    144000000.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Po = P/(rho*N**3*L**5)
    return Po


def Drag(F, A, V, rho):
    r'''Calculates drag coefficient `Cd` for a given drag force `F`,
    projected area `A`, characteristic velocity `V`, and density `rho`.

    .. math::
        C_D = \frac{F_d}{A\cdot\frac{1}{2}\rho V^2}

    Parameters
    ----------
    F : float
        Drag force, [N]
    A : float
        Projected area, [m^2]
    V : float
        Characteristic velocity, [m/s]
    rho : float
        Density, [kg/m^3]

    Returns
    -------
    Cd : float
        Drag coefficient, [-]

    Notes
    -----
    Used in flow around objects, or objects flowing within a fluid.

    .. math::
        C_D = \frac{\text{Drag forces}}{\text{Projected area}\cdot
        \text{Velocity head}}

    Examples
    --------
    >>> Drag(1000, 0.0001, 5, 2000)
    400.0

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Cd = F/(A*rho*V**2/2.)
    return Cd


def Capillary(V, mu, sigma):
    r'''Calculates Capillary number `Ca` for a characteristic velocity `V`,
    viscosity `mu`, and surface tension `sigma`.

    .. math::
        Ca = \frac{V \mu}{\sigma}

    Parameters
    ----------
    V : float
        Characteristic velocity, [m/s]
    mu : float
        Dynamic viscosity, [Pa*s]
    sigma : float
        Surface tension, [N/m]

    Returns
    -------
    Ca : float
        Capillary number, [-]

    Notes
    -----
    Used in porous media calculations and film flow calculations.
    Surface tension may gas-liquid, or liquid-liquid.

    .. math::
        Ca = \frac{\text{Viscous forces}}
        {\text{Surface forces}}

    Examples
    --------
    >>> Capillary(1.2, 0.01, .1)
    0.12

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Kundu, Pijush K., Ira M. Cohen, and David R. Dowling. Fluid
       Mechanics. Academic Press, 2012.
    '''
    Ca = V*mu/sigma
    return Ca


def Archimedes(L, rhof, rhop, mu, g=g):
    r'''Calculates Archimedes number, `Ar`, for a fluid and particle with the
    given densities, characteristic length, viscosity, and gravity
    (usually diameter of particle).

    .. math::
        Ar = \frac{L^3 \rho_f(\rho_p-\rho_f)g}{\mu^2}

    Parameters
    ----------
    L : float
        Characteristic length, typically particle diameter [m]
    rhof : float
        Density of fluid, [kg/m^3]
    rhop : float
        Density of particle, [kg/m^3]
    mu : float
        Viscosity of fluid, [N/m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    Ar : float
        Archimedes number []

    Notes
    -----
    Used in fluid-particle interaction calculations.

    .. math::
        Ar = \frac{\text{Gravitational force}}{\text{Viscous force}}

    Examples
    --------
    >>> Archimedes(0.002, 2., 3000, 1E-3)
    470.4053872

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Ar = L**3*rhof*(rhop-rhof)*g/mu**2
    return Ar


def Ohnesorge(L, rho, mu, sigma):
    r'''Calculates Ohnesorge number, `Oh`, for a fluid with the given
    characteristic length, density, viscosity, and surface tension.

    .. math::
         \text{Oh} = \frac{\mu}{\sqrt{\rho \sigma L }}

    Parameters
    ----------
    L : float
        Characteristic length [m]
    rho : float
        Density of fluid, [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*s]
    sigma : float
        Surface tension, [N/m]

    Returns
    -------
    Oh : float
        Ohnesorge number []

    Notes
    -----
    Often used in spray calculations. Sometimes given the symbol Z.

    .. math::
        Oh = \frac{\sqrt{\text{We}}}{\text{Re}}= \frac{\text{viscous forces}}
        {\sqrt{\text{Inertia}\cdot\text{Surface tension}} }

    Examples
    --------
    >>> Ohnesorge(1E-4, 1000., 1E-3, 1E-1)
    0.01

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    Oh = mu/(L*rho*sigma)**0.5
    return Oh

    
def Suratman(L, rho, mu, sigma):
    r'''Calculates Suratman number, `Su`, for a fluid with the given
    characteristic length, density, viscosity, and surface tension.

    .. math::
        \text{Su} = \frac{\rho\sigma L}{\mu^2}

    Parameters
    ----------
    L : float
        Characteristic length [m]
    rho : float
        Density of fluid, [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*s]
    sigma : float
        Surface tension, [N/m]

    Returns
    -------
    Su : float
        Suratman number []

    Notes
    -----
    Also known as Laplace number. Used in two-phase flow, especially the 
    bubbly-slug regime. No confusion regarding the definition of this group 
    has been observed.

    .. math::
        \text{Su} = \frac{\text{Re}^2}{\text{We}} =\frac{\text{Inertia}\cdot
        \text{Surface tension} }{\text{(viscous forces)}^2}

    The oldest reference to this group found by the author is in 1963, from 
    [2]_.

    Examples
    --------
    >>> Suratman(1E-4, 1000., 1E-3, 1E-1)
    10000.0

    References
    ----------
    .. [1] Sen, Nilava. "Suratman Number in Bubble-to-Slug Flow Pattern 
       Transition under Microgravity." Acta Astronautica 65, no. 3-4 (August 
       2009): 423-28. doi:10.1016/j.actaastro.2009.02.013.
    .. [2] Catchpole, John P., and George. Fulford. "DIMENSIONLESS GROUPS." 
       Industrial & Engineering Chemistry 58, no. 3 (March 1, 1966): 46-60. 
       doi:10.1021/ie50675a012.
    '''
    return rho*sigma*L/(mu*mu)

    
def Bejan_L(dP, L, mu, alpha):
    r'''Calculates Bejan number of a length or `Be_L` for a fluid with the
    given parameters flowing over a characteristic length `L` and experiencing
    a pressure drop `dP`.

    .. math::
        Be_L = \frac{\Delta P L^2}{\mu \alpha}

    Parameters
    ----------
    dP : float
        Pressure drop, [Pa]
    L : float
        Characteristic length, [m]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    alpha : float
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Be_L : float
        Bejan number with respect to length []

    Notes
    -----
    Termed a dimensionless number by someone in 1988.

    Examples
    --------
    >>> Bejan_L(1E4, 1, 1E-3, 1E-6)
    10000000000000.0

    References
    ----------
    .. [1] Awad, M. M. "The Science and the History of the Two Bejan Numbers."
       International Journal of Heat and Mass Transfer 94 (March 2016): 101-3.
       doi:10.1016/j.ijheatmasstransfer.2015.11.073.
    .. [2] Bejan, Adrian. Convection Heat Transfer. 4E. Hoboken, New Jersey:
       Wiley, 2013.
    '''
    Be_L = dP*L**2/(alpha*mu)
    return Be_L


def Bejan_p(dP, K, mu, alpha):
    r'''Calculates Bejan number of a permeability or `Be_p` for a fluid with
    the given parameters and a permeability `K` experiencing a pressure drop
    `dP`.

    .. math::
        Be_p = \frac{\Delta P K}{\mu \alpha}

    Parameters
    ----------
    dP : float
        Pressure drop, [Pa]
    K : float
        Permeability, [m^2]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    alpha : float
        Thermal diffusivity, [m^2/s]

    Returns
    -------
    Be_p : float
        Bejan number with respect to pore characteristics []

    Notes
    -----
    Termed a dimensionless number by someone in 1988.

    Examples
    --------
    >>> Bejan_p(1E4, 1, 1E-3, 1E-6)
    10000000000000.0

    References
    ----------
    .. [1] Awad, M. M. "The Science and the History of the Two Bejan Numbers."
       International Journal of Heat and Mass Transfer 94 (March 2016): 101-3.
       doi:10.1016/j.ijheatmasstransfer.2015.11.073.
    .. [2] Bejan, Adrian. Convection Heat Transfer. 4E. Hoboken, New Jersey:
       Wiley, 2013.
    '''
    Be_p = dP*K/(alpha*mu)
    return Be_p


def Boiling(G, q, Hvap):
    r'''Calculates Boiling number or `Bg` using heat flux, two-phase mass flux,
    and heat of vaporization of the fluid flowing. Used in two-phase heat 
    transfer calculations.

    .. math::
        \text{Bg} = \frac{q}{G_{tp} \Delta H_{vap}}

    Parameters
    ----------
    G : float
        Two-phase mass flux in a channel (combined liquid and vapor) [kg/m^2/s]
    q : float
        Heat flux [W/m^2]
    Hvap : float
        Heat of vaporization of the fluid [J/kg]

    Returns
    -------
    Bg : float
        Boiling number [-]

    Notes
    -----
    Most often uses the symbol `Bo` instead of `Bg`, but this conflicts with
    Bond number.
    
    .. math::
        \text{Bg} = \frac{\text{mass liquid evaporated / area heat transfer 
        surface}}{\text{mass flow rate fluid / flow cross sectional area}}
    
    First defined in [4]_, though not named.

    Examples
    --------
    >>> Boiling(300, 3000, 800000)
    1.25e-05

    References
    ----------
    .. [1] Winterton, Richard H.S. BOILING NUMBER. Thermopedia. Hemisphere, 
       2011. 10.1615/AtoZ.b.boiling_number
    .. [2] Collier, John G., and John R. Thome. Convective Boiling and 
       Condensation. 3rd edition. Clarendon Press, 1996.
    .. [3] Stephan, Karl. Heat Transfer in Condensation and Boiling. Translated
       by C. V. Green.. 1992 edition. Berlin; New York: Springer, 2013.
    .. [4] W. F. Davidson, P. H. Hardie, C. G. R. Humphreys, A. A. Markson, 
       A. R. Mumford and T. Ravese "Studies of heat transmission through boiler
       tubing at pressures from 500 to 3300 pounds" Trans. ASME, Vol. 65, 9, 
       February 1943, pp. 553-591. 
    '''
    return q/(G*Hvap)

    
def Dean(Re, Di, D):
    r'''Calculates Dean number, `De`, for a fluid with the Reynolds number `Re`,
    inner diameter `Di`, and a secondary diameter `D`. `D` may be the 
    diameter of curvature, the diameter of a spiral, or some other dimension.

    .. math::
        \text{De} = \sqrt{\frac{D_i}{D}} \text{Re} = \sqrt{\frac{D_i}{D}} 
        \frac{\rho v D}{\mu}
        
    Parameters
    ----------
    Re : float
        Reynolds number []
    Di : float
        Inner diameter []
    D : float
        Diameter of curvature or outer spiral or other dimension []

    Returns
    -------
    De : float
        Dean number [-]

    Notes
    -----
    Used in flow in curved geometry.

    .. math::
        \text{De} = \frac{\sqrt{\text{centripetal forces}\cdot 
        \text{inertial forces}}}{\text{viscous forces}}

    Examples
    --------
    >>> Dean(10000, 0.1, 0.4)
    5000.0

    References
    ----------
    .. [1] Catchpole, John P., and George. Fulford. "DIMENSIONLESS GROUPS." 
       Industrial & Engineering Chemistry 58, no. 3 (March 1, 1966): 46-60. 
       doi:10.1021/ie50675a012.
    '''
    return (Di/D)**0.5*Re


def relative_roughness(D, roughness=1.52e-06):
    r'''Calculates relative roughness `eD` using a diameter and the roughness
    of the material of the wall. Default roughness is that of steel.

    .. math::
        eD=\frac{\epsilon}{D}

    Parameters
    ----------
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    eD : float
        Relative Roughness, [-]

    Examples
    --------
    >>> relative_roughness(0.5, 1E-4)
    0.0002

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    eD = roughness/D
    return eD


### Misc utilities

def nu_mu_converter(rho, mu=None, nu=None):
    r'''Calculates either kinematic or dynamic viscosity, depending on inputs.
    Used when one type of viscosity is known as well as density, to obtain
    the other type. Raises an error if both types of viscosity or neither type
    of viscosity is provided.

    .. math::
        \nu = \frac{\mu}{\rho}

        \mu = \nu\rho

    Parameters
    ----------
    rho : float
        Density, [kg/m^3]
    mu : float, optional
        Dynamic viscosity, [Pa*s]
    nu : float, optional
        Kinematic viscosity, [m^2/s]

    Returns
    -------
    mu or nu : float
        Dynamic viscosity, [Pa*s] or Kinematic viscosity, [m^2/s]

    Examples
    --------
    >>> nu_mu_converter(998., nu=1.0E-6)
    0.000998

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    if (nu and mu) or not rho or (not nu and not mu):
        raise Exception('Inputs must be rho and one of mu and nu.')
    if mu:
        return mu/rho
    elif nu:
        return nu*rho


def gravity(latitude, H):
    r'''Calculates local acceleration due to gravity `g` according to [1]_.
    Uses latitude and height to calculate `g`.

    .. math::
        g = 9.780356(1 + 0.0052885\sin^2\phi - 0.0000059^22\phi)
        - 3.086\times 10^{-6} H

    Parameters
    ----------
    latitude : float
        Degrees, [degrees]
    H : float
        Height above earth's surface [m]

    Returns
    -------
    g : float
        Acceleration due to gravity, [m/s^2]

    Notes
    -----
    Better models, such as EGM2008 exist.

    Examples
    --------
    >>> gravity(55, 1E4)
    9.784151976863571

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    '''
    lat = latitude*pi/180
    g = 9.780356*(1+0.0052885*sin(lat)**2 -0.0000059*sin(2*lat)**2)-3.086E-6*H
    return g

### Friction loss conversion functions

def K_from_f(fd, L, D):
    r'''Calculates loss coefficient, K, for a given section of pipe
    at a specified friction factor.

    .. math::
        K = f_dL/D

    Parameters
    ----------
    fd : float
        friction factor of pipe, []
    L : float
        Length of pipe, [m]
    D : float
        Inner diameter of pipe, [m]

    Returns
    -------
    K : float
        Loss coefficient, []

    Notes
    -----
    For fittings with a specified L/D ratio, use D = 1 and set L to
    specified L/D ratio.

    Examples
    --------
    >>> K_from_f(fd=0.018, L=100., D=.3)
    6.0
    '''
    K = fd*L/D
    return K


def K_from_L_equiv(L_D, fd=0.015):
    r'''Calculates loss coefficient, for a given equivalent length (L/D).

    .. math::
        K = f_d \frac{L}{D}

    Parameters
    ----------
    L_D : float
        Length over diameter, []
    fd : float, optional
        Darcy friction factor, [-]

    Returns
    -------
    K : float
        Loss coefficient, []

    Notes
    -----
    Almost identical to `K_from_f`, but with a default friction factor for
    fully turbulent flow in steel pipes.

    Examples
    --------
    >>> K_from_L_equiv(240.)
    3.5999999999999996
    '''
    K = fd*L_D
    return K


def L_equiv_from_K(K, fd=0.015):
    r'''Calculates equivalent length of pipe (L/D), for a given loss 
    coefficient.

    .. math::
        \frac{L}{D} = \frac{K}{f_d}

    Parameters
    ----------
    K : float
        Loss coefficient, []
    fd : float, optional
        Darcy friction factor, [-]

    Returns
    -------
    L_D : float
        Length over diameter, []

    Notes
    -----
    Assumes a default friction factor for fully turbulent flow in steel pipes.

    Examples
    --------
    >>> L_equiv_from_K(3.6)
    240.00000000000003
    '''
    L_D = K/fd
    return L_D


def dP_from_K(K, rho, V):
    r'''Calculates pressure drop, for a given loss coefficient,
    at a specified density and velocity.

    .. math::
        dP = 0.5K\rho V^2

    Parameters
    ----------
    K : float
        Loss coefficient, []
    rho : float
        Density of fluid, [kg/m^3]
    V : float
        Velocity of fluid in pipe, [m/s]

    Returns
    -------
    dP : float
        Pressure drop, [Pa]

    Notes
    -----
    Loss ciefficient `K` is usually the sum of several factors, including
    the friction factor.

    Examples
    --------
    >>> dP_from_K(K=10, rho=1000, V=3)
    45000.0
    '''
    dP = K*0.5*rho*V**2
    return dP


def head_from_K(K, V, g=g):
    r'''Calculates head loss, for a given loss coefficient,
    at a specified velocity.

    .. math::
        \text{head} = \frac{K V^2}{2g}

    Parameters
    ----------
    K : float
        Loss coefficient, []
    V : float
        Velocity of fluid in pipe, [m/s]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    head : float
        Head loss, [m]

    Notes
    -----
    Loss ciefficient `K` is usually the sum of several factors, including
    the friction factor.

    Examples
    --------
    >>> head_from_K(K=10, V=1.5)
    1.1471807396001694
    '''
    head = K*0.5*V**2/g
    return head


def head_from_P(P, rho, g=g):
    r'''Calculates head for a fluid of specified density at specified
    pressure.

    .. math::
        \text{head} = {P\over{\rho g}}

    Parameters
    ----------
    P : float
        Pressure fluid in pipe, [Pa]
    rho : float
        Density of fluid, [kg/m^3]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    head : float
        Head, [m]

    Notes
    -----
    By definition. Head varies with location, inversely propertional to the
    increase in gravitational constant.

    Examples
    --------
    >>> head_from_P(P=98066.5, rho=1000)
    10.000000000000002
    '''
    head = P/rho/g
    return head


def P_from_head(head, rho, g=g):
    r'''Calculates head for a fluid of specified density at specified
    pressure.

    .. math::
        P = \rho g \cdot \text{head}

    Parameters
    ----------
    head : float
        Head, [m]
    rho : float
        Density of fluid, [kg/m^3]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    P : float
        Pressure fluid in pipe, [Pa]

    Notes
    -----

    Examples
    --------
    >>> P_from_head(head=5., rho=800.)
    39226.6
    '''
    P = head*rho*g
    return P



### Synonyms
alpha = thermal_diffusivity # synonym for thermal diffusivity
Pr = Prandtl # Synonym
