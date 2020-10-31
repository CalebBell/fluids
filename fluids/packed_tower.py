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

This module contains correlations and functions for calculating pressure drop
from packings and demisters; separation efficiency of demisters; demister
pressure drop; and demister geometry.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Packing Pressure Drop
---------------------
.. autofunction:: fluids.packed_tower.Robbins
.. autofunction:: fluids.packed_tower.Stichlmair_dry
.. autofunction:: fluids.packed_tower.Stichlmair_wet

Packing Flooding
----------------
.. autofunction:: fluids.packed_tower.Stichlmair_flood

Demister Pressure Drop
----------------------
.. autofunction:: fluids.packed_tower.dP_demister_dry_Setekleiv_Svendsen
.. autofunction:: fluids.packed_tower.dP_demister_dry_Setekleiv_Svendsen_lit
.. autofunction:: fluids.packed_tower.dP_demister_wet_ElDessouky

Demister Separation Efficiency
------------------------------
.. autofunction:: fluids.packed_tower.separation_demister_ElDessouky

Demister Geometry
-----------------
.. autofunction:: fluids.packed_tower.voidage_experimental
.. autofunction:: fluids.packed_tower.specific_area_mesh
"""

from __future__ import division
from math import log, sqrt
from fluids.constants import g, pi
from fluids.numerics import secant, newton_system, solve_2_direct, numpy as np

__all__ = ['voidage_experimental', 'specific_area_mesh',
'Stichlmair_dry', 'Stichlmair_wet', 'Stichlmair_flood', 'Robbins',
'dP_demister_dry_Setekleiv_Svendsen_lit',
'dP_demister_dry_Setekleiv_Svendsen',
'dP_demister_wet_ElDessouky', 'separation_demister_ElDessouky']

### Demister

def dP_demister_dry_Setekleiv_Svendsen(S, voidage, vs, rho, mu, L=1.0):
    r'''Calculates dry pressure drop across a demister, using the
    correlation in [1]_. This model is for dry demisters with no holdup only.

    .. math::
        \frac{\Delta P \epsilon^2}{\rho_f v^2} = 10.29 - \frac{565}
        {69.6SL - (SL)^2 - 779} - \frac{74.9}{160.9 - 4.85SL} + 45.33\left(
        \frac{\mu_f \epsilon S^2 L}{\rho_f v}\right)^{0.75}

    Parameters
    ----------
    S : float
        Specific area of the demister, normally ~250-1000 [m^2/m^3]
    voidage : float
        Voidage of bed of the demister material, normally ~0.98 []
    vs : float
        Superficial velocity of fluid, Q/A [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid [Pa*s]
    L : float, optional
        Length of the demister [m]

    Returns
    -------
    dP : float
        Pressure drop across a dry demister [Pa]

    Notes
    -----
    Useful at startup and in modeling. Dry pressure drop is normally negligible
    compared to wet pressure drop. Coefficients obtained by evolutionary
    programming and may not fit data outside of the limits of the variables.

    Examples
    --------
    >>> dP_demister_dry_Setekleiv_Svendsen(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
    320.3280788941329

    References
    ----------
    .. [1] Setekleiv, A. Eddie, and Hallvard F. Svendsen. "Dry Pressure Drop in
       Spiral Wound Wire Mesh Pads at Low and Elevated Pressures." Chemical
       Engineering Research and Design 109 (May 2016): 141-149.
       doi:10.1016/j.cherd.2016.01.019.
    '''
    term = 10.29 - 565./(69.6*S*L - (S*L)**2 - 779) - 74.9/(160.9 - 4.85*S*L)
    right = term + 45.33*(mu*voidage*S**2*L/rho/vs)**0.75
    return right*rho*vs**2/voidage**2


def dP_demister_dry_Setekleiv_Svendsen_lit(S, voidage, vs, rho, mu, L=1.0):
    r'''Calculates dry pressure drop across a demister, using the
    correlation in [1]_. This model is for dry demisters with no holdup only.
    Developed with literature data included as well as their own experimental
    data.

    .. math::
        \frac{\Delta P \epsilon^2}{\rho_f v^2} = 7.3 - \frac{320}
        {69.6SL - (SL)^2 - 779} - \frac{52.4}{161 - 4.85SL} + 27.2\left(
        \frac{\mu_f \epsilon S^2 L}{\rho_f v}\right)^{0.75}

    Parameters
    ----------
    S : float
        Specific area of the demister, normally ~250-1000 [m^2/m^3]
    voidage : float
        Voidage of bed of the demister material, normally ~0.98 []
    vs : float
        Superficial velocity of fluid, Q/A [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid [Pa*s]
    L : float, optional
        Length of the demister [m]

    Returns
    -------
    dP : float
        Pressure drop across a dry demister [Pa]

    Notes
    -----
    Useful at startup and in modeling. Dry pressure drop is normally negligible
    compared to wet pressure drop. Coefficients obtained by evolutionary
    programming and may not fit data outside of the limits of the variables.

    Examples
    --------
    >>> dP_demister_dry_Setekleiv_Svendsen_lit(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1.0)
    209.083848658307

    References
    ----------
    .. [1] Setekleiv, A. Eddie, and Hallvard F. Svendsen. "Dry Pressure Drop in
       Spiral Wound Wire Mesh Pads at Low and Elevated Pressures." Chemical
       Engineering Research and Design 109 (May 2016): 141-149.
       doi:10.1016/j.cherd.2016.01.019.
    '''
    term = 7.3 - 320./(69.6*S*L - (S*L)**2 - 779) - 52.4/(161 - 4.85*S*L)
    right = term + 27.2*(mu*voidage*S**2*L/rho/vs)**0.75
    return right*rho*vs**2/voidage**2


def dP_demister_wet_ElDessouky(vs, voidage, d_wire, L=1.0):
    r'''Calculates wet pressure drop across a demister, using the
    correlation in [1]_. Uses only their own experimental data.

    .. math::
        \frac{\Delta P}{L} = 0.002357(1-\epsilon)^{0.375798}(V)^{0.81317}
        (d_w)^{-1.56114147}

    Parameters
    ----------
    vs : float
        Superficial velocity of fluid, Q/A [m/s]
    voidage : float
        Voidage of bed of the demister material, normally ~0.98 []
    d_wire : float
        Diameter of mesh wire,[m]
    L : float, optional
        Length of the demister [m]

    Returns
    -------
    dP : float
        Pressure drop across a dry demister [Pa]

    Notes
    -----
    No dependency on the liquid properties is included here. Because of the
    exponential nature of the correlation, the limiting pressure drop as V
    is lowered is 0 Pa. A dry pressure drop correlation should be compared with
    results from this at low velocities, and the larger of the
    two pressure drops used.

    The correlation in [1]_ was presented as follows, with wire diameter in
    units of mm, density in kg/m^3, V in m/s, and dP in Pa/m.

    .. math::
        \Delta P = 3.88178(\rho_{mesh})^{0.375798}(V)^{0.81317}
        (d_w)^{-1.56114147}

    Here, the correlation is converted to base SI units and to use voidage;
    not all demisters are stainless steel as in [1]_. A density of 7999 kg/m^3
    was used in the conversion.

    In [1]_, V ranged from 0.98-7.5 m/s, rho from 80.317-208.16 kg/m^3, depth
    from 100 to 200 mm, wire diameter of 0.2mm to 0.32 mm, and particle
    diameter from 1 to 5 mm.


    Examples
    --------
    >>> dP_demister_wet_ElDessouky(6, 0.978, 0.00032)
    688.9216420105029

    References
    ----------
    .. [1] El-Dessouky, Hisham T, Imad M Alatiqi, Hisham M Ettouney, and Noura
       S Al-Deffeeri. "Performance of Wire Mesh Mist Eliminator." Chemical
       Engineering and Processing: Process Intensification 39, no. 2 (March
       2000): 129-39. doi:10.1016/S0255-2701(99)00033-1.
    '''
    return L*0.002356999643727531*(1-voidage)**0.375798*vs**0.81317*d_wire**-1.56114147


def separation_demister_ElDessouky(vs, voidage, d_wire, d_drop):
    r'''Calculates droplet removal by a demister as a fraction from 0 to 1,
    using the correlation in [1]_. Uses only their own experimental data.

    .. math::
        \eta = 0.85835(d_w)^{-0.28264}(1-\epsilon)^{0.099625}(V)^{0.106878}
        (d_p)^{0.383197}

    Parameters
    ----------
    vs : float
        Superficial velocity of fluid, Q/A [m/s]
    voidage : float
        Voidage of bed of the demister material, normally ~0.98 []
    d_wire : float
        Diameter of mesh wire,[m]
    d_drop : float
        Drop diameter, [m]

    Returns
    -------
    eta : float
        Fraction droplets removed by mass [-]

    Notes
    -----
    No dependency on the liquid properties is included here. Because of the
    exponential nature of the correlation, for smaller diameters separation
    quickly lowers. This correlation can predict a separation larger than 1
    for higher velocities, lower voidages, lower wire diameters, and large
    droplet sizes. This function truncates these larger values to 1.

    The correlation in [1]_ was presented as follows, with wire diameter in
    units of mm, density in kg/m^3, V in m/s, separation in %, and particle
    diameter in mm.

    .. math::
        \eta = 17.5047(d_w)^{-0.28264}(\rho_{mesh})^{0.099625}(V)^{0.106878}
        (d_p)^{0.383197}

    Here, the correlation is converted to base SI units and to use voidage;
    not all demisters are stainless steel as in [1]_. A density of 7999 kg/m^3
    was used in the conversion.

    In [1]_, V ranged from 0.98-7.5 m/s, rho from 80.317-208.16 kg/m^3, depth
    from 100 to 200 mm, wire diameter of 0.2 mm to 0.32 mm, and particle
    diameter from 1 to 5 mm.

    Examples
    --------
    >>> separation_demister_ElDessouky(1.35, 0.974, 0.0002, 0.005)
    0.8982892997640582

    References
    ----------
    .. [1] El-Dessouky, Hisham T, Imad M Alatiqi, Hisham M Ettouney, and Noura
       S Al-Deffeeri. "Performance of Wire Mesh Mist Eliminator." Chemical
       Engineering and Processing: Process Intensification 39, no. 2 (March
       2000): 129-39. doi:10.1016/S0255-2701(99)00033-1.
    '''
    eta = 0.858352355761947*d_wire**-0.28264*(1-voidage)**0.099625*vs**0.106878*d_drop**0.383197
    return min(eta, 1.0)


def voidage_experimental(m, rho, D, H):
    r'''Calculates voidage of a bed or mesh given an experimental weight and
    fixed density, diameter, and height, as shown in [1]_. The formula is also
    self-evident.

    .. math::
        \epsilon = 1 - \frac{\frac{m_{mesh}}{\frac{\pi}{4}d_{column}^2
        L_{mesh}}}{\rho_{material}}

    Parameters
    ----------
    m : float
        Mass of mesh or bed particles weighted, [kg]
    rho : float
        Density of solid particles or mesh [kg/m^3]
    D : float
        Diameter of the cylindrical bed [m]
    H : float
        Height of the demister or bed [m]

    Returns
    -------
    voidage : float
        Voidage of bed of the material []

    Notes
    -----
    Should be trusted over manufacturer data.

    Examples
    --------
    >>> voidage_experimental(m=126, rho=8000, D=1, H=1)
    0.9799464771704212

    References
    ----------
    .. [1] Helsør, T., and H. Svendsen. "Experimental Characterization of
       Pressure Drop in Dry Demisters at Low and Elevated Pressures." Chemical
       Engineering Research and Design 85, no. 3 (2007): 377-85.
       doi:10.1205/cherd06048.
    '''
    return 1 - m/(pi/4*D**2*H)/rho


def specific_area_mesh(voidage, d):
    r'''Calculates the specific area of a wire mesh, as used in demisters or
    filters. Shown in [1]_, and also self-evident and non-empirical.
    Makes the ideal assumption that wires never touch.

    .. math::
        S = \frac{4(1-\epsilon)}{d_{wire}}

    Parameters
    ----------
    voidage : float
        Voidage of the mesh []
    d : float
        Diameter of the wires making the mesh, [m]

    Returns
    -------
    S : float
        Specific area of the mesh [m^2/m^3]

    Notes
    -----
    Should be preferred over manufacturer data. Can also be used to show that
    manufacturer's data is inconsistent with their claimed voidage and wire
    diameter.

    Examples
    --------
    >>> specific_area_mesh(voidage=.934, d=3e-4)
    879.9999999999994

    References
    ----------
    .. [1] Helsør, T., and H. Svendsen. "Experimental Characterization of
       Pressure Drop in Dry Demisters at Low and Elevated Pressures." Chemical
       Engineering Research and Design 85, no. 3 (2007): 377-85.
       doi:10.1205/cherd06048.
    '''
    return 4*(1-voidage)/d

### Packing


def Stichlmair_dry(Vg, rhog, mug, voidage, specific_area, C1, C2, C3, H=1.):
    r'''Calculates dry pressure drop across a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area.

    Pressure drop is given by:

    .. math::
        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

    .. math::
        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

    .. math::
        d_p = \frac{6(1-\epsilon)}{a}

    Parameters
    ----------
    Vg : float
        Superficial velocity of gas, Q/A [m/s]
    rhog : float
        Density of gas [kg/m^3]
    mug : float
        Viscosity of gas [Pa*s]
    voidage : float
        Voidage of bed of packing material []
    specific_area : float
        Specific area of the packing material [m^2/m^3]
    C1 : float
        Packing-specific constant []
    C2 : float
        Packing-specific constant []
    C3 : float
        Packing-specific constant []
    H : float, optional
        Height of packing [m]

    Returns
    -------
    dP_dry : float
        Pressure drop across dry packing [Pa]

    Notes
    -----
    This model is used by most process simulation tools. If H is not provided,
    it defaults to 1. If Z is not provided, it defaults to 1.

    Examples
    --------
    >>> Stichlmair_dry(Vg=0.4, rhog=5., mug=5E-5, voidage=0.68,
    ... specific_area=260., C1=32., C2=7.0, C3=1.0)
    236.80904286559885

    References
    ----------
    .. [1] Stichlmair, J., J. L. Bravo, and J. R. Fair. "General Model for
       Prediction of Pressure Drop and Capacity of Countercurrent Gas/liquid
       Packed Columns." Gas Separation & Purification 3, no. 1 (March 1989):
       19-28. doi:10.1016/0950-4214(89)80016-7.
    '''
    dp = 6*(1-voidage)/specific_area
    Re = Vg*rhog*dp/mug
    f0 = C1/Re + C2/sqrt(Re) + C3
    return 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg**2


def _Stichlmair_wet_err(dP_irr, h0, c1, dP_dry, H, voidage, c):
    hT = h0*(1.0 + 20.0*dP_irr*dP_irr*c1)
    err = dP_dry/H*((1-voidage+hT)/(1.0 - voidage))**((2.0 + c)/3.)*(voidage/(voidage-hT))**4.65 -dP_irr/H
    return err

def Stichlmair_wet(Vg, Vl, rhog, rhol, mug, voidage, specific_area, C1, C2, C3, H=1.0):
    r'''Calculates dry pressure drop across a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area. This model is for irrigated
    columns only.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P_{irr}}{H} = \frac{\Delta P_{dry}}{H}\left(\frac
        {1-\epsilon + h_T}{1-\epsilon}\right)^{(2+c)/3}
        \left(\frac{\epsilon}{\epsilon-h_T}\right)^{4.65}

    .. math::
        h_T = h_0\left[1 + 20\left(\frac{\Delta Pirr}{H\rho_L g}\right)^2\right]

    .. math::
        Fr_L = \frac{V_L^2 a}{g \epsilon^{4.65}}

    .. math::
        h_0 = 0.555 Fr_L^{1/3}

    .. math::
        c = \frac{-C_1/Re_g - C_2/(2Re_g^{0.5})}{f_0}

    .. math::
        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

    .. math::
        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

    .. math::
        d_p = \frac{6(1-\epsilon)}{a}

    Parameters
    ----------
    Vg : float
        Superficial velocity of gas, Q/A [m/s]
    Vl : float
        Superficial velocity of liquid, Q/A [m/s]
    rhog : float
        Density of gas [kg/m^3]
    rhol : float
        Density of liquid [kg/m^3]
    mug : float
        Viscosity of gas [Pa*s]
    voidage : float
        Voidage of bed of packing material []
    specific_area : float
        Specific area of the packing material [m^2/m^3]
    C1 : float
        Packing-specific constant []
    C2 : float
        Packing-specific constant []
    C3 : float
        Packing-specific constant []
    H : float, optional
        Height of packing [m]

    Returns
    -------
    dP : float
        Pressure drop across irrigated packing [Pa]

    Notes
    -----
    This model is used by most process simulation tools. If H is not provided,
    it defaults to 1. If Z is not provided, it defaults to 1.
    A numerical solver is used and needed by this model. Its initial guess
    is the dry pressure drop. Convergence problems may occur.
    The model as described in [1]_ appears to have a typo, and could not match
    the example. As described in [2]_, however, the model works.

    Examples
    --------
    Example is from [1]_, matches.

    >>> Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5,
    ... voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    539.876823725352

    References
    ----------
    .. [1] Stichlmair, J., J. L. Bravo, and J. R. Fair. "General Model for
       Prediction of Pressure Drop and Capacity of Countercurrent Gas/liquid
       Packed Columns." Gas Separation & Purification 3, no. 1 (March 1989):
       19-28. doi:10.1016/0950-4214(89)80016-7.
    .. [2] Piche, Simon R., Faical Larachi, and Bernard P. A. Grandjean.
       "Improving the Prediction of Irrigated Pressure Drop in Packed
       Absorption Towers." The Canadian Journal of Chemical Engineering 79,
       no. 4 (August 1, 2001): 584-94. doi:10.1002/cjce.5450790417.
    '''
    dp = 6.0*(1.0 - voidage)/specific_area
    Re = Vg*rhog*dp/mug
    f0 = C1/Re + C2/sqrt(Re) + C3
    dP_dry = 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg*Vg
    c = (-C1/Re - C2/(2*sqrt(Re)))/f0
    Frl = Vl**2*specific_area/(g*voidage**4.65)
    h0 = 0.555*Frl**(1/3.)

    c1 = 1.0/(H*rhol*g)
    c1 *= c1
    return secant(_Stichlmair_wet_err, dP_dry, args=(h0, c1, dP_dry, H, voidage, c))


def _Stichlmair_flood_f(inputs, Vl, rhog, rhol, mug, voidage, specific_area,
                        C1, C2, C3, H):
    """Internal function which calculates the errors of the two Stichlmair
    objective functions, and their jacobian."""
    Vg, dP_irr = float(inputs[0]), float(inputs[1])
    dp = 6.0*(1.0 - voidage)/specific_area
    Re = Vg*rhog*dp/mug
    f0 = C1/Re + C2/sqrt(Re) + C3
    dP_dry = 0.75*f0*(1.0 - voidage)/voidage**4.65*rhog*H/dp*Vg*Vg
    c = (-C1/Re - 0.5*C2*1.0/sqrt(Re))/f0
    Frl = Vl*Vl*specific_area/(g*voidage**4.65)
    h0 = 0.555*Frl**(1/3.)
    hT = h0*(1.0 + 20.0*(dP_irr/H/rhol/g)**2)
    err1 = dP_dry/H*((1.0 - voidage + hT)/(1.0 - voidage))**((2.0 + c)/3.)*(voidage/(voidage-hT))**4.65 - dP_irr/H
    term = (dP_irr/(rhol*g*H))**2
    err2 = (1./term - 40.0*((2.0+c)/3.)*h0/(1.0 - voidage + h0*(1.0 + 20.0*term))
    - 186.0*h0/(voidage - h0*(1.0 + 20.0*term)))
    return err1, err2

def _Stichlmair_flood_f_and_jac(inputs, Vl, rhog, rhol, mug, voidage,
                                specific_area, C1, C2, C3, H):
    """Internal function which calculates the errors of the two Stichlmair
    objective functions, and their jacobian.

    Derived using SymPy on the main flooding function.
    """
    Vg, dP_irr = inputs[0], inputs[1]
    x0 = 1.0/H
    x1 = Vg*Vg
    x2 = voidage**(-4.65)
    x3 = specific_area*x2
    x4 = Vl*Vl*x3/g
    x5 = x4**0.333333333333333
    x6 = dP_irr*dP_irr
    x7 = H*H
    x8 = 1.0/x7
    x9 = g*g
    x10 = 1.0/x9
    x11 = rhol*rhol
    x12 = 1.0/x11
    x13 = x5*(20.0*x10*x12*x6*x8 + 1.0)
    x14 = 0.555*x13
    x15 = (voidage/(voidage - x14))**4.65
    x16 = 1.0/Vg
    x17 = 1.0/rhog
    x18 = voidage - 1.0
    x19 = 1.0/x18
    x20 = C1*mug*specific_area*x16*x17*x19
    x21 = 2.44948974278318*C2
    x22 = Vg*rhog/(mug*specific_area)
    x23 = x21*1.0/sqrt(-x18*x22)
    x24 = 6.0*C3 - x20 + x23
    x25 = 1.0 - voidage
    x26 = x14 + x25
    x27 = -x19*x26
    x28 = 2.0*C1*mug*specific_area*x16*x17/x25 + x21*1.0/sqrt(x22*x25)
    x29 = 1.0/x24
    x30 = x28*x29
    x31 = x27**(-0.166666666666667*x30 + 0.666666666666667)
    x32 = x11*x7*x9
    x33 = 200.0*voidage
    x34 = 111.0*x13
    x35 = x33 - x34
    x36 = 1.0/x35
    x37 = -x33 + x34 + 200.0
    x38 = 1.0/x37
    x39 = 2.0*x20
    x40 = -4.0*x20 + x23 + x29*(-x23 + x39)*(x23 - x39)
    x41 = dP_irr*rhog*specific_area*x0*x1*x10*x12*x15*x2*x24*x31
    x42 = dP_irr*x10*x12*x4**0.666666666666667*x8

    F1, F2, dF1_dVg, dF2_dVg, dF1_dP_irr, dF2_dP_irr = (
            -dP_irr*x0 + 0.0208333333333333*rhog*specific_area*x1*x15*x2*x24*x31,
             x32/x6 - 20646.0*x36*x5 - x38*x5*(2960.0 - 740.0*x28*x29),
             0.00173611111111111*Vg*rhog*x15*x3*x31*(144.0*C3 - 12.0*x20 + 18.0*x23 + x40*log(x27)),
             x0*(430.125*x36*x41*x5 - 15.4166666666667*x38*x41*x5*(x30 - 4.0) - 1.0),
             -1.85*x16*x29*x40*x5/x26,
             3285600.0*x42*(-x30 + 4.0)*x38*x38- 91668240.0*x42*x36*x36 - 2.0*x32/(dP_irr*x6))

    err = [0.0]*2
    err[0] = F1
    err[1] = F2

    jac = [[dF1_dVg, dF2_dVg], [dF1_dP_irr, dF2_dP_irr]]# numba: delete
#    jac = np.array([[dF1_dVg, dF2_dVg], [dF1_dP_irr, dF2_dP_irr]]) # numba: uncomment
    return err, jac



def Stichlmair_flood(Vl, rhog, rhol, mug, voidage, specific_area, C1, C2, C3,
                     H=1.0):
    r'''Calculates gas rate for flooding of a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P_{irr}}{H} = \frac{\Delta P_{dry}}{H}\left(\frac
        {1-\epsilon + h_T}{1-\epsilon}\right)^{(2+c)/3}
        \left(\frac{\epsilon}{\epsilon-h_T}\right)^{4.65}

    .. math::
        h_T = h_0\left[1 + 20\left(\frac{\Delta Pirr}{H\rho_L g}\right)^2\right]

    .. math::
        Fr_L = \frac{V_L^2 a}{g \epsilon^{4.65}}

    .. math::
        h_0 = 0.555 Fr_L^{1/3}

    .. math::
        c = \frac{-C_1/Re_g - C_2/(2Re_g^{0.5})}{f_0}

    .. math::
        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

    .. math::
        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

    .. math::
        d_p = \frac{6(1-\epsilon)}{a}

    Parameters
    ----------
    Vl : float
        Superficial velocity of liquid, Q/A [m/s]
    rhog : float
        Density of gas [kg/m^3]
    rhol : float
        Density of liquid [kg/m^3]
    mug : float
        Viscosity of gas [Pa*s]
    voidage : float
        Voidage of bed of packing material []
    specific_area : float
        Specific area of the packing material [m^2/m^3]
    C1 : float
        Packing-specific constant []
    C2 : float
        Packing-specific constant []
    C3 : float
        Packing-specific constant []
    H : float, optional
        Height of packing [m]

    Returns
    -------
    Vg : float
        Superficial velocity of gas, Q/A [m/s]

    Notes
    -----
    A numerical solver is used to solve this model.

    Examples
    --------
    Example is from [1]_, matches.

    >>> Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5,
    ... voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    0.6394323542746928

    References
    ----------
    .. [1] Stichlmair, J., J. L. Bravo, and J. R. Fair. "General Model for
       Prediction of Pressure Drop and Capacity of Countercurrent Gas/liquid
       Packed Columns." Gas Separation & Purification 3, no. 1 (March 1989):
       19-28. doi:10.1016/0950-4214(89)80016-7.
    '''
    guess = [0.0]*2
    guess[0] = Vl*100.0
    guess[1] = 1000.0
    return newton_system(_Stichlmair_flood_f_and_jac, x0=guess, jac=True,
                         args=(Vl, rhog, rhol, mug, voidage, specific_area, C1,
                         C2, C3, H), ytol=1e-11, solve_func=solve_2_direct)[0][0]


def Robbins(L, G, rhol, rhog, mul, H=1.0, Fpd=24.0):
    r'''Calculates pressure drop across a packed column, using the Robbins
    equation.

    Pressure drop is given by:

    .. math::
        \Delta P = C_3 G_f^2 10^{C_4L_f}+0.4[L_f/20000]^{0.1}[C_3G_f^210^{C_4L_f}]^4

    .. math::
        G_f=G[0.075/\rho_g]^{0.5}[F_{pd}/20]^{0.5}=986F_s[F_{pd}/20]^{0.5}

    .. math::
        L_f=L[62.4/\rho_L][F_{pd}/20]^{0.5}\mu^{0.1}

    .. math::
        F_s=V_s\rho_g^{0.5}

    Parameters
    ----------
    L : float
        Specific liquid mass flow rate [kg/s/m^2]
    G : float
        Specific gas mass flow rate [kg/s/m^2]
    rhol : float
        Density of liquid [kg/m^3]
    rhog : float
        Density of gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    H : float
        Height of packing [m]
    Fpd : float
        Robbins packing factor (tabulated for packings) [1/ft]

    Returns
    -------
    dP : float
        Pressure drop across packing [Pa]

    Notes
    -----
    Perry's displayed equation has a typo in a superscript.
    This model is based on the example in Perry's.

    Examples
    --------
    >>> Robbins(L=12.2, G=2.03, rhol=1000., rhog=1.1853, mul=0.001, H=2.0, Fpd=24.0)
    619.6624593438102

    References
    ----------
    .. [1] Robbins [Chem. Eng. Progr., p. 87 (May 1991) Improved Pressure Drop
       Prediction with a New Correlation.
    '''
    # Convert SI units to imperial for use in correlation
    L = L*737.33812 # kg/s/m^2 to lb/hr/ft^2
    G = G*737.33812 # kg/s/m^2 to lb/hr/ft^2
    rhol = rhol*0.062427961 # kg/m^3 to lb/ft^3
    rhog = rhog*0.062427961 # kg/m^3 to lb/ft^3
    mul = mul*1000.0 # Pa*s to cP

    C3 = 7.4E-8
    C4 = 2.7E-5
    Fpd_root_term = sqrt(.05*Fpd)
    Lf = L*(62.4/rhol)*Fpd_root_term*mul**0.1
    Gf = G*sqrt(0.075/rhog)*Fpd_root_term
    Gf2 = Gf*Gf
    C4LF_10_GF2_C3 = C3*Gf2*10.0**(C4*Lf)
    C4LF_10_GF2_C3_2 = C4LF_10_GF2_C3*C4LF_10_GF2_C3
    dP = C4LF_10_GF2_C3 + 0.4*(5e-5*Lf)**0.1*(C4LF_10_GF2_C3_2*C4LF_10_GF2_C3_2)
    return dP*817.22083*H # in. H2O to Pa/m
