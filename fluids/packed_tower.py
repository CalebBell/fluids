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
from scipy.constants import g, pi
from scipy.optimize import fsolve


__all__ = ['voidage_experimental', 'specific_area_mesh',
'Stichlmair_dry', 'Stichlmair_wet', 'Stichlmair_flood',
'dP_demister_dry_Setekleiv_Svendsen_lit',
'dP_demister_dry_Setekleiv_Svendsen']


### Demister

def dP_demister_dry_Setekleiv_Svendsen(S, voidage, vs, rho, mu, L=1):
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
        Viscosity of fluid [Pa*S]
    L : float, optional
        Length of the demister [m]

    Returns
    -------
    dP : float
        Pressure drop across a dry demister [Pa]

    Notes
    -----
    Useful at startup and in modeling. Dry pressure drop is normally neglible
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
    dP = right*rho*vs**2/voidage**2
    return dP


def dP_demister_dry_Setekleiv_Svendsen_lit(S, voidage, vs, rho, mu, L=1):
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
        Viscosity of fluid [Pa*S]
    L : float, optional
        Length of the demister [m]

    Returns
    -------
    dP : float
        Pressure drop across a dry demister [Pa]

    Notes
    -----
    Useful at startup and in modeling. Dry pressure drop is normally neglible
    compared to wet pressure drop. Coefficients obtained by evolutionary
    programming and may not fit data outside of the limits of the variables.

    Examples
    --------
    >>> dP_demister_dry_Setekleiv_Svendsen_lit(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
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
    dP = right*rho*vs**2/voidage**2
    return dP


def voidage_experimental(m, rho, D, H):
    r'''Calculates voidage of a bed or mesh given an experimental weight and
    fixed density, diameter, and height, as shown in [1]_. The formula is also
    self-evident.

    .. math::
        \epsilon = 1 - \frac{\frac{m_{mesh}}{\frac{\pi}{4}d_{column}
        L_{mesh}}}{\rho_{material}}

    Parameters
    ----------
    m : float
        Mass of mesh or bed particles weighted, [kg]
    rho : float
        Density of solid particles or mesh [kg/m^3]
    D : float
        Diameter of the cylindrical bed [m]
    L : float
        Length of the demister or bed [m]

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
    voidage = 1 - m/(pi/4*D*H)/rho
    return voidage


def specific_area_mesh(voidage, d):
    r'''Calculates the specific area of a wire mesh, as used in demisters or
    filters. Shown in [1]_, and also self-evident and non-emperical.
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
    Should be prefered over manufacturer data. Can also be used to show that
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
    S = 4*(1-voidage)/d
    return S

### Packing


def Stichlmair_dry(Vg, rhog, mug, voidage, specific_area, C1, C2, C3, H=1.):
    r'''Calculates dry pressure drop across a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area.

    Pressure drop is given by:

    .. math::
        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

        d_p = \frac{6(1-\epsilon)}{a}

    Parameters
    ----------
    Vg : float
        Superficial velocity of gas, Q/A [m/s]
    rhog : float
        Density of gas [kg/m^3]
    mug : float
        Viscosity of gas [Pa*S]
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
    ... specific_area=260., C1=32., C2=7, C3=1)
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
    f0 = C1/Re + C2/Re**0.5 + C3
    dP_dry = 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg**2
    return dP_dry


def Stichlmair_wet(Vg, Vl, rhog, rhol, mug, voidage, specific_area, C1, C2, C3, H=1):
    r'''Calculates dry pressure drop across a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area. This model is for irrigated
    columns only.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P_{irr}}{H} = \frac{\Delta P_{dry}}{H}\left(\frac
        {1-\epsilon + h_T}{1-\epsilon}\right)^{(2+c)/3}
        \left(\frac{\epsilon}{\epsilon-h_T}\right)^{4.65}

        h_T = h_0\left[1 + 20\left(\frac{\Delta Pirr}{H\rho_L g}\right)^2\right]

        Fr_L = \frac{V_L^2 a}{g \epsilon^{4.65}}

        h_0 = 0.555 Fr_L^{1/3}

        c = \frac{-C_1/Re_g - C_2/(2Re_g^{0.5})}{f_0}

        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

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
        Viscosity of gas [Pa*S]
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
    539.8768237253518

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
    dp = 6*(1-voidage)/specific_area
    Re = Vg*rhog*dp/mug
    f0 = C1/Re + C2/Re**0.5 + C3
    dP_dry = 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg**2
    c = (-C1/Re - C2/(2*Re**0.5))/f0
    Frl = Vl**2*specific_area/(g*voidage**4.65)
    h0 = 0.555*Frl**(1/3.)
    def to_zero(dP_irr):
        hT = h0*(1 + 20*(dP_irr/H/rhol/g)**2)
        err = dP_dry/H*((1-voidage+hT)/(1-voidage))**((2+c)/3.)*(voidage/(voidage-hT))**4.65 -dP_irr/H
        return err
    dP_irr = float(fsolve(to_zero, dP_dry))
    return dP_irr


def Stichlmair_flood(Vl, rhog, rhol, mug, voidage, specific_area, C1, C2, C3, H=1):
    r'''Calculates gas rate for flooding of a packed column, using the
    Stichlmair [1]_ correlation. Uses three regressed constants for each
    type of packing, and voidage and specific area.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P_{irr}}{H} = \frac{\Delta P_{dry}}{H}\left(\frac
        {1-\epsilon + h_T}{1-\epsilon}\right)^{(2+c)/3}
        \left(\frac{\epsilon}{\epsilon-h_T}\right)^{4.65}

        h_T = h_0\left[1 + 20\left(\frac{\Delta Pirr}{H\rho_L g}\right)^2\right]

        Fr_L = \frac{V_L^2 a}{g \epsilon^{4.65}}

        h_0 = 0.555 Fr_L^{1/3}

        c = \frac{-C_1/Re_g - C_2/(2Re_g^{0.5})}{f_0}

        \Delta P_{dry} = \frac{3}{4} f_0 \frac{1-\epsilon}{\epsilon^{4.65}}
        \rho_G \frac{H}{d_p}V_g^2

        f_0 = \frac{C_1}{Re_g} + \frac{C_2}{Re_g^{0.5}} + C_3

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
        Viscosity of gas [Pa*S]
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
    0.6394323542687361

    References
    ----------
    .. [1] Stichlmair, J., J. L. Bravo, and J. R. Fair. "General Model for
       Prediction of Pressure Drop and Capacity of Countercurrent Gas/liquid
       Packed Columns." Gas Separation & Purification 3, no. 1 (March 1989):
       19-28. doi:10.1016/0950-4214(89)80016-7.
    '''
    def to_zero(inputs):
        Vg, dP_irr = inputs
        dp = 6*(1-voidage)/specific_area
        Re = Vg*rhog*dp/mug
        f0 = C1/Re + C2/Re**0.5 + C3
        dP_dry = 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg**2
        c = (-C1/Re - C2/(2*Re**0.5))/f0
        Frl = Vl**2*specific_area/(g*voidage**4.65)
        h0 = 0.555*Frl**(1/3.)
        hT = h0*(1 + 20*(dP_irr/H/rhol/g)**2)
        err1 = dP_dry/H*((1-voidage+hT)/(1-voidage))**((2+c)/3.)*(voidage/(voidage-hT))**4.65 -dP_irr/H
        term = (dP_irr/(rhol*g*H))**2
        err2 = (1./term - 40*((2+c)/3.)*h0/(1 - voidage + h0*(1 + 20*term))
        - 186*h0/(voidage - h0*(1 + 20*term)))
        return err1, err2
    Vg = float(fsolve(to_zero, [Vl*100., 1000])[0])
    return Vg

#print [Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)]
