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

from scipy.constants import g
from scipy.optimize import fsolve

__all__ = ['Stichlmair_dry', 'Stichlmair_wet', 'Stichlmair_flood']


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

#print [dP_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)]


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
    A numerical solver is used in loops in in this model. The model is not
    well-behaved, and the solver nurmally provides an error. Also,
    there may be multiple solutions. This problem is well-suited to being
    investigated in a CAS environment.

    Examples
    --------
    Example is from [1]_, matches.

    >>> Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5,
    ... voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    0.6394380360583576

    References
    ----------
    .. [1] Stichlmair, J., J. L. Bravo, and J. R. Fair. "General Model for
       Prediction of Pressure Drop and Capacity of Countercurrent Gas/liquid
       Packed Columns." Gas Separation & Purification 3, no. 1 (March 1989):
       19-28. doi:10.1016/0950-4214(89)80016-7.
    '''
    dp = 6*(1-voidage)/specific_area
    def to_zero(Vg):
        dp = 6*(1-voidage)/specific_area
        Re = Vg*rhog*dp/mug
        f0 = C1/Re + C2/Re**0.5 + C3
        dP_dry = 3/4.*f0*(1-voidage)/voidage**4.65*rhog*H/dp*Vg**2
        c = (-C1/Re - C2/(2*Re**0.5))/f0
        Frl = Vl**2*specific_area/(g*voidage**4.65)
        h0 = 0.555*Frl**(1/3.)
        def to_zero_wet(dP_irr):
            hT = h0*(1 + 20*(dP_irr/H/rhol/g)**2)
            err = dP_dry/H*((1-voidage+hT)/(1-voidage))**((2+c)/3.)*(voidage/(voidage-hT))**4.65 -dP_irr/H
            return err
        dP_irr = float(fsolve(to_zero_wet, dP_dry))
        term = (dP_irr/(rhol*g*H))**2
        err = (1./term - 40*((2+c)/3.)*h0/(1 - voidage + h0*(1 + 20*term))
        - 186*h0/(voidage - h0*(1 + 20*term)))
        return err
    Vg = float(fsolve(to_zero, Vl*100.))
    return Vg
#print [Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)]
