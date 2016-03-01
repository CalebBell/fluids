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

__all__ = ['Ergun', 'Kuo_Nydegger', 'Jones_Krier', 'Carman', 'Hicks',
           'Brauer', 'Montillet', 'Idelchik', 'voidage_Benyahia_Oneil',
           'voidage_Benyahia_Oneil_spherical', 'voidage_Benyahia_Oneil_cylindrical']


def Ergun(Dp, voidage=0.4, sphericity=1, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed, using the famous Ergun
    equation.

    Pressure drop is given by:

    .. math::
        \Delta p=\frac{150\mu (1-\epsilon)^2 V_s L}{\epsilon^3 (\Phi D_p)^2 }
        + \frac{1.75 (1-\epsilon) \rho V_s^2 L}{\epsilon^3 (\Phi D_p)}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    sphericity : float
        Sphericity of particles in bed []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    The first term in this equation represents laminar loses, and the second,
    turbulent loses. Sphericity must be calculated.
    According to [1]_, developed using spheres, pulverized coke/coal, sand,
    cylinders and tablets for ranges of :math:`1 < RE_{ERg} <2300`. [1]_ cites
    a source claiming it should not be used above 500.

    Examples
    --------
    >>> # Custom example
    >>> Ergun(Dp=0.0008, voidage=0.4, sphericity=1., H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    892.3013355913797

    References
    ----------
    .. [1] Ergun, S. (1952) 'Fluid flow through packed columns',
       Chem. Eng. Prog., 48, 89-94.
    '''
    dP_laminar = 150*mu*(1.-voidage)**2*vs*H/voidage**3/(Dp*sphericity)**2
    dP_turbulent = 1.75*(1-voidage)*rho*vs**2*H/voidage**3/(Dp*sphericity)
    dP = dP_laminar + dP_turbulent
    return dP

#print Ergun(Dp=0.0008, voidage=0.4, sphericity=1., H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)

def Kuo_Nydegger(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
         \frac{\Delta P}{L} \frac{D_p^2}{\mu v_{s}}\left(\frac{\phi}{1-\phi}
         \right)^2 = 276 + 5.05 Re^{0.87}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    Does not share exact form with the Ergun equation.
    767 < Re_{ergun} < 24330

    Examples
    --------
    Custom example, outside lower limit of Re (Re = 1):

    >>> Kuo_Nydegger(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    1658.3187666274648

    Re = 4000 custom example:

    >>> Kuo_Nydegger(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    241.56063171630015

    References
    ----------
    .. [1] Kuo, K. K. and Nydegger, C., "Flow Resistance Measurement and
       Correlation in Packed Beds of WC 870 Ball Propellants," Journal of
       Ballistics , Vol. 2, No. 1, pp. 1-26, 1978.
    .. [2] Jones, D. P., and H. Krier. "Gas Flow Resistance Measurements
       Through Packed Beds at High Reynolds Numbers." Journal of Fluids
       Engineering 105, no. 2 (June 1, 1983): 168-172. doi:10.1115/1.3240959.
    '''
    vs = vs/voidage
    Re = rho*vs*Dp/mu*voidage
    Fv = 276.23 + 5.05*(Re/(1-voidage))**0.87
    others = Dp**2/H/mu/vs*(voidage/(1-voidage))**2
    dP = Fv/others
    return dP

#print [Kuo_Nydegger(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]

def Jones_Krier(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [1]_.

    Pressure drop is given by:

    .. math::
         \frac{\Delta P}{L} \frac{D_p^2}{\mu v_{s}}\left(\frac{\phi}{1-\phi}
         \right)^2 = 150 + 1.89 Re^{0.87}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    Does not share exact form with the Ergun equation.
    733 < Re_{ergun} < 126670

    Examples
    --------
    Custom example, outside lower limit of Re (Re = 1):

    >>> Jones_Krier(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    911.494265366317

    Re = 4000 custom example:

    >>> Jones_Krier(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    184.69401245425462

    References
    ----------
    .. [1] Jones, D. P., and H. Krier. "Gas Flow Resistance Measurements
       Through Packed Beds at High Reynolds Numbers." Journal of Fluids
       Engineering 105, no. 2 (June 1, 1983): 168-172. doi:10.1115/1.3240959.
    '''
    vs = vs/voidage
    Re = rho*vs*Dp/mu*voidage
    Fv = 150. + 3.89*(Re/(1-voidage))**0.87
    others = Dp**2/H/mu/vs*(voidage/(1-voidage))**2
    dP = Fv/others
    return dP

#print [Jones_Krier(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Jones_Krier(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]
#
#def ergun_2(Dp, voidage=0.4, sphericity=1, H=None, vs=None, rho=None, mu=None):
#    vs = vs/voidage
#    Re = rho*vs*Dp/mu*voidage
#    Fv = 150 + 1.75*Re/(1-voidage)
#    others = Dp**2/H/mu/vs*(voidage/(1-voidage))**2
#    dP = Fv/others
#    return dP
#print ergun_2(Dp=0.0008, voidage=0.4, sphericity=1., H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)


def Carman(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p \frac{\epsilon^3}{(1-\epsilon)}
        = \frac{180}{Re_{Erg}} + \frac{2.87}{Re_{Erg}^{0.1}}

        Re_{Erg} = \frac{\rho v_s  D_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`0.1 < RE_{ERg} <60000`

    Examples
    --------
    Custom example:

    >>> Carman(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    1077.0587868633704

    Re = 4000 custom example:

    >>> Carman(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    178.2490332160841

    References
    ----------
    .. [1] P.C. Carman, Fluid flow through granular beds, Transactions of the
       London Institute of Chemical Engineers 15 (1937) 150-166.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*Dp/mu/(1-voidage)
    right = 180./Re + 2.87/Re**0.1
    left = Dp/H/rho/vs**2*voidage**3/(1-voidage)
    dP = right/left
    return dP

#print [Carman(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Carman(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]

def Hicks(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p \frac{\epsilon^3}{(1-\epsilon)}
        = \frac{6.8}{Re_{Erg}^{0.2}}

        Re_{Erg} = \frac{\rho v_s  D_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`300 < RE_{ERg} <60000`

    Examples
    --------
    Custom example, outside lower limit of Re (Re = 1):

    >>> Hicks(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    62.534899706155834

    Re = 4000 custom example:

    >>> Hicks(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    171.20579747453397

    References
    ----------
    .. [1] Hicks, R. E. "Pressure Drop in Packed Beds of Spheres." Industrial
       Engineering Chemistry Fundamentals 9, no. 3 (August 1, 1970): 500-502.
       doi:10.1021/i160035a032.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*Dp/mu/(1-voidage)
    right = 6.8/Re**0.2
    left = Dp/H/rho/vs**2*voidage**3/(1-voidage)
    dP = right/left
    return dP

#print [Hicks(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Hicks(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]


def Brauer(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p \frac{\epsilon^3}{(1-\epsilon)} =
        \frac{160}{Re_{Erg}} + \frac{3.1}{Re_{Erg}^{0.1}}

        Re_{Erg} = \frac{\rho v_s  D_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`0.01 < RE_{ERg} <40000`

    Examples
    --------
    Custom example:

    >>> Brauer(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    962.7294566294247

    Re = 4000 custom example:

    >>> Brauer(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    191.77738833880164

    References
    ----------
    .. [1] H. Brauer, Grundlagen der Einphasen -und Mehrphasenstromungen,
       Sauerlander AG, Aarau, 1971.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*Dp/mu/(1-voidage)
    right = 160./Re + 3.1/Re**0.1
    left = Dp/H/rho/vs**2*voidage**3/(1-voidage)
    dP = right/left
    return dP


#print [Brauer(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Brauer(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]

def Montillet(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None, Dc=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p \frac{\epsilon^3}{(1-\epsilon)} = a\left(\frac{D_c}{D_p}\right)^{0.20}
        \left(\frac{1000}{Re_{p}} + \frac{60}{Re_{p}^{0.5}} + 12 \right)

        Re_{p} = \frac{\rho v_s  D_p}{\mu}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]
    Dc : float, optional
        Diameter of the column, [m]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`10 < REp <2500`
    if Dc/D > 50, set to 2.2.
    a = 0.061 for epsilon < 0.4, 0.050 for > 0.4.

    Examples
    --------
    Custom example:

    >>> Montillet(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    1148.1905244077548

    Re = 4000 custom example:

    >>> Montillet(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    212.67409611116554

    References
    ----------
    .. [1] Montillet, A., E. Akkari, and J. Comiti. "About a Correlating
       Equation for Predicting Pressure Drops through Packed Beds of Spheres
       in a Large Range of Reynolds Numbers." Chemical Engineering and
       Processing: Process Intensification 46, no. 4 (April 2007): 329-33.
       doi:10.1016/j.cep.2006.07.002.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*Dp/mu
    if voidage < 0.4:
        a = 0.061
    else:
        a = 0.05
    if not Dc or Dc/Dp > 50:
        Dterm = 2.2
    else:
        Dterm = (Dc/Dp)**0.2
    right = a*Dterm*(1000./Re + 60/Re**0.5 + 12)
    left = Dp/H/rho/vs**2*voidage**3/(1-voidage)
    dP = right/left
    return dP


#print [Montillet(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Montillet(Dp=0.08, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]


def Idelchik(Dp, voidage=0.4, H=None, vs=None, rho=None, mu=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    Pressure drop is given by:

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p  = \frac{0.765}{\epsilon^{4.2}}
        \left(\frac{30}{Re_l} + \frac{3}{Re_l^{0.7}} + 0.3\right)

        Re_l = (0.45/\epsilon^{0.5})Re_{Erg}

        Re_{Erg} = \frac{\rho v_s  D_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    Dp : float
        Particle diameter [m]
    voidage : float
        Void fraction of bed packing []
    H : float
        Height of packed bed [m]
    vs : float
        Superficial velocity of fluid [m/s]
    rho : float
        Density of fluid [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*S]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`0.001 < RE_{ERg} <1000`
    This equation is valid for void fractions between 0.3 and 0.8. Cited as
    by Bernshtein. This model is likely presented in [2]_ with a typo, as it
    varries greatly from other models.

    Examples
    --------
    Custom example, outside lower limit of Re (Re = 1):

    >>> Idelchik(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    56.13431404382615

    Re = 400 custom example:

    >>> Idelchik(Dp=0.008, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    120.55068459098145

    References
    ----------
    .. [1] Idelchik, I. E. Flow Resistance: A Design Guide for Engineers.
       Hemisphere Publishing Corporation, New York, 1989.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*Dp/mu/(1-voidage)
    Re = (0.45/voidage**0.5)*Re
    right = 0.765/voidage*(30./Re + 3./Re**0.7 + 0.3)
    left = Dp/H/rho/vs**2
    dP = right/left
    return dP


#print [Idelchik(Dp=0.0008, voidage=0.4, H=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)]
#print [Idelchik(Dp=0.008, voidage=0.4, H=0.5, vs=0.05, rho=1000., mu=1.00E-003)]
#

### Voidage correlations

def voidage_Benyahia_Oneil(Dpe, Dt, sphericity):
    r'''Calculates voidage of a bed of arbitraryily shaped uniform particles
    packed into a bed or tube of diameter `Dt`, with equivalent sphere diameter
    `Dp`. Shown in [1]_, and cited by various authors. Correlations exist
    also for spheres, solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.1504 + \frac{0.2024}{\phi} + \frac{1.0814}
        {\left(\frac{d_{t}}{d_{pe}}+0.1226\right)^2}

    Parameters
    ----------
    Dpe : float
        Equivalent spherical particle diameter, [m]
    Dt : float
        Diameter of the tube, [m]
    sphericity : float
        Sphericity of particles in bed []

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error of 5.2%; valid 1.5 < dtube/dp < 50 and 0.42 < sphericity < 1

    Examples
    --------
    >>> voidage_Benyahia_Oneil(1E-3, 1E-2, .8)
    0.41395363849210065

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    voidage = 0.1504 + 0.2024/sphericity + 1.0814/(Dt/Dpe + 0.1226)**2
    return voidage


def voidage_Benyahia_Oneil_spherical(Dp, Dt):
    r'''Calculates voidage of a bed of spheres
    packed into a bed or tube of diameter `Dt`, with sphere diameters
    `Dp`. Shown in [1]_, and cited by various authors. Correlations exist
    also for solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.390+\frac{1.740}{\left(\frac{d_{cyl}}{d_p}+1.140\right)^2}

    Parameters
    ----------
    Dp : float
        Spherical particle diameter, [m]
    Dt : float
        Diameter of the tube, [m]

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error 1.5%, 1.5 < ratio < 50.

    Examples
    --------
    >>> voidage_Benyahia_Oneil_spherical(.001, .05)
    0.3906653157443224

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    voidage = 0.390 + 1.740/(Dt/Dp + 1.140)**2
    return voidage


def voidage_Benyahia_Oneil_cylindrical(Dpe, Dt, sphericity):
    r'''Calculates voidage of a bed of cylindrical uniform particles
    packed into a bed or tube of diameter `Dt`, with equivalent sphere diameter
    `Dpe`. Shown in [1]_, and cited by various authors. Correlations exist
    also for spheres, solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.373+\frac{1.703}{\left(\frac{d_{cyl}}{d_p}+0.611\right)^2}

    Parameters
    ----------
    Dpe : float
        Equivalent spherical particle diameter, [m]
    Dt : float
        Diameter of the tube, [m]
    sphericity : float
        Sphericity of particles in bed []

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error 0.016%; 1.7 < ratio < 26.3.

    Examples
    --------
    >>> voidage_Benyahia_Oneil_cylindrical(.01, .1, .6)
    0.38812523109607894

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    voidage = 0.373 + 1.703/(Dt/Dpe + 0.611)**2
    return voidage
