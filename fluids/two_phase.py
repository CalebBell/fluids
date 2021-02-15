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

This module contains functions for calculating two-phase pressure drop. It also
contains correlations for flow regime.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.


.. contents:: :local:

Interfaces
----------
.. autofunction:: two_phase_dP
.. autofunction:: two_phase_dP_methods
.. autofunction:: two_phase_dP_acceleration
.. autofunction:: two_phase_dP_gravitational
.. autofunction:: two_phase_dP_dz_acceleration
.. autofunction:: two_phase_dP_dz_gravitational

Two Phase Pressure Drop Correlations
------------------------------------
.. autofunction:: Beggs_Brill
.. autofunction:: Lockhart_Martinelli
.. autofunction:: Friedel
.. autofunction:: Chisholm
.. autofunction:: Kim_Mudawar
.. autofunction:: Baroczy_Chisholm
.. autofunction:: Theissing
.. autofunction:: Muller_Steinhagen_Heck
.. autofunction:: Gronnerud
.. autofunction:: Lombardi_Pedrocchi
.. autofunction:: Jung_Radermacher
.. autofunction:: Tran
.. autofunction:: Chen_Friedel
.. autofunction:: Zhang_Webb
.. autofunction:: Xu_Fang
.. autofunction:: Yu_France
.. autofunction:: Wang_Chiang_Lu
.. autofunction:: Hwang_Kim
.. autofunction:: Zhang_Hibiki_Mishima
.. autofunction:: Mishima_Hibiki
.. autofunction:: Bankoff

Two Phase Flow Regime Correlations
----------------------------------
.. autofunction:: Mandhane_Gregory_Aziz_regime
.. autofunction:: Taitel_Dukler_regime

"""

from __future__ import division
__all__ = ['two_phase_dP', 'two_phase_dP_methods', 'two_phase_dP_acceleration',
           'two_phase_dP_dz_acceleration', 'two_phase_dP_gravitational',
           'two_phase_dP_dz_gravitational',
           'Beggs_Brill', 'Lockhart_Martinelli', 'Friedel', 'Chisholm',
           'Kim_Mudawar', 'Baroczy_Chisholm', 'Theissing',
           'Muller_Steinhagen_Heck', 'Gronnerud', 'Lombardi_Pedrocchi',
           'Jung_Radermacher', 'Tran', 'Chen_Friedel', 'Zhang_Webb', 'Xu_Fang',
           'Yu_France', 'Wang_Chiang_Lu', 'Hwang_Kim', 'Zhang_Hibiki_Mishima',
           'Mishima_Hibiki', 'Bankoff',
           'Mandhane_Gregory_Aziz_regime', 'Taitel_Dukler_regime']

from math import pi, log, exp, sin, cos, radians, log10, sqrt
from fluids.constants import g, deg2rad
from fluids.numerics import splev, implementation_optimize_tck
from fluids.friction import friction_factor
from fluids.core import Reynolds, Froude, Weber, Confinement, Bond, Suratman
from fluids.two_phase_voidage import homogeneous, Lockhart_Martinelli_Xtt


Beggs_Brill_dat = {'segregated': (0.98, 0.4846, 0.0868),
'intermittent': (0.845, 0.5351, 0.0173),
'distributed': (1.065, 0.5824, 0.0609)}

def _Beggs_Brill_holdup(regime, lambda_L, Fr, angle, LV):
    if regime == 0:
        a, b, c = 0.98, 0.4846, 0.0868
    elif regime == 2:
        a, b, c = 0.845, 0.5351, 0.0173
    elif regime == 3:
        a, b, c = 1.065, 0.5824, 0.0609
    HL0 = a*lambda_L**b*Fr**-c
    if HL0 < lambda_L:
        HL0 = lambda_L

    if angle > 0.0: # uphill
        # h used instead of g to avoid conflict with gravitational constant
        if regime == 0:
            d, e, f, h = 0.011, -3.768, 3.539, -1.614
        elif regime == 2:
            d, e, f, h = 2.96, 0.305, -0.4473, 0.0978
        elif regime == 3:
            # Dummy values for distributed - > psi = 1.
            d, e, f, h = 2.96, 0.305, -0.4473, 0.0978
    elif angle <= 0: # downhill
        d, e, f, h = 4.70, -0.3692, 0.1244, -0.5056

    C = (1.0 - lambda_L)*log(d*lambda_L**e*LV**f*Fr**h)
    if C < 0.0:
        C = 0.0

    # Correction factor for inclination angle
    Psi = 1.0 + C*(sin(1.8*angle) - 1.0/3.0*sin(1.8*angle)**3)
    if (angle > 0 and regime == 3) or angle == 0:
        Psi = 1.0
    Hl = HL0*Psi
    return Hl

def Beggs_Brill(m, x, rhol, rhog, mul, mug, sigma, P, D, angle, roughness=0.0,
                L=1.0, g=g, acceleration=True):
    r'''Calculates the two-phase pressure drop according to the Beggs-Brill
    correlation ([1]_, [2]_, [3]_).

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Mass quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    P : float
        Pressure of fluid (used only if `acceleration=True`), [Pa]
    D : float
        Diameter of pipe, [m]
    angle : float
        The angle of the pipe with respect to the horizontal, [degrees]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]
    acceleration : bool
        Whether or not to include the original acceleration component, [-]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    The original acceleration formula is fairly primitive and normally
    neglected. The model was developed assuming smooth pipe, so leaving
    `roughness` to zero may be wise.

    Note this is a "mechanistic" pressure drop model - the gravitational
    pressure drop cannot be separated from the frictional pressure drop.

    Examples
    --------
    >>> Beggs_Brill(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, P=1E7, D=0.05, angle=0, roughness=0.0, L=1.0)
    686.9724506803469

    References
    ----------
    .. [1] Beggs, D.H., and J.P. Brill. "A Study of Two-Phase Flow in Inclined
       Pipes." Journal of Petroleum Technology 25, no. 05 (May 1, 1973):
       607-17. https://doi.org/10.2118/4007-PA.
    .. [2] Brill, James P., and Howard Dale Beggs. Two-Phase Flow in Pipes,
       1994.
    .. [3] Shoham, Ovadia. Mechanistic Modeling of Gas-Liquid Two-Phase Flow in
       Pipes. Pap/Cdr edition. Richardson, TX: Society of Petroleum Engineers,
       2006.
    '''
    # 0 - segregated; 1 - transition; 2 - intermittent; 3 - distributed
    qg = x*m/rhog
    ql = (1.0 - x)*m/rhol

    A = 0.25*pi*D*D
    Vsg = qg/A
    Vsl = ql/A
    Vm = Vsg + Vsl
    Fr = Vm*Vm/(g*D)
    lambda_L = Vsl/Vm # no slip liquid holdup

    L1 = 316.0*lambda_L**0.302
    L2 = 0.0009252*lambda_L**-2.4684
    L3 = 0.1*lambda_L**-1.4516
    L4 = 0.5*lambda_L**-6.738
    if (lambda_L < 0.01 and Fr < L1) or (lambda_L >= 0.01 and Fr < L2):
        regime = 0
    elif (lambda_L >= 0.01 and L2 <= Fr <= L3):
        regime = 1
    elif (0.01 <= lambda_L < 0.4 and L3 < Fr <= L1) or (lambda_L >= 0.4 and L3 < Fr <= L4):
        regime = 2
    elif (lambda_L < 0.4 and Fr >= L1) or (lambda_L >= 0.4 and Fr > L4):
        regime = 3
    else:
        raise ValueError('Outside regime ranges')

    LV = Vsl*sqrt(sqrt(rhol/(g*sigma)))
    if angle is None: angle = 0.0
    angle = deg2rad*angle

    if regime != 1:
        Hl = _Beggs_Brill_holdup(regime, lambda_L, Fr, angle, LV)
    else:
        A = (L3 - Fr)/(L3 - L2)
        Hl = (A*_Beggs_Brill_holdup(0, lambda_L, Fr, angle, LV)
             + (1.0 - A)*_Beggs_Brill_holdup(2, lambda_L, Fr, angle, LV))

    rhos = rhol*Hl + rhog*(1.0 - Hl)
    mum = mul*lambda_L +  mug*(1.0 - lambda_L)
    rhom = rhol*lambda_L +  rhog*(1.0 - lambda_L)
    Rem = rhom*D/mum*Vm
    fn = friction_factor(Re=Rem, eD=roughness/D)
    x = lambda_L/(Hl*Hl)


    if 1.0 < x < 1.2:
        S = log(2.2*x - 1.2)
    else:
        logx = log(x)
        # from horner(-0.0523 + 3.182*log(x) - 0.8725*log(x)**2 + 0.01853*log(x)**4, x)
        S = logx/(logx*(logx*(0.01853*logx*logx - 0.8725) + 3.182) - 0.0523)
    if S > 7.0:
        S = 7.0  # Truncate S to avoid exp(S) overflowing
    ftp = fn*exp(S)
    dP_ele = g*sin(angle)*rhos*L
    dP_fric = ftp*L/D*0.5*rhom*Vm*Vm
    # rhos here is pretty clearly rhos according to Shoham
    if P is None: P = 101325.0
    if not acceleration:
        dP = dP_ele + dP_fric
    else:
        Ek = Vsg*Vm*rhos/P  # Confirmed this expression is dimensionless
        dP = (dP_ele + dP_fric)/(1.0 - Ek)
    return dP


def Friedel(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Friedel correlation.

    .. math::
        \Delta P_{friction} = \Delta P_{lo} \phi_{lo}^2

    .. math::
        \phi_{lo}^2 = E + \frac{3.24FH}{Fr^{0.0454} We^{0.035}}

    .. math::
        H = \left(\frac{\rho_l}{\rho_g}\right)^{0.91}\left(\frac{\mu_g}{\mu_l}
        \right)^{0.19}\left(1 - \frac{\mu_g}{\mu_l}\right)^{0.7}

    .. math::
        F = x^{0.78}(1 - x)^{0.224}

    .. math::
        E = (1-x)^2 + x^2\left(\frac{\rho_l f_{d,go}}{\rho_g f_{d,lo}}\right)

    .. math::
        Fr = \frac{G_{tp}^2}{gD\rho_H^2}

    .. math::
        We = \frac{G_{tp}^2 D}{\sigma \rho_H}

    .. math::
        \rho_H = \left(\frac{x}{\rho_g} + \frac{1-x}{\rho_l}\right)^{-1}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable to vertical upflow and horizontal flow. Known to work poorly
    when mul/mug > 1000. Gives mean errors on the order of 40%. Tested on data
    with diameters as small as 4 mm.

    The power of 0.0454 is given as 0.045 in [2]_, [3]_, [4]_, and [5]_; [6]_
    and [2]_ give 0.0454 and [2]_ also gives a similar correlation said to be
    presented in [1]_, so it is believed this 0.0454 was the original power.
    [6]_ also gives an expression for friction factor claimed to be presented
    in [1]_; it is not used here.

    Examples
    --------
    Example 4 in [6]_:

    >>> Friedel(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    738.6500525002241

    References
    ----------
    .. [1] Friedel, L. "Improved Friction Pressure Drop Correlations for
       Horizontal and Vertical Two-Phase Pipe Flow." , in: Proceedings,
       European Two Phase Flow Group Meeting, Ispra, Italy, 1979: 485-481.
    .. [2] Whalley, P. B. Boiling, Condensation, and Gas-Liquid Flow. Oxford:
       Oxford University Press, 1987.
    .. [3] Triplett, K. A., S. M. Ghiaasiaan, S. I. Abdel-Khalik, A. LeMouel,
       and B. N. McCord. "Gas-liquid Two-Phase Flow in Microchannels: Part II:
       Void Fraction and Pressure Drop.” International Journal of Multiphase
       Flow 25, no. 3 (April 1999): 395-410. doi:10.1016/S0301-9322(98)00055-X.
    .. [4] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [5] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    .. [6] Ghiaasiaan, S. Mostafa. Two-Phase Flow, Boiling, and Condensation:
        In Conventional and Miniature Systems. Cambridge University Press, 2007.
    '''
    # Liquid-only properties, for calculation of E, dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of E
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)

    F = x**0.78*(1-x)**0.224
    H = (rhol/rhog)**0.91*(mug/mul)**0.19*(1 - mug/mul)**0.7
    E = (1-x)**2 + x**2*(rhol*fd_go/(rhog*fd_lo))

    # Homogeneous properties, for Froude/Weber numbers
    voidage_h = homogeneous(x, rhol, rhog)
    rho_h = rhol*(1-voidage_h) + rhog*voidage_h
    Q_h = m/rho_h
    v_h = Q_h/(pi/4*D**2)

    Fr = Froude(V=v_h, L=D, squared=True) # checked with (m/(pi/4*D**2))**2/g/D/rho_h**2
    We = Weber(V=v_h, L=D, rho=rho_h, sigma=sigma) # checked with (m/(pi/4*D**2))**2*D/sigma/rho_h

    phi_lo2 = E + 3.24*F*H/(Fr**0.0454*We**0.035)
    return phi_lo2*dP_lo


def Gronnerud(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Gronnerud correlation as
    presented in [2]_, [3]_, and [4]_.

    .. math::
        \Delta P_{friction} = \Delta P_{gd} \phi_{lo}^2

    .. math::
        \phi_{gd} = 1 + \left(\frac{dP}{dL}\right)_{Fr}\left[
        \frac{\frac{\rho_l}{\rho_g}}{\left(\frac{\mu_l}{\mu_g}\right)^{0.25}}
        -1\right]

    .. math::
        \left(\frac{dP}{dL}\right)_{Fr} = f_{Fr}\left[x+4(x^{1.8}-x^{10}
        f_{Fr}^{0.5})\right]

    .. math::
        f_{Fr} = Fr_l^{0.3} + 0.0055\left(\ln \frac{1}{Fr_l}\right)^2

    .. math::
        Fr_l = \frac{G_{tp}^2}{gD\rho_l^2}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Developed for evaporators. Applicable from 0 < x < 1.

    In the model, if `Fr_l` is more than 1, `f_Fr` is set to 1.

    Examples
    --------
    >>> Gronnerud(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... D=0.05, roughness=0.0, L=1.0)
    384.12541144474085

    References
    ----------
    .. [1] Gronnerud, R. "Investigation of Liquid Hold-Up, Flow Resistance and
       Heat Transfer in Circulation Type Evaporators. 4. Two-Phase Flow
       Resistance in Boiling Refrigerants." Proc. Freudenstadt Meet., IIR/C.
       R. Réun. Freudenstadt, IIF. 1972-1: 127-138. 1972.
    .. [2] ASHRAE Handbook: Fundamentals. American Society of Heating,
       Refrigerating and Air-Conditioning Engineers, Incorporated, 2013.
    .. [3] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [4] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    '''
    G = m/(pi/4*D**2)
    V = G/rhol
    Frl = Froude(V=V, L=D, squared=True)
    if Frl >= 1:
        f_Fr = 1
    else:
        f_Fr = Frl**0.3 + 0.0055*(log(1./Frl))**2
    dP_dL_Fr = f_Fr*(x + 4*(x**1.8 - x**10*sqrt(f_Fr)))
    phi_gd = 1 + dP_dL_Fr*((rhol/rhog)/sqrt(sqrt(mul/mug)) - 1)

    # Liquid-only properties, for calculation of E, dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)
    return phi_gd*dP_lo


def Chisholm(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0,
             rough_correction=False):
    r'''Calculates two-phase pressure drop with the Chisholm (1973) correlation
    from [1]_, also in [2]_ and [3]_.

    .. math::
        \frac{\Delta P_{tp}}{\Delta P_{lo}} = \phi_{ch}^2

    .. math::
        \phi_{ch}^2 = 1 + (\Gamma^2 -1)\left\{B x^{(2-n)/2} (1-x)^{(2-n)/2}
        + x^{2-n} \right\}

    .. math::
        \Gamma ^2 = \frac{\left(\frac{\Delta P}{L}\right)_{go}}{\left(\frac{
        \Delta P}{L}\right)_{lo}}

    For Gamma < 9.5:

    .. math::
        B = \frac{55}{G_{tp}^{0.5}} \text{ for } G_{tp} > 1900

    .. math::
        B = \frac{2400}{G_{tp}} \text{ for } 500 < G_{tp} < 1900

    .. math::
        B = 4.8 \text{ for } G_{tp} < 500

    For 9.5 < Gamma < 28:

    .. math::
        B = \frac{520}{\Gamma G_{tp}^{0.5}} \text{ for } G_{tp} < 600

    .. math::
        B = \frac{21}{\Gamma} \text{ for } G_{tp} > 600

    For Gamma > 28:

    .. math::
        B = \frac{15000}{\Gamma^2 G_{tp}^{0.5}}

    If `rough_correction` is True, the following correction to B is applied:

    .. math::
        \frac{B_{rough}}{B_{smooth}} = \left[0.5\left\{1+ \left(\frac{\mu_g}
        {\mu_l}\right)^2 + 10^{-600\epsilon/D}\right\}\right]^{\frac{0.25-n}
        {0.25}}

    .. math::
        n = \frac{\ln \frac{f_{d,lo}}{f_{d,go}}}{\ln \frac{Re_{go}}{Re_{lo}}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]
    rough_correction : bool, optional
        Whether or not to use the roughness correction proposed in the 1968
        version of the correlation

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for  0 < x < 1. n = 0.25, the exponent in the Blassius equation.
    Originally developed for smooth pipes, a roughness correction is included
    as well from the Chisholm's 1968 work [4]_. Neither [2]_ nor [3]_ have any
    mention of the correction however.

    Examples
    --------
    >>> Chisholm(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    1084.1489922923738

    References
    ----------
    .. [1] Chisholm, D. "Pressure Gradients due to Friction during the Flow of
       Evaporating Two-Phase Mixtures in Smooth Tubes and Channels."
       International Journal of Heat and Mass Transfer 16, no. 2 (February
       1973): 347-58. doi:10.1016/0017-9310(73)90063-X.
    .. [2] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [3] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    .. [4] Chisholm, D. "Research Note: Influence of Pipe Surface Roughness on
       Friction Pressure Gradient during Two-Phase Flow." Journal of Mechanical
       Engineering Science 20, no. 6 (December 1, 1978): 353-354.
       doi:10.1243/JMES_JOUR_1978_020_061_02.
    '''
    G_tp = m/(pi/4*D**2)
    n = 0.25 # Blasius friction factor exponent
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of dP_go
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    Gamma = sqrt(dP_go/dP_lo)
    if Gamma <= 9.5:
        if G_tp <= 500:
            B = 4.8
        elif G_tp < 1900:
            B = 2400./G_tp
        else:
            B = 55.0/sqrt(G_tp)
    elif Gamma <= 28:
        if G_tp <= 600:
            B = 520./sqrt(G_tp)/Gamma
        else:
            B = 21./Gamma
    else:
        B = 15000./sqrt(G_tp)/Gamma**2

    if rough_correction:
        n = log(fd_lo/fd_go)/log(Re_go/Re_lo)
        B_ratio = (0.5*(1 + (mug/mul)**2 + 10**(-600*roughness/D)))**((0.25-n)/0.25)
        B = B*B_ratio

    phi2_ch = 1 + (Gamma**2-1)*(B*x**((2-n)/2.)*(1-x)**((2-n)/2.) + x**(2-n))
    return phi2_ch*dP_lo


def Baroczy_Chisholm(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Baroczy (1966) model.
    It was presented in graphical form originally; Chisholm (1973) made the
    correlation non-graphical. The model is also shown in [3]_.

    .. math::
        \frac{\Delta P_{tp}}{\Delta P_{lo}} = \phi_{ch}^2

    .. math::
        \phi_{ch}^2 = 1 + (\Gamma^2 -1)\left\{B x^{(2-n)/2} (1-x)^{(2-n)/2}
        + x^{2-n} \right\}

    .. math::
        \Gamma ^2 = \frac{\left(\frac{\Delta P}{L}\right)_{go}}{\left(\frac{
        \Delta P}{L}\right)_{lo}}

    For Gamma < 9.5:

    .. math::
        B = \frac{55}{G_{tp}^{0.5}}

    For 9.5 < Gamma < 28:

    .. math::
        B = \frac{520}{\Gamma G_{tp}^{0.5}}

    For Gamma > 28:

    .. math::
        B = \frac{15000}{\Gamma^2 G_{tp}^{0.5}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for  0 < x < 1. n = 0.25, the exponent in the Blassius equation.
    The `Chisholm_1973` function should be used in preference to this.

    Examples
    --------
    >>> Baroczy_Chisholm(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    1084.1489922923738

    References
    ----------
    .. [1] Baroczy, C. J. "A systematic correlation for two-phase pressure
       drop." In Chem. Eng. Progr., Symp. Ser., 62: No. 64, 232-49 (1966).
    .. [2] Chisholm, D. "Pressure Gradients due to Friction during the Flow of
       Evaporating Two-Phase Mixtures in Smooth Tubes and Channels."
       International Journal of Heat and Mass Transfer 16, no. 2 (February
       1973): 347-58. doi:10.1016/0017-9310(73)90063-X.
    .. [3] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    '''
    G_tp = m/(pi/4*D**2)
    n = 0.25 # Blasius friction factor exponent
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of dP_go
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    Gamma = sqrt(dP_go/dP_lo)
    if Gamma <= 9.5:
        B = 55.0/sqrt(G_tp)
    elif Gamma <= 28:
        B = 520./sqrt(G_tp)/Gamma
    else:
        B = 15000./sqrt(G_tp)/Gamma**2
    phi2_ch = 1 + (Gamma**2-1)*(B*x**((2-n)/2.)*(1-x)**((2-n)/2.) + x**(2-n))
    return phi2_ch*dP_lo


def Muller_Steinhagen_Heck(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Muller-Steinhagen and Heck
    (1986) correlation from [1]_, also in [2]_ and [3]_.

    .. math::
        \Delta P_{tp} = G_{MSH}(1-x)^{1/3} + \Delta P_{go}x^3

    .. math::
        G_{MSH} = \Delta P_{lo} + 2\left[\Delta P_{go} - \Delta P_{lo}\right]x

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for  0 < x < 1. Developed to be easily integrated. The
    contribution of each term to the overall pressure drop can be
    understood in this model.

    Examples
    --------
    >>> Muller_Steinhagen_Heck(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    793.4465457435081

    References
    ----------
    .. [1] Müller-Steinhagen, H, and K Heck. "A Simple Friction Pressure Drop
       Correlation for Two-Phase Flow in Pipes." Chemical Engineering and
       Processing: Process Intensification 20, no. 6 (November 1, 1986):
       297-308. doi:10.1016/0255-2701(86)80008-3.
    .. [2] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [3] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    '''
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of dP_go
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    G_MSH = dP_lo + 2*(dP_go - dP_lo)*x
    return G_MSH*(1-x)**(1/3.) + dP_go*x**3


def Lombardi_Pedrocchi(m, x, rhol, rhog, sigma, D, L=1.0):
    r'''Calculates two-phase pressure drop with the Lombardi-Pedrocchi (1972)
    correlation from [1]_ as shown in [2]_ and [3]_.

    .. math::
        \Delta P_{tp} = \frac{0.83 G_{tp}^{1.4} \sigma^{0.4} L}{D^{1.2}
        \rho_{h}^{0.866}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    This is a purely empirical method. [3]_ presents a review of this and other
    correlations. It did not perform best, but there were also correlations
    worse than it.

    Examples
    --------
    >>> Lombardi_Pedrocchi(m=0.6, x=0.1, rhol=915., rhog=2.67, sigma=0.045,
    ... D=0.05, L=1.0)
    1567.328374498781

    References
    ----------
    .. [1] Lombardi, C., and E. Pedrocchi. "Pressure Drop Correlation in Two-
       Phase Flow." Energ. Nucl. (Milan) 19: No. 2, 91-99, January 1, 1972.
    .. [2] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [3] Turgut, Oğuz Emrah, Mustafa Turhan Çoban, and Mustafa Asker.
       "Comparison of Flow Boiling Pressure Drop Correlations for Smooth
       Macrotubes." Heat Transfer Engineering 37, no. 6 (April 12, 2016):
       487-506. doi:10.1080/01457632.2015.1060733.
    '''
    voidage_h = homogeneous(x, rhol, rhog)
    rho_h = rhol*(1-voidage_h) + rhog*voidage_h
    G_tp = m/(pi/4*D**2)
    return 0.83*G_tp**1.4*sigma**0.4*L/(D**1.2*rho_h**0.866)


def Theissing(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Theissing (1980)
    correlation as shown in [2]_ and [3]_.

    .. math::
        \Delta P_{{tp}} = \left[ {\Delta P_{{lo}}^{{1/{n\epsilon}}} \left({1 -
        x} \right)^{{1/\epsilon}} + \Delta P_{{go}}^{{1/
        {(n\epsilon)}}} x^{{1/\epsilon}}} \right]^{n\epsilon}

    .. math::
        \epsilon = 3 - 2\left({\frac{{2\sqrt {{{\rho_{{l}}}/
        {\rho_{{g}}}}}}}{{1 + {{\rho_{{l}}}/{\rho_{{g}}}}}}}
        \right)^{{{0.7}/n}}

    .. math::
        n = \frac{{n_1 + n_2 \left({{{\Delta P_{{g}}}/{\Delta
        P_{{l}}}}} \right)^{0.1}}}{{1 + \left({{{\Delta P_{{g}}} /
        {\Delta P_{{l}}}}} \right)^{0.1}}}

    .. math::
        n_1 = \frac{{\ln \left({{{\Delta P_{{l}}}/
        {\Delta P_{{lo}}}}} \right)}}{{\ln \left({1 - x} \right)}}

    .. math::
        n_2 = \frac{\ln \left({\Delta P_{{g}} / \Delta P_{{go}}}
        \right)}{{\ln x}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for 0 < x < 1. Notable, as it can be used for two-phase liquid-
    liquid flow as well as liquid-gas flow.

    Examples
    --------
    >>> Theissing(m=0.6, x=.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... D=0.05, roughness=0.0, L=1.0)
    497.6156370699538

    References
    ----------
    .. [1] Theissing, Peter. "Eine Allgemeingültige Methode Zur Berechnung Des
       Reibungsdruckverlustes Der Mehrphasenströmung (A Generally Valid Method
       for Calculating Frictional Pressure Drop on Multiphase Flow)." Chemie
       Ingenieur Technik 52, no. 4 (January 1, 1980): 344-345.
       doi:10.1002/cite.330520414.
    .. [2] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    .. [3] Greco, A., and G. P. Vanoli. "Experimental Two-Phase Pressure
       Gradients during Evaporation of Pure and Mixed Refrigerants in a Smooth
       Horizontal Tube. Comparison with Correlations." Heat and Mass Transfer
       42, no. 8 (April 6, 2006): 709-725. doi:10.1007/s00231-005-0020-7.
    '''
    # Liquid-only flow
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only flow
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    # Handle x = 0, x=1:
    if x == 0:
        return dP_lo
    elif x == 1:
        return dP_go

    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor(Re=Re_g, eD=roughness/D)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    # The model
    n1 = log(dP_l/dP_lo)/log(1.-x)
    n2 = log(dP_g/dP_go)/log(x)
    n = (n1 + n2*(dP_g/dP_l)**0.1)/(1 + (dP_g/dP_l)**0.1)
    epsilon = 3 - 2*(2*sqrt(rhol/rhog)/(1.+rhol/rhog))**(0.7/n)
    dP = (dP_lo**(1./(n*epsilon))*(1-x)**(1./epsilon)
          + dP_go**(1./(n*epsilon))*x**(1./epsilon))**(n*epsilon)
    return dP


def Jung_Radermacher(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Jung-Radermacher (1989)
    correlation, also shown in [2]_ and [3]_.

    .. math::
        \frac{\Delta P_{tp}}{\Delta P_{lo}} = \phi_{tp}^2

    .. math::
        \phi_{tp}^2 = 12.82X_{tt}^{-1.47}(1-x)^{1.8}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for 0 < x < 1. Developed for the annular flow regime in
    turbulent-turbulent flow.

    Examples
    --------
    >>> Jung_Radermacher(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    552.0686123725568

    References
    ----------
    .. [1] Jung, D. S., and R. Radermacher. "Prediction of Pressure Drop during
       Horizontal Annular Flow Boiling of Pure and Mixed Refrigerants."
       International Journal of Heat and Mass Transfer 32, no. 12 (December 1,
       1989): 2435-46. doi:10.1016/0017-9310(89)90203-2.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11–12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Filip, Alina, Florin Băltăreţu, and Radu-Mircea Damian. "Comparison
       of Two-Phase Pressure Drop Models for Condensing Flows in Horizontal
       Tubes." Mathematical Modelling in Civil Engineering 10, no. 4 (2015):
       19-27. doi:10.2478/mmce-2014-0019.
    '''
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    Xtt = Lockhart_Martinelli_Xtt(x, rhol, rhog, mul, mug)
    phi_tp2 = 12.82*Xtt**-1.47*(1.-x)**1.8
    return phi_tp2*dP_lo


def Tran(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Tran (2000) correlation,
    also shown in [2]_ and [3]_.

    .. math::
        \Delta P = dP_{lo} \phi_{lo}^2

    .. math::
        \phi_{lo}^2 = 1 + (4.3\Gamma^2-1)[\text{Co} \cdot x^{0.875}
        (1-x)^{0.875}+x^{1.75}]

    .. math::
        \Gamma ^2 = \frac{\left(\frac{\Delta P}{L}\right)_{go}}{\left(\frac
        {\Delta P}{L}\right)_{lo}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Developed for boiling refrigerants in channels with hydraulic diameters of
    2.4 mm to 2.92 mm.

    Examples
    --------
    >>> Tran(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    423.2563312951232

    References
    ----------
    .. [1] Tran, T. N, M. -C Chyu, M. W Wambsganss, and D. M France. "Two-Phase
       Pressure Drop of Refrigerants during Flow Boiling in Small Channels: An
       Experimental Investigation and Correlation Development." International
       Journal of Multiphase Flow 26, no. 11 (November 1, 2000): 1739-54.
       doi:10.1016/S0301-9322(99)00119-6.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11–12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Choi, Kwang-Il, A. S. Pamitran, Chun-Young Oh, and Jong-Taek Oh.
       "Two-Phase Pressure Drop of R-410A in Horizontal Smooth Minichannels."
       International Journal of Refrigeration 31, no. 1 (January 2008): 119-29.
       doi:10.1016/j.ijrefrig.2007.06.006.
    '''
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of dP_go
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    Gamma2 = dP_go/dP_lo
    Co = Confinement(D=D, rhol=rhol, rhog=rhog, sigma=sigma)
    phi_lo2 = 1 + (4.3*Gamma2 -1)*(Co*x**0.875*(1-x)**0.875 + x**1.75)
    return dP_lo*phi_lo2


def Chen_Friedel(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Chen modification of the
    Friedel correlation, as given in [1]_ and also shown in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{Friedel}\Omega

    For Bo < 2.5:

    .. math::
        \Omega = \frac{0.0333Re_{lo}^{0.45}}{Re_g^{0.09}(1 + 0.4\exp(-Bo))}

    For Bo >= 2.5:

    .. math::
        \Omega = \frac{We^{0.2}}{2.5 + 0.06Bo}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable ONLY to mini/microchannels; yields drastically too low
    pressure drops for larger channels. For more details, see the `Friedel`
    correlation.

    It is not explicitly stated in [1]_ how to calculate the liquid mixture
    density for use in calculation of Weber number; the homogeneous model is
    assumed as it is used in the Friedel model.

    The bond number used here is 1/4 the normal value,  i.e.:

    .. math::
        Bo = \frac{g(\rho_l-\rho_g)D^2}{4\sigma}

    Examples
    --------
    >>> Chen_Friedel(m=.0005, x=0.9, rhol=950., rhog=1.4, mul=1E-3, mug=1E-5,
    ... sigma=0.02, D=0.003, roughness=0.0, L=1.0)
    6249.247540588871

    References
    ----------
    .. [1] Chen, Ing Youn, Kai-Shing Yang, Yu-Juei Chang, and Chi-Chung Wang.
       "Two-Phase Pressure Drop of Air–water and R-410A in Small Horizontal
       Tubes." International Journal of Multiphase Flow 27, no. 7 (July 2001):
       1293-99. doi:10.1016/S0301-9322(01)00004-0.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11–12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Choi, Kwang-Il, A. S. Pamitran, Chun-Young Oh, and Jong-Taek Oh.
       "Two-Phase Pressure Drop of R-410A in Horizontal Smooth Minichannels."
       International Journal of Refrigeration 31, no. 1 (January 2008): 119-29.
       doi:10.1016/j.ijrefrig.2007.06.006.
    '''
    # Liquid-only properties, for calculation of E, dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of E
    v_go = m/rhog/(pi/4*D**2)
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)

    F = x**0.78*(1-x)**0.224
    H = (rhol/rhog)**0.91*(mug/mul)**0.19*(1 - mug/mul)**0.7
    E = (1-x)**2 + x**2*(rhol*fd_go/(rhog*fd_lo))

    # Homogeneous properties, for Froude/Weber numbers
    rho_h = 1./(x/rhog + (1-x)/rhol)
    Q_h = m/rho_h
    v_h = Q_h/(pi/4*D**2)

    Fr = Froude(V=v_h, L=D, squared=True) # checked with (m/(pi/4*D**2))**2/g/D/rho_h**2
    We = Weber(V=v_h, L=D, rho=rho_h, sigma=sigma) # checked with (m/(pi/4*D**2))**2*D/sigma/rho_h

    phi_lo2 = E + 3.24*F*H/(Fr**0.0454*We**0.035)

    dP = phi_lo2*dP_lo

    # Chen modification; Weber number is the same as above
    # Weber is same
    Bo = Bond(rhol=rhol, rhog=rhog, sigma=sigma, L=D)/4 # Custom definition

    if Bo < 2.5:
        # Actual gas flow, needed for this case only.
        v_g = m*x/rhog/(pi/4*D**2)
        Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
        Omega = 0.0333*Re_lo**0.45/(Re_g**0.09*(1 + 0.5*exp(-Bo)))
    else:
        Omega = We**0.2/(2.5 + 0.06*Bo)
    return dP*Omega


def Zhang_Webb(m, x, rhol, mul, P, Pc, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Zhang-Webb (2001)
    correlation as shown in [1]_ and also given in [2]_.

    .. math::
        \phi_{lo}^2 = (1-x)^2 + 2.87x^2\left(\frac{P}{P_c}\right)^{-1}
        + 1.68x^{0.8}(1-x)^{0.25}\left(\frac{P}{P_c}\right)^{-1.64}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    P : float
        Pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Applicable for 0 < x < 1. Corresponding-states method developed with
    R-134A, R-22 and R-404A in tubes of hydraulic diameters of 2.13 mm,
    6.25 mm, and 3.25 mm. For the author's 119 data points, the mean deviation
    was 11.5%. Recommended for reduced pressures larger than 0.2 and tubes of
    diameter 1-7 mm.

    Does not require known properties for the gas phase.

    Examples
    --------
    >>> Zhang_Webb(m=0.6, x=0.1, rhol=915., mul=180E-6, P=2E5, Pc=4055000,
    ... D=0.05, roughness=0.0, L=1.0)
    712.0999804205617

    References
    ----------
    .. [1] Zhang, Ming, and Ralph L. Webb. "Correlation of Two-Phase Friction
       for Refrigerants in Small-Diameter Tubes." Experimental Thermal and
       Fluid Science 25, no. 3-4 (October 2001): 131-39.
       doi:10.1016/S0894-1777(01)00066-8.
    .. [2] Choi, Kwang-Il, A. S. Pamitran, Chun-Young Oh, and Jong-Taek Oh.
       "Two-Phase Pressure Drop of R-410A in Horizontal Smooth Minichannels."
       International Journal of Refrigeration 31, no. 1 (January 2008): 119-29.
       doi:10.1016/j.ijrefrig.2007.06.006.
    '''
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)
    Pr = 0.5 if (Pc is None or P is None) else P/Pc
    phi_lo2 = (1-x)**2 + 2.87*x**2/Pr + 1.68*x**0.8*sqrt(sqrt(1-x))*Pr**-1.64
    return dP_lo*phi_lo2


def Bankoff(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Bankoff (1960) correlation,
    as shown in [2]_, [3]_, and [4]_.

    .. math::
        \Delta P_{tp} = \phi_{l}^{7/4} \Delta P_{l}

    .. math::
        \phi_l = \frac{1}{1-x}\left[1 - \gamma\left(1 - \frac{\rho_g}{\rho_l}
        \right)\right]^{3/7}\left[1 + x\left(\frac{\rho_l}{\rho_g} - 1\right)
        \right]

    .. math::
        \gamma = \frac{0.71 + 2.35\left(\frac{\rho_g}{\rho_l}\right)}
        {1 + \frac{1-x}{x} \cdot \frac{\rho_g}{\rho_l}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    This correlation is not actually shown in [1]_. Its origin is unknown.
    The author recommends against using this.

    Examples
    --------
    >>> Bankoff(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... D=0.05, roughness=0.0, L=1.0)
    4746.0594424533965

    References
    ----------
    .. [1] Bankoff, S. G. "A Variable Density Single-Fluid Model for Two-Phase
       Flow With Particular Reference to Steam-Water Flow." Journal of Heat
       Transfer 82, no. 4 (November 1, 1960): 265-72. doi:10.1115/1.3679930.
    .. [2] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    .. [3] Moreno Quibén, Jesús. "Experimental and Analytical Study of Two-
       Phase Pressure Drops during Evaporation in Horizontal Tubes," 2005.
       doi:10.5075/epfl-thesis-3337.
    .. [4] Mekisso, Henock Mateos. "Comparison of Frictional Pressure Drop
       Correlations for Isothermal Two-Phase Horizontal Flow." Thesis, Oklahoma
       State University, 2013. https://shareok.org/handle/11244/11109.
    '''
    # Liquid-only properties, for calculation of dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    gamma = (0.71 + 2.35*rhog/rhol)/(1. + (1.-x)/x*rhog/rhol)
    phi_Bf = 1./(1.-x)*(1 - gamma*(1 - rhog/rhol))**(3/7.)*(1. + x*(rhol/rhog -1.))
    return dP_lo*phi_Bf**(7/4.)


def Xu_Fang(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Xu and Fang (2013)
    correlation. Developed after a comprehensive review of available
    correlations, likely meaning it is quite accurate.

    .. math::
        \Delta P = \Delta P_{lo} \phi_{lo}^2

    .. math::
        \phi_{lo}^2 = Y^2x^3 + (1-x^{2.59})^{0.632}[1 + 2x^{1.17}(Y^2-1)
        + 0.00775x^{-0.475} Fr_{tp}^{0.535} We_{tp}^{0.188}]

    .. math::
        Y^2 = \frac{\Delta P_{go}}{\Delta P_{lo}}

    .. math::
        Fr_{tp} = \frac{G_{tp}^2}{gD\rho_{tp}^2}

    .. math::
        We_{tp} = \frac{G_{tp}^2 D}{\sigma \rho_{tp}}

    .. math::
        \frac{1}{\rho_{tp}} = \frac{1-x}{\rho_l} + \frac{x}{\rho_g}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----


    Examples
    --------
    >>> Xu_Fang(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    604.0595632116267

    References
    ----------
    .. [1] Xu, Yu, and Xiande Fang. "A New Correlation of Two-Phase Frictional
       Pressure Drop for Condensing Flow in Pipes." Nuclear Engineering and
       Design 263 (October 2013): 87-96. doi:10.1016/j.nucengdes.2013.04.017.
    '''
    A = pi/4*D*D
    # Liquid-only properties, for calculation of E, dP_lo
    v_lo = m/rhol/A
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)

    # Gas-only properties, for calculation of E
    v_go = m/rhog/A
    Re_go = Reynolds(V=v_go, rho=rhog, mu=mug, D=D)
    fd_go = friction_factor(Re=Re_go, eD=roughness/D)
    dP_go = fd_go*L/D*(0.5*rhog*v_go**2)

    # Homogeneous properties, for Froude/Weber numbers
    voidage_h = homogeneous(x, rhol, rhog)
    rho_h = rhol*(1-voidage_h) + rhog*voidage_h

    Q_h = m/rho_h
    v_h = Q_h/A

    Fr = Froude(V=v_h, L=D, squared=True)
    We = Weber(V=v_h, L=D, rho=rho_h, sigma=sigma)
    Y2 = dP_go/dP_lo

    phi_lo2 = Y2*x**3 + (1-x**2.59)**0.632*(1 + 2*x**1.17*(Y2-1)
            + 0.00775*x**-0.475*Fr**0.535*We**0.188)

    return phi_lo2*dP_lo


def Yu_France(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Yu, France, Wambsganss,
    and Hull (2002) correlation given in [1]_ and reviewed in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        \phi_l^2 = X^{-1.9}

    .. math::
        X = 18.65\left(\frac{\rho_g}{\rho_l}\right)^{0.5}\left(\frac{1-x}{x}
        \right)\frac{Re_{g}^{0.1}}{Re_l^{0.5}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----

    Examples
    --------
    >>> Yu_France(m=0.6, x=.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... D=0.05, roughness=0.0, L=1.0)
    1146.9833225539571

    References
    ----------
    .. [1] Yu, W., D. M. France, M. W. Wambsganss, and J. R. Hull. "Two-Phase
       Pressure Drop, Boiling Heat Transfer, and Critical Heat Flux to Water in
       a Small-Diameter Horizontal Tube." International Journal of Multiphase
       Flow 28, no. 6 (June 2002): 927-41. doi:10.1016/S0301-9322(02)00019-8.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Xu, Yu, Xiande Fang, Xianghui Su, Zhanru Zhou, and Weiwei Chen.
       "Evaluation of Frictional Pressure Drop Correlations for Two-Phase Flow
       in Pipes." Nuclear Engineering and Design, SI : CFD4NRS-3, 253 (December
       2012): 86-97. doi:10.1016/j.nucengdes.2012.08.007.
    '''
    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)

    X = 18.65*sqrt(rhog/rhol)*(1-x)/x*Re_g**0.1/sqrt(Re_l)
    phi_l2 = X**-1.9
    return phi_l2*dP_l


def Wang_Chiang_Lu(m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Wang, Chiang, and Lu (1997)
    correlation given in [1]_ and reviewed in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{g} \phi_g^2

    .. math::
        \phi_g^2 = 1 + 9.397X^{0.62} + 0.564X^{2.45} \text{ for } G >= 200 kg/m^2/s

    .. math::
        \phi_g^2 = 1 + CX + X^2 \text{ for lower mass fluxes}

    .. math::
        C = 0.000004566X^{0.128}Re_{lo}^{0.938}\left(\frac{\rho_l}{\rho_g}
        \right)^{-2.15}\left(\frac{\mu_l}{\mu_g}\right)^{5.1}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----

    Examples
    --------
    >>> Wang_Chiang_Lu(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    448.29981978639137

    References
    ----------
    .. [1] Wang, Chi-Chuan, Ching-Shan Chiang, and Ding-Chong Lu. "Visual
       Observation of Two-Phase Flow Pattern of R-22, R-134a, and R-407C in a
       6.5-Mm Smooth Tube." Experimental Thermal and Fluid Science 15, no. 4
       (November 1, 1997): 395-405. doi:10.1016/S0894-1777(97)00007-1.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Xu, Yu, Xiande Fang, Xianghui Su, Zhanru Zhou, and Weiwei Chen.
       "Evaluation of Frictional Pressure Drop Correlations for Two-Phase Flow
       in Pipes." Nuclear Engineering and Design, SI : CFD4NRS-3, 253 (December
       2012): 86-97. doi:10.1016/j.nucengdes.2012.08.007.
    '''
    G_tp = m/(pi/4*D**2)

    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor(Re=Re_g, eD=roughness/D)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    X = sqrt(dP_l/dP_g)

    if G_tp >= 200:
        phi_g2 = 1 + 9.397*X**0.62 + 0.564*X**2.45
    else:
        # Liquid-only flow; Re_lo is oddly needed
        v_lo = m/rhol/(pi/4*D**2)
        Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
        C = 0.000004566*X**0.128*Re_lo**0.938*(rhol/rhog)**-2.15*(mul/mug)**5.1
        phi_g2 = 1 + C*X + X**2
    return dP_g*phi_g2


def Hwang_Kim(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Hwang and Kim (2006)
    correlation as in [1]_, also presented in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        C = 0.227 Re_{lo}^{0.452} X^{-0.32} Co^{-0.82}

    .. math::
        \phi_l^2 = 1 + \frac{C}{X} + \frac{1}{X^2}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Developed with data for microtubes of diameter 0.244 mm and 0.792 mm only.
    Not likely to be suitable to larger diameters.

    Examples
    --------
    >>> Hwang_Kim(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    798.302774184557

    References
    ----------
    .. [1] Hwang, Yun Wook, and Min Soo Kim. "The Pressure Drop in Microtubes
       and the Correlation Development."  International Journal of Heat and
       Mass Transfer 49, no. 11-12 (June 2006): 1804-12.
       doi:10.1016/j.ijheatmasstransfer.2005.10.040.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Xu, Yu, Xiande Fang, Xianghui Su, Zhanru Zhou, and Weiwei Chen.
       "Evaluation of Frictional Pressure Drop Correlations for Two-Phase Flow
       in Pipes." Nuclear Engineering and Design, SI : CFD4NRS-3, 253 (December
       2012): 86-97. doi:10.1016/j.nucengdes.2012.08.007.
    '''
    # Liquid-only flow
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)

    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor(Re=Re_g, eD=roughness/D)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    # Actual model
    X = sqrt(dP_l/dP_g)
    Co = Confinement(D=D, rhol=rhol, rhog=rhog, sigma=sigma)
    C = 0.227*Re_lo**0.452*X**-0.320*Co**-0.820
    phi_l2 = 1 + C/X + 1./X**2
    return dP_l*phi_l2


def Zhang_Hibiki_Mishima(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0,
                         L=1.0, flowtype='adiabatic vapor'):
    r'''Calculates two-phase pressure drop with the Zhang, Hibiki, Mishima and
    (2010) correlation as in [1]_, also presented in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        \phi_l^2 = 1 + \frac{C}{X} + \frac{1}{X^2}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    For adiabatic liquid-vapor two-phase flow:

    .. math::
        C = 21[1 - \exp(-0.142/Co)]

    For adiabatic liquid-gas two-phase flow:

    .. math::
        C = 21[1 - \exp(-0.674/Co)]

    For flow boiling:

    .. math::
        C = 21[1 - \exp(-0.358/Co)]

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]
    flowtype : str
        One of 'adiabatic vapor', 'adiabatic gas', or 'flow boiling'

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Seems fairly reliable.

    Examples
    --------
    >>> Zhang_Hibiki_Mishima(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    444.9718476894804

    References
    ----------
    .. [1] Zhang, W., T. Hibiki, and K. Mishima. "Correlations of Two-Phase
       Frictional Pressure Drop and Void Fraction in Mini-Channel."
       International Journal of Heat and Mass Transfer 53, no. 1-3 (January 15,
       2010): 453-65. doi:10.1016/j.ijheatmasstransfer.2009.09.011.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Xu, Yu, Xiande Fang, Xianghui Su, Zhanru Zhou, and Weiwei Chen.
       "Evaluation of Frictional Pressure Drop Correlations for Two-Phase Flow
       in Pipes." Nuclear Engineering and Design, SI : CFD4NRS-3, 253 (December
       2012): 86-97. doi:10.1016/j.nucengdes.2012.08.007.
    '''
    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor(Re=Re_g, eD=roughness/D)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    # Actual model
    X = sqrt(dP_l/dP_g)
    Co = Confinement(D=D, rhol=rhol, rhog=rhog, sigma=sigma)

    if flowtype == 'adiabatic vapor':
        C = 21*(1 - exp(-0.142/Co))
    elif flowtype == 'adiabatic gas':
        C = 21*(1 - exp(-0.674/Co))
    elif flowtype == 'flow boiling':
        C = 21*(1 - exp(-0.358/Co))
    else:
        raise ValueError("Only flow types 'adiabatic vapor', 'adiabatic gas, \
and 'flow boiling' are recognized.")

    phi_l2 = 1 + C/X + 1./X**2
    return dP_l*phi_l2


def Mishima_Hibiki(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1.0):
    r'''Calculates two-phase pressure drop with the Mishima and Hibiki (1996)
    correlation as in [1]_, also presented in [2]_ and [3]_.

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        C = 21[1 - \exp(-319D)]

    .. math::
        \phi_l^2 = 1 + \frac{C}{X} + \frac{1}{X^2}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----

    Examples
    --------
    >>> Mishima_Hibiki(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    732.4268200606265

    References
    ----------
    .. [1] Mishima, K., and T. Hibiki. "Some Characteristics of Air-Water Two-
       Phase Flow in Small Diameter Vertical Tubes." International Journal of
       Multiphase Flow 22, no. 4 (August 1, 1996): 703-12.
       doi:10.1016/0301-9322(96)00010-9.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [3] Xu, Yu, Xiande Fang, Xianghui Su, Zhanru Zhou, and Weiwei Chen.
       "Evaluation of Frictional Pressure Drop Correlations for Two-Phase Flow
       in Pipes." Nuclear Engineering and Design, SI : CFD4NRS-3, 253 (December
       2012): 86-97. doi:10.1016/j.nucengdes.2012.08.007.
    '''
    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor(Re=Re_l, eD=roughness/D)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor(Re=Re_g, eD=roughness/D)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    # Actual model
    X = sqrt(dP_l/dP_g)
    C = 21*(1 - exp(-0.319E3*D))
    phi_l2 = 1 + C/X + 1./X**2
    return dP_l*phi_l2

def friction_factor_Kim_Mudawar(Re):
    if Re < 2000:
        return 64./Re
    elif Re < 20000:
        return 0.316/sqrt(sqrt(Re))
    else:
        return 0.184*Re**-0.2


def Kim_Mudawar(m, x, rhol, rhog, mul, mug, sigma, D, L=1.0):
    r'''Calculates two-phase pressure drop with the Kim and Mudawar (2012)
    correlation as in [1]_, also presented in [2]_.

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        \phi_l^2 = 1 + \frac{C}{X} + \frac{1}{X^2}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    For turbulent liquid, turbulent gas:

    .. math::
        C = 0.39Re_{lo}^{0.03} Su_{go}^{0.10}\left(\frac{\rho_l}{\rho_g}
        \right)^{0.35}

    For turbulent liquid, laminar gas:

    .. math::
        C = 8.7\times 10^{-4} Re_{lo}^{0.17} Su_{go}^{0.50}\left(\frac{\rho_l}
        {\rho_g}\right)^{0.14}

    For laminar liquid, turbulent gas:

    .. math::
        C = 0.0015 Re_{lo}^{0.59} Su_{go}^{0.19}\left(\frac{\rho_l}{\rho_g}
        \right)^{0.36}

    For laminar liquid, laminar gas:

    .. math::
        C = 3.5\times 10^{-5} Re_{lo}^{0.44} Su_{go}^{0.50}\left(\frac{\rho_l}
        {\rho_g}\right)^{0.48}

    This model has its own friction factor calculations, to be consistent with
    its Reynolds number transition. As their model was regressed with these
    equations, more error is obtained when using any other friction factor
    calculation. The laminar equation 64/Re is used up to Re=2000, then the
    Blasius equation with a coefficient of 0.316, and above Re = 20000,

    .. math::
        f_d = \frac{0.184}{Re^{0.2}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]
    L : float, optional
        Length of pipe, [m]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    The critical Reynolds number in this model is 2000, with a Reynolds number
    definition using actual liquid and gas flows. This model also requires
    liquid-only Reynolds number to be calculated.

    No attempt to incorporate roughness into the model was made in [1]_.

    The model was developed with hydraulic diameter from 0.0695 to 6.22 mm,
    mass velocities 4 to 8528 kg/m^2/s, flow qualities from 0 to 1, reduced
    pressures from 0.0052 to 0.91, superficial liquid Reynolds numbers up to
    79202, superficial gas Reynolds numbers up to 253810, liquid-only Reynolds
    numbers up to 89798, 7115 data points from 36 sources and working fluids
    air, CO2, N2, water, ethanol, R12, R22, R134a, R236ea, R245fa, R404A, R407C,
    propane, methane, and ammonia.

    Examples
    --------
    >>> Kim_Mudawar(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05, L=1.0)
    840.4137796786074

    References
    ----------
    .. [1] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    # Actual Liquid flow
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    fd_l = friction_factor_Kim_Mudawar(Re=Re_l)
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)

    # Actual gas flow
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)
    fd_g = friction_factor_Kim_Mudawar(Re=Re_g)
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    # Liquid-only flow
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)

    Su = Suratman(L=D, rho=rhog, mu=mug, sigma=sigma)
    X = sqrt(dP_l/dP_g)
    Re_c = 2000 # Transition Reynolds number

    if Re_l < Re_c and Re_g < Re_c:
        C = 3.5E-5*Re_lo**0.44*sqrt(Su)*(rhol/rhog)**0.48
    elif Re_l < Re_c and Re_g >= Re_c:
        C = 0.0015*Re_lo**0.59*Su**0.19*(rhol/rhog)**0.36
    elif Re_l >= Re_c and Re_g < Re_c:
        C = 8.7E-4*Re_lo**0.17*sqrt(Su)*(rhol/rhog)**0.14
    else: # Turbulent case
        C = 0.39*Re_lo**0.03*Su**0.10*(rhol/rhog)**0.35

    phi_l2 = 1 + C/X + 1./X**2
    return dP_l*phi_l2


def Lockhart_Martinelli(m, x, rhol, rhog, mul, mug, D, L=1.0, Re_c=2000.0):
    r'''Calculates two-phase pressure drop with the Lockhart and Martinelli
    (1949) correlation as presented in non-graphical form by Chisholm (1967).

    .. math::
        \Delta P = \Delta P_{l} \phi_{l}^2

    .. math::
        \phi_l^2 = 1 + \frac{C}{X} + \frac{1}{X^2}

    .. math::
        X^2 = \frac{\Delta P_l}{\Delta P_g}

    +---------+---------+--+
    |Liquid   |Gas      |C |
    +=========+=========+==+
    |Turbulent|Turbulent|20|
    +---------+---------+--+
    |Laminar  |Turbulent|12|
    +---------+---------+--+
    |Turbulent|Laminar  |10|
    +---------+---------+--+
    |Laminar  |Laminar  |5 |
    +---------+---------+--+

    This model has its own friction factor calculations, to be consistent with
    its Reynolds number transition and the procedure specified in the original
    work. The equation 64/Re is used up to Re_c, and above it the Blasius
    equation is used as follows:

    .. math::
        f_d = \frac{0.184}{Re^{0.2}}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    L : float, optional
        Length of pipe, [m]
    Re_c : float, optional
        Transition Reynolds number, used to decide which friction factor
        equation to use and which C value to use from the table above.

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Notes
    -----
    Developed for horizontal flow. Very popular. Many implementations of this
    model assume turbulent-turbulent flow.

    The original model proposed that the transition Reynolds number was 1000
    for laminar flow, and 2000 for turbulent flow; it proposed no model
    for Re_l < 1000 and Re_g between 1000 and 2000 and also Re_g < 1000 and
    Re_l between 1000 and 2000.

    No correction is available in this model for rough pipe.

    [3]_ examined the original data in [1]_ again, and fit more curves to the
    data, separating them into different flow regimes. There were 229 datum
    in the turbulent-turbulent regime, 9 in the turbulent-laminar regime, 339
    in the laminar-turbulent regime, and 42 in the laminar-laminar regime.
    Errors from [3]_'s curves were 13.4%, 3.5%, 14.3%, and 12.0% for the above
    regimes, respectively. [2]_'s fits provide further error.

    Examples
    --------
    >>> Lockhart_Martinelli(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6,
    ... mug=14E-6, D=0.05, L=1.0)
    716.4695654888484

    References
    ----------
    .. [1] Lockhart, R. W. & Martinelli, R. C. (1949), "Proposed correlation of
       data for isothermal two-phase, two-component flow in pipes", Chemical
       Engineering Progress 45 (1), 39-48.
    .. [2] Chisholm, D."A Theoretical Basis for the Lockhart-Martinelli
       Correlation for Two-Phase Flow." International Journal of Heat and Mass
       Transfer 10, no. 12 (December 1967): 1767-78.
       doi:10.1016/0017-9310(67)90047-6.
    .. [3] Cui, Xiaozhou, and John J. J. Chen."A Re-Examination of the Data of
       Lockhart-Martinelli." International Journal of Multiphase Flow 36, no.
       10 (October 2010): 836-46. doi:10.1016/j.ijmultiphaseflow.2010.06.001.
    .. [4] Kim, Sung-Min, and Issam Mudawar. "Universal Approach to Predicting
       Two-Phase Frictional Pressure Drop for Adiabatic and Condensing Mini/
       Micro-Channel Flows." International Journal of Heat and Mass Transfer
       55, no. 11-12 (May 2012): 3246-61.
       doi:10.1016/j.ijheatmasstransfer.2012.02.047.
    '''
    v_l = m*(1-x)/rhol/(pi/4*D**2)
    Re_l = Reynolds(V=v_l, rho=rhol, mu=mul, D=D)
    v_g = m*x/rhog/(pi/4*D**2)
    Re_g = Reynolds(V=v_g, rho=rhog, mu=mug, D=D)

    if Re_l < Re_c and Re_g < Re_c:
        C = 5.0
    elif Re_l < Re_c and Re_g >= Re_c:
        # Liquid laminar, gas turbulent
        C = 12.0
    elif Re_l >= Re_c and Re_g < Re_c:
        # Liquid turbulent, gas laminar
        C = 10.0
    else: # Turbulent case
        C = 20.0

    # Frictoin factor as in the original model
    fd_l =  64./Re_l if Re_l < Re_c else 0.184*Re_l**-0.2
    dP_l = fd_l*L/D*(0.5*rhol*v_l**2)
    fd_g =  64./Re_g if Re_g < Re_c else 0.184*Re_g**-0.2
    dP_g = fd_g*L/D*(0.5*rhog*v_g**2)

    X = sqrt(dP_l/dP_g)

    phi_l2 = 1 + C/X + 1./X**2
    return dP_l*phi_l2


two_phase_correlations = {
    # 0 index, args are: m, x, rhol, mul, P, Pc, D, roughness=0.0, L=1
    'Zhang_Webb': (Zhang_Webb, 0),
    # 1 index, args are: m, x, rhol, rhog, mul, mug, D, L=1
    'Lockhart_Martinelli': (Lockhart_Martinelli, 1),
    # 2 index, args are: m, x, rhol, rhog, mul, mug, D, roughness=0.0, L=1
    'Bankoff': (Bankoff, 2),
    'Baroczy_Chisholm': (Baroczy_Chisholm, 2),
    'Chisholm': (Chisholm, 2),
    'Gronnerud': (Gronnerud, 2),
    'Jung_Radermacher': (Jung_Radermacher, 2),
    'Muller_Steinhagen_Heck': (Muller_Steinhagen_Heck, 2),
    'Theissing': (Theissing, 2),
    'Wang_Chiang_Lu': (Wang_Chiang_Lu, 2),
    'Yu_France': (Yu_France, 2),
    # 3 index, args are: m, x, rhol, rhog, mul, mug, sigma, D, L=1
    'Kim_Mudawar': (Kim_Mudawar, 3),
    # 4 index, args are: m, x, rhol, rhog, mul, mug, sigma, D, roughness=0.0, L=1
    'Friedel': (Friedel, 4),
    'Hwang_Kim': (Hwang_Kim, 4),
    'Mishima_Hibiki': (Mishima_Hibiki, 4),
    'Tran': (Tran, 4),
    'Xu_Fang': (Xu_Fang, 4),
    'Zhang_Hibiki_Mishima': (Zhang_Hibiki_Mishima, 4),
    'Chen_Friedel': (Chen_Friedel, 4),
    # 5 index: args are m, x, rhol, rhog, sigma, D, L=1
    'Lombardi_Pedrocchi': (Lombardi_Pedrocchi, 5),
    # Misc indexes:
    'Chisholm rough': (Chisholm, 101),
    'Zhang_Hibiki_Mishima adiabatic gas': (Zhang_Hibiki_Mishima, 102),
    'Zhang_Hibiki_Mishima flow boiling': (Zhang_Hibiki_Mishima, 103),
    'Beggs-Brill': (Beggs_Brill, 104)
}
_unknown_msg_two_phase = "Unknown method; available methods are %s" %(list(two_phase_correlations.keys()))

def two_phase_dP_methods(m, x, rhol, D, L=1.0, rhog=None, mul=None, mug=None,
                         sigma=None, P=None, Pc=None, roughness=0.0, angle=0,
                         check_ranges=False):
    r'''This function returns a list of names of correlations for two-phase
    liquid-gas pressure drop for flow inside channels.
    24 calculation methods are available, with varying input requirements.

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    L : float, optional
        Length of pipe, [m]
    rhog : float, optional
        Gas density, [kg/m^3]
    mul : float, optional
        Viscosity of liquid, [Pa*s]
    mug : float, optional
        Viscosity of gas, [Pa*s]
    sigma : float, optional
        Surface tension, [N/m]
    P : float, optional
        Pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    angle : float, optional
        The angle of the pipe with respect to the horizontal, [degrees]
    check_ranges : bool, optional
        Added for Future use only

    Returns
    -------
    methods : list
        List of methods which can be used to calculate two-phase pressure drop
        with the given inputs.

    Examples
    --------
    >>> len(two_phase_dP_methods(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0, angle=30.0, roughness=1e-4, P=1e5, Pc=1e6))
    24
    '''
    usable_indices = []
    if rhog is not None and sigma is not None:
        usable_indices.append(5)
    if rhog is not None and sigma is not None and mul is not None and mug is not None:
        usable_indices.extend([4, 3, 102, 103]) # Differs only in the addition of roughness
    if rhog is not None and mul is not None and mug is not None:
        usable_indices.extend([1,2, 101]) # Differs only in the addition of roughness
    if mul is not None and P is not None and Pc is not None:
        usable_indices.append(0)
    if (rhog is not None and mul is not None and mug is not None
        and sigma is not None and P is not None and angle is not None):
        usable_indices.append(104)
    return [key for key, value in two_phase_correlations.items() if value[1] in usable_indices]

def two_phase_dP(m, x, rhol, D, L=1.0, rhog=None, mul=None, mug=None, sigma=None,
                 P=None, Pc=None, roughness=0.0, angle=None, Method=None):
    r'''This function handles calculation of two-phase liquid-gas pressure drop
    for flow inside channels. 23 calculation methods are available, with
    varying input requirements. A correlation will be automatically selected if
    none is specified. The full list of correlation can be obtained with the
    `AvailableMethods` flag.

    If no correlation is selected, the following rules are used, with the
    earlier options attempted first:

        * If rhog, mul, mug, and sigma are specified, use the Kim_Mudawar model
        * If rhog, mul, and mug are specified, use the Chisholm model
        * If mul, P, and Pc are specified, use the Zhang_Webb model
        * If rhog and sigma are specified, use the Lombardi_Pedrocchi model

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    D : float
        Diameter of pipe, [m]
    L : float, optional
        Length of pipe, [m]
    rhog : float, optional
        Gas density, [kg/m^3]
    mul : float, optional
        Viscosity of liquid, [Pa*s]
    mug : float, optional
        Viscosity of gas, [Pa*s]
    sigma : float, optional
        Surface tension, [N/m]
    P : float, optional
        Pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    angle : float, optional
        The angle of the pipe with respect to the horizontal, [degrees]

    Returns
    -------
    dP : float
        Pressure drop of the two-phase flow, [Pa]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        two_phase_correlations.

    Notes
    -----
    These functions may be integrated over, with properties recalculated as
    the fluid's quality changes.

    This model considers only the frictional pressure drop, not that due to
    gravity or acceleration.

    Examples
    --------
    >>> two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6,
    ... sigma=0.0487, D=0.05, L=1.0)
    840.4137796786074
    '''
    if Method is None:
        if rhog is not None and mul is not None and mug is not None and sigma is not None:
            Method2 = 'Kim_Mudawar' # Kim_Mudawar preferred
        elif rhog is not None and mul is not None and mug is not None:
            Method2 = 'Chisholm' # Second choice, indexes 1 or 2
        elif mul is not None and P is not None and Pc is not None:
            Method2 = 'Zhang_Webb' # Not a good choice
        elif rhog is not None and sigma is not None:
            Method2 = 'Lombardi_Pedrocchi' # Last try
        else:
            raise ValueError('All possible methods require more information \
than provided; provide more inputs!')
    else:
        Method2 = Method

    if Method2 == "Zhang_Webb":
        return Zhang_Webb(m=m, x=x, rhol=rhol, mul=mul, P=P, Pc=Pc, D=D, roughness=roughness, L=L)
    elif Method2 == "Lockhart_Martinelli":
        return Lockhart_Martinelli(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, L=L)
    elif Method2 == "Bankoff":
        return Bankoff(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Baroczy_Chisholm":
        return Baroczy_Chisholm(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Chisholm":
        return Chisholm(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Gronnerud":
        return Gronnerud(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Jung_Radermacher":
        return Jung_Radermacher(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Muller_Steinhagen_Heck":
        return Muller_Steinhagen_Heck(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Theissing":
        return Theissing(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Wang_Chiang_Lu":
        return Wang_Chiang_Lu(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Yu_France":
        return Yu_France(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D, roughness=roughness, L=L)
    elif Method2 == "Kim_Mudawar":
        return Kim_Mudawar(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, L=L)
    elif Method2 == "Friedel":
        return Friedel(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Hwang_Kim":
        return Hwang_Kim(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Mishima_Hibiki":
        return Mishima_Hibiki(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Tran":
        return Tran(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Xu_Fang":
        return Xu_Fang(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Zhang_Hibiki_Mishima":
        return Zhang_Hibiki_Mishima(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Chen_Friedel":
        return Chen_Friedel(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, sigma=sigma, D=D, roughness=roughness, L=L)
    elif Method2 == "Lombardi_Pedrocchi":
        return Lombardi_Pedrocchi(m=m, x=x, rhol=rhol, rhog=rhog, sigma=sigma, D=D, L=L)
    elif Method2 == "Chisholm rough":
        return Chisholm(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug, D=D,
                     L=L, roughness=roughness, rough_correction=True)
    elif Method2 == "Zhang_Hibiki_Mishima adiabatic gas":
        return Zhang_Hibiki_Mishima(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                     sigma=sigma, D=D, L=L, roughness=roughness,
                     flowtype='adiabatic gas')
    elif Method2 == "Zhang_Hibiki_Mishima flow boiling":
        return Zhang_Hibiki_Mishima(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                     sigma=sigma, D=D, L=L, roughness=roughness,
                     flowtype='flow boiling')
    elif Method2 == "Beggs-Brill":
        return Beggs_Brill(m=m, x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug,
                     sigma=sigma, P=P, D=D, angle=angle, L=L,
                     roughness=roughness, acceleration=False, g=g)
    else:
        raise ValueError(_unknown_msg_two_phase)


def two_phase_dP_acceleration(m, D, xi, xo, alpha_i, alpha_o, rho_li, rho_gi,
                              rho_lo=None, rho_go=None):
    r'''This function handles calculation of two-phase liquid-gas pressure drop
    due to acceleration for flow inside channels. This is a discrete
    calculation for a segment with a known difference in quality (and ideally
    known inlet and outlet pressures so density dependence can be included).

    .. math::
        \Delta P_{acc} = G^2\left\{\left[\frac{(1-x_o)^2}{\rho_{l,o}
        (1-\alpha_o)} + \frac{x_o^2}{\rho_{g,o}\alpha_o} \right]
        - \left[\frac{(1-x_i)^2}{\rho_{l,i}(1-\alpha_i)}
        + \frac{x_i^2}{\rho_{g,i}\alpha_i} \right]\right\}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    D : float
        Diameter of pipe, [m]
    xi : float
        Quality of fluid at inlet, [-]
    xo : float
        Quality of fluid at outlet, [-]
    alpha_i : float
        Void fraction at inlet (area of gas / total area of channel), [-]
    alpha_o : float
        Void fraction at outlet (area of gas / total area of channel), [-]
    rho_li : float
        Liquid phase density at inlet, [kg/m^3]
    rho_gi : float
        Gas phase density at inlet, [kg/m^3]
    rho_lo : float, optional
        Liquid phase density at outlet, [kg/m^3]
    rho_go : float, optional
        Gas phase density at outlet, [kg/m^3]

    Returns
    -------
    dP : float
        Acceleration component of pressure drop for two-phase flow, [Pa]

    Notes
    -----
    The use of different gas and liquid phase densities at the inlet and outlet
    is optional; the outlet densities conditions will be assumed to be those of
    the inlet if they are not specified.

    There is a continuous variant of this method which can be integrated over,
    at the expense of a speed. The differential form of this is as follows
    ([1]_, [3]_):

    .. math::
        - \left(\frac{d P}{dz}\right)_{acc} = G^2 \frac{d}{dz} \left[\frac{
        (1-x)^2}{\rho_l(1-\alpha)} + \frac{x^2}{\rho_g\alpha}\right]

    Examples
    --------
    >>> two_phase_dP_acceleration(m=1, D=0.1, xi=0.372, xo=0.557, rho_li=827.1,
    ... rho_gi=3.919, alpha_i=0.992, alpha_o=0.996)
    706.8560377214725

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Awad, M. M., and Y. S. Muzychka. "Effective Property Models for
       Homogeneous Two-Phase Flows." Experimental Thermal and Fluid Science 33,
       no. 1 (October 1, 2008): 106-13.
       doi:10.1016/j.expthermflusci.2008.07.006.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    G = 4*m/(pi*D*D)
    if rho_lo is None:
        rho_lo = rho_li
    if rho_go is None:
        rho_go = rho_gi
    in_term = (1.-xi)**2/(rho_li*(1.-alpha_i)) + xi*xi/(rho_gi*alpha_i)
    out_term = (1.-xo)**2/(rho_lo*(1.-alpha_o)) + xo*xo/(rho_go*alpha_o)
    return G*G*(out_term - in_term)


def two_phase_dP_dz_acceleration(m, D, x, rhol, rhog, dv_dP_l, dv_dP_g, dx_dP,
                                 dP_dL, dA_dL):
    r'''This function handles calculation of two-phase liquid-gas pressure drop
    due to acceleration for flow inside channels. This is a continuous
    calculation, providing the differential in pressure per unit length and
    should be called as part of an integration routine ([1]_, [2]_, [3]_).

    .. math::
        -\left(\frac{\partial P}{\partial L}\right)_{A} = G^2
        \left(\left(\frac{1}{\rho_g} - \frac{1}{\rho_l}\right)\frac{\partial P}
        {\partial L}\frac{\partial x}{\partial P} +
        \frac{\partial P}{\partial L}\left[x \frac{\partial (1/\rho_g)}
        {\partial P}  + (1-x) \frac{\partial (1/\rho_l)}{\partial P}
        \right] \right) - \frac{G^2}{\rho_{hom}}\frac{1}{A}\frac{\partial A}
        {\partial L}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    D : float
        Diameter of pipe, [m]
    x : float
        Quality of fluid [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    dv_dP_l : float
        Derivative of mass specific volume of the liquid phase with respect to
        pressure, [m^3/(kg*Pa)]
    dv_dP_g : float
        Derivative of mass specific volume of the gas phase with respect to
        pressure, [m^3/(kg*Pa)]
    dx_dP : float
        Derivative of mass quality of the two-phase fluid with respect to
        pressure (numerical derivatives may be convenient for this), [1/Pa]
    dP_dL : float
        Pressure drop per unit length of pipe, [Pa/m]
    dA_dL : float
        Change in area of pipe per unit length of pipe, [m^2/m]

    Returns
    -------
    dP_dz : float
        Acceleration component of pressure drop for two-phase flow, [Pa/m]

    Notes
    -----

    This calculation has the `homogeneous` model built in to it as its
    derivation is shown in [1]_. The discrete calculation is more flexible as
    different void fractions may be used.

    Examples
    --------
    >>> two_phase_dP_dz_acceleration(m=1, D=0.1, x=0.372, rhol=827.1,
    ... rhog=3.919, dv_dP_l=-5e-12, dv_dP_g=-4e-7, dx_dP=-2e-7, dP_dL=120.0,
    ... dA_dL=0.0001)
    20.137876617489034

    References
    ----------
    .. [1] Shoham, Ovadia. Mechanistic Modeling of Gas-Liquid Two-Phase Flow in
       Pipes. Pap/Cdr edition. Richardson, TX: Society of Petroleum Engineers,
       2006.
    .. [2] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    A = 0.25*pi*D*D
    G = m/A
    t1 = (1.0/rhog - 1.0/rhol)*dP_dL*dx_dP + dP_dL*(x*dv_dP_g + (1.0 - x)*dv_dP_l)

    voidage_h = homogeneous(x, rhol, rhog)
    rho_h = rhol*(1.0 - voidage_h) + rhog*voidage_h
    return -G*G*(t1 - dA_dL/(rho_h*A))




def two_phase_dP_gravitational(angle, z, alpha_i, rho_li, rho_gi,
                               alpha_o=None, rho_lo=None, rho_go=None, g=g):
    r'''This function handles calculation of two-phase liquid-gas pressure drop
    due to gravitation for flow inside channels. This is a discrete
    calculation for a segment with a known difference in elevation (and ideally
    known inlet and outlet pressures so density dependence can be included).

    .. math::
        - \Delta P_{grav} =  g \sin \theta z \left\{\frac{ [\alpha_o\rho_{g,o}
        + (1-\alpha_o)\rho_{l,o}] + [\alpha_i\rho_{g,i} + (1-\alpha_i)\rho_{l,i}]}
        {2}\right\}

    Parameters
    ----------
    angle : float
        The angle of the pipe with respect to the horizontal, [degrees]
    z : float
        The total length of the pipe, [m]
    alpha_i : float
        Void fraction at inlet (area of gas / total area of channel), [-]
    rho_li : float
        Liquid phase density at inlet, [kg/m^3]
    rho_gi : float
        Gas phase density at inlet, [kg/m^3]
    alpha_o : float, optional
        Void fraction at outlet (area of gas / total area of channel), [-]
    rho_lo : float, optional
        Liquid phase density at outlet, [kg/m^3]
    rho_go : float, optional
        Gas phase density at outlet, [kg/m^3]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    dP : float
        Gravitational component of pressure drop for two-phase flow, [Pa]

    Notes
    -----
    The use of different gas and liquid phase densities and void fraction
    at the inlet and outlet is optional; the outlet densities and void fraction
    will be assumed to be those of the inlet if they are not specified. This
    does not add much accuracy.

    There is a continuous variant of this method which can be integrated over,
    at the expense of a speed. The differential form of this is as follows
    ([1]_, [2]_):

    .. math::
        -\left(\frac{dP}{dz} \right)_{grav} =  [\alpha\rho_g + (1-\alpha)
        \rho_l]g \sin \theta

    Examples
    --------
    Example calculation, page 13-2 from [3]_:

    >>> two_phase_dP_gravitational(angle=90, z=2, alpha_i=0.9685, rho_li=1518.,
    ... rho_gi=2.6)
    987.237416829999

    The same calculation, but using average inlet and outlet conditions:

    >>> two_phase_dP_gravitational(angle=90, z=2, alpha_i=0.9685, rho_li=1518.,
    ... rho_gi=2.6,  alpha_o=0.968, rho_lo=1517.9, rho_go=2.59)
    994.5416058829999

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    .. [3] Thome, John R. "Engineering Data Book III." Wolverine Tube Inc
       (2004). http://www.wlv.com/heat-transfer-databook/
    '''
    if rho_lo is None:
        rho_lo = rho_li
    if rho_go is None:
        rho_go = rho_gi
    if alpha_o is None:
        alpha_o = alpha_i
    angle = radians(angle)
    in_term = alpha_i*rho_gi + (1. - alpha_i)*rho_li
    out_term = alpha_o*rho_go + (1. - alpha_o)*rho_lo
    return g*z*sin(angle)*(out_term + in_term)/2.


def two_phase_dP_dz_gravitational(angle, alpha, rhol, rhog, g=g):
    r'''This function handles calculation of two-phase liquid-gas pressure drop
    due to gravitation for flow inside channels. This is a differential
    calculation for a segment with an infinitesimal difference in elevation for
    use in performing integration over a pipe as shown in [1]_ and [2]_.

    .. math::
        -\left(\frac{dP}{dz} \right)_{grav} =  [\alpha\rho_g + (1-\alpha)
        \rho_l]g \sin \theta

    Parameters
    ----------
    angle : float
        The angle of the pipe with respect to the horizontal, [degrees]
    alpha : float
        Void fraction (area of gas / total area of channel), [-]
    rhol : float
        Liquid phase density, [kg/m^3]
    rhog : float
        Gas phase density, [kg/m^3]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    dP_dz : float
        Gravitational component of pressure drop for two-phase flow, [Pa/m]

    Notes
    -----

    Examples
    --------
    >>> two_phase_dP_dz_gravitational(angle=90, alpha=0.9685, rhol=1518,
    ... rhog=2.6)
    493.6187084149995

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/Micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    angle = radians(angle)
    return g*sin(angle)*(alpha*rhog + (1. - alpha)*rhol)

Dukler_XA_tck = implementation_optimize_tck([[-2.4791105294648372, -2.4791105294648372, -2.4791105294648372,
                           -2.4791105294648372, 0.14360803483759585, 1.7199938263676038,
                           1.7199938263676038, 1.7199938263676038, 1.7199938263676038],
                 [0.21299880246561081, 0.16299733301915248, -0.042340970712679615,
                           -1.9967836909384598, -2.9917366639619414, 0.0, 0.0, 0.0, 0.0],
                 3])
Dukler_XC_tck = implementation_optimize_tck([[-1.8323873272724698, -1.8323873272724698, -1.8323873272724698,
                           -1.8323873272724698, -0.15428198203334137, 1.7031193462360779,
                           1.7031193462360779, 1.7031193462360779, 1.7031193462360779],
                 [0.2827776229531682, 0.6207113329042158, 1.0609541626742232,
                           0.44917638072891825, 0.014664597632360495, 0.0, 0.0, 0.0, 0.0],
                 3])
Dukler_XD_tck = implementation_optimize_tck([[0.2532652936901574, 0.2532652936901574, 0.2532652936901574,
                           0.2532652936901574, 3.5567847823070253, 3.5567847823070253,
                           3.5567847823070253, 3.5567847823070253],
                 [0.09054274779541564, -0.05102629221303253, -0.23907463153703945,
                           -0.7757156285450911, 0.0, 0.0, 0.0, 0.0],
                 3])

XA_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XA_tck))
XC_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XC_tck))
XD_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XD_tck))


def Taitel_Dukler_regime(m, x, rhol, rhog, mul, mug, D, angle, roughness=0.0,
                         g=g):
    r'''Classifies the regime of a two-phase flow according to Taitel and
    Dukler (1976) ([1]_, [2]_).

    The flow regimes in this method are 'annular', 'bubbly', 'intermittent',
    'stratified wavy', and 'stratified smooth'.

    The four dimensionless parameters used are 'X', 'T', 'F', and 'K'.

    .. math::
        X = \left[\frac{(dP/dL)_{l,s,f}}{(dP/dL)_{g,s,f}}\right]^{0.5}

    .. math::
        T = \left[\frac{(dP/dL)_{l,s,f}}{(\rho_l-\rho_g)g\cos\theta}\right]^{0.5}

    .. math::
        F = \sqrt{\frac{\rho_g}{(\rho_l-\rho_g)}} \frac{v_{g,s}}{\sqrt{D g \cos\theta}}

    .. math::
        K = F\left[\frac{D v_{l,s}}{\nu_l}  \right]^{0.5} = F \sqrt{Re_{l,s}}

    Note that 'l' refers to liquid, 'g' gas, 'f' friction-only, and 's'
    superficial (i.e. if only the mass flow of that phase were flowing in the
    pipe).

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Mass quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    D : float
        Diameter of pipe, [m]
    angle : float
        The angle of the pipe with respect to the horizontal, [degrees]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    regime : str
        One of 'annular', 'bubbly', 'intermittent', 'stratified wavy',
        'stratified smooth', [-]
    X : float
        `X` dimensionless group used in the calculation, [-]
    T : float
        `T` dimensionless group used in the calculation, [-]
    F : float
        `F` dimensionless group used in the calculation, [-]
    K : float
        `K` dimensionless group used in the calculation, [-]

    Notes
    -----
    The original friction factor used in this model is that of Blasius.

    Examples
    --------
    >>> Taitel_Dukler_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67,
    ... mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, angle=0)[0]
    'annular'

    References
    ----------
    .. [1] Taitel, Yemada, and A. E. Dukler. "A Model for Predicting Flow
       Regime Transitions in Horizontal and near Horizontal Gas-Liquid Flow."
       AIChE Journal 22, no. 1 (January 1, 1976): 47-55.
       doi:10.1002/aic.690220105.
    .. [2] Brill, James P., and Howard Dale Beggs. Two-Phase Flow in Pipes,
       1994.
    .. [3] Shoham, Ovadia. Mechanistic Modeling of Gas-Liquid Two-Phase Flow in
       Pipes. Pap/Cdr edition. Richardson, TX: Society of Petroleum Engineers,
       2006.
    '''
    angle = radians(angle)
    A = 0.25*pi*D*D
    # Liquid-superficial properties, for calculation of dP_ls, dP_ls
    # Paper and Brill Beggs 1991 confirms not v_lo but v_sg
    v_ls =  m*(1.0 - x)/(rhol*A)
    Re_ls = Reynolds(V=v_ls, rho=rhol, mu=mul, D=D)
    fd_ls = friction_factor(Re=Re_ls, eD=roughness/D)
    dP_ls = fd_ls/D*(0.5*rhol*v_ls*v_ls)

    # Gas-superficial properties, for calculation of dP_gs
    v_gs = m*x/(rhog*A)
    Re_gs = Reynolds(V=v_gs, rho=rhog, mu=mug, D=D)
    fd_gs = friction_factor(Re=Re_gs, eD=roughness/D)
    dP_gs = fd_gs/D*(0.5*rhog*v_gs*v_gs)

    X = sqrt(dP_ls/dP_gs)

    F = sqrt(rhog/(rhol-rhog))*v_gs/sqrt(D*g*cos(angle))

    # Paper only uses kinematic viscosity
    nul = mul/rhol

    T = sqrt(dP_ls/((rhol-rhog)*g*cos(angle)))
    K = sqrt(rhog*v_gs*v_gs*v_ls/((rhol-rhog)*g*nul*cos(angle)))

    F_A_at_X = XA_interp_obj(X)

    X_B_transition = 1.7917 # Roughly

    if F >= F_A_at_X and X <= X_B_transition:
        regime = 'annular'
    elif F >= F_A_at_X:
        T_D_at_X = XD_interp_obj(X)
        if T >= T_D_at_X:
            regime = 'bubbly'
        else:
            regime = 'intermittent'
    else:
        K_C_at_X = XC_interp_obj(X)
        if K >= K_C_at_X:
            regime = 'stratified wavy'
        else:
            regime = 'stratified smooth'

    return regime, X, T, F, K


def Mandhane_Gregory_Aziz_regime(m, x, rhol, rhog, mul, mug, sigma, D):
    r'''Classifies the regime of a two-phase flow according to Mandhane,
    Gregory, and Azis  (1974) flow map.

    The flow regimes in this method are 'elongated bubble', 'stratified',
    'annular mist', 'slug', 'dispersed bubble', and 'wave'.

    The parameters used are just the superficial liquid and gas velocity (i.e.
    if only the mass flow of that phase were flowing in the pipe).

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    x : float
        Mass quality of fluid, [-]
    rhol : float
        Liquid density, [kg/m^3]
    rhog : float
        Gas density, [kg/m^3]
    mul : float
        Viscosity of liquid, [Pa*s]
    mug : float
        Viscosity of gas, [Pa*s]
    sigma : float
        Surface tension, [N/m]
    D : float
        Diameter of pipe, [m]

    Returns
    -------
    regime : str
        One of 'elongated bubble', 'stratified', 'annular mist', 'slug',
        'dispersed bubble', or 'wave', [-]
    v_gs : float
        The superficial gas velocity in the pipe (x axis coordinate), [ft/s]
    v_ls : float
        The superficial liquid velocity in the pipe (x axis coordinate), [ft/s]

    Notes
    -----
    [1]_ contains a Fortran implementation of this model, which this has been
    validated against. This is a very fast flow map as all transitions were
    spelled out with clean transitions.

    Examples
    --------
    >>> Mandhane_Gregory_Aziz_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67,
    ... mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)
    ('slug', 0.9728397701853173, 42.05456634236875)

    References
    ----------
    .. [1] Mandhane, J. M., G. A. Gregory, and K. Aziz. "A Flow Pattern Map for
       Gas-liquid Flow in Horizontal Pipes." International Journal of
       Multiphase Flow 1, no. 4 (October 30, 1974): 537-53.
       doi:10.1016/0301-9322(74)90006-8.
    '''
    A = 0.25*pi*D*D
    Vsl =  m*(1.0 - x)/(rhol*A)
    Vsg = m*x/(rhog*A)

    # Convert to imperial units
    Vsl, Vsg = Vsl/0.3048, Vsg/0.3048
#    X1 = (rhog/0.0808)**0.333 * (rhol*72.4/62.4/sigma)**0.25 * (mug/0.018)**0.2
#    Y1 = (rhol*72.4/62.4/sigma)**0.25 * (mul/1.)**0.2
    X1 = (rhog/1.294292)**0.333 * sqrt(sqrt(rhol*0.0724/(999.552*sigma))) * (mug*1.8E5)**0.2
    Y1 = sqrt(sqrt(rhol*0.0724/999.552/sigma)) * (mul*1E3)**0.2

    if Vsl < 14.0*Y1:
        if Vsl <= 0.1:
            Y1345 = 14.0*(Vsl/0.1)**-0.368
        elif Vsl <= 0.2:
            Y1345 = 14.0*(Vsl/0.1)**-0.415
        elif Vsl <= 1.15:
            Y1345 = 10.5*(Vsl/0.2)**-0.816
        elif Vsl <= 4.8:
            Y1345 = 2.5
        else:
            Y1345 = 2.5*(Vsl/4.8)**0.248

        if Vsl <= 0.1:
            Y456 = 70.0*(Vsl/0.01)**-0.0675
        elif Vsl <= 0.3:
            Y456 = 60.0*(Vsl/0.1)**-0.415
        elif Vsl <= 0.56:
            Y456 = 38.0*(Vsl/0.3)**0.0813
        elif Vsl <= 1.0:
            Y456 = 40.0*(Vsl/0.56)**0.385
        elif Vsl <= 2.5:
            Y456 = 50.0*(Vsl/1.)**0.756
        else:
            Y456 = 100.0*(Vsl/2.5)**0.463

        Y45 = 0.3*Y1
        Y31 = 0.5/Y1
        Y1345 = Y1345*X1
        Y456 = Y456*X1

        if Vsg <= Y1345 and Vsl >= Y31:
            regime = 'elongated bubble'
        elif Vsg <= Y1345 and Vsl <= Y31:
            regime = 'stratified'
        elif Vsg >= Y1345 and Vsg <= Y456 and Vsl > Y45:
            regime = 'slug'
        elif Vsg >= Y1345 and Vsg <= Y456 and Vsl <= Y45:
            regime = 'wave'
        else:
            regime = 'annular mist'
    elif Vsg <= (230.*(Vsl/14.)**0.206)*X1:
        regime = 'dispersed bubble'
    else:
        regime = 'annular mist'
    return regime, Vsl, Vsg

Mandhane_Gregory_Aziz_regimes = {'elongated bubble': 1, 'stratified': 2,
                                 'slug':3, 'wave': 4,
                                 'annular mist': 5, 'dispersed bubble': 6}
