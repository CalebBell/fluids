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
__all__ = ['Friedel', 'Chisholm', 'Baroczy_Chisholm', 
           'Muller_Steinhagen_Heck', 'Gronnerud', 'Lombardi_Pedrocchi']

from math import pi, log
from fluids.friction import friction_factor
from fluids.core import Reynolds, Froude, Weber
from fluids.two_phase_voidage import homogeneous


def Friedel(m, x, rhol, rhog, mul, mug, sigma, D, roughness=0, L=1):
    r'''Calculates two-phase pressure drop with the Friedel correlation.
    
    .. math::
        \Delta P_{friction} = \Delta P_{lo} \phi_{lo}^2
        
        \phi_{lo}^2 = E + \frac{3.24FH}{Fr^{0.0454} We^{0.035}}
        
        H = \left(\frac{\rho_l}{\rho_g}\right)^{0.91}\left(\frac{\mu_g}{\mu_l}
        \right)^{0.19}\left(1 - \frac{\mu_g}{\mu_l}\right)^{0.7}
        
        F = x^{0.78}(1 - x)^{0.224}
        
        E = (1-x)^2 + x^2\left(\frac{\rho_l f_{d,go}}{\rho_g f_{d,lo}}\right)
        
        Fr = \frac{G_{tp}^2}{gD\rho_H^2}
        
        We = \frac{G_{tp}^2 D}{\sigma \rho_H}
        
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
    ... sigma=0.0487, D=0.05, roughness=0, L=1)
    738.6500525002243

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


def Gronnerud(m, x, rhol, rhog, mul, mug, D, roughness=0, L=1):
    r'''Calculates two-phase pressure drop with the Gronnerud correlation as
    presented in [2]_, [3]_, and [4]_.
    
    .. math::
        \Delta P_{friction} = \Delta P_{gd} \phi_{lo}^2
        
        \phi_{gd} = 1 + \left(\frac{dP}{dL}\right)_{Fr}\left[
        \frac{\frac{\rho_l}{\rho_g}}{\left(\frac{\mu_l}{\mu_g}\right)^{0.25}}
        -1\right]
        
        \left(\frac{dP}{dL}\right)_{Fr} = f_{Fr}\left[x+4(x^{1.8}-x^{10}
        f_{Fr}^{0.5})\right]
        
        f_{Fr} = Fr_l^{0.3} + 0.0055\left(\ln \frac{1}{Fr_l}\right)^2
        
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
    ... D=0.05, roughness=0, L=1)
    384.125411444741

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
    dP_dL_Fr = f_Fr*(x + 4*(x**1.8 - x**10*f_Fr**0.5))
    phi_gd = 1 + dP_dL_Fr*((rhol/rhog)/(mul/mug)**0.25 - 1)
    
    # Liquid-only properties, for calculation of E, dP_lo
    v_lo = m/rhol/(pi/4*D**2)
    Re_lo = Reynolds(V=v_lo, rho=rhol, mu=mul, D=D)
    fd_lo = friction_factor(Re=Re_lo, eD=roughness/D)
    dP_lo = fd_lo*L/D*(0.5*rhol*v_lo**2)
    return phi_gd*dP_lo

    
def Chisholm(m, x, rhol, rhog, mul, mug, D, roughness=0, L=1, 
             rough_correction=False):
    r'''Calculates two-phase pressure drop with the Chisholm (1973) correlation 
    from [1]_, also in [2]_ and [3]_.
    
    .. math::
        \frac{\Delta P_{tp}}{\Delta P_{lo}} = \phi_{ch}^2
        
        \phi_{ch}^2 = 1 + (\Gamma^2 -1)\left\{B x^{(2-n)/2} (1-x)^{(2-n)/2}
        + x^{2-n} \right\}
        
        \Gamma ^2 = \frac{\left(\frac{\Delta P}{L}\right)_{go}}{\left(\frac{
        \Delta P}{L}\right)_{lo}}
        
    For Gamma < 9.5:
    
    .. math::
        B = \frac{55}{G_{tp}^{0.5}} \text{ for } G_{tp} > 1900
        
        B = \frac{2400}{G_{tp}} \text{ for } 500 < G_{tp} < 1900
        
        B = 4.8 \text{ for } G_{tp} < 500

    For 9.5 < Gamma < 28:
        
    .. math::
        B = \frac{520}{\Gamma G_{tp}^{0.5}} \text{ for } G_{tp} < 600
        
        B = \frac{21}{\Gamma} \text{ for } G_{tp} > 600

    For Gamma > 28:
        
    .. math::
        B = \frac{15000}{\Gamma^2 G_{tp}^{0.5}}

    If `rough_correction` is True, the following correction to B is applied:
    
    .. math::
        \frac{B_{rough}}{B_{smooth}} = \left[0.5\left\{1+ \left(\frac{\mu_g}
        {\mu_l}\right)^2 + 10^{-600\epsilon/D}\right\}\right]^{\frac{0.25-n}
        {0.25}}
        
        n = \frac{\log \frac{f_{d,lo}}{f_{d,go}}}{\log \frac{Re_{go}}{Re_{lo}}}

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
    ... mug=14E-6, D=0.05, roughness=0, L=1)
    1084.1489922923736

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
    
    Gamma = (dP_go/dP_lo)**0.5
    if Gamma <= 9.5:
        if G_tp <= 500:
            B = 4.8
        elif G_tp < 1900:
            B = 2400./G_tp
        else:
            B = 55*G_tp**-0.5
    elif Gamma <= 28:
        if G_tp <= 600:
            B = 520.*G_tp**-0.5/Gamma
        else:
            B = 21./Gamma
    else:
        B = 15000.*G_tp**-0.5/Gamma**2
    
    if rough_correction:
        n = log(fd_lo/fd_go)/log(Re_go/Re_lo)
        B_ratio = (0.5*(1 + (mug/mul)**2 + 10**(-600*roughness/D)))**((0.25-n)/0.25)
        B = B*B_ratio
    
    phi2_ch = 1 + (Gamma**2-1)*(B*x**((2-n)/2.)*(1-x)**((2-n)/2.) + x**(2-n))
    return phi2_ch*dP_lo


def Baroczy_Chisholm(m, x, rhol, rhog, mul, mug, D, roughness=0, L=1):
    r'''Calculates two-phase pressure drop with the Baroczy (1966) model.
    It was presented in graphical form originally; Chisholm (1973) made the 
    correlation non-graphical. The model is also shown in [3]_.
    
    .. math::
        \frac{\Delta P_{tp}}{\Delta P_{lo}} = \phi_{ch}^2
        
        \phi_{ch}^2 = 1 + (\Gamma^2 -1)\left\{B x^{(2-n)/2} (1-x)^{(2-n)/2}
        + x^{2-n} \right\}
        
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
    ... mug=14E-6, D=0.05, roughness=0, L=1)
    1084.1489922923736

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
    
    Gamma = (dP_go/dP_lo)**0.5
    if Gamma <= 9.5:
        B = 55*G_tp**-0.5
    elif Gamma <= 28:
        B = 520.*G_tp**-0.5/Gamma
    else:
        B = 15000.*G_tp**-0.5/Gamma**2
    phi2_ch = 1 + (Gamma**2-1)*(B*x**((2-n)/2.)*(1-x)**((2-n)/2.) + x**(2-n))
    return phi2_ch*dP_lo

    
def Muller_Steinhagen_Heck(m, x, rhol, rhog, mul, mug, D, roughness=0, L=1):
    r'''Calculates two-phase pressure drop with the Muller-Steinhagen and Heck
    (1986) correlation from [1]_, also in [2]_ and [3]_.
    
    .. math::
        \Delta P_{tp} = G_{MSH}(1-x)^{1/3} + \Delta P_{go}x^3
        
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
    ... mug=14E-6, D=0.05, roughness=0, L=1)
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


def Lombardi_Pedrocchi(m, x, rhol, rhog, sigma, D, L=1):
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
    This is a purely emperical method. [3]_ presents a review of this and other
    correlations. It did not perform best, but there were also correlations 
    worse than it.
        
    Examples
    --------
    >>> Lombardi_Pedrocchi(m=0.6, x=0.1, rhol=915., rhog=2.67, sigma=0.045, 
    ... D=0.05, L=1)
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
