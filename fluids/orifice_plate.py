# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import cos, sin, tan, atan, pi, radians, exp
import numpy as np
from fluids.friction import friction_factor
from scipy.optimize import newton

__all__ = ['orifice_discharge', 'orifice_expansibility',
           'C_Reader_Harris_Gallagher', 'Reader_Harris_Gallagher_discharge']


def orifice_discharge(D, Do, P1, P2, rho, C, expansibility=1.0):
    r'''Calculates the flow rate of an orifice plate based on the geometry
    of the plate, measured pressures of the orifice, and the density of the
    fluid.
    
    .. math::
        m = \left(\frac{\pi D_o^2}{4}\right) C \frac{\sqrt{2\Delta P \rho_1}}
        {\sqrt{1 - \beta^4}}\cdot \epsilon
        
    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of orifice at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of orifice at the cross-section of
        the pressure tap, [Pa]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    C : float
        Coefficient of discharge of the orifice, [-]
    expansibility : float, optional
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Returns
    -------
    m : float
        Mass flow rate of fluid, [kg/s]

    Notes
    -----
    This is formula 1-12 in [1]_.

    Examples
    --------
    >>> orifice_discharge(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1646, 
    ... C=0.5988, expansibility=0.9975)
    0.01120390943807026

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement 
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    '''
    dP = P1 - P2
    beta = Do/D
    return (pi*Do*Do/4.)*C*(2*dP*rho)**0.5/(1.0 - beta**4)**0.5*expansibility


def orifice_expansibility(D, Do, P1, P2, k):
    r'''Calculates the expansibility factor for orifice plate calculations
    based on the geometry of the plate, measured pressures of the orifice, and
    the isentropic exponent of the fluid.
    
    .. math::
        \epsilon = 1 - (0.351 + 0.256\beta^4 + 0.93\beta^8)
        \left[1-\left(\frac{P_2}{P_1}\right)^{1/\kappa}\right]
        
    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of orifice at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of orifice at the cross-section of
        the pressure tap, [Pa]
    k : float
        Isentropic exponent of fluid, [-]

    Returns
    -------
    expansibility : float, optional
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Notes
    -----
    This formula was determined for the range of P2/P1 >= 0.80, and for fluids
    of air, steam, and natural gas. However, there is no objection to using
    it for other fluids.

    Examples
    --------
    >>> orifice_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    0.9974739057343425

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement 
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    '''
    beta = Do/D
    return (1.0 - (0.351 + 0.256*beta**4 + 0.93*beta**8)*(
            1.0 - (P2/P1)**(1./k)))


def C_Reader_Harris_Gallagher(D, Do, P1, P2, rho, mu, k, m, taps='corner'):
    r'''Calculates the coefficient of discharge of the orifice based on the 
    geometry of the plate, measured pressures of the orifice, mass flow rate
    through the orifice, and the density, viscosity, and isentropic exponent 
    of the fluid.
    
    .. math::
        C = 0.5961 + 0.0261\beta^2 - 0.216\beta^8 + 0.000521\left(\frac{
        10^6\beta}{Re_D}\right)^{0.7}\\
        + (0.0188 + 0.0063A)\beta^{3.5} \left(\frac{10^6}{Re_D}\right)^{0.3} \\
        +(0.043 + 0.080\exp(-10L_1) -0.123\exp(-7L_1))(1-0.11A)\frac{\beta^4}
        {1-\beta^4} \\
        -  0.031(M_2' - 0.8M_2'^{1.1})\beta^{1.3}
        
    .. math::
        M_2' = \frac{2L_2'}{1-\beta}
        
        A = \left(\frac{19000\beta}{Re_{D}}\right)^{0.8}
        
        Re_D = \frac{\rho v D}{\mu}
        
        
    If D < 71.12 mm (2.8 in.):
        
    .. math::
        C += 0.11(0.75-\beta)\left(2.8-\frac{D}{0.0254}\right)
        
    If the orifice has corner taps:
        
    .. math::
        L_1 = L_2' = 0
        
    If the orifice has D and D/2 taps:
        
    .. math::
        L_1 = 1
        
        L_2' = 0.47
    
    If the orifice has Flange taps:
    
    .. math::
        L_1 = L_2' = \frac{0.0254}{D}
        
    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of orifice at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of orifice at the cross-section of
        the pressure tap, [Pa]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    k : float
        Isentropic exponent of fluid, [-]
    m : float
        Mass flow rate of fluid through the orifice, [kg/s]
    taps : str
        The orientation of the taps; one of 'corner', 'flange', 'D', or 'D/2',
        [-]
        
    Returns
    -------
    C : float
        Coefficient of discharge of the orifice, [-]

    Notes
    -----
    The following limits apply to the orifice plate standard [1]_:
        
    The measured pressure difference for the orifice plate should be under
    250 kPa.
    
    There are roughness limits as well; the roughness should be under 6 
    micrometers, although there are many more conditions to that given in [1]_.
        
    For orifice plates with D and D/2 or corner pressure taps:
        
    * Orifice bore diameter muse be larger than 12.5 mm (0.5 inches)
    * Pipe diameter between 50 mm and 1 m (2 to 40 inches)
    * Beta between 0.1 and 0.75 inclusive
    * Reynolds number larger than 5000 (for :math:`0.10 \le \beta \le 0.56`)
      or for :math:`\beta \ge 0.56, Re_D \ge 16000\beta^2`
      
    For orifice plates with flange pressure taps:

    * Orifice bore diameter muse be larger than 12.5 mm (0.5 inches)
    * Pipe diameter between 50 mm and 1 m (2 to 40 inches)
    * Beta between 0.1 and 0.75 inclusive
    * Reynolds number larger than 5000 and also larger than :math:`170000\beta^2 D`.
    
    

    Examples
    --------
    >>> C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, P1=1E5, P2=9.9E4, 
    ... rho=1.165, mu=1.85E-5, m=0.12, k=1.4, taps='flange')
    0.5990326277163659
    
    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement 
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    '''
    A_pipe = pi/4.*D*D
    v = m/(A_pipe*rho)
    Re_D = rho*v*D/mu
    
    beta = Do/D
    dP = P1 - P2
    if taps == 'corner':
        L1, L2_prime = 0.0, 0.0
    elif taps == 'D' or taps == 'D/2':
        L1 = 1.0
        L2_prime = 0.47
    elif taps == 'flange':
        L1 = L2_prime = 0.0254/D
    else:
        raise Exception('Unsupported tap location')
        
    beta2 = beta*beta
    beta4 = beta2*beta2
    beta8 = beta4*beta4
    
    A = (19000.0*beta/Re_D)**0.8
    M2_prime = 2*L2_prime/(1.0 - beta)
    
    C = (0.5961 + 0.0261*beta2 - 0.216*beta8 + 0.000521*(1E6*beta/Re_D)**0.7
         + (0.0188 + 0.0063*A)*beta**3.5*(1E6/Re_D)**0.3
         + (0.043 + 0.080*exp(-1E1*L1) - 0.123*exp(-7.0*L1))*(1.0 - 0.11*A)*beta4/(1.0 - beta4)
         -0.031*(M2_prime - 0.8*M2_prime**1.1)*beta**1.3)
    if D < 0.07112:
        C += 0.011*(0.75 - beta)*(2.8 - D/0.0254)
    
    return C


def Reader_Harris_Gallagher_discharge(D, Do, P1, P2, rho, mu, k, taps='corner'):
    r'''Calculates the mass flow rate of fluid through an orifice based on the 
    geometry of the plate, measured pressures of the orifice, and the density, 
    viscosity, and isentropic exponent of the fluid. This solves an equation
    iteratively to obtain the correct flow rate.
        
    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of orifice at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of orifice at the cross-section of
        the pressure tap, [Pa]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    k : float
        Isentropic exponent of fluid, [-]
    taps : str
        The orientation of the taps; one of 'corner', 'flange', 'D', or 'D/2',
        [-]
        
    Returns
    -------
    m : float
        Mass flow rate of fluid through the orifice, [kg/s]

    Notes
    -----

    Examples
    --------
    >>> Reader_Harris_Gallagher_discharge(D=0.07366, Do=0.05, P1=200000.0, 
    ... P2=183000.0, rho=999.1, mu=0.0011, k=1.33, taps='D')
    7.702338035732167
    
    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement 
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    '''
    def to_solve(m):
        C = C_Reader_Harris_Gallagher(D=D, Do=Do, P1=P1, P2=P2, 
            rho=rho, mu=mu, m=m, k=k, taps=taps)
        epsilon = orifice_expansibility(D=D, Do=Do, P1=P1, P2=P2, k=k)
        m_calc = orifice_discharge(D=D, Do=Do, P1=P1, P2=P2, rho=rho, 
                                    C=C, expansibility=epsilon)
        return m - m_calc

    return newton(to_solve, 2.81)
