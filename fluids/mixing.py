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
from scipy.constants import g
from math import log, pi

__all__ = ['adjust_homogeneity', 'agitator_time_homogeneous',
'Kp_helical_ribbon_Rieger', 'time_helical_ribbon_Grenville', 'size_tee',
'COV_motionless_mixer', 'K_motionless_mixer']

max_Fo_for_turbulent = 1/1225.
min_regime_constant_for_turbulent = 6370.

def adjust_homogeneity(fraction):
    '''Base: 95% homogeneity'''
    multiplier = log(1-fraction)/log(0.05)
    return multiplier


def agitator_time_homogeneous(D=None, N=None, P=None, T=None, H=None, mu=None, rho=None, homogeneity=.95):
    r'''Calculates time for a fluid mizing in a tank with an impeller to
    reach a specified level of homogeneity, according to [1]_.

    .. math::
        N_p = \frac{Pg}{\rho N^3 D^5}

        Re_{imp} = \frac{\rho D^2 N}{\mu}

        \text{constant} = N_p^{1/3} Re_{imp}

        Fo = 5.2/\text{constant} \text{for turbulent regime}

        Fo = (183/\text{constant})^2 \text{for transition regime}

    Parameters
    ----------
    D : float
        Impeller diameter (optional) [m]
    N : float:
        Speed of impeller, [r/s]
    P : float
        Actual power required to mix, ignoring mechanical inefficiencies [W]
    T : float
        Tank diameter, [m]
    H : float
        Tank height, [m]
    mu : float
        Mixture viscosity, [Pa*s]
    rho : float
        Mixture density, [kg/m^3]
    homogeneity : float
        Fraction completion of mixing, optional, []

    Returns
    -------
    t : float
        Time for specified degree of homogeneity [s]

    Notes
    -----
    If impeller diameter is not specified, assumed to be 0.5 tank diameters.

    The first example is solved forward rather than backwards here. A rather
    different result is obtained, but is accurate.

    No check to see if the mixture if laminar is currently implemented.
    This would underpredict the required time.

    Examples
    --------
    >>> agitator_time_homogeneous(D=36*.0254, N=56/60., P=957., T=1.83, H=1.83, mu=0.018, rho=1020, homogeneity=.995)
    15.143198226374668

    >>> agitator_time_homogeneous(D=1, N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)
    67.7575069865228

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    '''
    if not D:
        D = T*0.5
    Np = P*g/rho/N**3/D**5
    Re_imp = rho/mu*D**2*N
    regime_constant = Np**(1/3.)*Re_imp
    if regime_constant >= min_regime_constant_for_turbulent:
        Fo = (5.2/regime_constant)
    else:
        Fo = (183./regime_constant)**2
    time = rho*T**1.5*H**0.5/mu*Fo
    multiplier = adjust_homogeneity(homogeneity)
    time = time*multiplier
    return time

#print [agitator_time_homogeneous(D=1, N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)]
#print 'example 2:'
#print [agitator_time_homogeneous(D=36*.0254, N=56/60., P=957., T=1.83, H=1.83, mu=0.018, rho=1020, homogeneity=.995)]

def Kp_helical_ribbon_Rieger(D=None, h=None, nb=None, pitch=None, width=None, T=None):
    r'''Calculates product of power number and reynolds number for a
    specified geometry for a heilical ribbon mixer in the laminar regime.
    One of several correlations listed in [1]_, it used more data than other
    listed correlations and was recommended.

    .. math::
        K_p = 82.8\frac{h}{D}\left(\frac{c}{D}\right)^{-0.38} \left(\frac{p}{D}\right)^{-0.35}
        \left(\frac{w}{D}\right)^{0.20} n_b^{0.78}

    Parameters
    ----------
    D : float
        Impeller diameter (optional) [m]
    h : float
        Ribbon mixer height, [m]
    nb : float:
        Number of blades, [-]
    pitch : float
         Height of one turn around a helix [m]
    width : float
         Width of one blade [m]
    T : float
        Tank diameter, [m]

    Returns
    -------
    Kp : float
        Product of power number and reynolds number for laminar regime []

    Notes
    -----
    Example is from example 9-6 in [1]_. Confirmed.

    Examples
    --------
    >>> Kp_helical_ribbon_Rieger(D=1.9, h=1.9, nb=2, pitch=1.9, width=.19, T=2)
    357.39749163259256

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    .. [2] Rieger, F., V. Novak, and D. Havelkov (1988). The influence of the
       geometrical shape on the power requirements of ribbon impellers,
       Int. Chem. Eng., 28, 376-383.
    '''
    c = (T-D)/2
    Kp = 82.8*h/D*(c/D)**-.38*(pitch/D)**-0.35*(width/D)**0.2*nb**0.78
    return Kp

#print [Kp_helical_ribbon_Rieger(D=1.9, h=1.9, nb=2, pitch=1.9, width=.19, T=2)]

def time_helical_ribbon_Grenville(Kp, N):
    r'''Calculates product of time required for mixing in a helical ribbon
    coil in the laminar regime according to the Grenville [2]_ method
    recommended in [1]_.

    .. math::
        t = 896\times10^3K_p^{-1.69}/N

    Parameters
    ----------
    Kp : float
        Product of power number and reynolds number for laminar regime []
    N : float:
        Speed of impeller, [r/s]

    Returns
    -------
    t : float
        Time for homogeneity [s]

    Notes
    -----
    Degree of homogeneity is not specified.
    Example is from example 9-6 in [1]_. Confirmed.

    Examples
    --------
    >>> time_helical_ribbon_Grenville(357.4, 4/60.)
    650.980654028894

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    .. [2] Grenville, R. K., T. M. Hutchinson, and R. W. Higbee (2001).
       Optimisation of helical ribbon geometry for blending in the laminar
       regime, presented at MIXING XVIII, NAMF.
    '''
    t = 896E3*Kp**-1.69/N
    return t

#print [time_helical_ribbon_Grenville(357.4, 4/60.)]


### Tee mixer

def size_tee(Q1=None, Q2=None, D=None, D2=None, n=1, pipe_diameters=5):
    r'''Calculates CoV of an optimal or specified tee for mixing at a tee
    according to [1]_. Assumes turbulent flow.
    The smaller stream in injected into the main pipe, which continues
    straight.
    COV calculation is according to [2]_.

    .. math::
        TODO

    Parameters
    ----------
    Q1 : float
        Volumetric flow rate of larger stream [m^3/s]
    Q2 : float
        Volumetric flow rate of smaller stream [m^3/s]
    D : float
        Diameter of pipe after tee [m]
    D2 : float
        Diameter of mixing inlet, optional (optimally calculated if not
        specified) [m]
    n : float
        Number of jets, 1 to 4 []
    pipe_diameters : float
        Number of diameters along tail pipe for CoV calculation, 0 to 5 []

    Returns
    -------
    CoV : float
        Standard deviation of dimentionless concentration [-]

    Notes
    -----
    Not specified if this works for liquid also, though probably not.
    Example is from example Example 9-6 in [1]_. Low precision used in example.

    Examples
    --------
    >>> size_tee(Q1=11.7, Q2=2.74, D=0.762, D2=None, n=1, pipe_diameters=5)
    0.2940930233038544

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    .. [2] Giorges, Aklilu T. G., Larry J. Forney, and Xiaodong Wang.
       "Numerical Study of Multi-Jet Mixing." Chemical Engineering Research and
       Design, Fluid Flow, 79, no. 5 (July 2001): 515-22.
       doi:10.1205/02638760152424280.
    '''
    V1 = Q1/(pi/4*D**2)
#    print 'V1', V1
    Cv = Q2/(Q1 + Q2)
    COV0 = ((1-Cv)/Cv)**0.5
#    print 'COV0', COV0
    if not D2:
        D2 = (Q2/Q1)**(2/3.)*D
    V2 = Q2/(pi/4*D2**2)
#    V2 = 45.67
#    print 'D2, V2', D2, V2
    B = n**2*(D2/D)**2*(V2/V1)**2
#    print 'B', B
    if not n == 1 and not n == 2 and not n == 3 and not n ==4:
        raise Exception('Only 1 or 4 side streams investigated')
    if n == 1:
        if B < 0.7:
            E = 1.33
        else:
            E = 1/33. + 0.95*log(B/0.7)
    elif n == 2:
        if B < 0.8:
            E = 1.44
        else:
            E = 1.44 + 0.95*log(B/0.8)**1.5
    elif n == 3:
        if B < 0.8:
            E = 1.75
        else:
            E = 1.75 + 0.95*log(B/0.8)**1.8
    else:
        if B < 2:
            E = 1.97
        else:
            E = 1.97 + 0.95*log(B/2.)**2
    COV = (0.32/B**0.86*(pipe_diameters)**-E )**0.5
    return COV

### Commercial motionless mixers
'''Data from:
Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
Handbook of Industrial Mixing: Science and Practice.
Hoboken, N.J.: Wiley-Interscience, 2004.'''
StatixMixers = {}
StatixMixers['KMS'] = {'Name': 'KMS', 'Vendor': 'Chemineer', 'Description': 'Twisted ribbon. Alternating left and right twists.', 'KL': 6.9, 'KiL': 0.87, 'KT': 150, 'KiT': 0.5}
StatixMixers['SMX'] = {'Name': 'SMX', 'Vendor': 'Koch-Glitsch', 'Description': 'Guide vanes 45 degrees to pipe axis. Adjacent elements rotated 90 degrees.', 'KL': 37.5, 'KiL': 0.63, 'KT': 500, 'KiT': 0.46}
StatixMixers['SMXL'] = {'Name': 'SMXL', 'Vendor': 'Koch-Glitsch', 'Description': 'Similar to SMX, but intersection bars at 30 degrees to pipe axis.', 'KL': 7.8, 'KiL': 0.85, 'KT': 100, 'KiT': 0.87}
StatixMixers['SMF'] = {'Name': 'SMF', 'Vendor': 'Koch-Glitsch', 'Description': 'Three guide vanes projecting from the tube wall in a way as to not contact. Designed for applications subject to plugging.', 'KL': 5.6, 'KiL': 0.83, 'KT': 130, 'KiT': 0.4}


def COV_motionless_mixer(Ki=None, Q1=None, Q2=None, pipe_diameters=None):
    r'''Calculates CoV of a motionless mixer with a regression parameter in
    [1]_ and originally in [2]_.

    .. math::
        \frac{CoV}{CoV_0} = K_i^{L/D}

    Parameters
    ----------
    Ki : float
        Correlation parameter specific to a mixer's design, [-]
    Q1 : float
        Volumetric flow rate of larger stream [m^3/s]
    Q2 : float
        Volumetric flow rate of smaller stream [m^3/s]
    pipe_diameters : float
        Number of diameters along tail pipe for CoV calculation, 0 to 5 []

    Returns
    -------
    CoV : float
        Standard deviation of dimentionless concentration [-]

    Notes
    -----
    Example 7-8.3.2 in [1]_, solved backwards.

    Examples
    --------
    >>> COV_motionless_mixer(Ki=.33, Q1=11.7, Q2=2.74, pipe_diameters=4.74/.762)
    0.0020900028665727685

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    .. [2] Streiff, F. A., S. Jaffer, and G. Schneider (1999). Design and
       application of motionless mixer technology, Proc. ISMIP3, Osaka,
       pp. 107-114.
    '''
    Cv = Q2/(Q1 + Q2)
    COV0 = ((1-Cv)/Cv)**0.5
    COVr = Ki**(pipe_diameters)
    COV = COV0*COVr
    return COV


def K_motionless_mixer(K=None, L=None, D=None, fd=None):
    r'''Calculates loss ciefficient of a motionless mixer with a regression
    parameter in [1]_ and originally in [2]_.

    .. math::
        K = K_{L/T}f\frac{L}{D}

    Parameters
    ----------
    K : float
        Correlation parameter specific to a mixer's design, [-]
        Also specific to laminar or turbulent regime.
    L : float
        Length of the motionless mixer [m]
    D : float
        Diameter of pipe [m]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient of mixer [-]

    Notes
    -----
    Related to example 7-8.3.2 in [1]_.

    Examples
    --------
    >>> K_motionless_mixer(K=150, L=.762*5, D=.762, fd=.01)
    7.5

    References
    ----------
    .. [1] Paul, Edward L, Victor A Atiemo-Obeng, and Suzanne M Kresta.
       Handbook of Industrial Mixing: Science and Practice.
       Hoboken, N.J.: Wiley-Interscience, 2004.
    .. [2] Streiff, F. A., S. Jaffer, and G. Schneider (1999). Design and
       application of motionless mixer technology, Proc. ISMIP3, Osaka,
       pp. 107-114.
    '''
    K = L/D*fd*K
    return K

#print K_motionless_mixer(K=150, L=.762*5, D=.762, fd=.01)