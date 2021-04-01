# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains correlations, standards, and solvers for orifice plates
and other flow metering devices. Both permanent and measured pressure drop
is included, and models work for both liquids and gases. A number of
non-standard devices are included, as well as limited two-phase functionality.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Flow Meter Solvers
------------------
.. autofunction:: differential_pressure_meter_solver

Flow Meter Interfaces
---------------------
.. autofunction:: differential_pressure_meter_dP
.. autofunction:: differential_pressure_meter_C_epsilon
.. autofunction:: differential_pressure_meter_beta
.. autofunction:: dP_orifice

Orifice Plate Correlations
--------------------------
.. autofunction:: C_Reader_Harris_Gallagher
.. autofunction:: C_eccentric_orifice_ISO_15377_1998
.. autofunction:: C_quarter_circle_orifice_ISO_15377_1998
.. autofunction:: C_Miller_1996
.. autofunction:: orifice_expansibility
.. autofunction:: orifice_expansibility_1989
.. autodata:: ISO_15377_CONICAL_ORIFICE_C

Nozzle Flow Meters
------------------
.. autofunction:: C_long_radius_nozzle
.. autofunction:: C_ISA_1932_nozzle
.. autofunction:: C_venturi_nozzle
.. autofunction:: nozzle_expansibility

Venturi Tube Meters
-------------------
.. autodata:: ROUGH_WELDED_CONVERGENT_VENTURI_TUBE_C
.. autodata:: MACHINED_CONVERGENT_VENTURI_TUBE_C
.. autodata:: AS_CAST_VENTURI_TUBE_C
.. autofunction:: dP_venturi_tube
.. autofunction:: C_Reader_Harris_Gallagher_wet_venturi_tube
.. autofunction:: dP_Reader_Harris_Gallagher_wet_venturi_tube

Cone Meters
-----------
.. autodata:: CONE_METER_C
.. autofunction:: diameter_ratio_cone_meter
.. autofunction:: cone_meter_expansibility_Stewart
.. autofunction:: dP_cone_meter

Wedge Meters
------------
.. autofunction:: C_wedge_meter_ISO_5167_6_2017
.. autofunction:: C_wedge_meter_Miller
.. autofunction:: diameter_ratio_wedge_meter
.. autofunction:: dP_wedge_meter

Flow Meter Utilities
--------------------
.. autofunction:: discharge_coefficient_to_K
.. autofunction:: K_to_discharge_coefficient
.. autofunction:: velocity_of_approach_factor
.. autofunction:: flow_coefficient
.. autofunction:: flow_meter_discharge
.. autodata:: all_meters

"""

from __future__ import division
from math import sqrt, cos, sin, tan, atan, pi, radians, exp, acos, log10, log
from fluids.friction import friction_factor
from fluids.core import Froude_densimetric
from fluids.numerics import interp, secant, brenth, NotBoundedError, implementation_optimize_tck, bisplev
from fluids.constants import g, inch, inch_inv, pi_inv

__all__ = ['C_Reader_Harris_Gallagher',
           'differential_pressure_meter_solver',
           'differential_pressure_meter_dP',
           'flow_meter_discharge', 'orifice_expansibility',
           'discharge_coefficient_to_K', 'K_to_discharge_coefficient',
           'dP_orifice', 'velocity_of_approach_factor',
           'flow_coefficient', 'nozzle_expansibility',
           'C_long_radius_nozzle', 'C_ISA_1932_nozzle', 'C_venturi_nozzle',
           'orifice_expansibility_1989', 'dP_venturi_tube',
           'diameter_ratio_cone_meter', 'diameter_ratio_wedge_meter',
           'cone_meter_expansibility_Stewart', 'dP_cone_meter',
           'C_wedge_meter_Miller', 'C_wedge_meter_ISO_5167_6_2017',
           'dP_wedge_meter',
           'C_Reader_Harris_Gallagher_wet_venturi_tube',
           'dP_Reader_Harris_Gallagher_wet_venturi_tube',
           'differential_pressure_meter_C_epsilon',
           'differential_pressure_meter_beta',
           'C_eccentric_orifice_ISO_15377_1998',
           'C_quarter_circle_orifice_ISO_15377_1998',
           'C_Miller_1996',
           'all_meters',
           ]


CONCENTRIC_ORIFICE = 'orifice' # normal
ECCENTRIC_ORIFICE = 'eccentric orifice'
CONICAL_ORIFICE = 'conical orifice'
SEGMENTAL_ORIFICE = 'segmental orifice'
QUARTER_CIRCLE_ORIFICE = 'quarter circle orifice'
CONDITIONING_4_HOLE_ORIFICE = 'Rosemount 4 hole self conditioing'
ORIFICE_HOLE_TYPES = [CONCENTRIC_ORIFICE, ECCENTRIC_ORIFICE, CONICAL_ORIFICE,
                      SEGMENTAL_ORIFICE, QUARTER_CIRCLE_ORIFICE]

ORIFICE_CORNER_TAPS = 'corner'
ORIFICE_FLANGE_TAPS = 'flange'
ORIFICE_D_AND_D_2_TAPS = 'D and D/2'
ORIFICE_PIPE_TAPS = 'pipe' # Not in ISO 5167
ORIFICE_VENA_CONTRACTA_TAPS = 'vena contracta' # Not in ISO 5167, normally segmental or eccentric orifices

# Used by miller; modifier on taps
TAPS_OPPOSITE = '180 degree'
TAPS_SIDE = '90 degree'


ISO_5167_ORIFICE = 'ISO 5167 orifice'
ISO_15377_ECCENTRIC_ORIFICE = 'ISO 15377 eccentric orifice'
ISO_15377_QUARTER_CIRCLE_ORIFICE = 'ISO 15377 quarter-circle orifice'
ISO_15377_CONICAL_ORIFICE = 'ISO 15377 conical orifice'

MILLER_ORIFICE = 'Miller orifice'
MILLER_ECCENTRIC_ORIFICE = 'Miller eccentric orifice'
MILLER_SEGMENTAL_ORIFICE = 'Miller segmental orifice'
MILLER_CONICAL_ORIFICE = 'Miller conical orifice'
MILLER_QUARTER_CIRCLE_ORIFICE = 'Miller quarter circle orifice'

UNSPECIFIED_METER = 'unspecified meter'


LONG_RADIUS_NOZZLE = 'long radius nozzle'
ISA_1932_NOZZLE = 'ISA 1932 nozzle'
VENTURI_NOZZLE = 'venuri nozzle'

AS_CAST_VENTURI_TUBE = 'as cast convergent venturi tube'
MACHINED_CONVERGENT_VENTURI_TUBE = 'machined convergent venturi tube'
ROUGH_WELDED_CONVERGENT_VENTURI_TUBE = 'rough welded convergent venturi tube'


HOLLINGSHEAD_ORIFICE = 'Hollingshead orifice'
HOLLINGSHEAD_VENTURI_SMOOTH = 'Hollingshead venturi smooth'
HOLLINGSHEAD_VENTURI_SHARP = 'Hollingshead venturi sharp'
HOLLINGSHEAD_CONE = 'Hollingshead v cone'
HOLLINGSHEAD_WEDGE = 'Hollingshead wedge'


CONE_METER = 'cone meter'
WEDGE_METER = 'wedge meter'
__all__.extend(['ISO_5167_ORIFICE','ISO_15377_ECCENTRIC_ORIFICE', 'MILLER_ORIFICE',
                'MILLER_ECCENTRIC_ORIFICE', 'MILLER_SEGMENTAL_ORIFICE',
                'LONG_RADIUS_NOZZLE', 'ISA_1932_NOZZLE',
                'VENTURI_NOZZLE', 'AS_CAST_VENTURI_TUBE',
                'MACHINED_CONVERGENT_VENTURI_TUBE',
                'ROUGH_WELDED_CONVERGENT_VENTURI_TUBE', 'CONE_METER',
                'WEDGE_METER', 'ISO_15377_CONICAL_ORIFICE',
                'MILLER_CONICAL_ORIFICE',
                'MILLER_QUARTER_CIRCLE_ORIFICE',
                'ISO_15377_QUARTER_CIRCLE_ORIFICE', 'UNSPECIFIED_METER',
                'HOLLINGSHEAD_ORIFICE', 'HOLLINGSHEAD_CONE', 'HOLLINGSHEAD_WEDGE',
                'HOLLINGSHEAD_VENTURI_SMOOTH', 'HOLLINGSHEAD_VENTURI_SHARP'])

__all__.extend(['ORIFICE_CORNER_TAPS', 'ORIFICE_FLANGE_TAPS',
                'ORIFICE_D_AND_D_2_TAPS', 'ORIFICE_PIPE_TAPS',
                'ORIFICE_VENA_CONTRACTA_TAPS', 'TAPS_OPPOSITE', 'TAPS_SIDE'])

__all__.extend(['CONCENTRIC_ORIFICE', 'ECCENTRIC_ORIFICE',
                'CONICAL_ORIFICE', 'SEGMENTAL_ORIFICE',
                'QUARTER_CIRCLE_ORIFICE'])


def flow_meter_discharge(D, Do, P1, P2, rho, C, expansibility=1.0):
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
    This is formula 1-12 in [1]_ and also [2]_.

    Examples
    --------
    >>> flow_meter_discharge(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1646,
    ... C=0.5988, expansibility=0.9975)
    0.01120390943807026

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    '''
    beta = Do/D
    beta2 = beta*beta
    return (0.25*pi*Do*Do)*C*expansibility*sqrt((2.0*rho*(P1 - P2))/(1.0 - beta2*beta2))


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

    It is said in [1]_ that for liquids this should not be used. The result
    can be forced by setting `k` to a really high number like 1E20.

    Examples
    --------
    >>> orifice_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    0.9974739057343425

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    '''
    beta = Do/D
    beta2 = beta*beta
    beta4 = beta2*beta2
    return (1.0 - (0.351 + beta4*(0.93*beta4 + 0.256))*(
            1.0 - (P2/P1)**(1./k)))


def orifice_expansibility_1989(D, Do, P1, P2, k):
    r'''Calculates the expansibility factor for orifice plate calculations
    based on the geometry of the plate, measured pressures of the orifice, and
    the isentropic exponent of the fluid.

    .. math::
        \epsilon = 1- (0.41 + 0.35\beta^4)\Delta P/\kappa/P_1

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
    expansibility : float
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Notes
    -----
    This formula was determined for the range of P2/P1 >= 0.75, and for fluids
    of air, steam, and natural gas. However, there is no objection to using
    it for other fluids.

    This is an older formula used to calculate expansibility factors for
    orifice plates.

    In this standard, an expansibility factor formula transformation in terms
    of the pressure after the orifice is presented as well. This is the more
    standard formulation in terms of the upstream conditions. The other formula
    is below for reference only:

    .. math::
        \epsilon_2 = \sqrt{1 + \frac{\Delta P}{P_2}} -  (0.41 + 0.35\beta^4)
        \frac{\Delta P}{\kappa P_2 \sqrt{1 + \frac{\Delta P}{P_2}}}

    [2]_ recommends this formulation for wedge meters as well.

    Examples
    --------
    >>> orifice_expansibility_1989(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    0.9970510687411718

    References
    ----------
    .. [1] American Society of Mechanical Engineers. MFC-3M-1989 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2005.
    .. [2] Miller, Richard W. Flow Measurement Engineering Handbook. 3rd
       edition. New York: McGraw-Hill Education, 1996.
    '''
    return 1.0 - (0.41 + 0.35*(Do/D)**4)*(P1 - P2)/(k*P1)


def C_Reader_Harris_Gallagher(D, Do, rho, mu, m, taps='corner'):
    r'''Calculates the coefficient of discharge of the orifice based on the
    geometry of the plate, measured pressures of the orifice, mass flow rate
    through the orifice, and the density and viscosity of the fluid.

    .. math::
        C = 0.5961 + 0.0261\beta^2 - 0.216\beta^8 + 0.000521\left(\frac{
        10^6\beta}{Re_D}\right)^{0.7}\\
        + (0.0188 + 0.0063A)\beta^{3.5} \left(\frac{10^6}{Re_D}\right)^{0.3} \\
        +(0.043 + 0.080\exp(-10L_1) -0.123\exp(-7L_1))(1-0.11A)\frac{\beta^4}
        {1-\beta^4} \\
        -  0.031(M_2' - 0.8M_2'^{1.1})\beta^{1.3}

    .. math::
        M_2' = \frac{2L_2'}{1-\beta}

    .. math::
        A = \left(\frac{19000\beta}{Re_{D}}\right)^{0.8}

    .. math::
        Re_D = \frac{\rho v D}{\mu}


    If D < 71.12 mm (2.8 in.) (Note this is a continuous addition; there is no
    discontinuity):

    .. math::
        C += 0.11(0.75-\beta)\left(2.8-\frac{D}{0.0254}\right)

    If the orifice has corner taps:

    .. math::
        L_1 = L_2' = 0

    If the orifice has D and D/2 taps:

    .. math::
        L_1 = 1

    .. math::
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
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
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
    * Reynolds number larger than 5000 and also larger than
      :math:`170000\beta^2 D`.

    This is also presented in Crane's TP410 (2009) publication, whereas the
    1999 and 1982 editions showed only a graph for discharge coefficients.

    Examples
    --------
    >>> C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5,
    ... m=0.12, taps='flange')
    0.5990326277163659

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    .. [3] Reader-Harris, M. J., "The Equation for the Expansibility Factor for
       Orifice Plates," Proceedings of FLOMEKO 1998, Lund, Sweden, 1998:
       209-214.
    .. [4] Reader-Harris, Michael. Orifice Plates and Venturi Tubes. Springer,
       2015.
    '''
    A_pipe = 0.25*pi*D*D
    v = m/(A_pipe*rho)
    Re_D = rho*v*D/mu
    Re_D_inv = 1.0/Re_D

    beta = Do/D
    if taps == 'corner':
        L1, L2_prime = 0.0, 0.0
    elif taps == 'flange':
        L1 = L2_prime = 0.0254/D
    elif taps  == 'D' or taps == 'D/2' or taps == ORIFICE_D_AND_D_2_TAPS:
        L1 = 1.0
        L2_prime = 0.47
    else:
        raise ValueError('Unsupported tap location')

    beta2 = beta*beta
    beta4 = beta2*beta2
    beta8 = beta4*beta4

    A = 2648.5177066967326*(beta*Re_D_inv)**0.8 # 19000.0^0.8 = 2648.51....
    M2_prime = 2.0*L2_prime/(1.0 - beta)

    # These two exps
    expnL1 = exp(-L1)
    expnL2 = expnL1*expnL1
    expnL3 = expnL1*expnL2
    delta_C_upstream = ((0.043 + expnL3*expnL2*expnL2*(0.080*expnL3 - 0.123))
            *(1.0 - 0.11*A)*beta4/(1.0 - beta4))

    # The max part is not in the ISO standard
    t1 = log10(3700.*Re_D_inv)
    if t1 < 0.0:
        t1 = 0.0
    delta_C_downstream = (-0.031*(M2_prime - 0.8*M2_prime**1.1)*beta**1.3
                          *(1.0 + 8.0*t1))

    # C_inf is discharge coefficient with corner taps for infinite Re
    # Cs, slope term, provides increase in discharge coefficient for lower
    # Reynolds numbers.
    x1 = 63.095734448019314*(Re_D_inv)**0.3 # 63.095... = (1e6)**0.3
    x2 = 22.7 - 0.0047*Re_D
    t2 = x1 if x1 > x2 else x2
    # max term is not in the ISO standard
    C_inf_C_s = (0.5961 + 0.0261*beta2 - 0.216*beta8
                 + 0.000521*(1E6*beta*Re_D_inv)**0.7
                 + (0.0188 + 0.0063*A)*beta2*beta*sqrt(beta)*(
                 t2))

    C = (C_inf_C_s + delta_C_upstream + delta_C_downstream)
    if D < 0.07112:
        # Limit is 2.8 inches, .1 inches smaller than the internal diameter of
        # a sched. 80 pipe.
        # Suggested to be required not becausue of any effect of small
        # diameters themselves, but because of edge radius differences.
        # max term is given in [4]_ Reader-Harris, Michael book
        # There is a check for t3 being negative and setting it to zero if so
        # in some sources but that only occurs when t3 is exactly the limit
        # (0.07112) so it is not needed
        t3 = (2.8 - D*inch_inv)
        delta_C_diameter = 0.011*(0.75 - beta)*t3
        C += delta_C_diameter

    return C


_Miller_1996_unsupported_type = "Supported orifice types are %s" %str(
        (CONCENTRIC_ORIFICE, SEGMENTAL_ORIFICE, ECCENTRIC_ORIFICE,
         CONICAL_ORIFICE, QUARTER_CIRCLE_ORIFICE))
_Miller_1996_unsupported_tap_concentric = "Supported taps for subtype '%s' are %s" %(
        CONCENTRIC_ORIFICE, (ORIFICE_CORNER_TAPS, ORIFICE_FLANGE_TAPS,
                             ORIFICE_D_AND_D_2_TAPS, ORIFICE_PIPE_TAPS))
_Miller_1996_unsupported_tap_pos_eccentric = "Supported tap positions for subtype '%s' are %s" %(
        ECCENTRIC_ORIFICE, (TAPS_OPPOSITE, TAPS_SIDE))
_Miller_1996_unsupported_tap_eccentric = "Supported taps for subtype '%s' are %s" %(
        ECCENTRIC_ORIFICE, (ORIFICE_FLANGE_TAPS, ORIFICE_VENA_CONTRACTA_TAPS))
_Miller_1996_unsupported_tap_segmental = "Supported taps for subtype '%s' are %s" %(
        SEGMENTAL_ORIFICE, (ORIFICE_FLANGE_TAPS, ORIFICE_VENA_CONTRACTA_TAPS))

def C_Miller_1996(D, Do, rho, mu, m, subtype='orifice',
                  taps=ORIFICE_CORNER_TAPS, tap_position=TAPS_OPPOSITE):
    r'''Calculates the coefficient of discharge of any of the orifice types
    supported by the Miller (1996) [1]_ correlation set. These correlations
    cover a wide range of industrial applications and sizes. Most of them are
    functions of `beta` ratio and Reynolds number. Unlike the ISO standards,
    these correlations do not come with well defined ranges of validity, so
    caution should be applied using there correlations.

    The base equation is as follows, and each orifice type and range has
    different values or correlations for :math:`C_{\infty}`, `b`, and `n`.

    .. math::
        C = C_{\infty} + \frac{b}{{Re}_D^n}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    m : float
        Mass flow rate of fluid through the orifice, [kg/s]
    subtype : str, optional
        One of 'orifice', 'eccentric orifice', 'segmental orifice',
        'conical orifice', or 'quarter circle orifice', [-]
    taps : str, optional
        The orientation of the taps; one of 'corner', 'flange',
        'D and D/2', 'pipe', or 'vena contracta'; not all orifice subtypes
        support the all tap types [-]
    tap_position : str, optional
        The rotation of the taps, used **only for the eccentric orifice case**
        where the pressure profile is are not symmetric; '180 degree' for the
        normal case where the taps are opposite the orifice bore, and
        '90 degree' for the case where, normally for operational reasons, the
        taps are near the bore [-]

    Returns
    -------
    C : float
        Coefficient of discharge of the orifice, [-]

    Notes
    -----
    Many of the correlations transition at a pipe diameter of 100 mm to
    different equations, which will lead to discontinuous behavior.

    It should also be noted the author of these correlations developed a
    commercial flow meter rating software package, at [2]_.
    He passed away in 2014, but contributed massively to the field of flow
    measurement.

    The numerous equations for the different cases are as follows:

    For all **regular (concentric) orifices**, the `b` equation is as follows
    and n = 0.75:

    .. math::
        b = 91.706\beta^{2.5}

    Regular (concentric) orifice, corner taps:

    .. math::
         C_{\infty} = 0.5959 + 0.0312\beta^2.1 - 0.184\beta^8

    Regular (concentric) orifice, flange taps, D > 58.4 mm:

    .. math::
         C_{\infty} = 0.5959 + 0.0312\beta^{2.1} - 0.184\beta^8
         + \frac{2.286\beta^4}{(D_{mm}(1.0 - \beta^4))}
         - \frac{0.856\beta^3}{D_{mm}}

    Regular (concentric) orifice, flange taps, D < 58.4 mm:

    .. math::
         C_{\infty} = 0.5959 + 0.0312\beta^{2.1} - 0.184\beta^8
         + \frac{0.039\beta^4}{(1.0 - \beta^4)} - \frac{0.856\beta^3}{D_{mm}}

    Regular (concentric) orifice, 'D and D/2' taps:

    .. math::
         C_{\infty} = 0.5959 + 0.0312\beta^{2.1} - 0.184\beta^8
         + \frac{0.039\beta^4}{(1.0 - \beta^4)} - 0.01584

    Regular (concentric) orifice, 'pipe' taps:

    .. math::
         C_{\infty} = 0.5959 + 0.461\beta^{2.1} + 0.48\beta^8
         + \frac{0.039\beta^4}{(1.0 - \beta^4)}

    For the case of a **conical orifice**, there is no tap dependence
    and one equation (`b` = 0, `n` = 0):

    .. math::
         C_{\infty} = 0.734 \text{ if } 250\beta \le Re \le 500\beta \text{ else } 0.730

    For the case of a **quarter circle orifice**, corner and flange taps have
    the same dependence (`b` = 0, `n` = 0):

    .. math::
         C_{\infty} = (0.7746 - 0.1334\beta^{2.1} + 1.4098\beta^8
                        + \frac{0.0675\beta^4}{(1 - \beta^4)} + 0.3865\beta^3)

    For all **segmental orifice** types, `b` = 0 and `n` = 0

    Segmental orifice, 'flange' taps, D < 10 cm:

    .. math::
         C_{\infty} = 0.6284 + 0.1462\beta^{2.1} - 0.8464\beta^8
         + \frac{0.2603\beta^4}{(1-\beta^4)} - 0.2886\beta^3

    Segmental orifice, 'flange' taps, D > 10 cm:

    .. math::
         C_{\infty} = 0.6276 + 0.0828\beta^{2.1} + 0.2739\beta^8
         - \frac{0.0934\beta^4}{(1-\beta^4)} - 0.1132\beta^3

    Segmental orifice, 'vena contracta' taps, D < 10 cm:

    .. math::
         C_{\infty} = 0.6261 + 0.1851\beta^{2.1} - 0.2879\beta^8
         + \frac{0.1170\beta^4}{(1-\beta^4)} - 0.2845\beta^3

    Segmental orifice, 'vena contracta' taps, D > 10 cm:

    .. math::
         C_{\infty} = 0.6276 + 0.0828\beta^{2.1} + 0.2739\beta^8
         - \frac{0.0934\beta^4}{(1-\beta^4)} - 0.1132\beta^3

    For all **eccentric orifice** types,  `n` = 0.75 and `b` is fit to a
    polynomial of `beta`.

    Eccentric orifice, 'flange' taps, 180 degree opposite taps, D < 10 cm:

    .. math::
        C_{\infty} = 0.5917 + 0.3061\beta^{2.1} + .3406\beta^8 -\frac{.1019\beta^4}{(1-\beta^4)} - 0.2715\beta^3

    .. math::
        b = 7.3 - 15.7\beta + 170.8\beta^2 - 399.7\beta^3 + 332.2\beta^4

    Eccentric orifice, 'flange' taps, 180 degree opposite taps, D > 10 cm:

    .. math::
        C_{\infty} = 0.6016 + 0.3312\beta^{2.1} -1.5581\beta^8 + \frac{0.6510\beta^4}{(1-\beta^4)} - 0.7308\beta^3

    .. math::
        b = -139.7 + 1328.8\beta - 4228.2\beta^2 + 5691.9\beta^3 - 2710.4\beta^4

    Eccentric orifice, 'flange' taps, 90 degree side taps, D < 10 cm:

    .. math::
        C_{\infty} = 0.5866 + 0.3917\beta^{2.1} + .7586\beta^8 - \frac{.2273\beta^4}{(1-\beta^4)} - .3343\beta^3

    .. math::
        b = 69.1 - 469.4\beta + 1245.6\beta^2 -1287.5\beta^3 + 486.2\beta^4

    Eccentric orifice, 'flange' taps, 90 degree side taps, D > 10 cm:

    .. math::
        C_{\infty} = 0.6037 + 0.1598\beta^{2.1} -.2918\beta^8 + \frac{0.0244\beta^4}{(1-\beta^4)} - 0.0790\beta^3

    .. math::
        b = -103.2 + 898.3\beta - 2557.3\beta^2 + 2977.0\beta^3 - 1131.3\beta^4

    Eccentric orifice, 'vena contracta' taps, 180 degree opposite taps, D < 10 cm:

    .. math::
        C_{\infty} = 0.5925 + 0.3380\beta^{2.1} + 0.4016\beta^8 - \frac{.1046\beta^4}{(1-\beta^4)} - 0.3212\beta^3

    .. math::
        b = 23.3 -207.0\beta + 821.5\beta^2 -1388.6\beta^3 + 900.3\beta^4

    Eccentric orifice, 'vena contracta' taps, 180 degree opposite taps, D > 10 cm:

    .. math::
        C_{\infty} = 0.5922 + 0.3932\beta^{2.1} + .3412\beta^8 - \frac{.0569\beta^4}{(1-\beta^4)} - 0.4628\beta^3

    .. math::
        b = 55.7 - 471.4\beta + 1721.8\beta^2 - 2722.6\beta^3 + 1569.4\beta^4

    Eccentric orifice, 'vena contracta' taps, 90 degree side taps, D < 10 cm:

    .. math::
        C_{\infty} = 0.5875 + 0.3813\beta^{2.1} + 0.6898\beta^8 - \frac{0.1963\beta^4}{(1-\beta^4)} - 0.3366\beta^3

    .. math::
        b = -69.3 + 556.9\beta - 1332.2\beta^2 + 1303.7\beta^3 - 394.8\beta^4

    Eccentric orifice, 'vena contracta' taps, 90 degree side taps, D > 10 cm:

    .. math::
        C_{\infty} = 0.5949 + 0.4078\beta^{2.1} + 0.0547\beta^8 + \frac{0.0955\beta^4}{(1-\beta^4)} - 0.5608\beta^3

    .. math::
        b = 52.8 - 434.2\beta + 1571.2\beta^2 - 2460.9\beta^3 + 1420.2\beta^4


    Examples
    --------
    >>> C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange', subtype='orifice')
    0.599065557156788

    References
    ----------
    .. [1] Miller, Richard W. Flow Measurement Engineering Handbook.
       McGraw-Hill Education, 1996.
    .. [2] "RW Miller & Associates." Accessed April 13, 2020.
       http://rwmillerassociates.com/.
    '''
    A_pipe = 0.25*pi*D*D
    v = m/(A_pipe*rho)
    Re = rho*v*D/mu
    D_mm = D*1000.0

    beta = Do/D
    beta3 = beta*beta*beta
    beta4 = beta*beta3
    beta8 = beta4*beta4
    beta21 = beta**2.1

    if subtype == MILLER_ORIFICE or subtype == CONCENTRIC_ORIFICE:
        b = 91.706*beta**2.5
        n = 0.75
        if taps == ORIFICE_CORNER_TAPS:
            C_inf = 0.5959 + 0.0312*beta21 - 0.184*beta8
        elif taps == ORIFICE_FLANGE_TAPS:
            if D_mm >= 58.4:
                C_inf = 0.5959 + 0.0312*beta21 - 0.184*beta8 + 2.286*beta4/(D_mm*(1.0 - beta4)) - 0.856*beta3/D_mm
            else:
                C_inf = 0.5959 + 0.0312*beta21 - 0.184*beta8 + 0.039*beta4/(1.0 - beta4) - 0.856*beta3/D_mm
        elif taps == ORIFICE_D_AND_D_2_TAPS:
            C_inf = 0.5959 + 0.0312*beta21 - 0.184*beta8 + 0.039*beta4/(1.0 - beta4) - 0.01584
        elif taps == ORIFICE_PIPE_TAPS:
            C_inf = 0.5959 + 0.461*beta21 + 0.48*beta8 + 0.039*beta4/(1.0 - beta4)
        else:
            raise ValueError(_Miller_1996_unsupported_tap_concentric)
    elif subtype == MILLER_ECCENTRIC_ORIFICE or subtype == ECCENTRIC_ORIFICE:
        if tap_position != TAPS_OPPOSITE and tap_position != TAPS_SIDE:
            raise ValueError(_Miller_1996_unsupported_tap_pos_eccentric)
        n = 0.75
        if taps == ORIFICE_FLANGE_TAPS:
            if tap_position == TAPS_OPPOSITE:
                if D < 0.1:
                    b = 7.3 - 15.7*beta + 170.8*beta**2 - 399.7*beta3 + 332.2*beta4
                    C_inf = 0.5917 + 0.3061*beta21 + .3406*beta8 -.1019*beta4/(1-beta4) - 0.2715*beta3
                else:
                    b = -139.7 + 1328.8*beta - 4228.2*beta**2 + 5691.9*beta3 - 2710.4*beta4
                    C_inf = 0.6016 + 0.3312*beta21 - 1.5581*beta8 + 0.6510*beta4/(1-beta4) - 0.7308*beta3
            elif tap_position == TAPS_SIDE:
                if D < 0.1:
                    b = 69.1 - 469.4*beta + 1245.6*beta**2 -1287.5*beta3 + 486.2*beta4
                    C_inf = 0.5866 + 0.3917*beta21 + 0.7586*beta8 -.2273*beta4/(1-beta4) - .3343*beta3
                else:
                    b = -103.2 + 898.3*beta - 2557.3*beta**2 + 2977.0*beta3 - 1131.3*beta4
                    C_inf = 0.6037 + 0.1598*beta21 - 0.2918*beta8 + 0.0244*beta4/(1-beta4) - 0.0790*beta3
        elif taps == ORIFICE_VENA_CONTRACTA_TAPS:
            if tap_position == TAPS_OPPOSITE:
                if D < 0.1:
                    b = 23.3 -207.0*beta + 821.5*beta**2 -1388.6*beta3 + 900.3*beta4
                    C_inf = 0.5925 + 0.3380*beta21 + 0.4016*beta8 -.1046*beta4/(1-beta4) - 0.3212*beta3
                else:
                    b = 55.7 - 471.4*beta + 1721.8*beta**2 - 2722.6*beta3 + 1569.4*beta4
                    C_inf = 0.5922 + 0.3932*beta21 + .3412*beta8 -.0569*beta4/(1-beta4) - 0.4628*beta3
            elif tap_position == TAPS_SIDE:
                if D < 0.1:
                    b = -69.3 + 556.9*beta - 1332.2*beta**2 + 1303.7*beta3 - 394.8*beta4
                    C_inf = 0.5875 + 0.3813*beta21 + 0.6898*beta8 -0.1963*beta4/(1-beta4) - 0.3366*beta3
                else:
                    b = 52.8 - 434.2*beta + 1571.2*beta**2 - 2460.9*beta3 + 1420.2*beta4
                    C_inf = 0.5949 + 0.4078*beta21 + 0.0547*beta8 +0.0955*beta4/(1-beta4) - 0.5608*beta3
        else:
            raise ValueError(_Miller_1996_unsupported_tap_eccentric)
    elif subtype == MILLER_SEGMENTAL_ORIFICE or subtype == SEGMENTAL_ORIFICE:
        n = b = 0.0
        if taps == ORIFICE_FLANGE_TAPS:
            if D < 0.1:
                C_inf = 0.6284 + 0.1462*beta21 - 0.8464*beta8 + 0.2603*beta4/(1-beta4) - 0.2886*beta3
            else:
                C_inf = 0.6276 + 0.0828*beta21 + 0.2739*beta8 - 0.0934*beta4/(1-beta4) - 0.1132*beta3
        elif taps == ORIFICE_VENA_CONTRACTA_TAPS:
            if D < 0.1:
                C_inf = 0.6261 + 0.1851*beta21 - 0.2879*beta8 + 0.1170*beta4/(1-beta4) - 0.2845*beta3
            else:
                # Yes these are supposed to be the same as the flange, large set
                C_inf = 0.6276 + 0.0828*beta21 + 0.2739*beta8 - 0.0934*beta4/(1-beta4) - 0.1132*beta3
        else:
            raise ValueError(_Miller_1996_unsupported_tap_segmental)
    elif subtype == MILLER_CONICAL_ORIFICE or subtype == CONICAL_ORIFICE:
        n = b = 0.0
        if 250.0*beta <= Re <= 500.0*beta:
            C_inf = 0.734
        else:
            C_inf = 0.730
    elif subtype == MILLER_QUARTER_CIRCLE_ORIFICE or subtype == QUARTER_CIRCLE_ORIFICE:
        n = b = 0.0
        C_inf = (0.7746 - 0.1334*beta21 + 1.4098*beta8
                 + 0.0675*beta4/(1.0 - beta4) + 0.3865*beta3)
    else:
        raise ValueError(_Miller_1996_unsupported_type)
    C = C_inf + b*Re**-n
    return C

# Data from: Discharge Coefficient Performance of Venturi, Standard Concentric Orifice Plate, V-Cone, and Wedge Flow Meters at Small Reynolds Numbers
orifice_std_Res_Hollingshead = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 100000.0,
    1000000.0, 10000000.0, 50000000.0
]
orifice_std_logRes_Hollingshead = [0.0, 1.6094379124341003, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221,
    4.382026634673881, 4.605170185988092, 5.298317366548036, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082, 8.006367567650246,
    8.517193191416238, 9.210340371976184, 11.512925464970229, 13.815510557964274, 16.11809565095832, 17.72753356339242
]

orifice_std_betas_Hollingshead = [0.5, 0.6, 0.65, 0.7]
orifice_std_beta_5_Hollingshead_Cs = [0.233, 0.478, 0.585, 0.654, 0.677, 0.688, 0.697, 0.700, 0.702, 0.699, 0.693, 0.684, 0.67, 0.648, 0.639, 0.632, 0.629,
    0.619, 0.615, 0.614, 0.614
]
orifice_std_beta_6_Hollingshead_Cs = [0.212, 0.448, 0.568, 0.657, 0.689, 0.707, 0.721, 0.725, 0.727, 0.725, 0.719, 0.707, 0.688, 0.658, 0.642, 0.633, 0.624,
    0.61, 0.605, 0.602, 0.595
]
orifice_std_beta_65_Hollingshead_Cs = [0.202, 0.425, 0.546, 0.648, 0.692, 0.715, 0.738, 0.748, 0.754, 0.764, 0.763, 0.755, 0.736, 0.685, 0.666, 0.656, 0.641,
    0.622, 0.612, 0.61, 0.607
]
orifice_std_beta_7_Hollingshead_Cs = [0.191, 0.407, 0.532, 0.644, 0.696, 0.726, 0.756, 0.772, 0.781, 0.795, 0.796, 0.788, 0.765, 0.7, 0.67, 0.659, 0.646, 0.623,
    0.616, 0.607, 0.604
]
orifice_std_Hollingshead_Cs = [orifice_std_beta_5_Hollingshead_Cs, orifice_std_beta_6_Hollingshead_Cs,
    orifice_std_beta_65_Hollingshead_Cs, orifice_std_beta_7_Hollingshead_Cs
]

orifice_std_Hollingshead_tck = implementation_optimize_tck([
    [0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7],
    [0.0, 0.0, 0.0, 0.0, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221, 4.382026634673881,
        4.605170185988092, 5.298317366548036, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082, 8.006367567650246,
        8.517193191416238, 9.210340371976184, 11.512925464970229, 13.815510557964274, 17.72753356339242, 17.72753356339242, 17.72753356339242,
        17.72753356339242
    ],
    [0.23300000000000026, 0.3040793845022822, 0.5397693379388018, 0.6509414325648643, 0.6761419937262648, 0.6901697401156808, 0.6972240707909276,
        0.6996759572505151, 0.7040223363705952, 0.7008741587711967, 0.692665226515394, 0.6826387818678974, 0.6727930643166521, 0.6490542161859936,
        0.6378780959698012, 0.6302027504736312, 0.6284904523610422, 0.616773266650063, 0.6144108030024114, 0.6137270770149181, 0.6140000000000004,
        0.21722222222222212, 0.26754856063815036, 0.547178981607613, 0.6825835849471493, 0.6848255120880751, 0.712775784969247, 0.7066842545008245,
        0.7020345744268808, 0.6931476737316041, 0.6710886785478944, 0.6501218695989138, 0.6257164975579488, 0.5888463567232898, 0.6237505336392806,
        0.578149766754485, 0.5761890160080455, 0.5922303103985014, 0.5657790974864929, 0.6013376373672517, 0.5693593555949975, 0.5528888888888888,
        0.206777777777778, 0.2644342350096853, 0.4630985572034346, 0.6306849522311501, 0.6899260188747366, 0.7092703879134302, 0.7331416654072416,
        0.7403866219900521, 0.7531493636395633, 0.7685019053395048, 0.771007019842085, 0.7649533772965396, 0.7707020081746302, 0.6897832472092346,
        0.6910618341373851, 0.6805763529796045, 0.6291884772151493, 0.6470904244660671, 0.5962879899497537, 0.6353096798316025, 0.6277777777777779,
        0.19100000000000003, 0.23712276889270198, 0.44482842661392175, 0.6337225464930397, 0.6926462978136392, 0.7316874888663132, 0.7542057211530093,
        0.77172737538752, 0.7876049778429112, 0.795143180926116, 0.7977570986094262, 0.7861445043222344, 0.777182818678971, 0.7057345800650827,
        0.6626698628526632, 0.6600690433654985, 0.6323396431072075, 0.6212684034830293, 0.616281323630018, 0.603728515722033, 0.6040000000000001
    ], 3, 3
])

def C_eccentric_orifice_ISO_15377_1998(D, Do):
    r'''Calculates the coefficient of discharge of an eccentric orifice based
    on the geometry of the plate according to ISO 15377, first introduced in
    1998 and also presented in the second 2007 edition. It also appears in BS
    1042-1.2: 1989.

    .. math::
        C = 0.9355 - 1.6889\beta + 3.0428\beta^2 - 1.7989\beta^3

    This type of plate is normally used to avoid obstructing entrained gas,
    liquid, or sediment.

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]

    Returns
    -------
    C : float
        Coefficient of discharge of the eccentric orifice, [-]

    Notes
    -----
    No correction for where the orifice bore is located is included.

    The following limits apply to the orifice plate standard [1]_:

    * Bore diameter above 50 mm.
    * Pipe diameter between 10 cm and 1 m.
    * Beta ratio between 0.46 and 0.84
    * :math:`2\times 10^5 \beta^2 \le Re_D \le 10^6 \beta`

    The uncertainty of this equation for `C` is said to be 1% if `beta` is
    under 0.75, otherwise 2%.

    The `orifice_expansibility` function should be used with this method as
    well.

    Additional specifications are:

    * The thickness of the orifice should be between 0.005`D` and 0.02`D`.
    * Corner tappings should be used, with hole diameter between 3 and 10 mm.
      The angular orientation of the tappings matters because the flow meter
      is not symmetrical. The angle should ideally be at the top or bottom of
      the plate, opposite which side the bore is on - but this can cause
      issues with deposition if the taps are on the bottom or gas bubbles if
      the taps are on the taps. The taps are often placed 30 degrees away from
      the ideal position to counteract this effect, with under an extra 2%
      error.

    Some comparisons with CFD results can be found in [2]_.

    Examples
    --------
    >>> C_eccentric_orifice_ISO_15377_1998(.2, .075)
    0.6351923828125

    References
    ----------
    .. [1] TC 30/SC 2, ISO. ISO/TR 15377:1998, Measurement of Fluid Flow by
       Means of Pressure-Differential Devices - Guide for the Specification of
       Nozzles and Orifice Plates beyond the Scope of ISO 5167-1.
    .. [2] Yashvanth, S., Varadarajan Seshadri, and J. YogeshKumarK. "CFD
       Analysis of Flow through Single and Multi Stage Eccentric Orifice Plate
       Assemblies," 2017.
    '''
    beta = Do/D
    C = beta*(beta*(3.0428 - 1.7989*beta) - 1.6889) + 0.9355
    return C

def C_quarter_circle_orifice_ISO_15377_1998(D, Do):
    r'''Calculates the coefficient of discharge of a quarter circle orifice based
    on the geometry of the plate according to ISO 15377, first introduced in
    1998 and also presented in the second 2007 edition. It also appears in BS
    1042-1.2: 1989.

    .. math::
        C = 0.73823 + 0.3309\beta - 1.1615\beta^2 + 1.5084\beta^3

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]

    Returns
    -------
    C : float
        Coefficient of discharge of the quarter circle orifice, [-]

    Notes
    -----
    The discharge coefficient of this type of orifice plate remains
    constant down to a lower than normal `Re`, as occurs in highly
    viscous applications.

    The following limits apply to the orifice plate standard [1]_:

    * Bore diameter >= 1.5 cm
    * Pipe diameter <= 50 cm
    * Beta ratio between 0.245 and 0.6
    * :math:`Re_d \le 10^5 \beta`

    There is also a table in [1]_ which lists increased minimum
    upstream pipe diameters for pipes of different roughnesses; the
    higher the roughness, the larger the pipe diameter required,
    and the table goes up to 20 cm for rusty cast iron.

    Corner taps should be used up to pipe diameters of 40 mm;
    for larger pipes, corner or flange taps can be used. No impact
    on the flow coefficient is included in the correlation.

    The recommended expansibility method for this type of orifice is
    :obj:`orifice_expansibility`.

    Examples
    --------
    >>> C_quarter_circle_orifice_ISO_15377_1998(.2, .075)
    0.77851484375000

    References
    ----------
    .. [1] TC 30/SC 2, ISO. ISO/TR 15377:1998, Measurement of Fluid Flow by
       Means of Pressure-Differential Devices - Guide for the Specification of
       Nozzles and Orifice Plates beyond the Scope of ISO 5167-1.
    '''
    beta = Do/D
    C = beta*(beta*(1.5084*beta - 1.16158) + 0.3309) + 0.73823
    return C

def discharge_coefficient_to_K(D, Do, C):
    r'''Converts a discharge coefficient to a standard loss coefficient,
    for use in computation of the actual pressure drop of an orifice or other
    device.

    .. math::
        K = \left[\frac{\sqrt{1-\beta^4(1-C^2)}}{C\beta^2} - 1\right]^2

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    C : float
        Coefficient of discharge of the orifice, [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the velocity and density of the fluid
        just upstream of the orifice, [-]

    Notes
    -----
    If expansibility is used in the orifice calculation, the result will not
    match with the specified pressure drop formula in [1]_; it can almost
    be matched by dividing the calculated mass flow by the expansibility factor
    and using that mass flow with the loss coefficient.

    Examples
    --------
    >>> discharge_coefficient_to_K(D=0.07366, Do=0.05, C=0.61512)
    5.2314291729754

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    '''
    beta = Do/D
    beta2 = beta*beta
    beta4 = beta2*beta2
    root_K = (sqrt(1.0 - beta4*(1.0 - C*C))/(C*beta2) - 1.0)
    return root_K*root_K


def K_to_discharge_coefficient(D, Do, K):
    r'''Converts a standard loss coefficient to a discharge coefficient.

    .. math::
        C = \sqrt{\frac{1}{2 \sqrt{K} \beta^{4} + K \beta^{4}}
        - \frac{\beta^{4}}{2 \sqrt{K} \beta^{4} + K \beta^{4}} }

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]
    K : float
        Loss coefficient with respect to the velocity and density of the fluid
        just upstream of the orifice, [-]

    Returns
    -------
    C : float
        Coefficient of discharge of the orifice, [-]

    Notes
    -----
    If expansibility is used in the orifice calculation, the result will not
    match with the specified pressure drop formula in [1]_; it can almost
    be matched by dividing the calculated mass flow by the expansibility factor
    and using that mass flow with the loss coefficient.

    This expression was derived with SymPy, and checked numerically. There were
    three other, incorrect roots.

    Examples
    --------
    >>> K_to_discharge_coefficient(D=0.07366, Do=0.05, K=5.2314291729754)
    0.6151200000000001

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    '''
    beta = Do/D
    beta2 = beta*beta
    beta4 = beta2*beta2
    root_K = sqrt(K)
    return sqrt((1.0 - beta4)/((2.0*root_K + K)*beta4))

def dP_orifice(D, Do, P1, P2, C):
    r'''Calculates the non-recoverable pressure drop of an orifice plate based
    on the pressure drop and the geometry of the plate and the discharge
    coefficient.

    .. math::
        \Delta\bar w = \frac{\sqrt{1-\beta^4(1-C^2)}-C\beta^2}
        {\sqrt{1-\beta^4(1-C^2)}+C\beta^2} (P_1 - P_2)

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
    C : float
        Coefficient of discharge of the orifice, [-]

    Returns
    -------
    dP : float
        Non-recoverable pressure drop of the orifice plate, [Pa]

    Notes
    -----
    This formula can be well approximated by:

    .. math::
        \Delta\bar w = \left(1 - \beta^{1.9}\right)(P_1 - P_2)

    The recoverable pressure drop should be recovered by 6 pipe diameters
    downstream of the orifice plate.

    Examples
    --------
    >>> dP_orifice(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, C=0.61512)
    9069.474705745388

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-2:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 2: Orifice Plates.
    '''
    beta = Do/D
    beta2 = beta*beta
    beta4 = beta2*beta2
    dP = P1 - P2
    delta_w = (sqrt(1.0 - beta4*(1.0 - C*C)) - C*beta2)/(
               sqrt(1.0 - beta4*(1.0 - C*C)) + C*beta2)*dP
    return delta_w


def velocity_of_approach_factor(D, Do):
    r'''Calculates a factor for orifice plate design called the `velocity of
    approach`.

    .. math::
        \text{Velocity of approach} = \frac{1}{\sqrt{1 - \beta^4}}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice at flow conditions, [m]

    Returns
    -------
    velocity_of_approach : float
        Coefficient of discharge of the orifice, [-]

    Notes
    -----

    Examples
    --------
    >>> velocity_of_approach_factor(D=0.0739, Do=0.0222)
    1.0040970074165514

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    '''
    return 1.0/sqrt(1.0 - (Do/D)**4)


def flow_coefficient(D, Do, C):
    r'''Calculates a factor for differential pressure flow meter design called
    the `flow coefficient`. This should not be confused with the flow
    coefficient often used when discussing valves.

    .. math::
        \text{Flow coefficient} = \frac{C}{\sqrt{1 - \beta^4}}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of flow meter characteristic dimension at flow conditions, [m]
    C : float
        Coefficient of discharge of the flow meter, [-]

    Returns
    -------
    flow_coefficient : float
        Differential pressure flow meter flow coefficient, [-]

    Notes
    -----
    This measure is used not just for orifices but for other differential
    pressure flow meters [2]_.

    It is sometimes given the symbol K. It is also equal to the product of the
    diacharge coefficient and the velocity of approach factor [2]_.

    Examples
    --------
    >>> flow_coefficient(D=0.0739, Do=0.0222, C=0.6)
    0.6024582044499308

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] Miller, Richard W. Flow Measurement Engineering Handbook. 3rd
       edition. New York: McGraw-Hill Education, 1996.
    '''
    return C*1.0/sqrt(1.0 - (Do/D)**4)


def nozzle_expansibility(D, Do, P1, P2, k, beta=None):
    r'''Calculates the expansibility factor for a nozzle or venturi nozzle,
    based on the geometry of the plate, measured pressures of the orifice, and
    the isentropic exponent of the fluid.

    .. math::
        \epsilon = \left\{\left(\frac{\kappa \tau^{2/\kappa}}{\kappa-1}\right)
        \left(\frac{1 - \beta^4}{1 - \beta^4 \tau^{2/\kappa}}\right)
        \left[\frac{1 - \tau^{(\kappa-1)/\kappa}}{1 - \tau}
        \right] \right\}^{0.5}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of orifice of the venturi or nozzle, [m]
    P1 : float
        Static pressure of fluid upstream of orifice at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of orifice at the cross-section of
        the pressure tap, [Pa]
    k : float
        Isentropic exponent of fluid, [-]
    beta : float, optional
        Optional `beta` ratio, which is useful to specify for wedge meters or
        flow meters which have a different beta ratio calculation, [-]

    Returns
    -------
    expansibility : float
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Notes
    -----
    This formula was determined for the range of P2/P1 >= 0.75.

    Examples
    --------
    >>> nozzle_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    0.9945702344566746

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-3:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 3: Nozzles and Venturi Nozzles.
    '''
    if beta is None:
        beta = Do/D
    beta2 = beta*beta
    beta4 = beta2*beta2
    tau = P2/P1
    term1 = k*tau**(2.0/k )/(k - 1.0)
    term2 = (1.0 - beta4)/(1.0 - beta4*tau**(2.0/k))
    if tau == 1.0:
        '''Avoid a zero division error.
        Obtained with:
            from sympy import *
            tau, k = symbols('tau, k')
            expr = (1 - tau**((k - 1)/k))/(1 - tau)
            limit(expr, tau, 1)
        '''
        term3 = (k - 1.0)/k
    else:
        term3 = (1.0 - tau**((k - 1.0)/k))/(1.0 - tau)
    return sqrt(term1*term2*term3)


def C_long_radius_nozzle(D, Do, rho, mu, m):
    r'''Calculates the coefficient of discharge of a long radius nozzle used
    for measuring flow rate of fluid, based on the geometry of the nozzle,
    mass flow rate through the nozzle, and the density and viscosity of the
    fluid.

    .. math::
        C = 0.9965 - 0.00653\beta^{0.5} \left(\frac{10^6}{Re_D}\right)^{0.5}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of long radius nozzle orifice at flow conditions, [m]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    m : float
        Mass flow rate of fluid through the nozzle, [kg/s]

    Returns
    -------
    C : float
        Coefficient of discharge of the long radius nozzle orifice, [-]

    Notes
    -----

    Examples
    --------
    >>> C_long_radius_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    0.9805503704679863

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-3:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 3: Nozzles and Venturi Nozzles.
    '''
    A_pipe = pi/4.*D*D
    v = m/(A_pipe*rho)
    Re_D = rho*v*D/mu
    beta = Do/D
    return 0.9965 - 0.00653*sqrt(beta)*sqrt(1E6/Re_D)


def C_ISA_1932_nozzle(D, Do, rho, mu, m):
    r'''Calculates the coefficient of discharge of an ISA 1932 style nozzle
    used for measuring flow rate of fluid, based on the geometry of the nozzle,
    mass flow rate through the nozzle, and the density and viscosity of the
    fluid.

    .. math::
        C = 0.9900 - 0.2262\beta^{4.1} - (0.00175\beta^2 - 0.0033\beta^{4.15})
        \left(\frac{10^6}{Re_D}\right)^{1.15}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of nozzle orifice at flow conditions, [m]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    m : float
        Mass flow rate of fluid through the nozzle, [kg/s]

    Returns
    -------
    C : float
        Coefficient of discharge of the nozzle orifice, [-]

    Notes
    -----

    Examples
    --------
    >>> C_ISA_1932_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    0.9635849973250495

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-3:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 3: Nozzles and Venturi Nozzles.
    '''
    A_pipe = pi/4.*D*D
    v = m/(A_pipe*rho)
    Re_D = rho*v*D/mu
    beta = Do/D
    C = (0.9900 - 0.2262*beta**4.1
         - (0.00175*beta**2 - 0.0033*beta**4.15)*(1E6/Re_D)**1.15)
    return C


def C_venturi_nozzle(D, Do):
    r'''Calculates the coefficient of discharge of an Venturi style nozzle
    used for measuring flow rate of fluid, based on the geometry of the nozzle.

    .. math::
        C = 0.9858 - 0.196\beta^{4.5}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of nozzle orifice at flow conditions, [m]

    Returns
    -------
    C : float
        Coefficient of discharge of the nozzle orifice, [-]

    Notes
    -----

    Examples
    --------
    >>> C_venturi_nozzle(D=0.07391, Do=0.0422)
    0.9698996454169576

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-3:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 3: Nozzles and Venturi Nozzles.
    '''
    beta = Do/D
    return 0.9858 - 0.198*beta**4.5


# Relative pressure loss as a function of beta reatio for venturi nozzles
# Venturi nozzles should be between 65 mm and 500 mm; there are high and low
# loss ratios , with the high losses corresponding to small diameters,
# low high losses corresponding to large diameters
# Interpolation can be performed.

venturi_tube_betas = [0.299160, 0.299470, 0.312390, 0.319010, 0.326580, 0.337290,
          0.342020, 0.347060, 0.359030, 0.365960, 0.372580, 0.384870,
          0.385810, 0.401250, 0.405350, 0.415740, 0.424250, 0.434010,
          0.447880, 0.452590, 0.471810, 0.473090, 0.493540, 0.499240,
          0.516530, 0.523800, 0.537630, 0.548060, 0.556840, 0.573890,
          0.582350, 0.597820, 0.601560, 0.622650, 0.626490, 0.649480,
          0.650990, 0.668700, 0.675870, 0.688550, 0.693180, 0.706180,
          0.713330, 0.723510, 0.749540, 0.749650]

venturi_tube_dP_high = [0.164534, 0.164504, 0.163591, 0.163508, 0.163439,
        0.162652, 0.162224, 0.161866, 0.161238, 0.160786,
        0.160295, 0.159280, 0.159193, 0.157776, 0.157467,
        0.156517, 0.155323, 0.153835, 0.151862, 0.151154,
        0.147840, 0.147613, 0.144052, 0.143050, 0.140107,
        0.138981, 0.136794, 0.134737, 0.132847, 0.129303,
        0.127637, 0.124758, 0.124006, 0.119269, 0.118449,
        0.113605, 0.113269, 0.108995, 0.107109, 0.103688,
        0.102529, 0.099567, 0.097791, 0.095055, 0.087681,
        0.087648]

venturi_tube_dP_low = [0.089232, 0.089218, 0.088671, 0.088435, 0.088206,
   0.087853, 0.087655, 0.087404, 0.086693, 0.086241,
   0.085813, 0.085142, 0.085102, 0.084446, 0.084202,
   0.083301, 0.082470, 0.081650, 0.080582, 0.080213,
   0.078509, 0.078378, 0.075989, 0.075226, 0.072700,
   0.071598, 0.069562, 0.068128, 0.066986, 0.064658,
   0.063298, 0.060872, 0.060378, 0.057879, 0.057403,
   0.054091, 0.053879, 0.051726, 0.050931, 0.049362,
   0.048675, 0.046522, 0.045381, 0.043840, 0.039913,
   0.039896]

#ratios_average = 0.5*(ratios_high + ratios_low)
D_bound_venturi_tube = [0.065, 0.5]


def dP_venturi_tube(D, Do, P1, P2):
    r'''Calculates the non-recoverable pressure drop of a venturi tube
    differential pressure meter based on the pressure drop and the geometry of
    the venturi meter.

    .. math::
        \epsilon =  \frac{\Delta\bar w }{\Delta P}

    The :math:`\epsilon` value is looked up in a table of values as a function
    of beta ratio and upstream pipe diameter (roughness impact).

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of venturi tube at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of venturi tube at the cross-section
        of the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of venturi tube at the
         cross-section of the pressure tap, [Pa]

    Returns
    -------
    dP : float
        Non-recoverable pressure drop of the venturi tube, [Pa]

    Notes
    -----
    The recoverable pressure drop should be recovered by 6 pipe diameters
    downstream of the venturi tube.

    Note there is some information on the effect of Reynolds number as well
    in [1]_ and [2]_, with a curve showing an increased pressure drop
    from 1E5-6E5 to with a decreasing multiplier from 1.75 to 1; the multiplier
    is 1 for higher Reynolds numbers. This is not currently included in this
    implementation.

    Examples
    --------
    >>> dP_venturi_tube(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0)
    1788.5717754177406

    References
    ----------
    .. [1] American Society of Mechanical Engineers. Mfc-3M-2004 Measurement
       Of Fluid Flow In Pipes Using Orifice, Nozzle, And Venturi. ASME, 2001.
    .. [2] ISO 5167-4:2003 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 4: Venturi Tubes.
    '''
    # Effect of Re is not currently included
    beta = Do/D
    epsilon_D65 = interp(beta, venturi_tube_betas, venturi_tube_dP_high)
    epsilon_D500 = interp(beta, venturi_tube_betas, venturi_tube_dP_low)
    epsilon = interp(D, D_bound_venturi_tube, [epsilon_D65, epsilon_D500])
    return epsilon*(P1 - P2)


def diameter_ratio_cone_meter(D, Dc):
    r'''Calculates the diameter ratio `beta` used to characterize a cone
    flow meter.

    .. math::
        \beta = \sqrt{1 - \frac{d_c^2}{D^2}}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Dc : float
        Diameter of the largest end of the cone meter, [m]

    Returns
    -------
    beta : float
        Cone meter diameter ratio, [-]

    Notes
    -----

    A mathematically equivalent formula often written is:

    .. math::
        \beta = \frac{\sqrt{D^2 - d_c^2}}{D}

    Examples
    --------
    >>> diameter_ratio_cone_meter(D=0.2575, Dc=0.184)
    0.6995709873957624

    References
    ----------
    .. [1] Hollingshead, Colter. "Discharge Coefficient Performance of Venturi,
       Standard Concentric Orifice Plate, V-Cone, and Wedge Flow Meters at
       Small Reynolds Numbers." May 1, 2011.
       https://digitalcommons.usu.edu/etd/869.
    '''
    D_ratio = Dc/D
    return sqrt(1.0 - D_ratio*D_ratio)


def cone_meter_expansibility_Stewart(D, Dc, P1, P2, k):
    r'''Calculates the expansibility factor for a cone flow meter,
    based on the geometry of the cone meter, measured pressures of the orifice,
    and the isentropic exponent of the fluid. Developed in [1]_, also shown
    in [2]_.

    .. math::
        \epsilon = 1 - (0.649 + 0.696\beta^4) \frac{\Delta P}{\kappa P_1}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Dc : float
        Diameter of the largest end of the cone meter, [m]
    P1 : float
        Static pressure of fluid upstream of cone meter at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid at the end of the center of the cone pressure
        tap, [Pa]
    k : float
        Isentropic exponent of fluid, [-]

    Returns
    -------
    expansibility : float
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Notes
    -----
    This formula was determined for the range of P2/P1 >= 0.75; the only gas
    used to determine the formula is air.

    Examples
    --------
    >>> cone_meter_expansibility_Stewart(D=1, Dc=0.9, P1=1E6, P2=8.5E5, k=1.2)
    0.9157343

    References
    ----------
    .. [1] Stewart, D. G., M. Reader-Harris, and NEL Dr RJW Peters. "Derivation
       of an Expansibility Factor for the V-Cone Meter." In Flow Measurement
       International Conference, Peebles, Scotland, UK, 2001.
    .. [2] ISO 5167-5:2016 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 5: Cone meters.
    '''
    dP = P1 - P2
    beta = diameter_ratio_cone_meter(D, Dc)
    beta *= beta
    beta *= beta
    return 1.0 - (0.649 + 0.696*beta)*dP/(k*P1)


def dP_cone_meter(D, Dc, P1, P2):
    r'''Calculates the non-recoverable pressure drop of a cone meter
    based on the measured pressures before and at the cone end, and the
    geometry of the cone meter according to [1]_.

    .. math::
        \Delta \bar \omega = (1.09 - 0.813\beta)\Delta P

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Dc : float
        Diameter of the largest end of the cone meter, [m]
    P1 : float
        Static pressure of fluid upstream of cone meter at the cross-section of
        the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid at the end of the center of the cone pressure
        tap, [Pa]

    Returns
    -------
    dP : float
        Non-recoverable pressure drop of the orifice plate, [Pa]

    Notes
    -----
    The recoverable pressure drop should be recovered by 6 pipe diameters
    downstream of the cone meter.

    Examples
    --------
    >>> dP_cone_meter(1, .7, 1E6, 9.5E5)
    25470.093437973323

    References
    ----------
    .. [1] ISO 5167-5:2016 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 5: Cone meters.
    '''
    dP = P1 - P2
    beta = diameter_ratio_cone_meter(D, Dc)
    return (1.09 - 0.813*beta)*dP


def diameter_ratio_wedge_meter(D, H):
    r'''Calculates the diameter ratio `beta` used to characterize a wedge
    flow meter as given in [1]_ and [2]_.

    .. math::
        \beta = \left(\frac{1}{\pi}\left\{\arccos\left[1 - \frac{2H}{D}
        \right] - 2 \left[1 - \frac{2H}{D}
        \right]\left(\frac{H}{D} - \left[\frac{H}{D}\right]^2
        \right)^{0.5}\right\}\right)^{0.5}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    H : float
        Portion of the diameter of the clear segment of the pipe up to the
        wedge blocking flow; the height of the pipe up to the wedge, [m]

    Returns
    -------
    beta : float
        Wedge meter diameter ratio, [-]

    Notes
    -----

    Examples
    --------
    >>> diameter_ratio_wedge_meter(D=0.2027, H=0.0608)
    0.5022531424646643

    References
    ----------
    .. [1] Hollingshead, Colter. "Discharge Coefficient Performance of Venturi,
       Standard Concentric Orifice Plate, V-Cone, and Wedge Flow Meters at
       Small Reynolds Numbers." May 1, 2011.
       https://digitalcommons.usu.edu/etd/869.
    .. [2] IntraWedge WEDGE FLOW METER Type: IWM. January 2011.
       http://www.intra-automation.com/download.php?file=pdf/products/technical_information/en/ti_iwm_en.pdf
    '''
    H_D = H/D
    t0 = 1.0 - 2.0*H_D
    t1 = acos(t0)
    t2 = t0 + t0
    t3 = sqrt(H_D - H_D*H_D)
    t4 = t1 - t2*t3
    return sqrt(pi_inv*t4)


def C_wedge_meter_Miller(D, H):
    r'''Calculates the coefficient of discharge of an wedge flow meter
    used for measuring flow rate of fluid, based on the geometry of the
    differential pressure flow meter.

    For half-inch lines:

    .. math::
        C = 0.7883 + 0.107(1 - \beta^2)

    For 1 to 1.5 inch lines:

    .. math::
        C = 0.6143 + 0.718(1 - \beta^2)

    For 1.5 to 24 inch lines:

    .. math::
        C = 0.5433 + 0.2453(1 - \beta^2)

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    H : float
        Portion of the diameter of the clear segment of the pipe up to the
        wedge blocking flow; the height of the pipe up to the wedge, [m]

    Returns
    -------
    C : float
        Coefficient of discharge of the wedge flow meter, [-]

    Notes
    -----
    There is an ISO standard being developed to cover wedge meters as of 2018.

    Wedge meters can have varying angles; 60 and 90 degree wedge meters have
    been reported. Tap locations 1 or 2 diameters (upstream and downstream),
    and 2D upstream/1D downstream have been used. Some wedges are sharp;
    some are smooth. [2]_ gives some experimental values.

    Examples
    --------
    >>> C_wedge_meter_Miller(D=0.1524, H=0.3*0.1524)
    0.7267069372687651

    References
    ----------
    .. [1] Miller, Richard W. Flow Measurement Engineering Handbook. 3rd
       edition. New York: McGraw-Hill Education, 1996.
    .. [2] Seshadri, V., S. N. Singh, and S. Bhargava. "Effect of Wedge Shape
       and Pressure Tap Locations on the Characteristics of a Wedge Flowmeter."
       IJEMS Vol.01(5), October 1994.
    '''
    beta = diameter_ratio_wedge_meter(D, H)
    beta *= beta
    if D <= 0.7*inch:
        # suggested limit 0.5 inch for this equation
        C = 0.7883 + 0.107*(1.0 - beta)
    elif D <= 1.4*inch:
        # Suggested limit is under 1.5 inches
        C = 0.6143 + 0.718*(1.0 - beta)
    else:
        C = 0.5433 + 0.2453*(1.0 - beta)
    return C


def C_wedge_meter_ISO_5167_6_2017(D, H):
    r'''Calculates the coefficient of discharge of an wedge flow meter
    used for measuring flow rate of fluid, based on the geometry of the
    differential pressure flow meter according to the ISO 5167-6 standard
    (draft 2017).

    .. math::
        C = 0.77 - 0.09\beta

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    H : float
        Portion of the diameter of the clear segment of the pipe up to the
        wedge blocking flow; the height of the pipe up to the wedge, [m]

    Returns
    -------
    C : float
        Coefficient of discharge of the wedge flow meter, [-]

    Notes
    -----
    This standard applies for wedge meters in line sizes between 50 and 600 mm;
    and height ratios between 0.2 and 0.6. The range of allowable Reynolds
    numbers is large; between 1E4 and 9E6. The uncertainty of the flow
    coefficient is approximately 4%. Usually a 10:1 span of flow can be
    measured accurately. The discharge and entry length of the meters must be
    at least half a pipe diameter. The wedge angle must be 90 degrees, plus or
    minus two degrees.

    The orientation of the wedge meter does not change the accuracy of this
    model.

    There should be a straight run of 10 pipe diameters before the wedge meter
    inlet, and two of the same pipe diameters after it.

    Examples
    --------
    >>> C_wedge_meter_ISO_5167_6_2017(D=0.1524, H=0.3*0.1524)
    0.724792059539853

    References
    ----------
    .. [1] ISO/DIS 5167-6 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 6: Wedge Meters.
    '''
    beta = diameter_ratio_wedge_meter(D, H)
    return 0.77 - 0.09*beta


def dP_wedge_meter(D, H, P1, P2):
    r'''Calculates the non-recoverable pressure drop of a wedge meter
    based on the measured pressures before and at the wedge meter, and the
    geometry of the wedge meter according to [1]_.

    .. math::
        \Delta \bar \omega = (1.09 - 0.79\beta)\Delta P

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    H : float
        Portion of the diameter of the clear segment of the pipe up to the
        wedge blocking flow; the height of the pipe up to the wedge, [m]
    P1 : float
        Static pressure of fluid upstream of wedge meter at the cross-section
        of the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid at the end of the wedge meter pressure tap, [
        Pa]

    Returns
    -------
    dP : float
        Non-recoverable pressure drop of the wedge meter, [Pa]

    Notes
    -----
    The recoverable pressure drop should be recovered by 5 pipe diameters
    downstream of the wedge meter.

    Examples
    --------
    >>> dP_wedge_meter(1, .7, 1E6, 9.5E5)
    20344.849697483587

    References
    ----------
    .. [1] ISO/DIS 5167-6 - Measurement of Fluid Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits Running
       Full -- Part 6: Wedge Meters.
    '''
    dP = P1 - P2
    beta = diameter_ratio_wedge_meter(D, H)
    return (1.09 - 0.79*beta)*dP


def C_Reader_Harris_Gallagher_wet_venturi_tube(mg, ml, rhog, rhol, D, Do, H=1):
    r'''Calculates the coefficient of discharge of the wet gas venturi tube
    based on the  geometry of the tube, mass flow rates of liquid and vapor
    through the tube, the density of the liquid and gas phases, and an
    adjustable coefficient `H`.

    .. math::
        C = 1 - 0.0463\exp(-0.05Fr_{gas, th}) \cdot \min\left(1,
        \sqrt{\frac{X}{0.016}}\right)

    .. math::
        Fr_{gas, th} = \frac{Fr_{\text{gas, densionetric }}}{\beta^{2.5}}

    .. math::
        \phi = \sqrt{1 + C_{Ch} X + X^2}

    .. math::
        C_{Ch} = \left(\frac{\rho_l}{\rho_{1,g}}\right)^n +
        \left(\frac{\rho_{1, g}}{\rho_{l}}\right)^n

    .. math::
        n = \max\left[0.583 - 0.18\beta^2 - 0.578\exp\left(\frac{-0.8
        Fr_{\text{gas, densiometric}}}{H}\right),0.392 - 0.18\beta^2 \right]

    .. math::
        X = \left(\frac{m_l}{m_g}\right) \sqrt{\frac{\rho_{1,g}}{\rho_l}}

    .. math::
        {Fr_{\text{gas, densiometric}}} = \frac{v_{gas}}{\sqrt{gD}}
        \sqrt{\frac{\rho_{1,g}}{\rho_l - \rho_{1,g}}}
        =  \frac{4m_g}{\rho_{1,g} \pi D^2 \sqrt{gD}}
        \sqrt{\frac{\rho_{1,g}}{\rho_l - \rho_{1,g}}}

    Parameters
    ----------
    mg : float
        Mass flow rate of gas through the venturi tube, [kg/s]
    ml : float
        Mass flow rate of liquid through the venturi tube, [kg/s]
    rhog : float
        Density of gas at `P1`, [kg/m^3]
    rhol : float
        Density of liquid at `P1`, [kg/m^3]
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of venturi tube at flow conditions, [m]
    H : float, optional
        A surface-tension effect coefficient used to adjust for different
        fluids, (1 for a hydrocarbon liquid, 1.35 for water, 0.79 for water in
        steam) [-]

    Returns
    -------
    C : float
        Coefficient of discharge of the wet gas venturi tube flow meter
        (includes flow rate of gas ONLY), [-]

    Notes
    -----
    This model has more error than single phase differential pressure meters.
    The model was first published in [1]_, and became ISO 11583 later.

    The limits of this correlation according to [2]_ are as follows:

    .. math::
        0.4 \le \beta \le 0.75

    .. math::
        0 < X \le 0.3

    .. math::
        Fr_{gas, th} > 3

    .. math::
        \frac{\rho_g}{\rho_l} > 0.02

    .. math::
        D \ge 50 \text{ mm}

    Examples
    --------
    >>> C_Reader_Harris_Gallagher_wet_venturi_tube(mg=5.31926, ml=5.31926/2,
    ... rhog=50.0, rhol=800., D=.1, Do=.06, H=1)
    0.9754210845876333

    References
    ----------
    .. [1] Reader-harris, Michael, and Tuv Nel. An Improved Model for
       Venturi-Tube Over-Reading in Wet Gas, 2009.
    .. [2] ISO/TR 11583:2012 Measurement of Wet Gas Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits.
    '''
    V = 4.0*mg/(rhog*pi*D*D)
    Frg = Froude_densimetric(V, L=D, rho1=rhol, rho2=rhog, heavy=False)
    beta = Do/D
    beta2 = beta*beta
    Fr_gas_th = Frg/(beta2*sqrt(beta))

    n = max(0.583 - 0.18*beta2 - 0.578*exp(-0.8*Frg/H),
            0.392 - 0.18*beta2)

    t0 = rhog/rhol
    t1 = (t0)**n
    C_Ch = t1 + 1.0/t1
    X =  ml/mg*sqrt(t0)
    OF = sqrt(1.0 + X*(C_Ch + X))

    C = 1.0 - 0.0463*exp(-0.05*Fr_gas_th)*min(1.0, sqrt(X/0.016))
    return C


def dP_Reader_Harris_Gallagher_wet_venturi_tube(D, Do, P1, P2, ml, mg, rhol,
                                                rhog, H=1.0):
    r'''Calculates the non-recoverable pressure drop of a wet gas venturi
    nozzle based on the pressure drop and the geometry of the venturi nozzle,
    the mass flow rates of liquid and gas through it, the densities of the
    vapor and liquid phase, and an adjustable coefficient `H`.

    .. math::
        Y = \frac{\Delta \bar \omega}{\Delta P} - 0.0896 - 0.48\beta^9

    .. math::
        Y_{max} = 0.61\exp\left[-11\frac{\rho_{1,g}}{\rho_l}
        - 0.045 \frac{Fr_{gas}}{H}\right]

    .. math::
        \frac{Y}{Y_{max}} = 1 - \exp\left[-35 X^{0.75} \exp
        \left( \frac{-0.28Fr_{gas}}{H}\right)\right]

    .. math::
        X = \left(\frac{m_l}{m_g}\right) \sqrt{\frac{\rho_{1,g}}{\rho_l}}

    .. math::
        {Fr_{\text{gas, densiometric}}} = \frac{v_{gas}}{\sqrt{gD}}
        \sqrt{\frac{\rho_{1,g}}{\rho_l - \rho_{1,g}}}
        =  \frac{4m_g}{\rho_{1,g} \pi D^2 \sqrt{gD}}
        \sqrt{\frac{\rho_{1,g}}{\rho_l - \rho_{1,g}}}

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    Do : float
        Diameter of venturi tube at flow conditions, [m]
    P1 : float
        Static pressure of fluid upstream of venturi tube at the cross-section
        of the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of venturi tube at the cross-
        section of the pressure tap, [Pa]
    ml : float
        Mass flow rate of liquid through the venturi tube, [kg/s]
    mg : float
        Mass flow rate of gas through the venturi tube, [kg/s]
    rhol : float
        Density of liquid at `P1`, [kg/m^3]
    rhog : float
        Density of gas at `P1`, [kg/m^3]
    H : float, optional
        A surface-tension effect coefficient used to adjust for different
        fluids, (1 for a hydrocarbon liquid, 1.35 for water, 0.79 for water in
        steam) [-]

    Returns
    -------
    C : float
        Coefficient of discharge of the wet gas venturi tube flow meter
        (includes flow rate of gas ONLY), [-]

    Notes
    -----
    The model was first published in [1]_, and became ISO 11583 later.

    Examples
    --------
    >>> dP_Reader_Harris_Gallagher_wet_venturi_tube(D=.1, Do=.06, H=1,
    ... P1=6E6, P2=6E6-5E4, ml=5.31926/2, mg=5.31926, rhog=50.0, rhol=800.,)
    16957.43843129572

    References
    ----------
    .. [1] Reader-harris, Michael, and Tuv Nel. An Improved Model for
       Venturi-Tube Over-Reading in Wet Gas, 2009.
    .. [2] ISO/TR 11583:2012 Measurement of Wet Gas Flow by Means of Pressure
       Differential Devices Inserted in Circular Cross-Section Conduits.
    '''
    dP = P1 - P2
    beta = Do/D
    X =  ml/mg*sqrt(rhog/rhol)

    V = 4*mg/(rhog*pi*D*D)
    Frg =  Froude_densimetric(V, L=D, rho1=rhol, rho2=rhog, heavy=False)

    Y_ratio = 1.0 - exp(-35.0*X**0.75*exp(-0.28*Frg/H))
    Y_max = 0.61*exp(-11.0*rhog/rhol - 0.045*Frg/H)
    Y = Y_max*Y_ratio
    rhs = -0.0896 - 0.48*beta**9
    dw = dP*(Y - rhs)
    return dw


# Venturi tube loss coefficients as a function of Re
as_cast_convergent_venturi_Res = [4E5, 6E4, 1E5, 1.5E5]
as_cast_convergent_venturi_Cs = [0.957, 0.966, 0.976, 0.982]

machined_convergent_venturi_Res = [5E4, 1E5, 2E5, 3E5,
                                   7.5E5, # 5E5 to 1E6
                                   1.5E6, # 1E6 to 2E6
                                   5E6] # 2E6 to 1E8
machined_convergent_venturi_Cs = [0.970, 0.977, 0.992, 0.998, 0.995, 1.000, 1.010]

rough_welded_convergent_venturi_Res = [4E4, 6E4, 1E5]
rough_welded_convergent_venturi_Cs = [0.96, 0.97, 0.98]

as_cast_convergent_entrance_machined_venturi_Res = [1E4, 6E4, 1E5, 1.5E5,
                                                    3.5E5, # 2E5 to 5E5
                                                    3.2E6] # 5E5 to 3.2E6
as_cast_convergent_entrance_machined_venturi_Cs = [0.963, 0.978, 0.98, 0.987, 0.992, 0.995]

venturi_Res_Hollingshead = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 30000.0, 50000.0, 75000.0, 100000.0, 1000000.0, 10000000.0, 50000000.0]
venturi_logRes_Hollingshead = [0.0, 1.6094379124341003, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221, 4.382026634673881, 4.605170185988092, 5.298317366548036, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082, 8.006367567650246, 8.517193191416238, 9.210340371976184, 10.308952660644293, 10.819778284410283, 11.225243392518447, 11.512925464970229, 13.815510557964274, 16.11809565095832, 17.72753356339242]
venturi_smooth_Cs_Hollingshead = [0.163, 0.336, 0.432, 0.515, 0.586, 0.625, 0.679, 0.705, 0.727, 0.803, 0.841, 0.881, 0.921, 0.937, 0.944, 0.954, 0.961, 0.967, 0.967, 0.97, 0.971, 0.973, 0.974, 0.975]
venturi_sharp_Cs_Hollingshead = [0.146, 0.3, 0.401, 0.498, 0.554, 0.596, 0.65, 0.688, 0.715, 0.801, 0.841, 0.884, 0.914, 0.94, 0.947, 0.944, 0.952, 0.959, 0.962, 0.963, 0.965, 0.967, 0.967, 0.967]


CONE_METER_C = 0.82
'''Constant loss coefficient for flow cone meters'''

ROUGH_WELDED_CONVERGENT_VENTURI_TUBE_C = 0.985
'''Constant loss coefficient for rough-welded convergent venturi tubes'''

MACHINED_CONVERGENT_VENTURI_TUBE_C = 0.995
'''Constant loss coefficient for machined convergent venturi tubes'''

AS_CAST_VENTURI_TUBE_C = 0.984
'''Constant loss coefficient for as-cast venturi tubes'''

ISO_15377_CONICAL_ORIFICE_C = 0.734
'''Constant loss coefficient for conical orifice plates according to ISO 15377'''

cone_Res_Hollingshead = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 7500.0,
    10000.0, 20000.0, 30000.0, 100000.0, 1000000.0, 10000000.0, 50000000.0
]
cone_logRes_Hollingshead = [0.0, 1.6094379124341003, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221,
    4.382026634673881, 4.605170185988092, 5.0106352940962555, 5.298317366548036, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082,
    8.006367567650246, 8.294049640102028, 8.517193191416238, 8.922658299524402, 9.210340371976184, 9.903487552536127, 10.308952660644293,
    11.512925464970229, 13.815510557964274, 16.11809565095832, 17.72753356339242
]
cone_betas_Hollingshead = [0.6611, 0.6995, 0.8203]

cone_beta_6611_Hollingshead_Cs = [0.066, 0.147, 0.207, 0.289, 0.349, 0.396, 0.462, 0.506, 0.537, 0.588, 0.622, 0.661, 0.7, 0.727, 0.75, 0.759, 0.763, 0.765,
    0.767, 0.773, 0.778, 0.789, 0.804, 0.803, 0.805, 0.802
]
cone_beta_6995_Hollingshead_Cs = [0.067, 0.15, 0.21, 0.292, 0.35, 0.394, 0.458, 0.502, 0.533, 0.584, 0.615, 0.645, 0.682, 0.721, 0.742, 0.75, 0.755, 0.757,
    0.763, 0.766, 0.774, 0.781, 0.792, 0.792, 0.79, 0.787
]
cone_beta_8203_Hollingshead_Cs = [0.057, 0.128, 0.182, 0.253, 0.303, 0.343, 0.4, 0.44, 0.472, 0.526, 0.557, 0.605, 0.644, 0.685, 0.705, 0.714, 0.721, 0.722,
    0.724, 0.723, 0.725, 0.731, 0.73, 0.73, 0.741, 0.734
]
cone_Hollingshead_Cs = [cone_beta_6611_Hollingshead_Cs, cone_beta_6995_Hollingshead_Cs,
    cone_beta_8203_Hollingshead_Cs
]

cone_Hollingshead_tck = implementation_optimize_tck([
    [0.6611, 0.6611, 0.6611, 0.8203, 0.8203, 0.8203],
    [0.0, 0.0, 0.0, 0.0, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221, 4.382026634673881,
        4.605170185988092, 5.0106352940962555, 5.298317366548036, 5.703782474656201, 6.214608098422191, 6.907755278982137, 7.600902459542082,
        8.006367567650246, 8.294049640102028, 8.517193191416238, 8.922658299524402, 9.210340371976184, 9.903487552536127, 10.308952660644293,
        11.512925464970229, 13.815510557964274, 17.72753356339242, 17.72753356339242, 17.72753356339242, 17.72753356339242
    ],
    [0.06600000000000003, 0.09181180887944293, 0.1406341453010674, 0.27319769866300025, 0.34177839953532274, 0.4025880076725502, 0.4563149328810349,
        0.5035445307357295, 0.5458473693359689, 0.583175639128474, 0.628052124545805, 0.6647198135005781, 0.7091524396786245, 0.7254729823419331,
        0.7487816963926843, 0.7588145502817809, 0.7628692532631826, 0.7660482147214834, 0.7644188319583379, 0.7782644144006241, 0.7721508139116487,
        0.7994728794028244, 0.8076742194714519, 0.7986221420822799, 0.8086240532850298, 0.802, 0.07016232064017663, 0.1059162635703894,
        0.1489681838592814, 0.28830815748629207, 0.35405213706957395, 0.40339795504063664, 0.4544570323055189, 0.5034637712201067, 0.5448190156693709,
        0.5840164245031125, 0.6211559598098063, 0.6218648844980823, 0.6621745760710729, 0.7282379546292953, 0.7340030734801267, 0.7396324865779599,
        0.7489736798953754, 0.7480726412914717, 0.7671564751169978, 0.756853660688892, 0.7787029642272745, 0.7742381131312691, 0.7887584162443445,
        0.7857610450218329, 0.7697076645551957, 0.7718300910596032, 0.05700000000000002, 0.07612544859943549, 0.12401733415778271, 0.24037452209595875,
        0.29662463502593156, 0.34859536586855205, 0.39480085719322505, 0.43661601622480606, 0.48091259102454764, 0.5240691286186233, 0.5590609288020619,
        0.6144556048716696, 0.6471713640567137, 0.6904158809061184, 0.7032590252050219, 0.712177974557301, 0.7221845303680273, 0.721505707129694,
        0.7249822376264551, 0.7218890085289907, 0.7221848475768714, 0.7371751354515526, 0.7252385062304629, 0.7278943803933404, 0.7496546607029086,
        0.7340000000000001
    ],
    2, 3
])

wedge_Res_Hollingshead = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 200.0, 300.0, 400.0, 500.0, 5000.0, 1.00E+04, 1.00E+05, 1.00E+06, 5.00E+07]
wedge_logRes_Hollingshead = [0.0, 1.6094379124341003, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221,
    4.382026634673881, 4.605170185988092, 5.298317366548036, 5.703782474656201, 5.991464547107982, 6.214608098422191, 8.517193191416238, 9.210340371976184,
    11.512925464970229, 13.815510557964274, 17.72753356339242
]

wedge_beta_5023_Hollingshead = [0.145, 0.318, 0.432, 0.551, 0.61, 0.641, 0.674, 0.69, 0.699, 0.716, 0.721, 0.725, 0.73, 0.729, 0.732, 0.732, 0.731, 0.733]
wedge_beta_611_Hollingshead = [0.127, 0.28, 0.384, 0.503, 0.567, 0.606, 0.645, 0.663, 0.672, 0.688, 0.694, 0.7, 0.705, 0.7, 0.702, 0.695, 0.699, 0.705]
wedge_betas_Hollingshead = [.5023, .611]
wedge_Hollingshead_Cs = [wedge_beta_5023_Hollingshead, wedge_beta_611_Hollingshead]

wedge_Hollingshead_tck = implementation_optimize_tck([
    [0.5023, 0.5023, 0.611, 0.611],
    [0.0, 0.0, 0.0, 0.0, 2.302585092994046, 2.995732273553991, 3.4011973816621555, 3.6888794541139363, 4.0943445622221, 4.382026634673881,
        4.605170185988092, 5.298317366548036, 5.703782474656201, 5.991464547107982, 6.214608098422191, 8.517193191416238, 9.210340371976184,
        11.512925464970229, 17.72753356339242, 17.72753356339242, 17.72753356339242, 17.72753356339242
    ],
    [0.14500000000000005, 0.18231832425722, 0.3339917130006919, 0.5379467710226973, 0.6077700659940896, 0.6459542943925077, 0.6729757007770231,
        0.6896405007576225, 0.7054863114589583, 0.7155740600632635, 0.7205446407610863, 0.7239576816068966, 0.7483627568160166, 0.7232963355919931,
        0.7366325320490953, 0.7264222143567053, 0.7339605394126009, 0.7330000000000001, 0.1270000000000001, 0.16939873865132285, 0.2828494933525669,
        0.4889107009077842, 0.5623120043524101, 0.6133092379676948, 0.6437092394687915, 0.6629923366662017, 0.6782934366011034, 0.687302374134782,
        0.6927470053128909, 0.6993992364234898, 0.7221204483546849, 0.6947577293284015, 0.7063701306810815, 0.6781614534359871, 0.7185326811948407,
        0.7050000000000001
    ],
    1, 3
])


beta_simple_meters = frozenset([ISO_5167_ORIFICE, ISO_15377_ECCENTRIC_ORIFICE,
                      ISO_15377_CONICAL_ORIFICE, ISO_15377_QUARTER_CIRCLE_ORIFICE,

                      MILLER_ORIFICE, MILLER_ECCENTRIC_ORIFICE,
                      MILLER_SEGMENTAL_ORIFICE, MILLER_CONICAL_ORIFICE,
                      MILLER_QUARTER_CIRCLE_ORIFICE,

                      CONCENTRIC_ORIFICE, ECCENTRIC_ORIFICE, CONICAL_ORIFICE,
                      SEGMENTAL_ORIFICE, QUARTER_CIRCLE_ORIFICE,
                      UNSPECIFIED_METER,
                      HOLLINGSHEAD_VENTURI_SHARP, HOLLINGSHEAD_VENTURI_SMOOTH, HOLLINGSHEAD_ORIFICE,

                      LONG_RADIUS_NOZZLE,
                      ISA_1932_NOZZLE, VENTURI_NOZZLE,
                      AS_CAST_VENTURI_TUBE,
                      MACHINED_CONVERGENT_VENTURI_TUBE,
                      ROUGH_WELDED_CONVERGENT_VENTURI_TUBE])

all_meters = frozenset(list(beta_simple_meters) + [CONE_METER, WEDGE_METER, HOLLINGSHEAD_CONE, HOLLINGSHEAD_WEDGE])
'''Set of string inputs representing all of the different supported flow meters
and their correlations.
'''
_unsupported_meter_msg = "Supported meter types are %s" % all_meters

def differential_pressure_meter_beta(D, D2, meter_type):
    r'''Calculates the beta ratio of a differential pressure meter.

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    D2 : float
        Diameter of orifice, or venturi meter orifice, or flow tube orifice,
        or cone meter end diameter, or wedge meter fluid flow height, [m]
    meter_type : str
        One of {'conical orifice', 'orifice', 'machined convergent venturi tube',
        'ISO 5167 orifice', 'Miller quarter circle orifice', 'Hollingshead venturi sharp',
        'segmental orifice', 'Miller conical orifice', 'Miller segmental orifice',
        'quarter circle orifice', 'Hollingshead v cone', 'wedge meter', 'eccentric orifice',
        'venuri nozzle', 'rough welded convergent venturi tube', 'ISA 1932 nozzle',
        'ISO 15377 quarter-circle orifice', 'Hollingshead venturi smooth',
        'Hollingshead orifice', 'cone meter', 'Hollingshead wedge', 'Miller orifice',
        'long radius nozzle', 'ISO 15377 conical orifice', 'unspecified meter',
        'as cast convergent venturi tube', 'Miller eccentric orifice',
        'ISO 15377 eccentric orifice'}, [-]

    Returns
    -------
    beta : float
        Differential pressure meter diameter ratio, [-]

    Notes
    -----

    Examples
    --------
    >>> differential_pressure_meter_beta(D=0.2575, D2=0.184,
    ... meter_type='cone meter')
    0.6995709873957624
    '''
    if meter_type in beta_simple_meters:
        beta = D2/D
    elif meter_type == CONE_METER or meter_type == HOLLINGSHEAD_CONE:
        beta = diameter_ratio_cone_meter(D=D, Dc=D2)
    elif meter_type == WEDGE_METER or meter_type == HOLLINGSHEAD_WEDGE:
        beta = diameter_ratio_wedge_meter(D=D, H=D2)
    else:
        raise ValueError(_unsupported_meter_msg)
    return beta


_meter_type_to_corr_default = {
    CONCENTRIC_ORIFICE: ISO_5167_ORIFICE,
    ECCENTRIC_ORIFICE: ISO_15377_ECCENTRIC_ORIFICE,
    CONICAL_ORIFICE: ISO_15377_CONICAL_ORIFICE,
    QUARTER_CIRCLE_ORIFICE: ISO_15377_QUARTER_CIRCLE_ORIFICE,
    SEGMENTAL_ORIFICE: MILLER_SEGMENTAL_ORIFICE,
    }

def differential_pressure_meter_C_epsilon(D, D2, m, P1, P2, rho, mu, k,
                                          meter_type, taps=None,
                                          tap_position=None, C_specified=None):
    r'''Calculates the discharge coefficient and expansibility of a flow
    meter given the mass flow rate, the upstream pressure, the second
    pressure value, and the orifice diameter for a differential
    pressure flow meter based on the geometry of the meter, measured pressures
    of the meter, and the density, viscosity, and isentropic exponent of the
    fluid.

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    D2 : float
        Diameter of orifice, or venturi meter orifice, or flow tube orifice,
        or cone meter end diameter, or wedge meter fluid flow height, [m]
    m : float
        Mass flow rate of fluid through the flow meter, [kg/s]
    P1 : float
        Static pressure of fluid upstream of differential pressure meter at the
        cross-section of the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of differential pressure meter or
        at the prescribed location (varies by type of meter) [Pa]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    k : float
        Isentropic exponent of fluid, [-]
    meter_type : str
        One of {'conical orifice', 'orifice', 'machined convergent venturi tube',
        'ISO 5167 orifice', 'Miller quarter circle orifice', 'Hollingshead venturi sharp',
        'segmental orifice', 'Miller conical orifice', 'Miller segmental orifice',
        'quarter circle orifice', 'Hollingshead v cone', 'wedge meter', 'eccentric orifice',
        'venuri nozzle', 'rough welded convergent venturi tube', 'ISA 1932 nozzle',
        'ISO 15377 quarter-circle orifice', 'Hollingshead venturi smooth',
        'Hollingshead orifice', 'cone meter', 'Hollingshead wedge', 'Miller orifice',
        'long radius nozzle', 'ISO 15377 conical orifice', 'unspecified meter',
        'as cast convergent venturi tube', 'Miller eccentric orifice',
        'ISO 15377 eccentric orifice'}, [-]
    taps : str, optional
        The orientation of the taps; one of 'corner', 'flange', 'D', or 'D/2';
        applies for orifice meters only, [-]
    tap_position : str, optional
        The rotation of the taps, used **only for the eccentric orifice case**
        where the pressure profile is are not symmetric; '180 degree' for the
        normal case where the taps are opposite the orifice bore, and
        '90 degree' for the case where, normally for operational reasons, the
        taps are near the bore [-]
    C_specified : float, optional
        If specified, the correlation for the meter type is not used - this
        value is returned for `C`

    Returns
    -------
    C : float
        Coefficient of discharge of the specified flow meter type at the
        specified conditions, [-]
    expansibility : float
        Expansibility factor (1 for incompressible fluids, less than 1 for
        real fluids), [-]

    Notes
    -----
    This function should be called by an outer loop when solving for a
    variable.

    The latest ISO formulations for `expansibility` are used with the Miller
    correlations.

    Examples
    --------
    >>> differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    ... P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    ... meter_type='ISO 5167 orifice', taps='D')
    (0.6151252900244296, 0.9711026966676307)
    '''
#    # Translate default meter type to implementation specific correlation
    if meter_type == CONCENTRIC_ORIFICE:
        meter_type = ISO_5167_ORIFICE
    elif meter_type == ECCENTRIC_ORIFICE:
        meter_type = ISO_15377_ECCENTRIC_ORIFICE
    elif meter_type == CONICAL_ORIFICE:
        meter_type = ISO_15377_CONICAL_ORIFICE
    elif meter_type == QUARTER_CIRCLE_ORIFICE:
        meter_type = ISO_15377_QUARTER_CIRCLE_ORIFICE
    elif meter_type == SEGMENTAL_ORIFICE:
        meter_type = MILLER_SEGMENTAL_ORIFICE

    if meter_type == ISO_5167_ORIFICE:
        C = C_Reader_Harris_Gallagher(D, D2, rho, mu, m, taps)
        epsilon = orifice_expansibility(D, D2, P1, P2, k)
    elif meter_type == ISO_15377_ECCENTRIC_ORIFICE:
        C = C_eccentric_orifice_ISO_15377_1998(D, D2)
        epsilon = orifice_expansibility(D, D2, P1, P2, k)
    elif meter_type == ISO_15377_QUARTER_CIRCLE_ORIFICE:
        C = C_quarter_circle_orifice_ISO_15377_1998(D, D2)
        epsilon = orifice_expansibility(D, D2, P1, P2, k)
    elif meter_type == ISO_15377_CONICAL_ORIFICE:
        C = ISO_15377_CONICAL_ORIFICE_C
        # Average of concentric square edge orifice and ISA 1932 nozzles
        epsilon = 0.5*(orifice_expansibility(D, D2, P1, P2, k)
                       + nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k))

    elif meter_type in (MILLER_ORIFICE, MILLER_ECCENTRIC_ORIFICE,
                      MILLER_SEGMENTAL_ORIFICE, MILLER_QUARTER_CIRCLE_ORIFICE):
        C = C_Miller_1996(D, D2, rho, mu, m, subtype=meter_type, taps=taps,
                          tap_position=tap_position)
        epsilon = orifice_expansibility(D, D2, P1, P2, k)
    elif meter_type == MILLER_CONICAL_ORIFICE:
        C = C_Miller_1996(D, D2, rho, mu, m, subtype=meter_type, taps=taps,
                          tap_position=tap_position)
        epsilon = 0.5*(orifice_expansibility(D, D2, P1, P2, k)
                       + nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k))
    elif meter_type == LONG_RADIUS_NOZZLE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = C_long_radius_nozzle(D=D, Do=D2, rho=rho, mu=mu, m=m)
    elif meter_type == ISA_1932_NOZZLE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = C_ISA_1932_nozzle(D=D, Do=D2, rho=rho, mu=mu, m=m)
    elif meter_type == VENTURI_NOZZLE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = C_venturi_nozzle(D=D, Do=D2)

    elif meter_type == AS_CAST_VENTURI_TUBE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = AS_CAST_VENTURI_TUBE_C
    elif meter_type == MACHINED_CONVERGENT_VENTURI_TUBE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = MACHINED_CONVERGENT_VENTURI_TUBE_C
    elif meter_type == ROUGH_WELDED_CONVERGENT_VENTURI_TUBE:
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
        C = ROUGH_WELDED_CONVERGENT_VENTURI_TUBE_C

    elif meter_type == CONE_METER:
        epsilon = cone_meter_expansibility_Stewart(D=D, Dc=D2, P1=P1, P2=P2, k=k)
        C = CONE_METER_C
    elif meter_type == WEDGE_METER:
        beta = diameter_ratio_wedge_meter(D=D, H=D2)
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P1, k=k, beta=beta)
        C = C_wedge_meter_ISO_5167_6_2017(D=D, H=D2)
    elif meter_type == HOLLINGSHEAD_ORIFICE:
        v = m/((0.25*pi*D*D)*rho)
        Re_D = rho*v*D/mu
        C = float(bisplev(D2/D, log(Re_D), orifice_std_Hollingshead_tck))
        epsilon = orifice_expansibility(D, D2, P1, P2, k)
    elif meter_type == HOLLINGSHEAD_VENTURI_SMOOTH:
        v = m/((0.25*pi*D*D)*rho)
        Re_D = rho*v*D/mu
        C = interp(log(Re_D), venturi_logRes_Hollingshead, venturi_smooth_Cs_Hollingshead, extrapolate=True)
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
    elif meter_type == HOLLINGSHEAD_VENTURI_SHARP:
        v = m/((0.25*pi*D*D)*rho)
        Re_D = rho*v*D/mu
        C = interp(log(Re_D), venturi_logRes_Hollingshead, venturi_sharp_Cs_Hollingshead, extrapolate=True)
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P2, k=k)
    elif meter_type == HOLLINGSHEAD_CONE:
        v = m/((0.25*pi*D*D)*rho)
        Re_D = rho*v*D/mu
        beta = diameter_ratio_cone_meter(D, D2)
        C = float(bisplev(beta, log(Re_D), cone_Hollingshead_tck))
        epsilon = cone_meter_expansibility_Stewart(D=D, Dc=D2, P1=P1, P2=P2, k=k)
    elif meter_type == HOLLINGSHEAD_WEDGE:
        v = m/((0.25*pi*D*D)*rho)
        Re_D = rho*v*D/mu
        beta = diameter_ratio_wedge_meter(D=D, H=D2)
        C = float(bisplev(beta, log(Re_D), wedge_Hollingshead_tck))
        epsilon = nozzle_expansibility(D=D, Do=D2, P1=P1, P2=P1, k=k, beta=beta)
    elif meter_type == UNSPECIFIED_METER:
        epsilon = orifice_expansibility(D, D2, P1, P2, k) # Default to orifice type expansibility
        if C_specified is None:
            raise ValueError("For unspecified meter type, C_specified is required")
    else:
        raise ValueError(_unsupported_meter_msg)
    if C_specified is not None:
        C = C_specified
    return C, epsilon



def err_dp_meter_solver_m(m_D, D, D2, P1, P2, rho, mu, k, meter_type, taps, tap_position, C_specified):
    m = m_D*D
    C, epsilon = differential_pressure_meter_C_epsilon(D, D2, m, P1, P2, rho,
                                                  mu, k, meter_type,
                                                  taps=taps, tap_position=tap_position,
                                                  C_specified=C_specified)
    m_calc = flow_meter_discharge(D=D, Do=D2, P1=P1, P2=P2, rho=rho,
                                C=C, expansibility=epsilon)
    err =  m - m_calc
    return err

def err_dp_meter_solver_P2(P2, D, D2, m, P1, rho, mu, k, meter_type, taps, tap_position, C_specified):
    C, epsilon = differential_pressure_meter_C_epsilon(D, D2, m, P1, P2, rho,
                                                  mu, k, meter_type,
                                                  taps=taps, tap_position=tap_position,
                                                  C_specified=C_specified)
    m_calc = flow_meter_discharge(D=D, Do=D2, P1=P1, P2=P2, rho=rho,
                                C=C, expansibility=epsilon)
    return m - m_calc

def err_dp_meter_solver_D2(D2, D, m, P1, P2, rho, mu, k, meter_type, taps, tap_position, C_specified):
    C, epsilon = differential_pressure_meter_C_epsilon(D, D2, m, P1, P2, rho,
                                                  mu, k, meter_type,
                                                  taps=taps, tap_position=tap_position, C_specified=C_specified)
    m_calc = flow_meter_discharge(D=D, Do=D2, P1=P1, P2=P2, rho=rho,
                                C=C, expansibility=epsilon)
    return m - m_calc

def err_dp_meter_solver_P1(P1, D, D2, m, P2, rho, mu, k, meter_type, taps, tap_position, C_specified):
    C, epsilon = differential_pressure_meter_C_epsilon(D, D2, m, P1, P2, rho,
                                                  mu, k, meter_type,
                                                  taps=taps, tap_position=tap_position, C_specified=C_specified)
    m_calc = flow_meter_discharge(D=D, Do=D2, P1=P1, P2=P2, rho=rho,
                                C=C, expansibility=epsilon)
    return m - m_calc

def differential_pressure_meter_solver(D, rho, mu, k, D2=None, P1=None, P2=None,
                                       m=None, meter_type=ISO_5167_ORIFICE,
                                       taps=None, tap_position=None,
                                       C_specified=None):
    r'''Calculates either the mass flow rate, the upstream pressure, the second
    pressure value, or the orifice diameter for a differential
    pressure flow meter based on the geometry of the meter, measured pressures
    of the meter, and the density, viscosity, and isentropic exponent of the
    fluid. This solves an equation iteratively to obtain the correct flow rate.

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    rho : float
        Density of fluid at `P1`, [kg/m^3]
    mu : float
        Viscosity of fluid at `P1`, [Pa*s]
    k : float
        Isentropic exponent of fluid, [-]
    D2 : float, optional
        Diameter of orifice, or venturi meter orifice, or flow tube orifice,
        or cone meter end diameter, or wedge meter fluid flow height, [m]
    P1 : float, optional
        Static pressure of fluid upstream of differential pressure meter at the
        cross-section of the pressure tap, [Pa]
    P2 : float, optional
        Static pressure of fluid downstream of differential pressure meter or
        at the prescribed location (varies by type of meter) [Pa]
    m : float, optional
        Mass flow rate of fluid through the flow meter, [kg/s]
    meter_type : str
        One of {'conical orifice', 'orifice', 'machined convergent venturi tube',
        'ISO 5167 orifice', 'Miller quarter circle orifice', 'Hollingshead venturi sharp',
        'segmental orifice', 'Miller conical orifice', 'Miller segmental orifice',
        'quarter circle orifice', 'Hollingshead v cone', 'wedge meter', 'eccentric orifice',
        'venuri nozzle', 'rough welded convergent venturi tube', 'ISA 1932 nozzle',
        'ISO 15377 quarter-circle orifice', 'Hollingshead venturi smooth',
        'Hollingshead orifice', 'cone meter', 'Hollingshead wedge', 'Miller orifice',
        'long radius nozzle', 'ISO 15377 conical orifice', 'unspecified meter',
        'as cast convergent venturi tube', 'Miller eccentric orifice',
        'ISO 15377 eccentric orifice'}, [-]
    taps : str, optional
        The orientation of the taps; one of 'corner', 'flange', 'D', or 'D/2';
        applies for orifice meters only, [-]
    tap_position : str, optional
        The rotation of the taps, used **only for the eccentric orifice case**
        where the pressure profile is are not symmetric; '180 degree' for the
        normal case where the taps are opposite the orifice bore, and
        '90 degree' for the case where, normally for operational reasons, the
        taps are near the bore [-]
    C_specified : float, optional
        If specified, the correlation for the meter type is not used - this
        value is used for `C`

    Returns
    -------
    ans : float
        One of `m`, the mass flow rate of the fluid; `P1`, the pressure
        upstream of the flow meter; `P2`, the second pressure
        tap's value; and `D2`, the diameter of the measuring device; units
        of respectively, kg/s, Pa, Pa, or m

    Notes
    -----
    See the appropriate functions for the documentation for the formulas and
    references used in each method.

    The solvers make some assumptions about the range of values answers may be
    in.

    Note that the solver for the upstream pressure uses the provided values of
    density, viscosity and isentropic exponent; whereas these values all
    depend on pressure (albeit to a small extent). An outer loop should be
    added with pressure-dependent values calculated in it for maximum accuracy.

    It would be possible to solve for the upstream pipe diameter, but there is
    no use for that functionality.

    If a meter has already been calibrated to have a known `C`, this may be
    provided and it will be used in place of calculating one.

    Examples
    --------
    >>> differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0,
    ... P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
    ... meter_type='ISO 5167 orifice', taps='D')
    7.702338035732167

    >>> differential_pressure_meter_solver(D=0.07366, m=7.702338, P1=200000.0,
    ... P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
    ... meter_type='ISO 5167 orifice', taps='D')
    0.04999999990831885
    '''
    if m is None and D is not None and D2 is not None and P1 is not None and P2 is not None:
        # Diameter to mass flow ratio
        m_D_guess = 40
        if rho < 100.0:
            m_D_guess *= 1e-2
        return secant(err_dp_meter_solver_m, m_D_guess, args=(D, D2, P1, P2, rho, mu, k, meter_type, taps, tap_position, C_specified))*D
    elif D2 is None and D is not None and m is not None and P1 is not None and P2 is not None:
        args = (D, m, P1, P2, rho, mu, k, meter_type, taps, tap_position, C_specified)
        try:
            return brenth(err_dp_meter_solver_D2, D*(1-1E-9), D*5E-3, args=args)
        except:
            try:
                return secant(err_dp_meter_solver_D2, D*.3, args=args, high=D, low=D*1e-10)
            except:
                return secant(err_dp_meter_solver_D2, D*.75, args=args, high=D, low=D*1e-10)
    elif P2 is None and D is not None and D2 is not None and m is not None and P1 is not None:
        args = (D, D2, m, P1, rho, mu, k, meter_type, taps, tap_position, C_specified)
        try:
            return brenth(err_dp_meter_solver_P2, P1*(1-1E-9), P1*0.5, args=args)
        except:
            return secant(err_dp_meter_solver_P2, P1*0.5, low=P1*1e-10, args=args, high=P1, bisection=True)
    elif P1 is None and D is not None and D2 is not None and m is not None and P2 is not None:
        args = (D, D2, m, P2, rho, mu, k, meter_type, taps, tap_position, C_specified)
        try:
            return brenth(err_dp_meter_solver_P1, P2*(1+1E-9), P2*1.4, args=args)
        except:
            return secant(err_dp_meter_solver_P1, P2*1.5, args=args, low=P2, bisection=True)
    else:
        raise ValueError('Solver is capable of solving for one of P1, P2, D2, or m only.')

# Set of orifice types that get their dP calculated with `dP_orifice`.
_dP_orifice_set = set([ISO_5167_ORIFICE, ISO_15377_ECCENTRIC_ORIFICE,
                  ISO_15377_CONICAL_ORIFICE, ISO_15377_QUARTER_CIRCLE_ORIFICE,

                  MILLER_ORIFICE, MILLER_ECCENTRIC_ORIFICE,
                  MILLER_SEGMENTAL_ORIFICE, MILLER_CONICAL_ORIFICE,
                  MILLER_QUARTER_CIRCLE_ORIFICE,

                  HOLLINGSHEAD_ORIFICE,

                  CONCENTRIC_ORIFICE, ECCENTRIC_ORIFICE, CONICAL_ORIFICE,
                  SEGMENTAL_ORIFICE, QUARTER_CIRCLE_ORIFICE])

_missing_C_msg = "Parameter C is required for this orifice type"

def differential_pressure_meter_dP(D, D2, P1, P2, C=None,
                                   meter_type=ISO_5167_ORIFICE):
    r'''Calculates the non-recoverable pressure drop of a differential
    pressure flow meter based on the geometry of the meter, measured pressures
    of the meter, and for most models the meter discharge coefficient.

    Parameters
    ----------
    D : float
        Upstream internal pipe diameter, [m]
    D2 : float
        Diameter of orifice, or venturi meter orifice, or flow tube orifice,
        or cone meter end diameter, or wedge meter fluid flow height, [m]
    P1 : float
        Static pressure of fluid upstream of differential pressure meter at the
        cross-section of the pressure tap, [Pa]
    P2 : float
        Static pressure of fluid downstream of differential pressure meter or
        at the prescribed location (varies by type of meter) [Pa]
    C : float, optional
        Coefficient of discharge (used only in orifice plates, and venturi
        nozzles), [-]
    meter_type : str
        One of {'conical orifice', 'orifice', 'machined convergent venturi tube',
        'ISO 5167 orifice', 'Miller quarter circle orifice', 'Hollingshead venturi sharp',
        'segmental orifice', 'Miller conical orifice', 'Miller segmental orifice',
        'quarter circle orifice', 'Hollingshead v cone', 'wedge meter', 'eccentric orifice',
        'venuri nozzle', 'rough welded convergent venturi tube', 'ISA 1932 nozzle',
        'ISO 15377 quarter-circle orifice', 'Hollingshead venturi smooth',
        'Hollingshead orifice', 'cone meter', 'Hollingshead wedge', 'Miller orifice',
        'long radius nozzle', 'ISO 15377 conical orifice', 'unspecified meter',
        'as cast convergent venturi tube', 'Miller eccentric orifice',
        'ISO 15377 eccentric orifice'}, [-]

    Returns
    -------
    dP : float
        Non-recoverable pressure drop of the differential pressure flow
        meter, [Pa]

    Notes
    -----
    See the appropriate functions for the documentation for the formulas and
    references used in each method.

    Wedge meters, and venturi nozzles do not have standard formulas available
    for pressure drop computation.

    Examples
    --------
    >>> differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0,
    ... P2=183000.0, meter_type='as cast convergent venturi tube')
    1788.5717754177406
    '''
    if meter_type in _dP_orifice_set:
        if C is None: raise ValueError(_missing_C_msg)
        dP = dP_orifice(D=D, Do=D2, P1=P1, P2=P2, C=C)
    elif meter_type == LONG_RADIUS_NOZZLE:
        if C is None: raise ValueError(_missing_C_msg)
        dP = dP_orifice(D=D, Do=D2, P1=P1, P2=P2, C=C)
    elif meter_type == ISA_1932_NOZZLE:
        if C is None: raise ValueError(_missing_C_msg)
        dP = dP_orifice(D=D, Do=D2, P1=P1, P2=P2, C=C)
    elif meter_type == VENTURI_NOZZLE:
        raise NotImplementedError("Venturi meter does not have an implemented pressure drop correlation")

    elif (meter_type == AS_CAST_VENTURI_TUBE
          or meter_type == MACHINED_CONVERGENT_VENTURI_TUBE
          or meter_type == ROUGH_WELDED_CONVERGENT_VENTURI_TUBE
          or meter_type == HOLLINGSHEAD_VENTURI_SMOOTH
          or meter_type == HOLLINGSHEAD_VENTURI_SHARP):
        dP = dP_venturi_tube(D=D, Do=D2, P1=P1, P2=P2)

    elif meter_type == CONE_METER or meter_type == HOLLINGSHEAD_CONE:
        dP = dP_cone_meter(D=D, Dc=D2, P1=P1, P2=P2)
    elif meter_type == WEDGE_METER or meter_type == HOLLINGSHEAD_WEDGE:
        dP = dP_wedge_meter(D=D, H=D2, P1=P1, P2=P2)
    else:
        raise ValueError(_unsupported_meter_msg)
    return dP
