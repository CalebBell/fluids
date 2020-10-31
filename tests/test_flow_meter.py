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

from fluids import *
from fluids.constants import inch
from math import log10, log, exp
from fluids.numerics import secant, linspace, logspace, assert_close, isclose, assert_close1d, assert_close2d
import pytest

def test_flow_meter_discharge():
    m = flow_meter_discharge(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1646, C=0.5988, expansibility=0.9975)
    assert_close(m, 0.01120390943807026)

def test_orifice_expansibility():
    epsilon = orifice_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    assert_close(epsilon, 0.9974739057343425)
    # Tested against a value in the standard

def test_orifice_expansibility_1989():
    # No actual sample points
    epsilon = orifice_expansibility_1989(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    assert_close(epsilon, 0.9970510687411718)

def test_C_Reader_Harris_Gallagher():
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='corner' )
    assert_close(C, 0.6000085121444034)

    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='D' )
    assert_close(C, 0.5988219225153976)

    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='flange' )
    assert_close(C, 0.5990042535666878)
#
#def test_Reader_Harris_Gallagher_discharge():
#    m = Reader_Harris_Gallagher_discharge(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, taps='D')
#    assert_close(m, 7.702338035732167)

    with pytest.raises(Exception):
        C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5,  m=0.12, taps='NOTALOCATION')

    # Test continuity at the low-diameter function
    kwargs = dict(Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='corner')
    C1 = C_Reader_Harris_Gallagher(D=0.07112, **kwargs)
    C2 = C_Reader_Harris_Gallagher(D=0.07112-1e-13, **kwargs)
    assert_close(C1, C2)

def test_C_Miller_1996():
    C_flange_ISO = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')
    C_corner_ISO = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='corner')
    C_D_D2_ISO = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps=ORIFICE_D_AND_D_2_TAPS)

    C_flange = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ORIFICE, taps=ORIFICE_FLANGE_TAPS)
    C_flange_2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype='orifice', taps=ORIFICE_FLANGE_TAPS)
    assert C_flange == C_flange_2
    assert_close(C_flange, 0.599065557156788, rtol=1e-12)
    assert_close(C_flange, C_flange_ISO, rtol=2e-4)

    C_flange_small_ISO = C_Reader_Harris_Gallagher(D=0.04, Do=0.02, rho=1.165, mu=1.85E-5, m=0.2, taps='flange')
    C_flange_small = C_Miller_1996(D=0.04, Do=0.02, rho=1.165, mu=1.85E-5, m=0.2, subtype=MILLER_ORIFICE, taps=ORIFICE_FLANGE_TAPS)
    assert_close(C_flange_small, 0.6035249226284967, rtol=1e-12)
    assert_close(C_flange_small_ISO, C_flange_small, rtol=1e-2)

    C_corner = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ORIFICE, taps=ORIFICE_CORNER_TAPS)
    assert_close(C_corner, 0.5991255880475622, rtol=1e-12)
    assert_close(C_corner, C_corner_ISO, rtol=2e-3)

    C_D_D2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ORIFICE, taps=ORIFICE_D_AND_D_2_TAPS)
    assert_close(C_D_D2, 0.5836056345693277, rtol=1e-12)
    assert_close(C_D_D2, C_D_D2_ISO, rtol=3e-2)

    C_pipe = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ORIFICE, taps=ORIFICE_PIPE_TAPS)
    assert_close(C_pipe, 0.6338716097225481, rtol=1e-12)



    C_flange_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_SEGMENTAL_ORIFICE, taps=ORIFICE_FLANGE_TAPS)
    C_flange_small2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype='segmental orifice', taps=ORIFICE_FLANGE_TAPS)
    assert C_flange_small == C_flange_small

    C_flange_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_SEGMENTAL_ORIFICE, taps=ORIFICE_FLANGE_TAPS)
    assert_close(C_flange_small, 0.6343546437000684, rtol=1e-12)
    assert_close(C_flange_large, 0.6301688962913937, rtol=1e-12)

    C_vc_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_SEGMENTAL_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS)
    C_vc_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_SEGMENTAL_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS)
    assert_close(C_vc_small, 0.6341386019820933, rtol=1e-12)
    assert_close(C_vc_large, 0.6301688962913937, rtol=1e-12)

    C_flange_opp_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps='flange', tap_position=TAPS_OPPOSITE)
    C_flange_opp_small2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype='eccentric orifice', taps='flange', tap_position=TAPS_OPPOSITE)
    assert_close(C_flange_opp_small, 0.6096299230744815, rtol=1e-12)
    C_flange_opp_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_ECCENTRIC_ORIFICE, taps='flange', tap_position=TAPS_OPPOSITE)
    assert_close(C_flange_opp_large, 0.6196903510975135, rtol=1e-12)

    C_flange_side_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps='flange', tap_position=TAPS_SIDE)
    C_flange_side_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_ECCENTRIC_ORIFICE, taps='flange', tap_position=TAPS_SIDE)
    assert_close(C_flange_side_small, 0.6086231594104639, rtol=1e-12)
    assert_close(C_flange_side_large, 0.6227796822413327, rtol=1e-12)


    C_vc_opp_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS, tap_position=TAPS_OPPOSITE)
    assert_close(C_vc_opp_small, 0.6108105171632562, rtol=1e-12)
    C_vc_opp_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS, tap_position=TAPS_OPPOSITE)
    assert_close(C_vc_opp_large, 0.6190713098741648, rtol=1e-12)

    C_vc_side_small = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS, tap_position=TAPS_SIDE)
    C_vc_side_large = C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_VENA_CONTRACTA_TAPS, tap_position=TAPS_SIDE)
    assert_close(C_vc_side_small, 0.6089351556538237, rtol=1e-12)
    assert_close(C_vc_side_large, 0.6214809940486437, rtol=1e-12)

    # Error testing
    with pytest.raises(ValueError):
        C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ORIFICE, taps='NOTATAP')

    with pytest.raises(ValueError):
        C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps='NOTATAP')

    with pytest.raises(ValueError):
        C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_FLANGE_TAPS, tap_position='NOTAPOSITION')

    with pytest.raises(ValueError):
        C_Miller_1996(D=0.2, Do=0.08, rho=1.165, mu=1.85E-5, m=2, subtype=MILLER_SEGMENTAL_ORIFICE, taps='BADTAP')

    with pytest.raises(ValueError):
        C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype='BADTYPE')

    # Conical
    C_high = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_CONICAL_ORIFICE)
    assert C_high == 0.73
    C_low = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.0001, subtype=MILLER_CONICAL_ORIFICE)
    assert C_low == 0.734
    C_low2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.0001, subtype='conical orifice')
    assert C_low2 == C_low
    # Quarter circle
    C_circ = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype=MILLER_QUARTER_CIRCLE_ORIFICE)
    assert_close(C_circ, 0.7750496225919683)
    C_circ2 = C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, subtype='quarter circle orifice')
    assert C_circ == C_circ2

def test_differential_pressure_meter_discharge():
    # Orifice
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=ISO_5167_ORIFICE, taps='D')
    assert_close(m, 7.702338035732167)

    # Nozzle meters
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(m, 11.86828167015467)

    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=ISA_1932_NOZZLE)
    assert_close(m, 11.370262314304702)

    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=VENTURI_NOZZLE)
    assert_close(m, 11.471786198133566)

    # Venturi tubes
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=AS_CAST_VENTURI_TUBE)
    assert_close(m, 11.867774156238344)

    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=MACHINED_CONVERGENT_VENTURI_TUBE)
    assert_close(m, 12.000442363269464)

    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=ROUGH_WELDED_CONVERGENT_VENTURI_TUBE)
    assert_close(m, 11.879834902332082)

    # Cone meter
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=CONE_METER)
    assert_close(m, 9.997923896460703)

    # wedge meter
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=WEDGE_METER)
    assert_close(m, 8.941980099523539)

    with pytest.raises(ValueError):
        differential_pressure_meter_solver(D=.07366, m=7.702338, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')


def test_differential_pressure_meter_diameter():
    # ISO 5167 orifice
    D2 = differential_pressure_meter_solver(D=0.07366, m=7.702338035732167, P1=200000.0,  P2=183000.0, rho=999.1, mu=0.0011, k=1.33,  meter_type=ISO_5167_ORIFICE, taps='D')
    assert_close(D2, 0.05)

    # Nozzle meters
    D2 = differential_pressure_meter_solver(D=0.07366, m= 11.86828167015467, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(D2, 0.05)

    D2 = differential_pressure_meter_solver(D=0.07366, m=11.370262314304702, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=ISA_1932_NOZZLE)
    assert_close(D2, 0.05)

    D2 = differential_pressure_meter_solver(D=0.07366, m=11.471786198133566, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=VENTURI_NOZZLE)
    assert_close(D2, 0.05)

    # Venturi tubes
    D2 = differential_pressure_meter_solver(D=0.07366, m=11.867774156238344, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=AS_CAST_VENTURI_TUBE)
    assert_close(D2, 0.05)

    D2 = differential_pressure_meter_solver(D=0.07366, m=12.000442363269464, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=MACHINED_CONVERGENT_VENTURI_TUBE)
    assert_close(D2, 0.05)

    D2 = differential_pressure_meter_solver(D=0.07366, m=11.879834902332082, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=ROUGH_WELDED_CONVERGENT_VENTURI_TUBE)
    assert_close(D2, 0.05)

    # Cone meter
    D2 = differential_pressure_meter_solver(D=0.07366, m=9.997923896460703, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=CONE_METER)
    assert_close(D2, 0.05)

    # wedge meter
    D2 = differential_pressure_meter_solver(D=0.07366, m=8.941980099523539, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type=WEDGE_METER)
    assert_close(D2, 0.05)


def test_differential_pressure_meter_P2():
    P2 = differential_pressure_meter_solver(D=0.07366, m=7.702338035732167, P1=200000.0,  D2=0.05, rho=999.1, mu=0.0011, k=1.33,  meter_type=ISO_5167_ORIFICE, taps='D')
    assert_close(P2, 183000.0)

    # Nozzle meters
    P2 = differential_pressure_meter_solver(D=0.07366, m= 11.86828167015467, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(P2, 183000.0)

    P2 = differential_pressure_meter_solver(D=0.07366, m=11.370262314304702, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=ISA_1932_NOZZLE)
    assert_close(P2, 183000.0)

    P2 = differential_pressure_meter_solver(D=0.07366, m=11.471786198133566, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=VENTURI_NOZZLE)
    assert_close(P2, 183000.0)

    # Venturi tubes
    P2 = differential_pressure_meter_solver(D=0.07366, m=11.867774156238344, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=AS_CAST_VENTURI_TUBE)
    assert_close(P2, 183000.0)

    P2 = differential_pressure_meter_solver(D=0.07366, m=12.000442363269464, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=MACHINED_CONVERGENT_VENTURI_TUBE)
    assert_close(P2, 183000.0)

    P2 = differential_pressure_meter_solver(D=0.07366, m=11.879834902332082, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=ROUGH_WELDED_CONVERGENT_VENTURI_TUBE)
    assert_close(P2, 183000.0)

    # Cone meter
    P2 = differential_pressure_meter_solver(D=0.07366, m=9.997923896460703, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=CONE_METER)
    assert_close(P2, 183000.0)

    # Wedge meter
    P2 = differential_pressure_meter_solver(D=0.07366, m=8.941980099523539, P1=200000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=WEDGE_METER)
    assert_close(P2, 183000.0)

def test_differential_pressure_meter_P1():
    P1 = differential_pressure_meter_solver(D=0.07366, m=7.702338035732167, P2=183000.0,  D2=0.05, rho=999.1, mu=0.0011, k=1.33,  meter_type=ISO_5167_ORIFICE, taps='D')
    assert_close(P1, 200000)

    # Nozzle meters
    P1 = differential_pressure_meter_solver(D=0.07366, m=11.86828167015467, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(P1, 200000)

    P1 = differential_pressure_meter_solver(D=0.07366, m=11.370262314304702, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=ISA_1932_NOZZLE)
    assert_close(P1, 200000)

    P1 = differential_pressure_meter_solver(D=0.07366, m=11.471786198133566, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=VENTURI_NOZZLE)
    assert_close(P1, 200000)

    # Venturi tubes
    P1 = differential_pressure_meter_solver(D=0.07366, m=11.867774156238344, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=AS_CAST_VENTURI_TUBE)
    assert_close(P1, 200000)

    P1 = differential_pressure_meter_solver(D=0.07366, m=12.000442363269464, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=MACHINED_CONVERGENT_VENTURI_TUBE)
    assert_close(P1, 200000)

    P1 = differential_pressure_meter_solver(D=0.07366, m=11.879834902332082, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=ROUGH_WELDED_CONVERGENT_VENTURI_TUBE)
    assert_close(P1, 200000)

    # Cone meter
    P1 = differential_pressure_meter_solver(D=0.07366, m=9.997923896460703, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=CONE_METER)
    assert_close(P1, 200000)

    # Wedge meter
    P1 = differential_pressure_meter_solver(D=0.07366, m=8.941980099523539, P2=183000.0, D2=0.05, rho=999.1, mu=0.0011, k=1.33, meter_type=WEDGE_METER)
    assert_close(P1, 200000)




def test_differential_pressure_meter_solver_limits():
    # ISO 5167 orifice - How low can P out go?
    P_out = differential_pressure_meter_solver(D=0.07366, m=7.702338, P1=200000.0, D2=0.0345, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')
    assert_close(P_out, 37914.15989971644)

    # same point
    D2_recalc = differential_pressure_meter_solver(D=0.07366, m=7.702338, P1=200000.0, P2=37914.15989971644, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')
    assert_close(D2_recalc, 0.0345)

    P1_recalc = differential_pressure_meter_solver(D=0.07366, m=7.702338, P2=37914.15989971644, D2=0.0345, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')
    assert_close(P1_recalc, 200000.0)

    m_recalc = differential_pressure_meter_solver(D=0.07366, P1=200000, P2=37914.15989971644, D2=0.0345, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')
    assert_close(m_recalc, 7.702338)

def test_differential_pressure_meter_solver_misc():
    # Test for types

    m_expect = 7.918128618951788
    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, rho=999.1, mu=0.0011,
                                       k=1.33, meter_type=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_FLANGE_TAPS, tap_position=TAPS_SIDE)
    assert_close(m, m_expect)

    P1 = differential_pressure_meter_solver(m=7.918128618951788, D=0.07366, D2=0.05,  P2=183000.0, rho=999.1, mu=0.0011,
                                       k=1.33, meter_type=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_FLANGE_TAPS, tap_position=TAPS_SIDE)
    assert_close(P1, 200000)
    P2 = differential_pressure_meter_solver(m=7.918128618951788, D=0.07366, D2=0.05,  P1=200000.0, rho=999.1, mu=0.0011,
                                       k=1.33, meter_type=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_FLANGE_TAPS, tap_position=TAPS_SIDE)
    assert_close(P2, 183000)

    D2 = differential_pressure_meter_solver(m=7.918128618951788, D=0.07366, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011,
                                       k=1.33, meter_type=MILLER_ECCENTRIC_ORIFICE, taps=ORIFICE_FLANGE_TAPS, tap_position=TAPS_SIDE)
    assert_close(D2, 0.05)

    m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, rho=1.2, mu=0.00011, k=1.33, meter_type='ISO 5167 orifice', taps='D')
    assert_close(m, 0.2695835697819371)

def test_unspecified_meter_C_specified():
    for t in ('unspecified meter', 'ISO 5167 orifice'):
        m = differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0,
         P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
        meter_type=t, taps='D', C_specified=0.6)
        assert_close(m, 7.512945567976503)

        D2 = differential_pressure_meter_solver(D=0.07366, m=7.512945567976503, D2=None, P1=200000.0,
         P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
        meter_type=t, taps='D', C_specified=0.6)
        assert_close(D2, 0.05)

        P1 = differential_pressure_meter_solver(D=0.07366, D2=0.05, m=7.512945567976503,
         P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
        meter_type=t, taps='D', C_specified=0.6)
        assert_close(P1, 200000.0)

        P2 = differential_pressure_meter_solver(D=0.07366, D2=0.05, m=7.512945567976503,
         P1=200000.0, rho=999.1, mu=0.0011, k=1.33,
        meter_type=t, taps='D', C_specified=0.6)
        assert_close(P2, 183000.0)

    with pytest.raises(ValueError):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0,
         P2=183000.0, rho=999.1, mu=0.0011, k=1.33,
        meter_type='unspecified meter', taps='D', C_specified=None)



def test_C_eccentric_orifice_ISO_15377_1998():
    C =  C_eccentric_orifice_ISO_15377_1998(.2, .075)
    assert_close(C, 0.6351923828125)

    # Does not perfectly match - like error in ISO.
    D = 1.0
    betas = [1e-2*i for i in range(46, 85, 1)]
    Cs_expect = [0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.628, 0.628, 0.628, 0.628, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.628, 0.628, 0.627, 0.626, 0.625, 0.624, 0.623, 0.621, 0.620, 0.618, 0.616, 0.613, 0.611, 0.608, 0.605, 0.601, 0.597]
    Cs_calc = [C_eccentric_orifice_ISO_15377_1998(D=D, Do=beta_i) for beta_i in betas]
    for Ci, Cj in zip(Cs_expect, Cs_calc):
        assert isclose(Ci, Cj, rel_tol=1.02e-3)

def test_C_quarter_circle_orifice_ISO_15377_1998():
    C = C_quarter_circle_orifice_ISO_15377_1998(.2, .075)
    assert_close(C, 0.7785148437500001, rtol=1e-12)

    betas = [0.245, 0.250, 0.260, 0.270, 0.280, 0.290, 0.300, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.390, 0.400, 0.410, 0.420, 0.430, 0.440, 0.450, 0.460, 0.470, 0.480, 0.490, 0.500, 0.510, 0.520, 0.530, 0.540, 0.550, 0.560, 0.570, 0.580, 0.590, 0.600]
    Cs_expect = [0.772, 0.772, 0.772, 0.773, 0.773, 0.773, 0.774, 0.774, 0.775, 0.775, 0.776, 0.776, 0.777, 0.778, 0.779, 0.780, 0.781, 0.783, 0.784, 0.786, 0.787, 0.789, 0.791, 0.794, 0.796, 0.799, 0.802, 0.805, 0.808, 0.812, 0.816, 0.820, 0.824, 0.829, 0.834, 0.839, 0.844]
    for Do, C_expect in zip(betas, Cs_expect):
        C = C_quarter_circle_orifice_ISO_15377_1998(D=1, Do=Do)
        assert (round(C, 3) == C_expect)

def test_K_to_discharge_coefficient():
    C = K_to_discharge_coefficient(D=0.07366, Do=0.05, K=5.2314291729754)
    assert_close(C, 0.6151200000000001)

def test_discharge_coefficient_to_K():
    K = discharge_coefficient_to_K(D=0.07366, Do=0.05, C=0.61512)
    assert_close(K, 5.2314291729754)

def test_dP_orifice():
    dP = dP_orifice(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, C=0.61512)
    assert_close(dP, 9069.474705745388)

def test_velocity_of_approach_factor():
    factor = velocity_of_approach_factor(D=0.0739, Do=0.0222)
    assert_close(factor, 1.0040970074165514)

def test_flow_coefficient():
    factor = flow_coefficient(D=0.0739, Do=0.0222, C=0.6)
    assert_close(factor, 0.6024582044499308)

def test_nozzle_expansibility():
    epsilon = nozzle_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    assert_close(epsilon, 0.9945702344566746)

    assert_close(nozzle_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=1e5, k=1.4), 1, rtol=1e-14)

def test_C_long_radius_nozzle():
    C = C_long_radius_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    assert_close(C, 0.9805503704679863)

def test_C_ISA_1932_nozzle():
    C = C_ISA_1932_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    assert_close(C, 0.9635849973250495)

def test_C_venturi_nozzle():
    C = C_venturi_nozzle(D=0.07391, Do=0.0422)
    assert_close(C, 0.9698996454169576)


def test_diameter_ratio_cone_meter():
    beta = diameter_ratio_cone_meter(D=0.2575, Dc=0.184)
    assert_close(beta, 0.6995709873957624)
    # Example in 1 matches exactly;
    beta = diameter_ratio_cone_meter(D=10.137*inch, Dc=7.244*inch)
    assert_close(beta, 0.6995232442563669)


def test_diameter_ratio_wedge_meter():
    beta = diameter_ratio_wedge_meter(D=6.065*inch, H=1.82*inch)
    assert_close(beta, 0.5024062047655528)

    beta = diameter_ratio_wedge_meter(D=7.981*inch, H=3.192*inch)
    assert_close(beta, 0.6111198863284705)

    beta = diameter_ratio_wedge_meter(D=7.981*inch, H=2.394*inch)
    assert_close(beta, 0.5022667856496335)


def test_cone_meter_expansibility_Stewart():
    eps = cone_meter_expansibility_Stewart(D=1, Dc=0.8930285549745876, P1=1E6, P2=1E6*.85, k=1.2)
    assert_close(eps, 0.91530745625)

def test_wedge_meter_expansibility():
    data = [[1.0000, 0.9871, 0.9741, 0.9610, 0.9478, 0.9345, 0.9007, 0.8662, 0.8308],
            [1.0000, 0.9863, 0.9726, 0.9588, 0.9449, 0.9310, 0.8957, 0.8599, 0.8234],
            [1.0000, 0.9848, 0.9696, 0.9544, 0.9393, 0.9241, 0.8860, 0.8479, 0.8094],
            [1.0000, 0.9820, 0.9643, 0.9467, 0.9292, 0.9119, 0.8692, 0.8272, 0.7857],
            [1.0000, 0.9771, 0.9547, 0.9329, 0.9117, 0.8909, 0.8408, 0.7930, 0.7472]]

    h_ds = [0.2, 0.3, 0.4, 0.5, 0.6]
    pressure_ratios = [1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8, 0.75]
    calculated = []
    for i, h_d in enumerate(h_ds):
        row = []
        beta = diameter_ratio_wedge_meter(D=1, H=h_d)
        for j, p_ratio in enumerate(pressure_ratios):

            ans = nozzle_expansibility(D=1, Do=h_d, P1=1E5, P2=1E5*p_ratio, k=1.2, beta=beta)
            row.append(ans)
        calculated.append(row)

    assert_close2d(data, calculated, rtol=1e-4)


def test_dP_wedge_meter():
    dP = dP_wedge_meter(1, .7, 1E6, 9.5E5)
    assert_close(dP, 20344.849697483587)


def test_dP_cone_meter():
    dP = dP_cone_meter(1, .7, 1E6, 9.5E5)
    assert_close(dP, 25470.093437973323)


def test_C_wedge_meter_Miller():
    # Large bore
    D = 0.15239999999999998
    C = C_wedge_meter_Miller(D=D, H=0.3*D)
    assert_close(C, 0.7267069372687651)

    # Tiny bore
    C = C_wedge_meter_Miller(D=.6*inch, H=0.3*.6*inch)
    assert_close(C, 0.8683022107124251)

    # Medium bore
    C = C_wedge_meter_Miller(D=1.3*inch, H=0.3*1.3*inch)
    assert_close(C, 1.15113726440674)


def test_C_wedge_meter_ISO_5167_6_2017():
    C = C_wedge_meter_ISO_5167_6_2017(D=0.1524, H=0.3*0.1524)
    assert_close(C, 0.724792059539853)

def test_dP_venturi_tube():
    dP = dP_venturi_tube(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0)
    assert_close(dP, 1788.5717754177406)


def test_C_Reader_Harris_Gallagher_wet_venturi_tube():
    # Example 1
    # Works don't change anything
    C = C_Reader_Harris_Gallagher_wet_venturi_tube(mg=5.31926, ml=5.31926/2, rhog=50.0, rhol=800., D=.1, Do=.06, H=1)
    assert_close(C, 0.9754210845876333)

    # From ISO 5167-4:2003, 5.6,
    # epsilon = 0.994236
    # nozzle_expansibility, orifice_expansibility
    epsilon = nozzle_expansibility(D=.1, Do=.06, P1=60E5, P2=59.5E5, k=1.3)
    assert_close(epsilon, 0.994236, rtol=0, atol=.0000001)

    # Example 2
    # Had to solve backwards to get ml, but C checks out perfectly
    C = C_Reader_Harris_Gallagher_wet_venturi_tube(ml=0.434947009566078, mg=6.3817, rhog=50.0, rhol=1000., D=.1, Do=.06, H=1.35)
    # Don't know what the ml is
    #  0,976 992 is C
    assert_close(C, 0.9769937323602329)


def test_dP_Reader_Harris_Gallagher_wet_venturi_tube():
    dP = dP_Reader_Harris_Gallagher_wet_venturi_tube(ml=5.31926/2, mg=5.31926, rhog=50.0, rhol=800., D=.1, Do=.06, H=1.0,  P1=6E6, P2=6E6-5E4)
    assert_close(dP, 16957.43843129572)


def test_differential_pressure_meter_dP():
    for m in [AS_CAST_VENTURI_TUBE, MACHINED_CONVERGENT_VENTURI_TUBE, ROUGH_WELDED_CONVERGENT_VENTURI_TUBE, HOLLINGSHEAD_VENTURI_SMOOTH, HOLLINGSHEAD_VENTURI_SHARP]:
        dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, meter_type=m)
        assert_close(dP, 1788.5717754177406)

    dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, C=0.61512, meter_type=ISO_5167_ORIFICE)
    assert_close(dP, 9069.474705745388)

    dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, C=0.61512, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(dP, 9069.474705745388)

    dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, C=0.61512, meter_type=ISA_1932_NOZZLE)
    assert_close(dP, 9069.474705745388)

    for m in (CONE_METER, HOLLINGSHEAD_CONE):
        dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0,  meter_type=m)
        assert_close(dP, 8380.848307054845)

    for m in (WEDGE_METER, HOLLINGSHEAD_WEDGE):
        dP = differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0,  meter_type=m)
        assert_close(dP, 7112.927753356824)

    with pytest.raises(Exception):
        differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, meter_type=VENTURI_NOZZLE)

    with pytest.raises(ValueError):
        differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, meter_type='NOTAMETER')



def test_differential_pressure_meter_beta():
    beta = differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=LONG_RADIUS_NOZZLE)
    assert_close(beta, 0.7145631067961165)

    beta = differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=WEDGE_METER)
    assert_close(beta, 0.8743896375172885)

    beta = differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=CONE_METER)
    assert_close(beta, 0.6995709873957624)

    with pytest.raises(ValueError):
        differential_pressure_meter_beta(D=0.07366, D2=0.05, meter_type='NOTAMETER')

    assert_close(differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=HOLLINGSHEAD_CONE),
        differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=CONE_METER))

    assert_close(differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=HOLLINGSHEAD_WEDGE),
        differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type=WEDGE_METER))



def test_cone_meter_expansibility_Stewart_full():
    err = lambda Dc, beta : diameter_ratio_cone_meter(D=1, Dc=Dc) - beta

    solve_Dc = lambda beta : float(secant(err, .7, args=(beta,)))

    # Accidentally missed the beta ratio 0.75, oops
    vals = [[1.0000, 0.9887, 0.9774, 0.9661, 0.9548, 0.9435, 0.9153, 0.8871, 0.8588],
    [1.0000, 0.9885, 0.9769, 0.9654, 0.9538, 0.9423, 0.9134, 0.8846, 0.8557],
    [1.0000, 0.9881, 0.9762, 0.9644, 0.9525, 0.9406, 0.9109, 0.8812, 0.8515],
    [1.0000, 0.9877, 0.9754, 0.9630, 0.9507, 0.9384, 0.9076, 0.8768, 0.8460],
    [1.0000, 0.9871, 0.9742, 0.9613, 0.9485, 0.9356, 0.9033, 0.8711, 0.8389],
    [1.0000, 0.9864, 0.9728, 0.9592, 0.9456, 0.9320, 0.8980, 0.8640, 0.8300]]
    pressure_ratios = [1, 0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8, 0.75]
    betas = [.45, .5, .55, .6, .65, .7, .75]

    k = 1.2
    for i, beta in enumerate(betas[:-1]):
        Dc = solve_Dc(beta)
        for j, pr in enumerate(pressure_ratios):
            eps = cone_meter_expansibility_Stewart(D=1, Dc=Dc, P1=1E5, P2=pr*1E5, k=1.2)
            eps = round(eps, 4)
            assert eps == vals[i][j]


def test_C_ISA_1932_nozzle_full():
    Cs = [[0.9616, 0.9692, 0.9750, 0.9773, 0.9789, 0.9813, 0.9820, 0.9821, 0.9822],
    [0.9604, 0.9682, 0.9741, 0.9764, 0.9781, 0.9805, 0.9812, 0.9813, 0.9814],
    [0.9592, 0.9672, 0.9731, 0.9755, 0.9773, 0.9797, 0.9804, 0.9805, 0.9806],
    [0.9579, 0.9661, 0.9722, 0.9746, 0.9763, 0.9788, 0.9795, 0.9797, 0.9797],
    [0.9567, 0.9650, 0.9711, 0.9736, 0.9754, 0.9779, 0.9786, 0.9787, 0.9788],
    [0.9554, 0.9638, 0.9700, 0.9726, 0.9743, 0.9769, 0.9776, 0.9777, 0.9778],
    [0.9542, 0.9626, 0.9689, 0.9715, 0.9733, 0.9758, 0.9766, 0.9767, 0.9768],
    [0.9529, 0.9614, 0.9678, 0.9703, 0.9721, 0.9747, 0.9754, 0.9756, 0.9757],
    [0.9516, 0.9602, 0.9665, 0.9691, 0.9709, 0.9735, 0.9743, 0.9744, 0.9745],
    [0.9503, 0.9589, 0.9653, 0.9678, 0.9696, 0.9722, 0.9730, 0.9731, 0.9732],
    [0.9490, 0.9576, 0.9639, 0.9665, 0.9683, 0.9709, 0.9717, 0.9718, 0.9719],
    [0.9477, 0.9562, 0.9626, 0.9651, 0.9669, 0.9695, 0.9702, 0.9704, 0.9705],
    [0.9464, 0.9548, 0.9611, 0.9637, 0.9655, 0.9680, 0.9688, 0.9689, 0.9690],
    [0.9451, 0.9534, 0.9596, 0.9621, 0.9639, 0.9664, 0.9672, 0.9673, 0.9674],
    [0.9438, 0.9520, 0.9581, 0.9606, 0.9623, 0.9648, 0.9655, 0.9656, 0.9657],
    [0.9424, 0.9505, 0.9565, 0.9589, 0.9606, 0.9630, 0.9638, 0.9639, 0.9640],
    [0.9411, 0.9490, 0.9548, 0.9572, 0.9588, 0.9612, 0.9619, 0.9620, 0.9621],
    [0.9398, 0.9474, 0.9531, 0.9554, 0.9570, 0.9593, 0.9600, 0.9601, 0.9602],
    [0.9385, 0.9458, 0.9513, 0.9535, 0.9550, 0.9573, 0.9579, 0.9580, 0.9581],
    [0.9371, 0.9442, 0.9494, 0.9515, 0.9530, 0.9551, 0.9558, 0.9559, 0.9560],
    [0.9358, 0.9425, 0.9475, 0.9495, 0.9509, 0.9529, 0.9535, 0.9536, 0.9537],
    [0.9345, 0.9408, 0.9455, 0.9473, 0.9487, 0.9506, 0.9511, 0.9512, 0.9513],
    [0.9332, 0.9390, 0.9434, 0.9451, 0.9464, 0.9481, 0.9487, 0.9487, 0.9488],
    [0.9319, 0.9372, 0.9412, 0.9428, 0.9440, 0.9456, 0.9460, 0.9461, 0.9462],
    [0.9306, 0.9354, 0.9390, 0.9404, 0.9414, 0.9429, 0.9433, 0.9434, 0.9435],
    [0.9293, 0.9335, 0.9367, 0.9379, 0.9388, 0.9401, 0.9405, 0.9405, 0.9406],
    [0.9280, 0.9316, 0.9343, 0.9353, 0.9361, 0.9372, 0.9375, 0.9375, 0.9376],
    [0.9268, 0.9296, 0.9318, 0.9326, 0.9332, 0.9341, 0.9344, 0.9344, 0.9344],
    [0.9255, 0.9276, 0.9292, 0.9298, 0.9303, 0.9309, 0.9311, 0.9311, 0.9312],
    [0.9243, 0.9256, 0.9265, 0.9269, 0.9272, 0.9276, 0.9277, 0.9277, 0.9278],
    [0.9231, 0.9235, 0.9238, 0.9239, 0.9240, 0.9241, 0.9242, 0.9242, 0.9242],
    [0.9219, 0.9213, 0.9209, 0.9208, 0.9207, 0.9205, 0.9205, 0.9205, 0.9205],
    [0.9207, 0.9192, 0.9180, 0.9176, 0.9172, 0.9168, 0.9166, 0.9166, 0.9166],
    [0.9195, 0.9169, 0.9150, 0.9142, 0.9136, 0.9128, 0.9126, 0.9126, 0.9125],
    [0.9184, 0.9147, 0.9118, 0.9107, 0.9099, 0.9088, 0.9084, 0.9084, 0.9083],
    [0.9173, 0.9123, 0.9086, 0.9071, 0.9060, 0.9045, 0.9041, 0.9040, 0.9040],
    [0.9162, 0.9100, 0.9053, 0.9034, 0.9020, 0.9001, 0.8996, 0.8995, 0.8994]]


    def C_ISA_1932_nozzle(D, Do, Re_D):
        beta = Do/D
        C = (0.9900 - 0.2262*beta**4.1
             - (0.00175*beta**2 - 0.0033*beta**4.15)*(1E6/Re_D)**1.15)
        return C

    Rd_values = [2E4, 3E4, 5E4, 7E4, 1E5, 3E5, 1E6, 2E6, 1E7]
    betas = [i/100. for i in range(44, 81)]

    for i in range(len(betas)):
        Cs_expect = Cs[i]
        beta = betas[i]
        Cs_calc = [round(C_ISA_1932_nozzle(D=1, Do=beta, Re_D=i), 4) for i in Rd_values]
        assert_close1d(Cs_expect, Cs_calc, atol=1E-4)

    # There were three typos in there in the values for beta of 0.77 or 0.78.
    # values: 0.9215, 0.9412, 0.9803

def test_C_long_radius_nozzle_full():
    Cs = [[0.9673, 0.9759, 0.9834, 0.9873, 0.9900, 0.9924, 0.9936, 0.9952, 0.9956],
    [0.9659, 0.9748, 0.9828, 0.9868, 0.9897, 0.9922, 0.9934, 0.9951, 0.9955],
    [0.9645, 0.9739, 0.9822, 0.9864, 0.9893, 0.9920, 0.9933, 0.9951, 0.9955],
    [0.9632, 0.9730, 0.9816, 0.9860, 0.9891, 0.9918, 0.9932, 0.9950, 0.9954],
    [0.9619, 0.9721, 0.9810, 0.9856, 0.9888, 0.9916, 0.9930, 0.9950, 0.9954],
    [0.9607, 0.9712, 0.9805, 0.9852, 0.9885, 0.9914, 0.9929, 0.9949, 0.9954],
    [0.9596, 0.9704, 0.9800, 0.9848, 0.9882, 0.9913, 0.9928, 0.9948, 0.9953],
    [0.9584, 0.9696, 0.9795, 0.9845, 0.9880, 0.9911, 0.9927, 0.9948, 0.9953],
    [0.9573, 0.9688, 0.9790, 0.9841, 0.9877, 0.9910, 0.9926, 0.9947, 0.9953],
    [0.9562, 0.9680, 0.9785, 0.9838, 0.9875, 0.9908, 0.9925, 0.9947, 0.9952],
    [0.9552, 0.9673, 0.9780, 0.9834, 0.9873, 0.9907, 0.9924, 0.9947, 0.9952],
    [0.9542, 0.9666, 0.9776, 0.9831, 0.9870, 0.9905, 0.9923, 0.9946, 0.9952],
    [0.9532, 0.9659, 0.9771, 0.9828, 0.9868, 0.9904, 0.9922, 0.9946, 0.9951],
    [0.9523, 0.9652, 0.9767, 0.9825, 0.9866, 0.9902, 0.9921, 0.9945, 0.9951],
    [0.9513, 0.9645, 0.9763, 0.9822, 0.9864, 0.9901, 0.9920, 0.9945, 0.9951],
    [0.9503, 0.9639, 0.9759, 0.9819, 0.9862, 0.9900, 0.9919, 0.9944, 0.9950],
    [0.9499, 0.9635, 0.9756, 0.9818, 0.9861, 0.9899, 0.9918, 0.9944, 0.9950],
    [0.9494, 0.9632, 0.9754, 0.9816, 0.9860, 0.9898, 0.9918, 0.9944, 0.9950],
    [0.9490, 0.9629, 0.9752, 0.9815, 0.9859, 0.9898, 0.9917, 0.9944, 0.9950],
    [0.9485, 0.9626, 0.9750, 0.9813, 0.9858, 0.9897, 0.9917, 0.9944, 0.9950],
    [0.9481, 0.9623, 0.9748, 0.9812, 0.9857, 0.9897, 0.9917, 0.9943, 0.9950],
    [0.9476, 0.9619, 0.9746, 0.9810, 0.9856, 0.9896, 0.9916, 0.9943, 0.9950],
    [0.9472, 0.9616, 0.9745, 0.9809, 0.9855, 0.9895, 0.9916, 0.9943, 0.9949],
    [0.9468, 0.9613, 0.9743, 0.9808, 0.9854, 0.9895, 0.9915, 0.9943, 0.9949],
    [0.9463, 0.9610, 0.9741, 0.9806, 0.9853, 0.9894, 0.9915, 0.9943, 0.9949],
    [0.9459, 0.9607, 0.9739, 0.9805, 0.9852, 0.9893, 0.9914, 0.9942, 0.9949],
    [0.9455, 0.9604, 0.9737, 0.9804, 0.9851, 0.9893, 0.9914, 0.9942, 0.9949],
    [0.9451, 0.9601, 0.9735, 0.9802, 0.9850, 0.9892, 0.9914, 0.9942, 0.9949],
    [0.9447, 0.9599, 0.9733, 0.9801, 0.9849, 0.9892, 0.9913, 0.9942, 0.9949],
    [0.9443, 0.9596, 0.9731, 0.9800, 0.9848, 0.9891, 0.9913, 0.9942, 0.9948],
    [0.9439, 0.9593, 0.9730, 0.9799, 0.9847, 0.9891, 0.9912, 0.9941, 0.9948],
    [0.9435, 0.9590, 0.9728, 0.9797, 0.9846, 0.9890, 0.9912, 0.9941, 0.9948],
    [0.9430, 0.9587, 0.9726, 0.9796, 0.9845, 0.9889, 0.9912, 0.9941, 0.9948],
    [0.9427, 0.9584, 0.9724, 0.9795, 0.9845, 0.9889, 0.9911, 0.9941, 0.9948],
    [0.9423, 0.9581, 0.9722, 0.9793, 0.9844, 0.9888, 0.9911, 0.9941, 0.9948],
    [0.9419, 0.9579, 0.9721, 0.9792, 0.9843, 0.9888, 0.9910, 0.9941, 0.9948],
    [0.9415, 0.9576, 0.9719, 0.9791, 0.9842, 0.9887, 0.9910, 0.9940, 0.9948],
    [0.9411, 0.9573, 0.9717, 0.9790, 0.9841, 0.9887, 0.9910, 0.9940, 0.9947],
    [0.9407, 0.9570, 0.9715, 0.9789, 0.9840, 0.9886, 0.9909, 0.9940, 0.9947],
    [0.9403, 0.9568, 0.9714, 0.9787, 0.9839, 0.9886, 0.9909, 0.9940, 0.9947],
    [0.9399, 0.9565, 0.9712, 0.9786, 0.9839, 0.9885, 0.9908, 0.9940, 0.9947],
    [0.9396, 0.9562, 0.9710, 0.9785, 0.9838, 0.9884, 0.9908, 0.9940, 0.9947],
    [0.9392, 0.9560, 0.9709, 0.9784, 0.9837, 0.9884, 0.9908, 0.9939, 0.9947],
    [0.9388, 0.9557, 0.9707, 0.9783, 0.9836, 0.9883, 0.9907, 0.9939, 0.9947],
    [0.9385, 0.9555, 0.9705, 0.9781, 0.9835, 0.9883, 0.9907, 0.9939, 0.9947],
    [0.9381, 0.9552, 0.9704, 0.9780, 0.9834, 0.9882, 0.9907, 0.9939, 0.9947]]

    Rd_values = [1E4, 2E4, 5E4, 1E5, 2E5, 5E5, 1E6, 5E6, 1E7]
    betas = [i/100. for i in list(range(20, 51, 2)) + list(range(51, 81))]

    def C_long_radius_nozzle(D, Do, Re_D):
        beta = Do/D
        return 0.9965 - 0.00653*beta**0.5*(1E6/Re_D)**0.5


    for i in range(len(betas)):
        Cs_expect = Cs[i]
        beta = betas[i]
        Cs_calc = [round(C_long_radius_nozzle(D=1, Do=beta, Re_D=i), 4) for i in Rd_values]
        assert_close1d(Cs_expect, Cs_calc, atol=1E-4)

    # Errata:
    # 0.9834 to 0.9805
    # 0.9828 to 9800
    # 0.9822 to 0.9795
    # 0.9816 to 0.979
    # 0.981 to 0.9785
    # 0.9805 to 0.9780
    # 0.98   to 0.9776
    # 0.9795 to 0.9771
    # 0.979 to 0.9767
    # 0.9785 to 0.9763
    #  9.9607 to 0.9607
    # 0.9875 to 0.9785

def test_C_venturi_nozzle_full():
    # Many values do not match well, but the equation has been checked with both standards.
    betas = [0.32, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78]
    Cs = [0.9847, 0.9846, 0.9845, 0.9843, 0.9841, 0.9838, 0.9836, 0.9833, 0.9830, 0.9826, 0.9823, 0.9818, 0.9814, 0.9809, 0.9804, 0.9798, 0.9792, 0.9786, 0.9779, 0.9771, 0.9763, 0.9755, 0.9745, 0.9736, 0.9725, 0.9714, 0.9702, 0.9689, 0.9676, 0.9661, 0.9646, 0.9630, 0.9613, 0.9595, 0.9576, 0.9556, 0.9535, 0.9512, 0.9489, 0.9464, 0.9438, 0.9411, 0.9382, 0.9352, 0.9321, 0.9288, 0.9253, 0.9236]
    Cs_calc = [C_venturi_nozzle(D=1, Do=beta) for beta in betas]
    assert_close1d(Cs, Cs_calc, rtol=5E-3)

def test_differential_pressure_meter_C_epsilon():
    # Some random cases
    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    meter_type=ISO_15377_ECCENTRIC_ORIFICE)
    assert_close(C, 0.6284616939680627)
    assert_close(eps, 0.9711026966676307)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    meter_type=ISO_15377_QUARTER_CIRCLE_ORIFICE)
    assert_close(C, 0.899402420975695)
    assert_close(eps, 0.9711026966676307)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    meter_type=ISO_15377_CONICAL_ORIFICE)
    assert_close(C, 0.734)
    assert_close(eps, 0.9532330165749132)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    meter_type=MILLER_ORIFICE, taps='corner')
    assert_close(C, 0.6068011224659587)
    assert_close(eps, 0.9711026966676307)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,
    meter_type=MILLER_CONICAL_ORIFICE)
    assert_close(C, 0.73)
    assert_close(eps, 0.9532330165749132)


    # Test one case of the default translation
    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,taps='corner',
    meter_type=CONCENTRIC_ORIFICE)
    C_iso, eps_iso = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168,taps='corner',
    meter_type=CONCENTRIC_ORIFICE)

    assert C == C_iso
    assert eps == eps_iso

    with pytest.raises(ValueError):
        differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
                                              P2=183000.0, rho=999.1, mu=0.0011,
                                              k=1.33, m=7.702338035732168, meter_type='NOTAREAMETER')


    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=.01,
        meter_type=HOLLINGSHEAD_ORIFICE)
    assert_close(C, 0.7809066489631418)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=.01,
        meter_type=HOLLINGSHEAD_VENTURI_SMOOTH)
    assert_close(C, 0.7765555753764869)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=.01,
        meter_type=HOLLINGSHEAD_VENTURI_SHARP)
    assert_close(C, 0.7710760458207614)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=.01,
        meter_type=HOLLINGSHEAD_CONE)
    assert_close(C, 0.5796605776735264)

    C, eps = differential_pressure_meter_C_epsilon(D=0.07366, D2=0.025, P1=200000.0,
    P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=.01,
        meter_type=HOLLINGSHEAD_WEDGE)
    assert_close(C, 0.7002380207294499)



@pytest.mark.fuzz
@pytest.mark.slow
def test_fuzz_K_to_discharge_coefficient():
    '''
    # Testing the different formulas
    from sympy import *
    C, beta, K = symbols('C, beta, K')

    expr = Eq(K, (sqrt(1 - beta**4*(1 - C*C))/(C*beta**2) - 1)**2)
    solns = solve(expr, C)
    [i.subs({'K': 5.2314291729754, 'beta': 0.05/0.07366}) for i in solns]

    [-sqrt(-beta**4/(-2*sqrt(K)*beta**4 + K*beta**4) + 1/(-2*sqrt(K)*beta**4 + K*beta**4)),
 sqrt(-beta**4/(-2*sqrt(K)*beta**4 + K*beta**4) + 1/(-2*sqrt(K)*beta**4 + K*beta**4)),
 -sqrt(-beta**4/(2*sqrt(K)*beta**4 + K*beta**4) + 1/(2*sqrt(K)*beta**4 + K*beta**4)),
 sqrt(-beta**4/(2*sqrt(K)*beta**4 + K*beta**4) + 1/(2*sqrt(K)*beta**4 + K*beta**4))]

    # Getting the formula
    from sympy import *
    C, beta, K = symbols('C, beta, K')

    expr = Eq(K, (sqrt(1 - beta**4*(1 - C*C))/(C*beta**2) - 1)**2)
    print(latex(solve(expr, C)[3]))
    '''

    Ds = logspace(log10(1-1E-9), log10(1E-9), 8)
    for D_ratio in Ds:
        Ks = logspace(log10(1E-9), log10(50000), 8)
        Ks_recalc = []
        for K in Ks:
            C = K_to_discharge_coefficient(D=1.0, Do=D_ratio, K=K)
            K_calc = discharge_coefficient_to_K(D=1.0, Do=D_ratio, C=C)
            Ks_recalc.append(K_calc)
        assert_close1d(Ks, Ks_recalc)

@pytest.mark.scipy
@pytest.mark.slow
def test_orifice_std_Hollingshead_fit():
    from scipy.interpolate import RectBivariateSpline, bisplev
    from fluids.flow_meter import orifice_std_Hollingshead_tck, orifice_std_logRes_Hollingshead, orifice_std_betas_Hollingshead, orifice_std_Hollingshead_Cs
    import numpy as np

    obj = RectBivariateSpline(orifice_std_betas_Hollingshead, orifice_std_logRes_Hollingshead,
                              np.array(orifice_std_Hollingshead_Cs), s=0, kx=3, ky=3)

    assert_close(obj(.55, log(1e3))[0][0], bisplev(.55, log(1e3), orifice_std_Hollingshead_tck))

    assert_close1d(obj.tck[0], orifice_std_Hollingshead_tck[0])
    assert_close1d(obj.tck[1], orifice_std_Hollingshead_tck[1])
    assert_close1d(obj.tck[2], orifice_std_Hollingshead_tck[2])


@pytest.mark.scipy
@pytest.mark.slow
def test_wedge_Hollingshead_fit():
    from scipy.interpolate import RectBivariateSpline, bisplev
    import numpy as np
    from fluids.flow_meter import wedge_betas_Hollingshead, wedge_logRes_Hollingshead, wedge_Hollingshead_Cs, wedge_Hollingshead_tck

    obj = RectBivariateSpline(wedge_betas_Hollingshead, wedge_logRes_Hollingshead,
                              np.array(wedge_Hollingshead_Cs), s=0, kx=1, ky=3)
    assert_close(obj(.55, log(1e4)), bisplev(.55, log(1e4), wedge_Hollingshead_tck))

    assert_close1d(obj.tck[0], wedge_Hollingshead_tck[0])
    assert_close1d(obj.tck[1], wedge_Hollingshead_tck[1])
    assert_close1d(obj.tck[2], wedge_Hollingshead_tck[2])

@pytest.mark.scipy
@pytest.mark.slow
def test_cone_Hollingshead_fit():
    from scipy.interpolate import RectBivariateSpline, bisplev
    import numpy as np
    from fluids.flow_meter import cone_logRes_Hollingshead, cone_betas_Hollingshead, cone_Hollingshead_Cs, cone_Hollingshead_tck

    obj = RectBivariateSpline(cone_betas_Hollingshead, cone_logRes_Hollingshead,
                              np.array(cone_Hollingshead_Cs), s=0, kx=2, ky=3)
    assert_close(obj(.77, log(1e4)), bisplev(.77, log(1e4), cone_Hollingshead_tck))

    assert_close1d(obj.tck[0], cone_Hollingshead_tck[0])
    assert_close1d(obj.tck[1], cone_Hollingshead_tck[1])
    assert_close1d(obj.tck[2], cone_Hollingshead_tck[2])
