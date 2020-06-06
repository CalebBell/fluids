# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids import *
import fluids.vectorized
from math import *
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d
import pytest
try:
    import numba
    import fluids.numba
    import fluids.numba_vectorized
except:
    numba = None
import numpy as np

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Clamond_numba():
    assert_close(fluids.numba.Clamond(10000.0, 2.0), 
                 fluids.Clamond(10000.0, 2.0), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, True),
                 fluids.Clamond(10000.0, 2.0, True), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, False),
                 fluids.Clamond(10000.0, 2.0, False), rtol=5e-15)
    
    Res = np.array([1e5, 1e6])
    eDs = np.array([1e-5, 1e-6])
    fast = np.array([False]*2)
    assert_close1d(fluids.numba_vectorized.Clamond(Res, eDs, fast), 
                   fluids.vectorized.Clamond(Res, eDs, fast), rtol=1e-14)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_string_error_message_outside_function():
    fluids.numba.entrance_sharp('Miller')
    fluids.numba.entrance_sharp()
    
    fluids.numba.entrance_angled(30, 'Idelchik')
    fluids.numba.entrance_angled(30, None)
    fluids.numba.entrance_angled(30.0)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_interp():

    assert_close(fluids.numba.CSA_motor_efficiency(100*hp, closed=True, poles=6, high_efficiency=True), 0.95)
    
    # Should take ~10 us
    powers = np.array([70000]*100)
    closed = np.array([True]*100)
    poles = np.array([6]*100)
    high_efficiency = np.array([True]*100)
    fluids.numba_vectorized.CSA_motor_efficiency(powers, closed, poles, high_efficiency)
    
    assert_close(fluids.numba.bend_rounded_Crane(Di=.4020, rc=.4*5, angle=30),
                 fluids.bend_rounded_Crane(Di=.4020, rc=.4*5, angle=30))
    

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_constants():
    assert_close(fluids.numba.K_separator_demister_York(975000), 0.09635076944244816)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_calling_function_in_other_module():
    assert_close(fluids.numba.ft_Crane(.5), 0.011782458726227104, rtol=1e-4)
    
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_None_is_not_multiplied_add_check_on_is_None():
    assert_close(fluids.numba.polytropic_exponent(1.4, eta_p=0.78), 1.5780346820809246, rtol=1e-5)
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_core_from_other_module():
    assert_close(fluids.numba.helical_turbulent_fd_Srinivasan(1E4, 0.01, .02), 0.0570745212117107)
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_string_branches():
    # Currently slower
    assert_close(fluids.numba.C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange'),  0.5990326277163659)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_interp_with_own_list():
    assert_close(fluids.numba.dP_venturi_tube(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0), 1788.5717754177406)
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_C_Reader_Harris_Gallagher_wet_venturi_tube_numba():
    assert_close(fluids.numba.C_Reader_Harris_Gallagher_wet_venturi_tube(mg=5.31926, ml=5.31926/2,  rhog=50.0, rhol=800., D=.1, Do=.06, H=1), 0.9754210845876333)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_rename_constant():
    assert_close(fluids.numba.friction_plate_Martin_1999(Re=20000, plate_enlargement_factor=1.15), 2.284018089834135)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_list_in_list_constant_converted():
    assert_close(fluids.numba.friction_plate_Kumar(Re=2000, chevron_angle=30),
                 friction_plate_Kumar(Re=2000, chevron_angle=30))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_have_to_make_zero_division_a_check():
    # Manually requires changes, and is unpythonic
    assert_close(fluids.numba.SA_ellipsoidal_head(2, 1.5), 
                 SA_ellipsoidal_head(2, 1.5))
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_functions_used_to_return_different_return_value_signatures_changed():
    assert_close1d(fluids.numba.SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',sideB_a=0.5), 
                    SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',sideB_a=0.5))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Colebrook_ignored():
    fd = fluids.numba.Colebrook(1e5, 1e-5)
    assert_close(fd, 0.018043802895063684, rtol=1e-14)



@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_secant_runs():
    # Really feel like the kwargs should work in object mode, but it doesn't
    # Just gets slower
    @numba.jit
    def to_solve(x):
        return sin(x*.3) - .5
    fluids.numba.secant(to_solve, .3, ytol=1e-10)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_brenth_runs():
    @numba.njit
    def to_solve(x, goal):
        return sin(x*.3) - goal
    
    ans = fluids.numba.brenth(to_solve, .3, 2, args=(.45,))
    assert_close(ans, 1.555884463490988)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_lambertw_runs():
    assert_close(fluids.numba.numerics.lambertw(5.0), 1.3267246652422002)
    
    
    assert_close(fluids.numba.Prandtl_von_Karman_Nikuradse(1e7), 0.008102669430874914)
    

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_ellipe_runs():
    assert_close(fluids.numba.plate_enlargement_factor(amplitude=5E-4, wavelength=3.7E-3),
                 1.1611862034509677, rtol=1e-10)
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_control_valve_noise():
    dB = fluids.numba.control_valve_noise_l_2015(m=40, P1=1E6, P2=6.5E5, Psat=2.32E3, rho=997, c=1400, Kv=77.848, d=0.1, Di=0.1071, FL=0.92, Fd=0.42, t_pipe=0.0036, rho_pipe=7800.0, c_pipe=5000.0,rho_air=1.293, c_air=343.0, An=-4.6)
    assert_close(dB, 81.58200097996539)
    
    dB = fluids.numba.control_valve_noise_g_2011(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3, gamma=1.22, MW=19.8, Kv=77.85,  d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98, Fd=0.296, t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0, rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(dB, 91.67702674629604)
    
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_friction_factor():
    fluids.numba.friction_factor(1e5, 1e-3)
    
    
    assert_close(fluids.numba.friction.friction_factor(1e4, 1e-4, Method='Churchill_1973'),
                 fluids.friction_factor(1e4, 1e-4, Method='Churchill_1973'))
    assert_close(fluids.numba.friction.friction_factor(1e4, 1e-4),
                 fluids.friction_factor(1e4, 1e-4))
    assert_close(fluids.numba.friction.friction_factor(1e2, 1e-4),
                 fluids.friction_factor(1e2, 1e-4))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_AvailableMethods_removal():
    assert_close(fluids.numba.friction_factor_curved(Re=1E5, Di=0.02, Dc=0.5),
                 fluids.friction_factor_curved(Re=1E5, Di=0.02, Dc=0.5))

    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_bisplev_uses():
    K = fluids.numba.entrance_beveled(Di=0.1, l=0.003, angle=45, method='Idelchik') 
    assert_close(K, 0.39949999999999997)
    
    assert_close(fluids.numba.VFD_efficiency(100*hp, load=0.2), 
                 fluids.VFD_efficiency(100*hp, load=0.2))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_splev_uses():
    methods = ['Rennels', 'Miller', 'Idelchik', 'Harris',  'Crane']
    Ks = [fluids.numba.entrance_distance(Di=0.1, t=0.0005, method=m) for m in methods]
    Ks_orig = [fluids.fittings.entrance_distance(Di=0.1, t=0.0005, method=m) for m in methods]
    assert_close1d(Ks, Ks_orig)

    # Same speed
    assert_close(fluids.numba.entrance_rounded(Di=0.1, rc=0.0235),
                 fluids.fittings.entrance_rounded(Di=0.1, rc=0.0235))
    
    
    # Got 10x faster! no strings.
    assert_close(fluids.numba.bend_rounded_Miller(Di=.6, bend_diameters=2, angle=90,  Re=2e6, roughness=2E-5, L_unimpeded=30*.6),
                 fluids.bend_rounded_Miller(Di=.6, bend_diameters=2, angle=90,  Re=2e6, roughness=2E-5, L_unimpeded=30*.6))
    

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_fittings():
    methods = ['Rennels', 'Miller', 'Crane', 'Blevins']
    assert_close1d([fluids.numba.bend_miter(Di=.6, angle=45, Re=1e6, roughness=1e-5, L_unimpeded=20, method=m) for m in methods],
                   [fluids.fittings.bend_miter(Di=.6, angle=45, Re=1e6, roughness=1e-5, L_unimpeded=20, method=m) for m in methods])

    assert_close(fluids.numba.contraction_round_Miller(Di1=1, Di2=0.4, rc=0.04),
                 fluids.contraction_round_Miller(Di1=1, Di2=0.4, rc=0.04))

    assert_close(fluids.numba.contraction_round(Di1=1, Di2=0.4, rc=0.04),
                 fluids.contraction_round(Di1=1, Di2=0.4, rc=0.04))

    assert_close(fluids.numba.contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120),
                 fluids.contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120),)
    
    assert_close(fluids.numba.diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07),
                 fluids.diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07),)

    assert_close(fluids.numba.K_gate_valve_Crane(D1=.1, D2=.146, angle=13.115),
                 fluids.K_gate_valve_Crane(D1=.1, D2=.146, angle=13.115))

    assert_close(fluids.numba.v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight'),
                 fluids.v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight'))

    assert_close(fluids.numba.K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633),
                 fluids.K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633),)

    assert_close(fluids.numba.bend_rounded(Di=4.020, rc=4.0*5, angle=30, Re=1E5),
                 fluids.bend_rounded(Di=4.020, rc=4.0*5, angle=30, Re=1E5))
    
    assert_close(fluids.numba.contraction_conical_Crane(Di1=0.0779, Di2=0.0525, l=0),
             fluids.contraction_conical_Crane(Di1=0.0779, Di2=0.0525, l=0))

    assert_close(fluids.numba.contraction_conical(Di1=0.1, Di2=0.04, l=0.04, Re=1E6),
                 fluids.contraction_conical(Di1=0.1, Di2=0.04, l=0.04, Re=1E6))
                 
    assert_close(fluids.numba.diffuser_conical(Di1=1/3., Di2=1.0, angle=50.0, Re=1E6),
                 fluids.diffuser_conical(Di1=1/3., Di2=1.0, angle=50.0, Re=1E6))
    assert_close(fluids.numba.diffuser_conical(Di1=1., Di2=10.,l=9, fd=0.01),
                 fluids.diffuser_conical(Di1=1., Di2=10.,l=9, fd=0.01))

    assert_close(fluids.numba.diffuser_conical_staged(Di1=1., Di2=10., DEs=np.array([2,3,4]), ls=np.array([1.1,1.2,1.3, 1.4]), fd=0.01),
                 fluids.diffuser_conical_staged(Di1=1., Di2=10., DEs=np.array([2,3,4]), ls=np.array([1.1,1.2,1.3, 1.4]), fd=0.01))

    assert_close(fluids.numba.K_globe_stop_check_valve_Crane(.1, .02, style=1),
                 fluids.K_globe_stop_check_valve_Crane(.1, .02, style=1))
    
    assert_close(fluids.numba.K_angle_stop_check_valve_Crane(.1, .02, style=1),
                fluids.K_angle_stop_check_valve_Crane(.1, .02, style=1))
    
    assert_close(fluids.numba.K_diaphragm_valve_Crane(D=.1, style=0),
                fluids.K_diaphragm_valve_Crane(D=.1, style=0))
    
    assert_close(fluids.numba.K_foot_valve_Crane(D=0.2, style=0),
                 fluids.K_foot_valve_Crane(D=0.2, style=0))
    
    assert_close(fluids.numba.K_butterfly_valve_Crane(D=.1, style=2),
                 fluids.K_butterfly_valve_Crane(D=.1, style=2))
    
    assert_close(fluids.numba.K_plug_valve_Crane(D1=.01, D2=.02, angle=50),
                 fluids.K_plug_valve_Crane(D1=.01, D2=.02, angle=50))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_filters_numba():
    assert_close(fluids.numba.round_edge_screen(0.5, 100, 45), 
                 fluids.round_edge_screen(0.5, 100, 45))
    assert_close(fluids.numba.round_edge_screen(0.5, 100), 
                 fluids.round_edge_screen(0.5, 100))

    assert_close(fluids.numba.round_edge_open_mesh(0.96, angle=33.),
                 fluids.round_edge_open_mesh(0.96, angle=33.))
    
    assert_close(fluids.numba.square_edge_grill(.45, l=.15, Dh=.002, fd=.0185),
                 fluids.square_edge_grill(.45, l=.15, Dh=.002, fd=.0185))
    
    assert_close(fluids.numba.round_edge_grill(.4, l=.15, Dh=.002, fd=.0185),
                 fluids.round_edge_grill(.4, l=.15, Dh=.002, fd=.0185))
    
    assert_close(fluids.numba.square_edge_screen(0.99), 
                 fluids.square_edge_screen(0.99))
    
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_pump_numba():
    assert_close(fluids.numba.motor_efficiency_underloaded(10.1*hp,  .1),
                 fluids.motor_efficiency_underloaded(10.1*hp,  .1),)
    
    assert_close(fluids.numba.current_ideal(V=120, P=1E4, PF=1, phase=1),
                 fluids.current_ideal(V=120, P=1E4, PF=1, phase=1))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_separator_numba():
    assert_close(fluids.numba.K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True),
                 fluids.K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_mixing_numba():
    assert_close(fluids.numba.size_tee(Q1=11.7, Q2=2.74, D=0.762, D2=None, n=1, pipe_diameters=5),
                 fluids.size_tee(Q1=11.7, Q2=2.74, D=0.762, D2=None, n=1, pipe_diameters=5))
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_compressible():
    assert_close(fluids.numba.isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78),
                 fluids.isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78),)

    assert_close(fluids.numba.isentropic_efficiency(1E5, 1E6, 1.4, eta_p=0.78),
                 fluids.isentropic_efficiency(1E5, 1E6, 1.4, eta_p=0.78))

    assert_close(fluids.numba.polytropic_exponent(1.4, eta_p=0.78),
                 fluids.polytropic_exponent(1.4, eta_p=0.78))
    
    assert_close(fluids.numba.Panhandle_A(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15),
                 fluids.Panhandle_A(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15))
    
    assert_close(fluids.numba.Panhandle_B(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15),
                 fluids.Panhandle_B(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15))
    
    assert_close(fluids.numba.Weymouth(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15),
                 fluids.Weymouth(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15))
    
    assert_close(fluids.numba.Spitzglass_high(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15),
                 fluids.Spitzglass_high(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15))
    
    assert_close(fluids.numba.Spitzglass_low(D=0.154051, P1=6720.3199, P2=0, L=54.864, SG=0.6, Tavg=288.7),
                 fluids.Spitzglass_low(D=0.154051, P1=6720.3199, P2=0, L=54.864, SG=0.6, Tavg=288.7))   
    
    assert_close(fluids.numba.Oliphant(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15),
                 fluids.Oliphant(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_control_valve():
    # Not working - size_control_valve_g, size_control_valve_l
    # Can take the functions out, but the dictionary return remains problematic
    # fluids.numba.control_valve_choke_P_l(69682.89291024722, 22048320.0, 0.6, P2=458887.5306077305) # Willing to change this error message if the other can pass

#    fluids.numba.size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,
#Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05,
#FL=0.85, Fd=0.42, xT=0.60)
    
    assert_close(fluids.numba.Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=False),
                 fluids.Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=False))

    assert_close(fluids.numba.convert_flow_coefficient(10, 'Kv', 'Av'),
                 fluids.convert_flow_coefficient(10, 'Kv', 'Av'))


@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_safety_valve():
    assert_close(fluids.numba.API520_round_size(1E-4),
                 fluids.API520_round_size(1E-4))
    assert_close(fluids.numba.API520_SH(593+273.15, 1066.325E3),
                fluids.API520_SH(593+273.15, 1066.325E3))
    assert_close(fluids.numba.API520_W(1E6, 3E5),
                fluids.API520_W(1E6, 3E5))
    
    assert_close(fluids.numba.API520_B(1E6, 5E5),
                 fluids.API520_B(1E6, 5E5))
    
    assert_close(fluids.numba.API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1),
                fluids.API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1))
    
    assert_close(fluids.numba.API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1),
                 fluids.API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1))


@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_packed_bed():
    assert_close(fluids.numba.Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2),
                 fluids.Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2))
    
    assert_close(fluids.numba.Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003),
                 fluids.Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003))
    
    assert_close(fluids.numba.dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01),
                 fluids.dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01))
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_packed_tower():
    # 12.8 us CPython, 1.4 PyPy, 1.85 numba
    assert_close(fluids.numba.Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.),
                 fluids.Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.),)

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_flow_meter():
    assert_close(fluids.numba.differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type='cone meter'),
                 fluids.differential_pressure_meter_beta(D=0.2575, D2=0.184, meter_type='cone meter'))
    
    assert_close(fluids.numba.C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange', subtype='orifice'),
                 fluids.C_Miller_1996(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange', subtype='orifice'))
    
    assert_close1d(fluids.numba.differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168, meter_type='ISO 5167 orifice', taps='D'), 
                   fluids.differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168, meter_type='ISO 5167 orifice', taps='D'))

    assert_close(fluids.numba.differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, meter_type='as cast convergent venturi tube'),
                 fluids.differential_pressure_meter_dP(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, meter_type='as cast convergent venturi tube'))
    
    assert_close(fluids.numba.differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D'),
                 fluids.differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D'))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_core():
    # All these had issues
    assert_close(fluids.numba.Reynolds(2.5, 0.25, nu=1.636e-05),
                 fluids.Reynolds(2.5, 0.25, nu=1.636e-05))
    
    assert_close(fluids.numba.Peclet_heat(1.5, 2, 1000., 4000., 0.6),
                 fluids.Peclet_heat(1.5, 2, 1000., 4000., 0.6))
    
    assert_close(fluids.numba.Fourier_heat(t=1.5, L=2, rho=1000., Cp=4000., k=0.6),
                 fluids.Fourier_heat(t=1.5, L=2, rho=1000., Cp=4000., k=0.6))
    
    assert_close(fluids.numba.Graetz_heat(1.5, 0.25, 5, 800., 2200., 0.6),
                 fluids.Graetz_heat(1.5, 0.25, 5, 800., 2200., 0.6))
    
    assert_close(fluids.numba.Schmidt(D=2E-6, mu=4.61E-6, rho=800),
                 fluids.Schmidt(D=2E-6, mu=4.61E-6, rho=800))
    
    assert_close(fluids.numba.Lewis(D=22.6E-6, alpha=19.1E-6),
                 fluids.Lewis(D=22.6E-6, alpha=19.1E-6))
    
    assert_close(fluids.numba.Confinement(0.001, 1077, 76.5, 4.27E-3),
                 fluids.Confinement(0.001, 1077, 76.5, 4.27E-3))
    
    assert_close(fluids.numba.Prandtl(Cp=1637., k=0.010, nu=6.4E-7, rho=7.1),
                 fluids.Prandtl(Cp=1637., k=0.010, nu=6.4E-7, rho=7.1))
    
    assert_close(fluids.numba.Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613, mu=1.9E-5),
                 fluids.Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613, mu=1.9E-5))
    
    assert_close(fluids.numba.Froude(1.83, L=2., squared=True),
                 fluids.Froude(1.83, L=2., squared=True))
    
    assert_close(fluids.numba.nu_mu_converter(998., nu=1.0E-6),
                 fluids.nu_mu_converter(998., nu=1.0E-6))
    
    assert_close(fluids.numba.gravity(55, 1E4),
                 fluids.gravity(55, 1E4))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_drag():
    assert_close(fluids.numba.drag_sphere(200),
                 fluids.drag_sphere(200))
    
    assert_close(fluids.numba.drag_sphere(1e6, Method='Almedeij'),
                 fluids.drag_sphere(1e6, Method='Almedeij'))


    assert_close(fluids.numba.v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3),
                 fluids.v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3))
    
    assert_close(fluids.numba.time_v_terminal_Stokes(D=1e-7, rhop=2200., rho=1.2, mu=1.78E-5, V0=1),
                 fluids.time_v_terminal_Stokes(D=1e-7, rhop=2200., rho=1.2, mu=1.78E-5, V0=1), rtol=1e-1 )

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_two_phase_voidage():
    assert_close(fluids.numba.gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2, Method='Duckler'),
                 fluids.gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2, Method='Duckler'))

    assert_close(fluids.numba.liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05),
                 fluids.liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05))


@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_misc_two_phase():
    assert_close(fluids.numba.Beggs_Brill(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, P=1E7, D=0.05, angle=0, roughness=0, L=1),
                 fluids.Beggs_Brill(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, P=1E7, D=0.05, angle=0, roughness=0, L=1))

    assert_close(fluids.numba.Kim_Mudawar(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1),
                 fluids.Kim_Mudawar(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1))

    reg_numba = fluids.numba.Mandhane_Gregory_Aziz_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67,  mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)
    reg_normal = fluids.Mandhane_Gregory_Aziz_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67,  mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)
    assert reg_numba == reg_normal
    
    reg_numba = fluids.numba.Taitel_Dukler_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0, angle=0)
    reg_normal = fluids.Taitel_Dukler_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0, angle=0)
    assert reg_numba == reg_normal

    assert_close(fluids.numba.two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1),
                 fluids.two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1))
    
    assert_close(fluids.numba.two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1, P=1e6),
                fluids.two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1, P=1e6))


'''Completely working submodles:
* core
* filters
* separator
* saltation
* mixing
* safety_valve
* open_flow
* pump (except CountryPower)
* flow_meter
* packed_bed
* two_phase_voidage
* two_phase

Near misses:
* fittings - Hooper2K, Darby3K
* drag - integrate_drag_sphere (odeint)

* compressible - P_isothermal_critical_flow, isothermal_gas (need lambertw, change solvers)
* packed_tower - Stichlmair_flood (newton_system)
* geometry - double quads

Not supported:
* particle_size_distribution
* atmosphere
* friction - Only nearest_material_roughness, material_roughness, roughness_Farshad
* piping - all dictionary lookups
'''

'''
Functions not working:
    
# Almost workk, needs support for new branches of lambertw
fluids.numba.P_isothermal_critical_flow(P=1E6, fd=0.00185, L=1000., D=0.5)
fluids.numba.lambertw(.5)

# newton_system not working
fluids.numba.Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)

# Using dictionaries outside is broken
# Also, nopython is broken for this case - https://github.com/numba/numba/issues/5377
fluids.numba.roughness_Farshad('Cr13, bare', 0.05)

piping.nearest_pipe -> Multiplication of None type; checking of type to handle in inputs;
 dictionary lookup of schedule coefficients; function in function; doesn't like something about the data either

piping.gauge_from_t -> numba type dict; once that's inside function, dying on checking
 "in" of a now-numpy array; same for t_from_gauge

fluids.numba.liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, Method='Xu Fang voidage')
* some raaguments can be done

fluids.numba.two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1)

Most classes which have different input types
Double quads not yet supported - almost!
'''





'''Global dictionary lookup:
Darby3K, Hooper2K, 


# Feels like this should work
from numba import njit, typeof, typed, types
Darby =  typed.Dict.empty(types.string, types.UniTuple(types.float64, 3))
Darby['Elbow, 90°, threaded, standard, (r/D = 1)'] = (800.0, 0.14, 4.0)
Darby['Elbow, 90°, threaded, long radius, (r/D = 1.5)'] = (800.0, 0.071, 4.2)

@numba.njit
def Darby3K(NPS, Re, name):
    K1, Ki, Kd = Darby[name]
    return K1/Re + Ki*(1. + Kd*NPS**-0.3)

Darby3K(NPS=12., Re=10000., name='Elbow, 90°, threaded, standard, (r/D = 1)')'''


'''
numba is not up to speeding up the various solvers!

I was able to contruct a secant version which numba would optimize, mostly.
However, it took 30x the time.

Trying to improve this, it was found reducing the number of arguments to secant
imroves things ~20%. Removing ytol or the exceptions did not improve things at all.

Eventually it was discovered, the rtol and xtol arguments should be fixed values inside the function.
This makes little sense, but it is what happened.
Slighyly better performance was found than in pure-python that way, although definitely not vs. pypy.


'''

'''Having a really hard time getting newton_system to work...
@numba.njit
def to_solve_jac(x0):
    return np.array([5.0*x0[0] - 3]), np.array([5.0])
# fluids.numerics.newton_system(to_solve_jac, x0=[1.0], jac=True)
fluids.numba.newton_system(to_solve_jac, x0=[1.0], jac=True)
'''