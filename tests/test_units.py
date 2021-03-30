# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import types
from fluids.numerics import assert_close, assert_close1d, assert_close2d
import pytest
import fluids
from fluids.units import *
from fluids.units import kwargs_to_args

def test_kwargs_to_args():
    sig = ['rho', 'mu', 'nu']
    args = (1,)
    kwargs = {'mu': 2.2}
    assert [1, 2.2, None] == kwargs_to_args(args, kwargs, sig)

    kwargs = {'nu': 2.2}
    assert [1, None, 2.2] == kwargs_to_args(args, kwargs, sig)

    assert [12.2, 2.2, 5.5] == kwargs_to_args(tuple(), {'mu': 2.2, 'nu': 5.5, 'rho': 12.2}, sig)
    assert [None, None, None] == kwargs_to_args(tuple(), {}, sig)

    assert [12.2, 2.2, 5.5] == kwargs_to_args((12.2, 2.2, 5.5), {}, sig)



def assert_pint_allclose(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def assert_pint_allclose1d(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close1d(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def assert_pint_allclose2d(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close2d(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def test_in_right_units():
    assert u.default_system == 'mks'

def test_nondimensional_reduction():
    Re = 171.8865229090909 *u.meter * u.pound / u.centipoise / u.foot ** 2 / u.second
    eD = 0.0005937067088858105*u.inch/u.meter
    assert_close(friction_factor(Re, eD).magnitude, 0.012301598061848239)


def test_convert_input():
    from fluids.units import convert_input

    ans = convert_input(5, 'm', u, False)
    assert ans == 5
    with pytest.raises(Exception):
        convert_input(5, 'm', u, True)


def test_sample_cases():
    Re = Reynolds(V=3.5*u.m/u.s, D=2*u.m, rho=997.1*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
    assert_close(Re.to_base_units().magnitude, 6979700.0)
    assert dict(Re.dimensionality) == {}


#    vs = hwm93(5E5*u.m, 45*u.degrees, 50*u.degrees, 365*u.day)
#    vs_known = [-73.00312042236328, 0.1485661268234253]
#    for v_known, v_calc in zip(vs_known, vs):
#        assert_close(v_known, v_calc.to_base_units().magnitude)
#        assert dict(v_calc.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}

    A = API520_A_g(m=24270*u.kg/u.hour, T=348.*u.K, Z=0.90, MW=51.*u.g/u.mol, k=1.11, P1=670*u.kPa, Kb=1, Kc=1)
    assert_close(A.to_base_units().magnitude, 0.00369904606468)
    assert dict(A.dimensionality) == {u'[length]': 2.0}

    T = T_critical_flow(473*u.K, 1.289)
    assert_close(T.to_base_units().magnitude, 413.280908694)
    assert dict(T.dimensionality) == {u'[temperature]': 1.0}

    T2 = T_critical_flow(473*u.K, 1.289*u.dimensionless)

    assert T == T2

    with pytest.raises(Exception):
        T_critical_flow(473, 1.289)

    with pytest.raises(Exception):
        T_critical_flow(473*u.m, 1.289)

    # boolean
    P1 = 8*u.bar + 1*u.atm
    P2 = 1*u.atm
    assert True == is_critical_flow(P1, P2, k=1.4*u.dimensionless)

    A = size_control_valve_g(T=433.*u.K, MW=44.01*u.g/u.mol, mu=1.4665E-4*u.Pa*u.s, gamma=1.30,
    Z=0.988, P1=680*u.kPa, P2=310*u.kPa, Q=38/36.*u.m**3/u.s, D1=0.08*u.m, D2=0.1*u.m, d=0.05*u.m,
    FL=0.85, Fd=0.42, xT=0.60)
    assert_close(A.to_base_units().magnitude, 0.0201629570705307)
    assert dict(A.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}

    A = API520_round_size(A=1E-4*u.m**2)
    assert_close(A.to_base_units().magnitude, 0.00012645136)
    assert dict(A.dimensionality) == {u'[length]': 2.0}

    SS = specific_speed(0.0402*u.m**3/u.s, 100*u.m, 3550*u.rpm)
    assert_close(SS.to_base_units().magnitude, 2.3570565251512066)
    assert dict(SS.dimensionality) == {u'[length]': 0.75, u'[time]': -1.5}

    v = Geldart_Ling(1.*u.kg/u.s, 1.2*u.kg/u.m**3, 0.1*u.m, 2E-5*u.Pa*u.s)
    assert_close(v.to_base_units().magnitude, 7.467495862402707)
    assert dict(v.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}

    s = speed_synchronous(50*u.Hz, poles=12)
    assert_close(s.to_base_units().magnitude, 157.07963267948966)
    assert dict(s.dimensionality) == {u'[time]': -1.0}

    t = t_from_gauge(.2, False, 'AWG')
    assert_close(t.to_base_units().magnitude, 0.5165)
    assert dict(t.dimensionality) == {u'[length]': 1.0}

    dP = Robbins(G=2.03*u.kg/u.m**2/u.s, rhol=1000*u.kg/u.m**3, Fpd=24/u.ft, L=12.2*u.kg/u.m**2/u.s, rhog=1.1853*u.kg/u.m**3, mul=0.001*u.Pa*u.s, H=2*u.m)
    assert_close(dP.to_base_units().magnitude, 619.662459344 )
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}

    dP = dP_packed_bed(dp=8E-4*u.m, voidage=0.4, vs=1E-3*u.m/u.s, rho=1E3*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
    assert_close(dP.to_base_units().magnitude, 1438.28269588 )
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}

    dP = dP_packed_bed(dp=8E-4*u.m, voidage=0.4*u.dimensionless, vs=1E-3*u.m/u.s, rho=1E3*u.kg/u.m**3, mu=1E-3*u.Pa*u.s, Dt=0.01*u.m)
    assert_close(dP.to_base_units().magnitude, 1255.16256625)
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}

    n = C_Chezy_to_n_Manning(26.15*u.m**0.5/u.s, Rh=5*u.m)
    assert_close(n.to_base_units().magnitude, 0.05000613713238358)
    assert dict(n.dimensionality) == {u'[length]': -0.3333333333333333, u'[time]': 1.0}

    Q = Q_weir_rectangular_SIA(0.2*u.m, 0.5*u.m, 1*u.m, 2*u.m)
    assert_close(Q.to_base_units().magnitude, 1.0408858453811165)
    assert dict(Q.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}

    t = agitator_time_homogeneous(D=36*.0254*u.m, N=56/60.*u.revolutions/u.second, P=957.*u.W, T=1.83*u.m, H=1.83*u.m, mu=0.018*u.Pa*u.s, rho=1020*u.kg/u.m**3, homogeneity=.995)
    assert_close(t.to_base_units().magnitude, 15.143198226374668)
    assert dict(t.dimensionality) == {u'[time]': 1.0}

    K = K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True)
    assert_close(K.to_base_units().magnitude, 0.07951613600476297, rtol=1e-2)
    assert dict(K.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}

    A = current_ideal(V=120*u.V, P=1E4*u.W, PF=1, phase=1)
    assert_close(A.to_base_units().magnitude, 83.33333333333333)
    assert dict(A.dimensionality) == {u'[current]': 1.0}

    fd = friction_factor(Re=1E5, eD=1E-4)
    assert_close(fd.to_base_units().magnitude, 0.01851386607747165)
    assert dict(fd.dimensionality) == {}

    K = Cv_to_K(2.712*u.gallon/u.minute, .015*u.m)
    assert_close(K.to_base_units().magnitude, 14.719595348352552)
    assert dict(K.dimensionality) == {}

    Cv = K_to_Cv(16, .015*u.m)
    assert_close(Cv.to_base_units().magnitude, 0.0001641116865931214)
    assert dict(Cv.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}

    Cd = drag_sphere(200)
    assert_close(Cd.to_base_units().magnitude, 0.7682237950389874)
    assert dict(Cd.dimensionality) == {}

    V, D = integrate_drag_sphere(D=0.001*u.m, rhop=2200.*u.kg/u.m**3, rho=1.2*u.kg/u.m**3, mu=1.78E-5*u.Pa*u.s, t=0.5*u.s, V=30*u.m/u.s, distance=True)
    assert_close(V.to_base_units().magnitude, 9.686465044063436)
    assert dict(V.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}
    assert_close(D.to_base_units().magnitude, 7.829454643649386)
    assert dict(D.dimensionality) == {u'[length]': 1.0}

    Bo = Bond(1000*u.kg/u.m**3, 1.2*u.kg/u.m**3, .0589*u.N/u.m, 2*u.m)
    assert_close(Bo.to_base_units().magnitude, 665187.2339558573)
    assert dict(Bo.dimensionality) == {}

    head = head_from_P(P=98066.5*u.Pa, rho=1000*u.kg/u.m**3)
    assert_close(head.to_base_units().magnitude, 10.000000000000002)
    assert dict(head.dimensionality) == {u'[length]': 1.0}

    roughness = roughness_Farshad('Cr13, bare', 0.05*u.m)
    assert_close(roughness.to_base_units().magnitude, 5.3141677781137006e-05)
    assert dict(roughness.dimensionality) == {u'[length]': 1.0}



def test_custom_wraps():
    A = A_multiple_hole_cylinder(0.01*u.m, 0.1*u.m, [(0.005*u.m, 1)])
    assert_close(A.to_base_units().magnitude, 0.004830198704894308)
    assert dict(A.dimensionality) == {u'[length]': 2.0}

    V = V_multiple_hole_cylinder(0.01*u.m, 0.1*u.m, [(0.005*u.m, 1)])
    assert_close(V.to_base_units().magnitude, 5.890486225480862e-06)
    assert dict(V.dimensionality) == {u'[length]': 3.0}

    # custom compressible flow model wrappers
    functions = [Panhandle_A, Panhandle_B, Weymouth, Spitzglass_high, Oliphant, Fritzsche]
    values = [42.56082051195928, 42.35366178004172, 32.07729055913029, 29.42670246281681, 28.851535408143057, 39.421535157535565]
    for f, v in zip(functions, values):
        ans = f(D=0.340*u.m, P1=90E5*u.Pa, P2=20E5*u.Pa, L=160E3*u.m, SG=0.693, Tavg=277.15*u.K)
        assert_pint_allclose(ans, v, {u'[length]': 3.0, u'[time]': -1.0})

    ans = IGT(D=0.340*u.m, P1=90E5*u.Pa, P2=20E5*u.Pa, L=160E3*u.m, SG=0.693, mu=1E-5*u.Pa*u.s, Tavg=277.15*u.K)
    assert_pint_allclose(ans, 48.92351786788815, {u'[length]': 3.0, u'[time]': -1.0})

    ans = Muller(D=0.340*u.m, P1=90E5*u.Pa, P2=20E5*u.Pa, L=160E3*u.m, SG=0.693, mu=1E-5*u.Pa*u.s, Tavg=277.15*u.K)
    assert_pint_allclose(ans, 60.45796698148659, {u'[length]': 3.0, u'[time]': -1.0})


    nu = nu_mu_converter(rho=1000*u.kg/u.m**3, mu=1E-4*u.Pa*u.s)
    assert_pint_allclose(nu, 1E-7, {u'[length]': 2.0, u'[time]': -1.0})

    mu = nu_mu_converter(rho=1000*u.kg/u.m**3, nu=1E-7*u.m**2/u.s)
    assert_pint_allclose(mu, 1E-4, {u'[time]': -1.0, u'[length]': -1.0, u'[mass]': 1.0})

    SA = SA_tank(D=1.*u.m, L=0*u.m, sideA='ellipsoidal', sideA_a=2*u.m, sideB='ellipsoidal', sideB_a=2*u.m)[0]
    assert_pint_allclose(SA, 10.124375616183064, {u'[length]': 2.0})

    SA, sideA_SA, sideB_SA, lateral_SA = SA_tank(D=1.*u.m, L=0*u.m, sideA='ellipsoidal', sideA_a=2*u.m, sideB='ellipsoidal', sideB_a=2*u.m)
    expect = [10.124375616183064, 5.062187808091532, 5.062187808091532, 0]
    for value, expected in zip([SA, sideA_SA, sideB_SA, lateral_SA], expect):
        assert_pint_allclose(value, expected, {u'[length]': 2.0})


    m = isothermal_gas(rho=11.3*u.kg/u.m**3, fd=0.00185*u.dimensionless, P1=1E6*u.Pa, P2=9E5*u.Pa, L=1000*u.m, D=0.5*u.m)
    assert_pint_allclose(m, 145.484757, {u'[mass]': 1.0, u'[time]': -1.0})


def test_db_functions():
    # dB
    ans = control_valve_noise_g_2011(m=2.22*u.kg/u.s, P1=1E6*u.Pa, P2=7.2E5*u.Pa, T1=450*u.K, rho=5.3*u.kg/u.m**3,
                        gamma=1.22, MW=19.8*u.g/u.mol, Kv=77.85*u.m**3/u.hour,  d=0.1*u.m, Di=0.2031*u.m, FL=None, FLP=0.792,
                         FP=0.98, Fd=0.296, t_pipe=0.008*u.m, rho_pipe=8000.0*u.kg/u.m**3, c_pipe=5000.0*u.m/u.s,
                        rho_air=1.293*u.kg/u.m**3, c_air=343.0*u.m/u.s, An=-3.8, Stp=0.2)
#    assert_pint_allclose(ans, 91.67702674629604, {})



def test_check_signatures():
    from fluids.units import check_args_order
    bad_names = set(['__getattr__'])
    for name in dir(fluids):
        if name in bad_names:
            continue
        obj = getattr(fluids, name)
        if isinstance(obj, types.FunctionType):
            if hasattr(obj, 'func_name') and obj.func_name == '<lambda>':
                continue  # 2
            if hasattr(obj, '__name__') and obj.__name__ == '<lambda>':
                continue # 3
            check_args_order(obj)



def test_differential_pressure_meter_solver():
    m = differential_pressure_meter_solver(D=0.07366*u.m, D2=0.05*u.m, P1=200000.0*u.Pa,
        P2=183000.0*u.Pa, rho=999.1*u.kg/u.m**3, mu=0.0011*u.Pa*u.s, k=1.33*u.dimensionless,
        meter_type='ISO 5167 orifice', taps='D')

    assert_pint_allclose(m, 7.702338035732167, {'[mass]': 1, '[time]': -1})

    P1 = differential_pressure_meter_solver(D=0.07366*u.m, D2=0.05*u.m, m=m,
        P2=183000.0*u.Pa, rho=999.1*u.kg/u.m**3, mu=0.0011*u.Pa*u.s, k=1.33*u.dimensionless,
        meter_type='ISO 5167 orifice', taps='D')
    assert_pint_allclose(P1, 200000, {'[length]': -1, '[mass]': 1, '[time]': -2})

    P2 = differential_pressure_meter_solver(D=0.07366*u.m, D2=0.05*u.m, P1=200000.0*u.Pa,
        m=m, rho=999.1*u.kg/u.m**3, mu=0.0011*u.Pa*u.s, k=1.33*u.dimensionless,
        meter_type='ISO 5167 orifice', taps='D')

    assert_pint_allclose(P2, 183000, {'[length]': -1, '[mass]': 1, '[time]': -2})

    D2 = differential_pressure_meter_solver(D=0.07366*u.m, m=m, P1=200000.0*u.Pa,
        P2=183000.0*u.Pa, rho=999.1*u.kg/u.m**3, mu=0.0011*u.Pa*u.s, k=1.33*u.dimensionless,
        meter_type='ISO 5167 orifice', taps='D')
    assert_pint_allclose(D2, .05, {'[length]': 1})



def test_Tank_units_full():

    T1 = TANK(L=3*u.m, D=150*u.cm, horizontal=True, sideA=None, sideB=None)

    # test all methods
    V = T1.V_from_h(0.1*u.m, 'full')
    assert_pint_allclose(V, 0.151783071377, u.m**3)

    h = T1.h_from_V(0.151783071377*u.m**3, method='brenth')
    assert_pint_allclose(h, 0.1, u.m)
    h = T1.h_from_V(0.151783071377*u.m**3, 'brenth')
    assert_pint_allclose(h, 0.1, u.m)

    # Check the table and approximations
    T1.set_table(dx=1*u.cm)
    assert 151 == len(T1.volumes)
    assert_pint_allclose1d(T1.heights[0:3], [0, 0.01, 0.02], u.m)
    T1.set_table(n=10)
    assert 10 == len(T1.volumes)
    T1.set_table(n=10*u.dimensionless)
    assert 10 == len(T1.volumes)

    T1.set_chebyshev_approximators(8, 8)
    T1.set_chebyshev_approximators(8*u.dimensionless, 8)
    T1.set_chebyshev_approximators(8, 8*u.dimensionless)

    assert 16 == len(T1.c_forward)
    assert 16 == len(T1.c_backward)

    # Check the properties

    assert_pint_allclose(T1.h_max, 1.5, u.m)
    assert_pint_allclose(T1.V_total, 5.301437602932776, u.m**3)
    assert_pint_allclose(T1.L_over_D, 2, u.dimensionless)
    assert_pint_allclose(T1.A_sideA, 1.76714586764, u.m**2)
    assert_pint_allclose(T1.A_sideB, 1.76714586764, u.m**2)
    assert_pint_allclose(T1.A_lateral, 14.1371669412, u.m**2)
    assert_pint_allclose(T1.A, 17.6714586764, u.m**2)



def test_HelicalCoil_units():
    C2 = HelicalCoil(Do=30*u.cm, H=20*u.cm, pitch=5*u.cm, Dt=2*u.cm)
    C3 = HelicalCoil(2*u.cm, 30*u.cm, 5*u.cm, 20*u.cm)

    for C1 in [C2, C3]:
        assert_pint_allclose(C1.Dt, 0.02, u.m)
        assert_pint_allclose(C1.Do, 0.3, u.m)
        assert_pint_allclose(C1.Do_total, 0.32, u.m)
        assert_pint_allclose(C1.pitch, 0.05, u.m)
        assert_pint_allclose(C1.H, 0.2, u.m)
        assert_pint_allclose(C1.H_total, 0.22, u.m)
        assert_pint_allclose(C1.N, 4, u.dimensionless)
        assert_pint_allclose(C1.tube_circumference, 0.942477796077, u.m)
        assert_pint_allclose(C1.tube_length, 3.7752126215, u.m)
        assert_pint_allclose(C1.surface_area, 0.237203604749 , u.m**2)
        assert_pint_allclose(C1.curvature, 0.06, u.dimensionless)
        assert_pint_allclose(C1.helix_angle, 0.0530019606897, u.radians)


def test_ATMOSPHERE_1976_units():
    five_km = ATMOSPHERE_1976(5000*u.m)
    assert_pint_allclose(five_km.T, 255.675543222, u.K)
    assert_pint_allclose(five_km.P, 54048.2861458, u.Pa)
    assert_pint_allclose(five_km.rho, 0.73642842078, u.kg/u.m**3)
    assert_pint_allclose(five_km.g, 9.79124107698, u.m/u.s**2)
    assert_pint_allclose(five_km.mu, 1.62824813536e-05, u.Pa*u.s)
    assert_pint_allclose(five_km.k, 0.0227319029514, u.W/u.K/u.m)
    assert_pint_allclose(five_km.v_sonic, 320.54551967, u.m/u.s)
    assert_pint_allclose(five_km.sonic_velocity(300*u.K), 347.220809082, u.m/u.s)

    # Test the staticmethod works alone
    assert_pint_allclose(ATMOSPHERE_1976.sonic_velocity(300*u.K), 347.220809082, u.m/u.s)

    # Check AttribtueError is property raised on __getstate__ for classes
    # as they now have a __getattr_ method
    import copy
    copy.copy(five_km)
    copy.deepcopy(five_km)


def test_ATMOSPHERE_NRLMSISE00():
    a = ATMOSPHERE_NRLMSISE00(Z=1E3*u.m, latitude=45*u.degrees, longitude=45*u.degrees, day=150*u.day)
    assert_pint_allclose(a.T, 285.544086062, u.K)
    assert_pint_allclose(a.rho, 1.10190620264, u.kg/u.m**3)
    assert_pint_allclose(a.O2_density, 4.80470350725e+24, u.count/u.m**3)
    assert_pint_allclose(a.day, 12960000, u.day)