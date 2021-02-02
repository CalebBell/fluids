# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import assert_close, assert_close1d, assert_close2d, isclose
import pytest

def test_control_valve():
    from fluids.control_valve import cavitation_index, FF_critical_pressure_ratio_l, is_choked_turbulent_l, is_choked_turbulent_g, Reynolds_valve, loss_coefficient_piping, Reynolds_factor
    CI = cavitation_index(1E6, 8E5, 2E5)
    assert_close(CI, 4.0)

    FF = FF_critical_pressure_ratio_l(70100.0, 22120000.0)
    assert_close(FF, 0.9442375225233299)

    F = is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.9)
    assert not F
    T = is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.6)
    assert T

    with pytest.raises(Exception):
        is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299)

    # Example 4, compressible flow - small flow trim sized for gas flow:
    assert False == is_choked_turbulent_g(0.536, 1.193, 0.8)
    # Custom example
    assert True == is_choked_turbulent_g(0.9, 1.193, 0.7)

    with pytest.raises(Exception):
        is_choked_turbulent_g(0.544, 0.929)

    Rev = Reynolds_valve(3.26e-07, 360, 100.0, 0.6, 0.98, 238.05817216710483)
    assert_close(Rev, 6596953.826574914)

    Rev = Reynolds_valve(3.26e-07, 360, 150.0, 0.9, 0.46, 164.9954763704956)
    assert_close(Rev, 2967024.346783506)

    K = loss_coefficient_piping(0.05, 0.08, 0.1)
    assert_close(K, 0.6580810546875)

    ### Reynolds factor (laminar)
    # In Example 4, compressible flow with small flow trim sized for gas flow
    # (Cv in the problem was converted to Kv here to make FR match with N32, N2):
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=False)
    assert_close(f, 0.7148753122302025)

    # Custom, same as above but with full trim:
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=True)
    assert_close(f, 0.9875328782172637)

    # Example 4 with Rev < 10:
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=8., full_trim=False)
    assert_close(f, 0.08339546213461975)

    # Same, with full_trim
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=8., full_trim=True)
    assert_close(f, 43.619397389803986)

def test_control_valve_size_l():
    ### Control valve liquid
    # From [1]_, matching example 1 for a globe, parabolic plug,
    # flow-to-open valve.

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.15, D2=0.15, d=0.15, FL=0.9, Fd=0.46)
    assert_close(Kv, 164.9954763704956)

    # Same as above - diameters removed
    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1)
    assert_close(Kv, 164.9954763704956)

    # From [1]_, matching example 2 for a ball, segmented ball,
    # flow-to-open valve.

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.1, FL=0.6, Fd=0.98)
    assert_close(Kv, 238.05817216710483)

    # Modified example 1 with non-choked flow, with reducer and expander

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4,  P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.09, d=0.08, FL=0.9, Fd=0.46)
    assert_close(Kv, 177.44417090966715)

    # Modified example 2 with non-choked flow, with reducer and expander

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.095, FL=0.6, Fd=0.98)
    assert_close(Kv, 241.6812562245056)

    # Same, test intermediate values
    ans = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.095, FL=0.6, Fd=0.98, full_output=True)
    del ans['choked']
    del ans['FR']
    ans_expect = {'FF': 0.9442375225233299,
                  'FLP': 0.5912597868996382,
                  'FP': 0.9969139124178094,
                  'Kv': 241.6812562245056,
                  'Rev': 6596962.21111206,
                  'laminar': False}

    for k in ans_expect.keys():
        assert_close(ans[k], ans_expect[k])

    # Modified example 2 with laminar flow at 100x viscosity, 100th flow rate, and 1/10th diameters:

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-2, P1=680E3, P2=220E3, Q=0.001, D1=0.01, D2=0.01, d=0.01, FL=0.6, Fd=0.98)
    assert_close(Kv, 3.0947562381723626)

    # Last test, laminar full trim
    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-2, P1=680E3, P2=220E3, Q=0.001, D1=0.01, D2=0.01, d=0.02, FL=0.6, Fd=0.98)
    assert_close(Kv, 3.0947562381723626)

    # TODO: find a test where the following is tested, or remove it as unnecessary.
    # if C/FR >= Ci:
    #    Ci = iterate_piping_laminar(Ci)
    # Efforts to make this happen have been unsuccessful.



    # Test the ignore choked option
    ans = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.1, FL=0.6, Fd=0.98, allow_choked=False, full_output=True)
    assert_close(ans['Kv'], 164.9954763704956)
    assert_close(ans['Rev'], 7805019.992655547)
    assert ans['choked'] == True # Still true even though the choke is ignored
    assert ans['FF']
    assert ans['FLP'] is None
    assert ans['FP'] is None
    assert ans['FR'] is None

    # Test the laminar switch
    for Kv, boolean in zip((0.014547698964079439, 0.011190537664676491), (True, False)):
        ans = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=670E3, Q=0.000001, D1=0.1, D2=0.1, d=0.1, FL=0.6, Fd=0.98, allow_laminar=boolean, full_output=True)
        assert_close(ans['Kv'], Kv)

    # Test for too many iterations, does not converge
    kwargs = {'allow_laminar': True, 'allow_choked': True, 'Fd': 1.0, 'FL': 1.0, 'D1': 0.1, 'D2': 0.1,
              'd': 0.09, 'full_output': True, 'P1': 1000000.0, 'mu': 0.0008512512422708317,
              'rho': 995.4212225776154, 'Pc': 22048320.0, 'Q': 0.004018399356246507,
              'Psat': 3537.075987237396, 'P2': 999990.0}
    res = size_control_valve_l(**kwargs)
    assert 'warning' in res

    # Test 'choked' in results
    kwargs = {'allow_laminar': True, 'allow_choked': True, 'Fd': 0.42, 'FL': 0.85,
              'D1': 0.08, 'D2': 0.1, 'd': 0.05, 'full_output': True, 'P1': 680000.0,
              'mu': 2.099826023627934e-05, 'T': 433.0, 'MW': 44.0095,
              'gamma': 1.2567580165935908, 'Z': 0.9896087377962121, 'xT': 0.6,
              'Q': 1.0632314140418966, 'P2': 313744.8065927219}
    res = size_control_valve_g(**kwargs)
    assert res['choked']

def test_control_valve_size_g():
    # From [1]_, matching example 3 for non-choked gas flow with attached
    # fittings  and a rotary, eccentric plug, flow-to-open control valve:

    Kv = size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,  Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)
    assert_close(Kv, 72.58664545391052)

    # From [1]_, roughly matching example 4 for a small flow trim sized tapered
    # needle plug valve. Difference is 3% and explained by the difference in
    # algorithms used.

    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., D1=0.015, D2=0.015, d=0.015, FL=0.98, Fd=0.07, xT=0.8)
    assert_close(Kv, 0.016498765335995726)

    # Diameters removed
    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., xT=0.8)
    assert_close(Kv, 0.012691357950765944)
    ans = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., xT=0.8, full_output=True)
    assert ans['laminar'] == False
    assert ans['choked'] == False
    assert ans['FP'] is None
    assert ans['FR'] is None
    assert ans['xTP'] is None
    assert ans['Rev'] is None

    # Choked custom example
    Kv = size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30, Z=0.988, P1=680E3, P2=30E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)
    assert_close(Kv, 70.67468803987839)


    # Laminar custom example
    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., D1=0.015, D2=0.015, d=0.001, FL=0.98, Fd=0.07, xT=0.8)
    assert_close(Kv, 0.016498765335995726)

    # Laminar custom example with iteration
    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=2.7E5, Q=0.1/3600., D1=0.015, D2=0.015, d=0.001, FL=0.98, Fd=0.07, xT=0.8)
    assert_close(Kv, 0.989125783445497)

    # test not allowing chokes
    ans_choked = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1e4, Q=0.46/3600., D1=0.015, D2=0.015, d=0.015, FL=0.98, Fd=0.07, xT=0.8, full_output=True, allow_choked=True)
    ans = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1e4, Q=0.46/3600., D1=0.015, D2=0.015, d=0.015, FL=0.98, Fd=0.07, xT=0.8, full_output=True, allow_choked=False)
    assert not isclose(ans_choked['Kv'], ans['Kv'], rel_tol=1E-4)

    # Test not allowing laminar
    for Kv, boolean in zip((0.001179609179354541, 0.00090739167642657), (True, False)):
        ans = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1e4, Q=1e-5, D1=0.015, D2=0.015, d=0.015, FL=0.98, Fd=0.07, xT=0.8, full_output=True, allow_laminar=boolean)
        assert_close(Kv, ans['Kv'])

    assert ans['choked'] # Still true even though the choke is ignored
    assert ans['xTP'] is None
    assert ans['Y']
    assert ans['FP'] is None
    assert ans['FR'] is None
    assert ans['Rev']

    # Test a warning is issued and a solution is still returned when in an unending loop
    # Ends with C ratio converged to 0.907207790871228


    args = {'P1': 680000.0, 'full_output': True, 'allow_choked': True,
            'Q': 0.24873053149856303, 'T': 433.0, 'Z': 0.9908749375670418,
            'FL': 0.85, 'allow_laminar': True, 'd': 0.05, 'mu': 2.119519588834806e-05,
            'MW': 44.0095, 'Fd': 0.42, 'gamma': 1.2431389717945152, 'D2': 0.1,
            'xT': 0.6, 'D1': 0.08}
    ans = size_control_valve_g(P2=678000., **args)
    assert ans['warning']

    # Test Kv does not reach infinity
    kwargs = {'P2': 310000.0028935982, 'P1': 680000.0, 'full_output': True, 'allow_choked': True, 'T': 433.0, 'Z': 0.9896087377962123, 'FL': 0.85, 'allow_laminar': True, 'd': 0.05, 'mu': 2.119519588834806e-05, 'MW': 44.0095, 'Fd': 0.42, 'gamma': 1.2431389717945152, 'D2': 0.1, 'xT': 0.6, 'D1': 0.08}
    size_control_valve_g(Q=1000000000.0, **kwargs)


def test_control_valve_choke_P_l():
    P2 = control_valve_choke_P_l(69682.89291024722, 22048320.0, 0.6, 680000.0)
    assert_close(P2, 458887.5306077305)
    P1 = control_valve_choke_P_l(69682.89291024722, 22048320.0, 0.6, P2=P2)
    assert_close(P1, 680000.0)

def test_control_valve_choke_P_g():
    P2 = control_valve_choke_P_g(1.0, 1.3, 1E5)
    assert_close(P2, 7142.857142857143)
    P1 = control_valve_choke_P_g(1.0, 1.3, P2=P2)
    assert_close(P1, 100000.0)


def test_control_valve_noise_l_2015():
    m = 30 # kg/s
    P1 = 1E6
    P2 = 8E5

    Psat = 2.32E3
    rho = 997.0
    c = 1400.0
    Kv = Cv_to_Kv(90)
    d = .1
    Di =.1071
    FL = 0.92
    Fd = 0.42
    rho_air = 1.293
    c_air = 343.0
    t_pipe = .0036

    # Example 1
    noise = control_valve_noise_l_2015(m, P1, P2, Psat, rho, c, Kv, d, Di, FL, Fd,
                                   t_pipe, rho_pipe=7800.0, c_pipe=5000.0,
                                   rho_air=rho_air, c_air=343.0, xFz=None, An=-4.6)
    assert_close(noise, 65.47210071692108)

    # Example 2
    m = 40 # kg/s
    P1 = 1E6
    P2 = 6.5E5
    noise = control_valve_noise_l_2015(m, P1, P2, Psat, rho, c, Kv, d, Di, FL, Fd,
                                   t_pipe, rho_pipe=7800.0, c_pipe=5000.0,
                                   rho_air=rho_air, c_air=343.0, xFz=None, An=-4.6)
    assert_close(noise, 81.58199982219298)

    # Example 3
    m = 40 # kg/s
    P1 = 1E6
    P2 = 6.5E5
    noise = control_valve_noise_l_2015(m, P1, P2, Psat, rho, c, Kv, d, Di, FL, Fd,
                                   t_pipe, rho_pipe=7800.0, c_pipe=5000.0,
                                   rho_air=rho_air, c_air=343.0, xFz=0.254340899267+0.1, An=-4.6)
    assert_close(noise, 69.93930269695811)

def test_control_valve_noise_g_2011():

    ans = control_valve_noise_g_2011(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(90.0),
                                   d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98,
                                   Fd=0.2959450058448346,
                                   t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                                   rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(ans, 91.67631681476502)

    ans = control_valve_noise_g_2011(m=2.29, P1=1E6, P2=6.9E5, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(90.0),
                               d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98,
                               Fd=0.2959450058448346,
                               t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                               rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(ans, 92.80027236454005)

    ans = control_valve_noise_g_2011(m=2.59, P1=1E6, P2=4.8E5, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(90.0),
                               d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98,
                               Fd=0.2959450058448346,
                               t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                               rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(97.65988432967984, ans)

    ans = control_valve_noise_g_2011(m=1.18, P1=1E6, P2=4.2E5, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(40.0),
                               d=0.2031, Di=0.2031, FL=None, FLP=0.792, FP=0.98,
                               Fd=0.2959450058448346,
                               t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                               rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(94.16189978031449, ans)# should be 94

    ans = control_valve_noise_g_2011(m=1.19, P1=1E6, P2=5E4, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(40.0),
                               d=0.2031, Di=0.2031, FL=None, FLP=0.792, FP=0.98,
                               Fd=0.2959450058448346,
                               t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                               rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(ans, 97.48317214321824)

    ans = control_valve_noise_g_2011(m=0.89, P1=1E6, P2=5E4, T1=450, rho=5.3,
                                     gamma=1.22, MW=19.8, Kv=Cv_to_Kv(30.0),
                               d=0.1, Di=0.15, FL=None, FLP=0.792, FP=0.98,
                               Fd=0.2959450058448346,
                               t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0,
                               rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
    assert_close(ans, 93.38835049261132)

@pytest.mark.scipy
def test_opening_quick_data():
    # Add some tolerance to tests after failures on arm64 https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=976558
    from scipy.interpolate import splrep
    from fluids.control_valve import opening_quick_tck, opening_quick, frac_CV_quick
    tck_recalc = splrep(opening_quick, frac_CV_quick, k=3, s=0)
    [assert_close1d(i, j, atol=1e-10) for i, j in zip(opening_quick_tck[:-1], tck_recalc[:-1])]

@pytest.mark.scipy
def test_opening_equal_data():
    # Add some tolerance to tests after failures on arm64 https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=976558
    from scipy.interpolate import splrep
    from fluids.control_valve import opening_equal, frac_CV_equal, opening_equal_tck
    tck_recalc = splrep(opening_equal, frac_CV_equal, k=3, s=0)
    [assert_close1d(i, j, atol=1e-10) for i, j in zip(opening_equal_tck[:-1], tck_recalc[:-1])]
