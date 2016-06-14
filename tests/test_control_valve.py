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

from fluids import *
from numpy.testing import assert_allclose
import pytest

def test_control_valve():
    from fluids.control_valve import cavitation_index, FF_critical_pressure_ratio_l, is_choked_turbulent_l, is_choked_turbulent_g, Reynolds_valve, loss_coefficient_piping, Reynolds_factor
    CI = cavitation_index(1E6, 8E5, 2E5)
    assert_allclose(CI, 4.0)

    FF = FF_critical_pressure_ratio_l(70100.0, 22120000.0)
    assert_allclose(FF, 0.9442375225233299)

    F = is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.9)
    T = is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299, 0.6)
    assert_allclose([False, True], [F, T])

    with pytest.raises(Exception):
        is_choked_turbulent_l(460.0, 680.0, 70.1, 0.9442375225233299)

    # Example 4, compressible flow - small flow trim sized for gas flow:
    assert False == is_choked_turbulent_g(0.536, 1.193, 0.8)
    # Custom example
    assert True == is_choked_turbulent_g(0.9, 1.193, 0.7)

    with pytest.raises(Exception):
        is_choked_turbulent_g(0.544, 0.929)

    Rev = Reynolds_valve(3.26e-07, 360, 100.0, 0.6, 0.98, 238.05817216710483)
    assert_allclose(Rev, 6596953.826574914)

    Rev = Reynolds_valve(3.26e-07, 360, 150.0, 0.9, 0.46, 164.9954763704956)
    assert_allclose(Rev, 2967024.346783506)

    K = loss_coefficient_piping(0.05, 0.08, 0.1)
    assert_allclose(K, 0.6580810546875)

    ### Reynolds factor (laminar)
    # In Example 4, compressible flow with small flow trim sized for gas flow
    # (Cv in the problem was converted to Kv here to make FR match with N32, N2):
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=False)
    assert_allclose(f, 0.7148753122302025)

    # Custom, same as above but with full trim:
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=1202., full_trim=True)
    assert_allclose(f, 0.9875328782172637)

    # Example 4 with Rev < 10:
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=8., full_trim=False)
    assert_allclose(f, 0.08339546213461975)

    # Same, with full_trim
    f = Reynolds_factor(FL=0.98, C=0.015483, d=15., Rev=8., full_trim=True)
    assert_allclose(f, 43.619397389803986)

def test_control_valve_size_l():
    ### Control valve liquid
    # From [1]_, matching example 1 for a globe, parabolic plug,
    # flow-to-open valve.

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.15, D2=0.15, d=0.15, FL=0.9, Fd=0.46)
    assert_allclose(Kv, 164.9954763704956)

    # From [1]_, matching example 2 for a ball, segmented ball,
    # flow-to-open valve.

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.1, FL=0.6, Fd=0.98)
    assert_allclose(Kv, 238.05817216710483)

    # Modified example 1 with non-choked flow, with reducer and expander

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4,  P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.09, d=0.08, FL=0.9, Fd=0.46)
    assert_allclose(Kv, 177.44417090966715)

    # Modified example 2 with non-choked flow, with reducer and expander

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.1, D2=0.1, d=0.95, FL=0.6, Fd=0.98)
    assert_allclose(Kv, 230.1734424266345)

    # Modified example 2 with laminar flow at 100x viscosity, 100th flow rate, and 1/10th diameters:

    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-2, P1=680E3, P2=220E3, Q=0.001, D1=0.01, D2=0.01, d=0.01, FL=0.6, Fd=0.98)
    assert_allclose(Kv, 3.0947562381723626)

    # Last test, laminar full trim
    Kv = size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-2, P1=680E3, P2=220E3, Q=0.001, D1=0.01, D2=0.01, d=0.02, FL=0.6, Fd=0.98)
    assert_allclose(Kv, 3.0947562381723626)

    # TODO: find a test where the following is tested, or remove it as unnecessary.
    # if C/FR >= Ci:
    #    Ci = iterate_piping_laminar(Ci)
    # Efforts to make this happen have been unsuccessful.


def test_control_valve_size_g():
    # From [1]_, matching example 3 for non-choked gas flow with attached
    # fittings  and a rotary, eccentric plug, flow-to-open control valve:

    Kv = size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,  Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)
    assert_allclose(Kv, 72.58664545391052)

    # From [1]_, roughly matching example 4 for a small flow trim sized tapered
    # needle plug valve. Difference is 3% and explained by the difference in
    # algorithms used.

    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., D1=0.015, D2=0.015, d=0.015, FL=0.98, Fd=0.07, xT=0.8)
    assert_allclose(Kv, 0.016498765335995726)

    # Choked custom example
    Kv = size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30, Z=0.988, P1=680E3, P2=30E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)
    assert_allclose(Kv, 70.67468803987839)

    # Laminar custom example
    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=1.3E5, Q=0.46/3600., D1=0.015, D2=0.015, d=0.001, FL=0.98, Fd=0.07, xT=0.8)
    assert_allclose(Kv, 0.016498765335995726)

    # Laminar custom example with iteration
    Kv = size_control_valve_g(T=320., MW=39.95, mu=5.625E-5, gamma=1.67, Z=1.0, P1=2.8E5, P2=2.7E5, Q=0.1/3600., D1=0.015, D2=0.015, d=0.001, FL=0.98, Fd=0.07, xT=0.8)
    assert_allclose(Kv, 0.989125783445497)
