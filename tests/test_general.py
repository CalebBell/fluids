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

from __future__ import division
from fluids import *
import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_compressible():
    T = T_critical_flow(473, 1.289)
    assert_allclose(T, 413.2809086937528)

    P = P_critical_flow(1400000, 1.289)
    assert_allclose(P, 766812.9022792266)

    TF = [is_critical_flow(670E3, 532E3, 1.11), is_critical_flow(670E3, 101E3, 1.11)]
    assert_allclose(TF, [False, True])

    SE = stagnation_energy(125)
    assert_allclose(SE, 7812.5)

    PST = P_stagnation(54050., 255.7, 286.8, 1.4)
    assert_allclose(PST, 80772.80495900588)

    Tst = T_stagnation(286.8, 54050, 54050*8, 1.4)
    assert_allclose(Tst, 519.5230938217768)

    Tstid = T_stagnation_ideal(255.7, 250, 1005.)
    assert_allclose(Tstid, 286.79452736318405)


def test_control_valve():
    from fluids.control_valve import cavitation_index, FF_critical_pressure_ratio_l, is_choked_turbulent_l, is_choked_turbulent_g, Reynolds_valve, loss_coefficient_piping, Reynolds_factor, size_control_valve_l, size_control_valve_g
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


def test_core_misc():
    alpha = thermal_diffusivity(0.02, 1., 1000.)
    assert_allclose(alpha, 2e-05)

    c = c_ideal_gas(1.4, 303., 28.96)
    assert_allclose(c, 348.9820361755092, rtol=1e-05)


def test_core_dimensionless():
    Re = Reynolds(2.5, 0.25, 1.1613, 1.9E-5)
    assert_allclose(Re, 38200.65789473684)
    Re = Reynolds(2.5, 0.25, nu=1.636e-05)
    assert_allclose(Re, 38202.93398533008)
    with pytest.raises(Exception):
        Reynolds(2.5, 0.25, 1.1613)

    PeH = Peclet_heat(1.5, 2, 1000., 4000., 0.6)
    assert_allclose(PeH, 20000000.0)
    PeH = Peclet_heat(1.5, 2, alpha=1E-7)
    assert_allclose(PeH, 30000000.0)
    with pytest.raises(Exception):
        Peclet_heat(1.5, 2, 1000., 4000.)

    PeM = Peclet_mass(1.5, 2, 1E-9)
    assert_allclose(PeM, 3000000000)

    FH1 = Fourier_heat(1.5, 2, 1000., 4000., 0.6)
    FH2 = Fourier_heat(1.5, 2, alpha=1E-7)
    assert_allclose([FH1, FH2], [5.625e-08, 3.75e-08])
    with pytest.raises(Exception):
        Fourier_heat(1.5, 2, 1000., 4000.)

    FHM = Fourier_mass(1.5, 2, 1E-9)
    assert_allclose(FHM,  3.7500000000000005e-10)

    GZh1 = Graetz_heat(1.5, 0.25, 5, 800., 2200., 0.6)
    GZh2 = Graetz_heat(1.5, 0.25, 5, alpha=1E-7)
    assert_allclose([GZh1, GZh2], [55000.0, 187500.0])
    with pytest.raises(Exception):
        Graetz_heat(1.5, 0.25, 5, 800., 2200.)

    Sc1 = Schmidt(D=2E-6, mu=4.61E-6, rho=800)
    Sc2 = Schmidt(D=1E-9, nu=6E-7)
    assert_allclose([Sc1, Sc2], [0.00288125, 600.])
    with pytest.raises(Exception):
        Schmidt(D=2E-6, mu=4.61E-6)

    Le1 = Lewis(D=22.6E-6, alpha=19.1E-6)
    Le2 = Lewis(D=22.6E-6, rho=800., k=.2, Cp=2200)
    assert_allclose([Le1, Le2], [0.8451327433628318, 0.00502815768302494])
    with pytest.raises(Exception):
        Lewis(D=22.6E-6, rho=800., k=.2)

    We = Weber(0.18, 0.001, 900., 0.01)
    assert_allclose(We, 2.916)

    Ma = Mach(33., 330)
    assert_allclose(Ma, 0.1)

    Kn = Knudsen(1e-10, .001)
    assert_allclose(Kn, 1e-07)

    Pr1 = Prandtl(Cp=1637., k=0.010, mu=4.61E-6)
    Pr2 = Prandtl(Cp=1637., k=0.010, nu=6.4E-7, rho=7.1)
    Pr3 = Prandtl(nu=6.3E-7, alpha=9E-7)
    assert_allclose([Pr1, Pr2, Pr3], [0.754657, 0.7438528, 0.7])
    with pytest.raises(Exception):
        Prandtl(Cp=1637., k=0.010)

    Gr1 = Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613, mu=1.9E-5)
    Gr2 = Grashof(L=0.9144, beta=0.000933, T1=378.2, T2=200, nu=1.636e-05)
    assert_allclose([Gr1, Gr2], [4656936556.178915, 4657491516.530312])
    with pytest.raises(Exception):
        Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613)

    Bo1 = Bond(1000., 1.2, .0589, 2)
    assert_allclose(Bo1, 665187.2339558573)

    Ra1 = Rayleigh(1.2, 4.6E9)
    assert_allclose(Ra1, 5520000000)

    Fr1 = Froude(1.83, 2., 1.63)
    Fr2 = Froude(1.83, L=2., squared=True)
    assert_allclose([Fr1, Fr2], [1.0135432593877318, 0.17074638128208924])

    St = Strouhal(8, 2., 4.)
    assert_allclose(St, 4.0)

    Nu1 = Nusselt(1000., 1.2, 300.)
    Nu2 = Nusselt(10000., .01, 4000.)
    assert_allclose([Nu1, Nu2], [4.0, 0.025])

    Sh = Sherwood(1000., 1.2, 300.)
    assert_allclose(Sh, 4.0)

    Bi1 = Biot(1000., 1.2, 300.)
    Bi2 = Biot(10000., .01, 4000.)
    assert_allclose([Bi1, Bi2], [4.0, 0.025])

    St1 = Stanton(5000, 5, 800, 2000.)
    assert_allclose(St1, 0.000625)

    Eu1 = Euler(1E5, 1000., 4)
    assert_allclose(Eu1, 6.25)

    Ca1 = Cavitation(2E5, 1E4, 1000, 10)
    assert_allclose(Ca1, 3.8)

    Ec1 = Eckert(10, 2000., 25.)
    assert_allclose(Ec1, 0.002)

    Ja1 = Jakob(4000., 2E6, 10.)
    assert_allclose(Ja1, 0.02)

    Po1 = Power_number(P=180, L=0.01, N=2.5, rho=800.)
    assert_allclose(Po1, 144000000)

    Cd1 = Drag(1000, 0.0001, 5, 2000)
    assert_allclose(Cd1, 400)

    Ca1 = Capillary(1.2, 0.01, .1)
    assert_allclose(Ca1, 0.12)

    Ar1 = Archimedes(0.002, 0.2804, 2699.37, 4E-5)
    Ar2 = Archimedes(0.002, 2., 3000, 1E-3)
    assert_allclose([Ar1, Ar2], [37109.575890227665, 470.4053872])

    Oh1 = Ohnesorge(1E-4, 1000., 1E-3, 1E-1)
    assert_allclose(Oh1, 0.01)

    BeL1 = Bejan_L(1E4, 1, 1E-3, 1E-6)
    assert_allclose(BeL1, 10000000000000)

    Bep1 = Bejan_p(1E4, 1, 1E-3, 1E-6)
    assert_allclose(Bep1, 10000000000000)

    e_D1 = relative_roughness(0.0254)
    e_D2 = relative_roughness(0.5, 1E-4)
    assert_allclose([e_D1, e_D2], [5.9842519685039374e-05, 0.0002])

def test_core_misc2():
    mu1 = nu_mu_converter(998., nu=1.0E-6)
    nu1 = nu_mu_converter(998., mu=0.000998)
    assert_allclose([mu1, nu1], [0.000998, 1E-6])
    with pytest.raises(Exception):
        nu_mu_converter(990)
    with pytest.raises(Exception):
        nu_mu_converter(990, 0.000998, 1E-6)

    g1 = gravity(55, 1E4)
    assert_allclose(g1, 9.784151976863571)

    K = K_from_f(f=0.018, L=100., D=.3)
    assert_allclose(K, 6.0)

    K = K_from_L_equiv(240.)
    assert_allclose(K, 3.6)

    dP = dP_from_K(K=10, rho=1000, V=3)
    assert_allclose(dP, 45000)

    head = head_from_K(K=10, V=1.5)
    assert_allclose(head, 1.1471807396001694)

    head = head_from_P(P=98066.5, rho=1000)
    assert_allclose(head, 10.0)

    P = P_from_head(head=5., rho=800.)
    assert_allclose(P, 39226.6)

def test_filters():
    K1 = round_edge_screen(0.5, 100)
    K2 = round_edge_screen(0.5, 100, 45)
    K3 = round_edge_screen(0.5, 100, 85)

    assert_allclose([K1, K2, K3], [2.0999999999999996, 1.05, 0.18899999999999997])

    Ks =  [round_edge_open_mesh(0.88, i) for i in ['round bar screen', 'diamond pattern wire', 'knotted net', 'knotless net']]
    K_values = [0.11687999999999998, 0.09912, 0.15455999999999998, 0.11664]
    assert_allclose(Ks, K_values)

    K1 = round_edge_open_mesh(0.96, angle=33.)
    K2 = round_edge_open_mesh(0.96, angle=50)
    assert_allclose([K1, K2], [0.02031327712601458, 0.012996000000000014])

    with pytest.raises(Exception):
        round_edge_open_mesh(0.96, subtype='not_filter', angle=33.)

    K = square_edge_screen(0.99)
    assert_allclose(K, 0.008000000000000009)

    K1 = square_edge_grill(.45)
    K2 = square_edge_grill(.45, l=.15, Dh=.002, fd=.0185)
    assert_allclose([K1, K2], [5.296296296296296, 12.148148148148147])

    K1 = round_edge_grill(.4)
    K2 = round_edge_grill(.4, l=.15, Dh=.002, fd=.0185)
    assert_allclose([K1, K2], [1.0, 2.3874999999999997])


def test_fittings():
    assert_allclose(entrance_sharp(), 0.57)

    K1 = entrance_distance(d=0.1, t=0.0005)
    assert_allclose(K1, 1.0154100000000004)
    with pytest.raises(Exception):
        entrance_distance(d=0.1, l=0.005, t=0.0005)
    with pytest.raises(Exception):
        entrance_distance(d=0.1,  t=0.05)

    assert_allclose(entrance_angled(30), 0.9798076211353316)

    K =  entrance_rounded(Di=0.1, rc=0.0235)
    assert_allclose(K, 0.09839534618360923)

    K = entrance_beveled(Di=0.1, l=0.003, angle=45)
    assert_allclose(K, 0.45086864221916984)

    ### Exits
    assert_allclose(exit_normal(), 1.0)

    ### Bends
    K_5_rc = [bend_rounded(Di=4.020, rc=4.0*5, angle=i, fd=0.0163) for i in [15, 30, 45, 60, 75, 90]]
    K_5_rc_values = [0.07038212630028828, 0.10680196344492195, 0.13858204974134541, 0.16977191374717754, 0.20114941557508642, 0.23248382866658507]
    assert_allclose(K_5_rc, K_5_rc_values)

    K_10_rc = [bend_rounded(Di=34.500, rc=36*10, angle=i, fd=0.0106) for i in [15, 30, 45, 60, 75, 90]]
    K_10_rc_values =  [0.061075866683922314, 0.10162621862720357, 0.14158887563243763, 0.18225270014527103, 0.22309967045081655, 0.26343782210280947]
    assert_allclose(K_10_rc, K_10_rc_values)

    K = bend_rounded(Di=4.020, bend_diameters=5, angle=30, fd=0.0163)
    assert_allclose(K, 0.106920213333191)

    K_miters =  [bend_miter(i) for i in [150, 120, 90, 75, 60, 45, 30, 15]]
    K_miter_values = [2.7128147734758103, 2.0264994448555864, 1.2020815280171306, 0.8332188430731828, 0.5299999999999998, 0.30419633092708653, 0.15308822558050816, 0.06051389308126326]
    assert_allclose(K_miters, K_miter_values)

    K_helix = helix(Di=0.01, rs=0.1, pitch=.03, N=10, fd=.0185)
    assert_allclose(K_helix, 14.525134924495514)

    K_spiral = spiral(Di=0.01, rmax=.1, rmin=.02, pitch=.01, fd=0.0185)
    assert_allclose(K_spiral, 7.950918552775473)

    ### Contractions
    K_sharp = contraction_sharp(Di1=1, Di2=0.4)
    assert_allclose(K_sharp, 0.5301269161591805)

    K_round = contraction_round(Di1=1, Di2=0.4, rc=0.04)
    assert_allclose(K_round, 0.1783332490866574)

    K_conical1 = contraction_conical(Di1=0.1, Di2=0.04, l=0.04, fd=0.0185)
    K_conical2 = contraction_conical(Di1=0.1, Di2=0.04, angle=73.74, fd=0.0185)
    assert_allclose([K_conical1, K_conical2], [0.15779041548350314, 0.15779101784158286])
    with pytest.raises(Exception):
        contraction_conical(Di1=0.1, Di2=0.04, fd=0.0185)

    K_beveled = contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120)
    assert_allclose(K_beveled, 0.40946469413070485)

    ### Expansions (diffusers)
    K_sharp = diffuser_sharp(Di1=.5, Di2=1)
    assert_allclose(K_sharp, 0.5625)

    K1 = diffuser_conical(Di1=.1**0.5, Di2=1, angle=10., fd=0.020)
    K2 = diffuser_conical(Di1=1/3., Di2=1, angle=50, fd=0.03) # 2
    K3 = diffuser_conical(Di1=2/3., Di2=1, angle=40, fd=0.03) # 3
    K4 = diffuser_conical(Di1=1/3., Di2=1, angle=120, fd=0.0185) # #4
    K5 = diffuser_conical(Di1=2/3., Di2=1, angle=120, fd=0.0185) # Last
    K6 = diffuser_conical(Di1=.1**0.5, Di2=1, l=3.908, fd=0.020)
    Ks = [0.12301652230915454, 0.8081340270019336, 0.32533470783539786, 0.812308728765127, 0.3282650135070033, 0.12300865396254032]
    assert_allclose([K1, K2, K3, K4, K5, K6], Ks)
    with pytest.raises(Exception):
        diffuser_conical(Di1=.1, Di2=0.1, angle=1800., fd=0.020)

    K1 = diffuser_conical_staged(Di1=1., Di2=10., DEs=[2,3,4,5,6,7,8,9], ls=[1,1,1,1,1,1,1,1,1], fd=0.01)
    K2 = diffuser_conical(Di1=1., Di2=10.,l=9, fd=0.01)
    Ks = [1.7681854713484308, 0.973137914861591]
    assert_allclose([K1, K2], Ks)

    K = diffuser_curved(Di1=.25**0.5, Di2=1., l=2.)
    assert_allclose(K, 0.2299781250000002)

    K = diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07)
    assert_allclose(K, 0.06873244301714816)

    # Misc
    K1 = Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
    K2 = Darby3K(NPS=12., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
    K3 = Darby3K(NPS=12., Re=10000., K1=950,  Ki=0.25,  Kd=4)
    Ks = [1.1572523963562353, 0.819510280626355, 0.819510280626355]
    assert_allclose([K1, K2, K3], Ks)

    with pytest.raises(Exception):
        Darby3K(NPS=12., Re=10000)
    with pytest.raises(Exception):
        Darby3K(NPS=12., Re=10000, name='fail')

    tot = sum([Darby3K(NPS=2., Re=1000, name=i) for i in Darby.keys()])
    assert_allclose(tot, 67.96442287975898)

    K1 = Hooper2K(Di=2., Re=10000., name='Valve, Globe, Standard')
    K2 = Hooper2K(Di=2., Re=10000., K1=900, Kinfty=4)
    assert_allclose([K1, K2], [6.15, 6.09])
    tot = sum([Hooper2K(Di=2., Re=10000., name=i) for i in Hooper.keys()])
    assert_allclose(tot, 46.18)

    with pytest.raises(Exception):
        Hooper2K(Di=2, Re=10000)
    with pytest.raises(Exception):
        Hooper2K(Di=2., Re=10000, name='fail')

    Cv = Kv_to_Cv(2)
    assert_allclose(Cv, 2.3121984567081197)
    Kv = Cv_to_Kv(2.312)
    assert_allclose(Kv, 1.9998283393819036)
    K = Kv_to_K(2.312, .015)
    assert_allclose(K, 15.1912580369009)
    Kv = K_to_Kv(15.1912580369009, .015)
    assert_allclose(Kv, 2.312)


def test_friction():
    assert_allclose(Moody(1E5, 1E-4), 0.01809185666808665)
    assert_allclose(Alshul_1952(1E5, 1E-4), 0.018382997825686878)
    assert_allclose(Wood_1966(1E5, 1E-4), 0.021587570560090762)
    assert_allclose(Churchill_1973(1E5, 1E-4), 0.01846708694482294)
    assert_allclose(Eck_1973(1E5, 1E-4), 0.01775666973488564)
    assert_allclose(Jain_1976(1E5, 1E-4), 0.018436560312693327)
    assert_allclose(Swamee_Jain_1976(1E5, 1E-4), 0.018452424431901808)
    assert_allclose(Churchill_1977(1E5, 1E-4), 0.018462624566280075)
    assert_allclose(Chen_1979(1E5, 1E-4), 0.018552817507472126)
    assert_allclose(Round_1980(1E5, 1E-4), 0.01831475391244354)
    assert_allclose(Shacham_1980(1E5, 1E-4), 0.01860641215097828)
    assert_allclose(Barr_1981(1E5, 1E-4), 0.01849836032779929)
    assert_allclose(Zigrang_Sylvester_1(1E5, 1E-4), 0.018646892425980794)
    assert_allclose(Zigrang_Sylvester_2(1E5, 1E-4), 0.01850021312358548)
    assert_allclose(Haaland(1E5, 1E-4), 0.018265053014793857)
    assert_allclose(Serghides_1(1E5, 1E-4), 0.01851358983180063)
    assert_allclose(Serghides_2(1E5, 1E-4), 0.018486377560664482)
    assert_allclose(Tsal_1989(1E5, 1E-4), 0.018382997825686878)
    assert_allclose(Tsal_1989(1E8, 1E-4), 0.012165854627780102)
    assert_allclose(Manadilli_1997(1E5, 1E-4), 0.01856964649724108)
    assert_allclose(Romeo_2002(1E5, 1E-4), 0.018530291219676177)
    assert_allclose(Sonnad_Goudar_2006(1E5, 1E-4), 0.0185971269898162)
    assert_allclose(Rao_Kumar_2007(1E5, 1E-4), 0.01197759334600925)
    assert_allclose(Buzzelli_2008(1E5, 1E-4), 0.018513948401365277)
    assert_allclose(Avci_Karagoz_2009(1E5, 1E-4), 0.01857058061066499)
    assert_allclose(Papaevangelo_2010(1E5, 1E-4), 0.015685600818488177)
    assert_allclose(Brkic_2011_1(1E5, 1E-4), 0.01812455874141297)
    assert_allclose(Brkic_2011_2(1E5, 1E-4), 0.018619745410688716)
    assert_allclose(Fang_2011(1E5, 1E-4), 0.018481390682985432)

    assert_allclose(sum(_roughness.values()), 0.01504508)

    assert_allclose(friction_factor(Re=1E5, eD=1E-4), 0.018513948401365277)
    assert friction_factor(Re=1E5, eD=1E-4, AvailableMethods=True) == ['Manadilli_1997', 'Haaland', 'Alshul_1952', 'Avci_Karagoz_2009', 'Rao_Kumar_2007', 'Zigrang_Sylvester_2', 'Eck_1973', 'Buzzelli_2008', 'Tsal_1989', 'Papaevangelo_2010', 'Barr_1981', 'Jain_1976', 'Moody', 'Brkic_2011_2', 'Brkic_2011_1', 'Swamee_Jain_1976', 'Wood_1966', 'Shacham_1980', 'Romeo_2002', 'Chen_1979', 'Fang_2011', 'Round_1980', 'Sonnad_Goudar_2006', 'Churchill_1973', 'Churchill_1977', 'Serghides_2', 'Serghides_1', 'Zigrang_Sylvester_1']
    assert_allclose(friction_factor(Re=1E5, eD=1E-4, Darcy=False), 0.018513948401365277*4)


def test_mixing():
    t1 = agitator_time_homogeneous(D=36*.0254, N=56/60., P=957., T=1.83, H=1.83, mu=0.018, rho=1020, homogeneity=.995)
    t2 = agitator_time_homogeneous(D=1, N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)
    t3 = agitator_time_homogeneous(N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)

    assert_allclose([t1, t2, t3], [15.143198226374668, 67.7575069865228, 51.70865552491966])

    Kp = Kp_helical_ribbon_Rieger(D=1.9, h=1.9, nb=2, pitch=1.9, width=.19, T=2)
    assert_allclose(Kp, 357.39749163259256)

    t = time_helical_ribbon_Grenville(357.4, 4/60.)
    assert_allclose(t, 650.980654028894)

    CoV = size_tee(Q1=11.7, Q2=2.74, D=0.762, D2=None, n=1, pipe_diameters=5)
    assert_allclose(CoV, 0.2940930233038544)

    CoV = COV_motionless_mixer(Ki=.33, Q1=11.7, Q2=2.74, pipe_diameters=4.74/.762)
    assert_allclose(CoV, 0.0020900028665727685)

    K = K_motionless_mixer(K=150, L=.762*5, D=.762, fd=.01)
    assert_allclose(K, 7.5)