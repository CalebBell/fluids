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
    methods_1 = friction_factor(Re=1E5, eD=1E-4, AvailableMethods=True)
    methods_1.sort()

    methods_2 = ['Manadilli_1997', 'Haaland', 'Alshul_1952', 'Avci_Karagoz_2009', 'Rao_Kumar_2007', 'Zigrang_Sylvester_2', 'Eck_1973', 'Buzzelli_2008', 'Tsal_1989', 'Papaevangelo_2010', 'Barr_1981', 'Jain_1976', 'Moody', 'Brkic_2011_2', 'Brkic_2011_1', 'Swamee_Jain_1976', 'Wood_1966', 'Shacham_1980', 'Romeo_2002', 'Chen_1979', 'Fang_2011', 'Round_1980', 'Sonnad_Goudar_2006', 'Churchill_1973', 'Churchill_1977', 'Serghides_2', 'Serghides_1', 'Zigrang_Sylvester_1']
    methods_2.sort()
    assert methods_1 == methods_2

    assert_allclose(friction_factor(Re=1E5, eD=1E-4, Darcy=False), 0.018513948401365277*4)


def test_geometry():
    SA1 = SA_partial_sphere(1., 0.7)
    SA2 = SA_partial_sphere(2, 1) # One spherical head's surface area:
    assert_allclose([SA1, SA2], [2.199114857512855, 6.283185307179586])

    V1 = V_partial_sphere(1., 0.7)
    assert_allclose(V1, 0.4105014400690663)

    # Two examples from [1]_, and at midway, full, and empty.
    Vs_horiz_conical1 = [V_horiz_conical(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs_horiz_conical1s = [2041.1923581273443, 6180.540773905826, 3648.490668241736, 7296.981336483472, 0.0]
    assert_allclose(Vs_horiz_conical1, Vs_horiz_conical1s)

    # Head only custom example:
    V_head1 = V_horiz_conical(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 = V_horiz_conical(108., 156., 42., 84., headonly=True)/231.
    assert_allclose([V_head1, V_head2], [508.8239000645628]*2)

    # Two examples from [1]_, and at midway, full, and empty.
    Vs_horiz_ellipsoidal = [V_horiz_ellipsoidal(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs_horiz_ellipsoidals = [2380.9565415578145, 7103.445235921378, 4203.695769930696, 8407.391539861392, 0.0]
    assert_allclose(Vs_horiz_ellipsoidal, Vs_horiz_ellipsoidals)

    #Head only custom example:
    V_head1 = V_horiz_ellipsoidal(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 = V_horiz_ellipsoidal(108., 156., 42., 84., headonly=True)/231.
    assert_allclose([V_head1, V_head2], [970.2761310723387]*2)

    # Two examples from [1]_, and at midway, full, and empty.
    V_calc = [V_horiz_guppy(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs = [1931.7208029476762, 5954.110515329029, 3412.8543046053724, 7296.981336483472, 0.0]
    assert_allclose(V_calc, Vs)

    # Head only custom example:
    V_head1 = V_horiz_guppy(D=108., L=156., a=42., h=36, headonly=True)/231.
    V_head2 = V_horiz_guppy(108., 156., 42., 36, headonly=True)/231.
    assert_allclose([V_head1, V_head2], [63.266257496613804]*2)


    # Two examples from [1]_, and at midway, full, and empty.
    V_calc = [V_horiz_spherical(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs = [2303.9615116986183, 6935.163365275476, 4094.025626387197, 8188.051252774394, 0.0]
    assert_allclose(V_calc, Vs)

    # Test when the integration function is called, on its limits:
    Vs = [V_horiz_spherical(D=108., L=156., a=i, h=84.)/231. for i in (108*.009999999, 108*.01000001)]
    V_calc = [5201.54341872961, 5201.543461255985]
    assert_allclose(Vs, V_calc)

    # Head only custom example:
    V_head1 =  V_horiz_spherical(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 =  V_horiz_spherical(108., 156., 42., 84., headonly=True)/231.
    assert_allclose([V_head1, V_head2], [886.1351957493874]*2)


    # Two examples from [1]_, and at midway, full, empty, and 1 inch; covering
    # all code cases.
    V_calc  = [V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=i)/231. for i in [36, 84, 54, 108, 0, 1]]
    Vs = [2028.626670842139, 5939.897910157917, 3534.9973314622794, 7069.994662924554, 0.0, 9.580013820942611]
    assert_allclose(V_calc, Vs)

    # Head only custom example:
    V_head1 = V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=36, headonly=True)/231.
    V_head2 = V_horiz_torispherical(108., 156., 1., 0.06, 36, headonly=True)/231.
    assert_allclose([V_head1, V_head2], [111.71919144384525]*2)

    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_conical(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [250.67461381371024, 2251.175535772343, 0.0, 6516.560761446257]
    assert_allclose(Vs_calc, Vs)

    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_ellipsoidal(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [783.3581681678445, 2902.831611916969, 0.0, 7168.216837590883]
    assert_allclose(Vs_calc, Vs)

    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_spherical(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [583.6018352850442, 2658.4605833627343, 0.0, 6923.845809036648]
    assert_allclose(Vs_calc, Vs)

    # Two examples from [1]_, and at empty, 1, 22, and h=D.
    Vs_calc = [V_vertical_torispherical(132., 1.0, 0.06, i)/231. for i in [24, 60, 0, 1, 22, 132]]
    Vs = [904.0688283793511, 3036.7614412163075, 0.0, 1.7906624793188568, 785.587561468186, 7302.146666890221]
    assert_allclose(Vs_calc, Vs)

    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_conical_concave(113., -33, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [251.15825565795188, 614.6068425492208, 1693.1654406426783, 0.0, 4428.278844757774]
    assert_allclose(Vs_calc, Vs)

    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_ellipsoidal_concave(113., -33, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [44.84968851034856, 207.6374468071692, 1215.605957384487, 0.0, 3950.7193614995826]
    assert_allclose(Vs_calc, Vs)

    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_spherical_concave(113., -33, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [112.81405437348528, 341.7056403375114, 1372.9286894955042, 0.0, 4108.042093610599]
    assert_allclose(Vs_calc, Vs)

    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_torispherical_concave(D=113., f=0.71, k=0.081, h=i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [103.88569287163769, 388.72142877582087, 1468.762358198084, 0.0, 4203.87576231318]
    assert_allclose(Vs_calc, Vs)

    SA1 = SA_ellipsoidal_head(2, 1)
    SA2 = SA_ellipsoidal_head(2, 0.999)
    SAs = [6.283185307179586, 6.278996936093318]
    assert_allclose([SA1, SA2], SAs)

    SA1 = SA_conical_head(2, 1)
    SAs = 4.442882938158366
    assert_allclose(SA1, SAs)

    SA1 = SA_guppy_head(2, 1)
    assert_allclose(SA1, 6.654000019110157)

    SA1 = SA_torispheroidal(D=2.54, fd=1.039370079, fk=0.062362205)
    assert_allclose(SA1, 6.00394283477063)

    SA1 = SA_tank(D=2, L=2)
    SA2 = SA_tank(D=1., L=0, sideA='ellipsoidal', sideA_a=2, sideB='ellipsoidal', sideB_a=2)
    SA3 = SA_tank(D=1., L=5, sideA='conical', sideA_a=2, sideB='conical', sideB_a=2)
    SA4 = SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical', sideB_a=0.5)
    SAs = [18.84955592153876, 28.480278854014387, 22.18452243965656, 18.84955592153876]
    assert_allclose([SA1, SA2, SA3, SA4], SAs)

    SA1, (SA2, SA3, SA4) = SA_tank(D=2.54, L=5, sideA='torispherical', sideB='torispherical', sideA_f=1.039370079, sideA_k=0.062362205, sideB_f=1.039370079, sideB_k=0.062362205, full_output=True)
    SAs = [51.90611237013163, 6.00394283477063, 6.00394283477063, 39.89822670059037]
    assert_allclose([SA1, SA2, SA3, SA4], SAs)

    SA1 = SA_tank(D=1., L=5, sideA='guppy', sideA_a=0.5, sideB='guppy', sideB_a=0.5)
    assert_allclose(SA1, 19.034963277504044)

    a1 = a_torispherical(D=96., f=0.9, k=0.2)
    a2 = a_torispherical(D=108., f=1., k=0.06)
    ais = [25.684268924767125, 18.288462280484797]
    assert_allclose([a1, a2], ais)

    # Horizontal configurations, compared with TankCalc - Ellipsoidal*2,
    # Ellipsoidal/None, spherical/conical, None/None. Final test is guppy/torispherical,
    # no checks available.

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=2, sideB_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [108.05249928250362, 416.5904542901302, 1086.4674593664702, 1756.34446444281, 2172.9349187329403]
    assert_allclose(Vs_calc, Vs)
    Vs_calc =[V_from_h(h=h, D=10., L=25., horizontal=True, sideA='ellipsoidal', sideA_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [105.12034613915314, 400.22799255268336, 1034.1075818066402, 1667.9871710605971, 2068.2151636132803]
    assert_allclose(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='spherical', sideB='conical', sideA_a=2, sideB_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [104.20408244287965, 400.47607362329063, 1049.291946298991, 1698.107818974691, 2098.583892597982]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideB='spherical', sideA='conical', sideB_a=2, sideA_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    assert_allclose(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=True) for h in [0, 0.75, 1.5]]
    Vs = [0.0, 4.417864669110647, 8.835729338221293]
    assert_allclose(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='guppy', sideB='torispherical', sideA_a=2, sideB_f=1., sideB_k=0.06) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [104.68706323659293, 399.0285611453449, 1037.3160340613756, 1683.391972469731, 2096.854290344973]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideB='guppy', sideA='torispherical', sideB_a=2, sideA_f=1., sideA_k=0.06) for h in [1, 2.5, 5, 7.5, 10]]
    assert_allclose(Vs_calc, Vs)

    with pytest.raises(Exception):
        V_from_h(h=7, D=1.5, L=5)


    # Vertical configurations, compared with TankCalc - conical*2, spherical*2,
    # ellipsoidal*2. Torispherical*2 has no check. None*2 checks.

    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.) for h in [0, 1, 2, 5., 7, 7.2, 8]]
    Vs = [0.0, 0.14726215563702155, 1.1780972450961726, 6.4795348480289485, 10.013826583317465, 10.301282311120932, 10.602875205865551]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='spherical', sideB='spherical', sideA_a=3., sideB_a=4.) for h in [0, 1.5, 3, 8.5, 13., 15., 16.2, 17]]
    Vs = [0.0, 25.91813939211579, 89.5353906273091, 365.99554414321085, 592.190215201676, 684.3435997069765, 718.7251897078633, 726.2315017548405]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=3., sideB_a=4.) for h in [0, 1.5, 3, 8.5, 13., 15., 16.2, 17]]
    Vs = [0.0, 31.41592653589793, 100.53096491487338, 376.99111843077515, 603.1857894892403, 695.3391739945409, 729.7207639954277, 737.2270760424049]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='torispherical', sideB='torispherical', sideA_a=1.3547, sideB_a=1.3547, sideA_f=1.,  sideA_k=0.06, sideB_f=1., sideB_k=0.06) for h in [0, 1.3, 9.3, 10.1, 10.7094, 12]]
    Vs = [0.0, 38.723353379954276, 440.84578224136413, 481.0581682073135, 511.68995321687544, 573.323556832692]
    assert_allclose(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=False) for h in [0, 2.5, 5]]
    Vs = [0, 4.417864669110647, 8.835729338221293]
    assert_allclose(Vs_calc, Vs)

    with pytest.raises(Exception):
        V_from_h(h=7, D=1.5, L=5., horizontal=False)




def test_geometry_tank():
    V1 = TANK(D=1.2, L=4, horizontal=False).V_total
    assert_allclose(V1, 4.523893421169302)

    V2 = TANK(D=1.2, L=4, horizontal=False).V_from_h(.5)
    assert_allclose(V2, 0.5654866776461628)

    V3 = TANK(D=1.2, L=4, horizontal=False).h_from_V(.5)
    assert_allclose(V3, 0.44209706414415373)

    T1 = TANK(V=10, L_over_D=0.7, sideB='conical', sideB_a=0.5)
    T1.set_table(dx=0.001)
    things_calc = T1.A, T1.A_sideA, T1.A_sideB, T1.A_lateral
    things = (24.94775907657148, 5.118555935958284, 5.497246519930003, 14.331956620683194)
    assert_allclose(things_calc, things)

    L1 = TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L
    D1 = TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D
    L2 = TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L
    assert_allclose([L1, D1, L2], [4.699531057009146, 9.999999999999407, 4.69953105700979])

    L1 = TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L
    D1 = TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D
    L2 = TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L
    assert_allclose([L1, D1, L2], [4.699531057009146, 9.999999999999407, 4.69953105700979])

    # Test L_over_D setting simple cases
    L1 = TANK(D=1.2, L_over_D=3.5, horizontal=False).L
    D1 = TANK(L=1.2, L_over_D=3.5, horizontal=False).D
    assert_allclose([L1, D1], [4.2, 0.342857142857])
    # Test toripsherical a calculation
    V = TANK(L=1.2, L_over_D=3.5, sideA='torispherical', sideB='torispherical', sideA_f=1.,  sideA_k=0.06, sideB_f=1., sideB_k=0.06).V_total
    assert_allclose(V, 0.117318265914)

    with pytest.raises(Exception):
        # Test overdefinition case
        TANK(V=10, L=10, D=10)
    with pytest.raises(Exception):
        # Test sides specified with V solving
        TANK(V=10, L=10, sideA='conical', sideB_a=0.5)
    with pytest.raises(Exception):
        TANK(V=10, L=10, sideA='conical', sideA_a_ratio=None)


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


def test_open_flow():
    Q1 = Q_weir_V_Shen(0.6, angle=45)
    Q2 = Q_weir_V_Shen(1.2)

    assert_allclose([Q1, Q2], [0.21071725775478228, 2.8587083148501078])

    Q1 = Q_weir_rectangular_Kindsvater_Carter(0.2, 0.5, 1)
    assert_allclose(Q1, 0.15545928949179422)

    Q1 = Q_weir_rectangular_SIA(0.2, 0.5, 1, 2)
    assert_allclose(Q1, 1.0408858453811165)

    Q1 = Q_weir_rectangular_full_Ackers(h1=0.9, h2=0.6, b=5)
    Q2 = Q_weir_rectangular_full_Ackers(h1=0.3, h2=0.4, b=2)
    assert_allclose([Q1, Q2], [9.251938159899948, 0.6489618999846898])

    Q1 = Q_weir_rectangular_full_SIA(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 1.1875825055400384)

    Q1 = Q_weir_rectangular_full_Rehbock(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 0.6486856330601333)

    Q1 = Q_weir_rectangular_full_Kindsvater_Carter(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 0.641560300081563)

    V1 = V_Manning(0.2859, 0.005236, 0.03)*0.5721
    V2 = V_Manning(0.2859, 0.005236, 0.03)
    V3 = V_Manning(Rh=5, S=0.001, n=0.05)
    assert_allclose([V1, V2, V3], [0.5988618058239864, 1.0467781958118971, 1.8493111942973235])

    C = n_Manning_to_C_Chezy(0.05, Rh=5)
    assert_allclose(C, 26.15320972023661)

    n = C_Chezy_to_n_Manning(26.15, Rh=5)
    assert_allclose(n, 0.05000613713238358)

    V = V_Chezy(Rh=5, S=0.001, C=26.153)
    assert_allclose(V, 1.8492963648371776)

    n_tot = np.sum(np.concatenate(np.array([list(val.values()) for thing in n_dicts for val in thing.values()])))
    assert_allclose(n_tot, 11.115999999999984)


def test_packed_bed():
    dP = Ergun(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1338.8671874999995)

    dP = Kuo_Nydegger(dp=8E-1, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 0.025651460973648624)

    dP = Jones_Krier(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1362.2719449873746)

    dP = Carman(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1614.721678121775)

    dP = Hicks(dp=0.01, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 3.631703956680737)

    dP = Brauer(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1441.5479196020563)

    dP = KTA(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1440.409277034248)

    dP = Erdim_Akgiray_Demir(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1438.2826958844414)

    dP = Fahien_Schriver(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1470.6175541844711)

    dP = Idelchik(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1571.909125999067)

    dP1 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2)
    assert_allclose([dP1, dP2], [1104.6473821473724, 1255.1625662548427])

    dP1 = Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    dP2 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.4, L=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    dP3 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.3, L=0.5, vs=0.05, rho=1000., mu=1.00E-003, Dt=1)
    assert_allclose([dP1, dP2, dP3], [1148.1905244077548, 212.67409611116554, 540.501305905986])

    dP1 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01)
    dP3 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP4 = dP_packed_bed(dp=8E-4, voidage=0.4, sphericity=0.6, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP5 = dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3)
    assert_allclose([dP1, dP2, dP3, dP4, dP5], [1438.2826958844414, 1255.1625662548427, 1338.8671874999995, 3696.2890624999986, 1438.2826958844414])

    methods_dP = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, AvailableMethods=True)
    methods_dP.sort()
    methods_dP_val = ['Harrison, Brunner & Hecker', 'Carman', 'Hicks', 'Montillet, Akkari & Comiti', 'Idelchik', 'Erdim, Akgiray & Demir', 'KTA', 'Kuo & Nydegger', 'Ergun', 'Brauer', 'Fahien & Schriver', 'Jones & Krier']
    methods_dP_val.sort()
    assert methods_dP == methods_dP_val

    with pytest.raises(Exception):
        dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3, Method='Fail')

    v = voidage_Benyahia_Oneil(1E-3, 1E-2, .8)
    assert_allclose(v, 0.41395363849210065)
    v = voidage_Benyahia_Oneil_spherical(.001, .05)
    assert_allclose(v, 0.3906653157443224)
    v = voidage_Benyahia_Oneil_cylindrical(.01, .1, .6)
    assert_allclose(v, 0.38812523109607894)


def test_packed_tower():
    dP = dP_demister_dry_Setekleiv_Svendsen(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
    assert_allclose(dP, 320.3280788941329)
    dP = dP_demister_dry_Setekleiv_Svendsen_lit(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
    assert_allclose(dP, 209.083848658307)
    dP = voidage_experimental(m=126, rho=8000, D=1, H=1)
    assert_allclose(dP, 0.9799464771704212)

    S = specific_area_mesh(voidage=.934, d=3e-4)
    assert_allclose(S, 879.9999999999994)

    dP_dry = Stichlmair_dry(Vg=0.4, rhog=5., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7, C3=1)
    assert_allclose(dP_dry, 236.80904286559885)

    dP_wet = Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    assert_allclose(dP_wet, 539.8768237253518)

    Vg = Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    assert_allclose(Vg, 0.6394323542687361)

def test_piping():
    P1 = nearest_pipe(Di=0.021)
    assert_allclose(P1, (1, 0.02664, 0.0334, 0.0033799999999999998))
    P2 = nearest_pipe(Do=.273, schedule='5S')
    assert_allclose(P2, (10, 0.26630000000000004, 0.2731, 0.0034))



    g1s = gauge_from_t(.5, False, 'BWG'), gauge_from_t(0.005588, True)
    assert_allclose(g1s, (0.2, 5))
    g2s = gauge_from_t(0.5165, False, 'AWG'), gauge_from_t(0.00462026, True, 'AWG')
    assert_allclose(g2s, (0.2, 5))
    g3s = gauge_from_t(.4305, False, 'SWG'), gauge_from_t(0.0052578, True, 'SWG')
    assert_allclose(g3s, (0.2, 5))
    g4s = gauge_from_t(.005, False, 'MWG'), gauge_from_t(0.0003556, True, 'MWG')
    assert_allclose(g4s, (0.2, 5))
    g5s = gauge_from_t(.432, False, 'BSWG'), gauge_from_t(0.0053848, True, 'BSWG')
    assert_allclose(g5s, (0.2, 5))
    g6s = gauge_from_t(0.227, False, 'SSWG'), gauge_from_t(0.0051816, True, 'SSWG')
    assert_allclose(g6s, (1, 5))

    with pytest.raises(Exception):
        gauge_from_t(.5, False, 'FAIL') # Not in schedule
    with pytest.raises(Exception):
        gauge_from_t(0.02) # Too large

    g1 = gauge_from_t(0.002) # not in index; gauge 14, 2 mm
    g2 = gauge_from_t(0.00185) # not in index, gauge 15, within tol (10% default)
    # Limits between them are 0.0018288 and 0.0021082 m.
    g3 = gauge_from_t(0.00002)
    assert_allclose([g1, g2, g3], [14, 15, 0.004])



    t1s = t_from_gauge(.2, False, 'BWG'), t_from_gauge(5, True)
    assert_allclose(t1s, (0.5, 0.005588))

    t2s = t_from_gauge(.2, False, 'AWG'), t_from_gauge(5, True, 'AWG')
    assert_allclose(t2s, (0.5165, 0.00462026))

    t3s = t_from_gauge(.2, False, 'SWG'), t_from_gauge(5, True, 'SWG')
    assert_allclose(t3s, (0.4305, 0.0052578))

    t4s = t_from_gauge(.2, False, 'MWG'), t_from_gauge(5, True, 'MWG')
    assert_allclose(t4s, (0.005, 0.0003556))

    t5s = t_from_gauge(.2, False, 'BSWG'), t_from_gauge(5, True, 'BSWG')
    assert_allclose(t5s, (0.432, 0.0053848))

    t6s = t_from_gauge(1, False, 'SSWG'), t_from_gauge(5, True, 'SSWG')
    assert_allclose(t6s, (0.227, 0.0051816))

    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='FAIL')
    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='MWG')




def test_safety_valve():
    A = API520_round_size(1E-4)
    assert_allclose(A, 0.00012645136)
    assert 'E' == API526_letters[API526_A.index(API520_round_size(1E-4))]
    with pytest.raises(Exception):
        API520_round_size(1)

    C1, C2 = API520_C(1.35), API520_C(1.)
    Cs = [0.02669419967057233, 0.023945830445454768]
    assert_allclose([C1, C2], Cs)

    F2 = API520_F2(1.8, 1E6, 7E5)
    assert_allclose(F2, 0.8600724121105563)

    Kv_calcs = [API520_Kv(100), API520_Kv(4525), API520_Kv(1E5)]
    Kvs = [0.6157445891444229, 0.9639390032437682, 0.9973949303006829]
    assert_allclose(Kv_calcs, Kvs)

    KN = API520_N(1774700)
    assert_allclose(KN, 0.9490406958152466)

    with pytest.raises(Exception):
        API520_SH(593+273.15, 21E6)
    with pytest.raises(Exception):
        API520_SH(1000, 1066E3)
    # Test under 15 psig sat case
    assert API520_SH(320, 5E4) == 1

    from fluids.safety_valve import _KSH_Pa, _KSH_tempKs
    KSH_tot =  sum([API520_SH(T, P) for P in _KSH_Pa[:-1] for T in _KSH_tempKs])
    assert_allclose(229.93, KSH_tot)

    KW = [API520_W(1E6, 3E5), API520_W(1E6, 1E5)]
    assert_allclose(KW, [0.9511471848008564, 1])

    B_calc = [API520_B(1E6, 3E5), API520_B(1E6, 5E5), API520_B(1E6, 5E5, overpressure=.16), API520_B(1E6, 5E5, overpressure=.21)]
    Bs = [1, 0.7929945420944432, 0.94825439189912, 1]
    assert_allclose(B_calc, Bs)

    with pytest.raises(Exception):
        API520_B(1E6, 5E5, overpressure=.17)
    with pytest.raises(Exception):
        API520_B(1E6, 7E5, overpressure=.16)

    A1 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)
    A2 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1, Kc=1)
    As = [0.0036990460646834414, 0.004248358775943481]
    assert_allclose([A1, A2], As)

    A = API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1, Kc=1)
    assert_allclose(A, 0.0011034712423692733)

