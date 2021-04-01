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
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d
import pytest
from math import log, pi


def test_isothermal_work_compression():
    assert_close(isothermal_work_compression(1E5, 1E6, 300.0), 5743.425357533477, rtol=1e-05)


def test_isentropic_work_compression():
    dH = isentropic_work_compression(P1=1E5, P2=1E6, T1=300.0, k=1.4, eta=1)
    assert_close(dH, 8125.161295388634, rtol=1e-05)

    dH = isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78)
    assert_close(dH, 10416.873455626454, rtol=1e-05)

    dH = isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78, Z=0.9)
    assert_close(dH, 9375.186110063809, rtol=1e-05)

    # Other solutions - P1, P2, and eta
    P1 = isentropic_work_compression(W=9375.186110063809, P2=1E6, T1=300., k=1.4, eta=0.78, Z=0.9)
    assert_close(P1, 1E5, rtol=1E-5)

    P2 = isentropic_work_compression(W=9375.186110063809, P1=1E5, T1=300., k=1.4, eta=0.78, Z=0.9)
    assert_close(P2, 1E6, rtol=1E-5)

    eta = isentropic_work_compression(W=9375.186110063809, P1=1E5, P2=1E6, T1=300, k=1.4, Z=0.9, eta=None)
    assert_close(eta, 0.78, rtol=1E-5)

    with pytest.raises(Exception):
        isentropic_work_compression(P1=1E5, P2=1E6, k=1.4, T1=None)


def test_isentropic_T_rise_compression():
    T2 = isentropic_T_rise_compression(286.8, 54050.0, 432400., 1.4)
    assert_close(T2, 519.5230938217768, rtol=1e-05)

    T2 = isentropic_T_rise_compression(286.8, 54050, 432400, 1.4, eta=0.78)
    assert_close(T2, 585.1629407971498, rtol=1e-05)

    # Test against the simpler formula for eta=1:
    # T2 = T2*(P2/P1)^((k-1)/k)
    T2_ideal = 286.8*((432400/54050)**((1.4-1)/1.4))
    assert_close(T2_ideal, 519.5230938217768, rtol=1e-05)


def test_isentropic_efficiency():
    eta_s = isentropic_efficiency(1E5, 1E6, 1.4, eta_p=0.78)
    assert_close(eta_s, 0.7027614191263858)
    eta_p = isentropic_efficiency(1E5, 1E6, 1.4, eta_s=0.7027614191263858)
    assert_close(eta_p, 0.78)

    with pytest.raises(Exception):
        isentropic_efficiency(1E5, 1E6, 1.4)

    # Example 7.6 of the reference:
    eta_s = isentropic_efficiency(1E5, 3E5, 1.4, eta_p=0.75)
    assert_close(eta_s, 0.7095085923615653)
    eta_p =  isentropic_efficiency(1E5, 3E5, 1.4, eta_s=eta_s)
    assert_close(eta_p, 0.75)


def test_polytropic_exponent():
    assert_close(polytropic_exponent(1.4, eta_p=0.78), 1.5780346820809246)
    assert_close(polytropic_exponent(1.4, n=1.5780346820809246), 0.78)
    with pytest.raises(Exception):
        polytropic_exponent(1.4)


def test_compressible():
    T = T_critical_flow(473., 1.289)
    assert_close(T, 413.2809086937528)

    P = P_critical_flow(1400000., 1.289)
    assert_close(P, 766812.9022792266)

    assert not is_critical_flow(670E3, 532E3, 1.11)
    assert is_critical_flow(670E3, 101E3, 1.11)

    SE = stagnation_energy(125.)
    assert_close(SE, 7812.5)

    PST = P_stagnation(54050., 255.7, 286.8, 1.4)
    assert_close(PST, 80772.80495900588)

    Tst = T_stagnation(286.8, 54050., 54050*8., 1.4)
    assert_close(Tst, 519.5230938217768)

    Tstid = T_stagnation_ideal(255.7, 250., 1005.)
    assert_close(Tstid, 286.79452736318405)


def test_Panhandle_A():
    # Example 7-18 Gas of Crane TP 410M
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 42.56082051195928

    # Test all combinations of relevant missing inputs
    assert_close(Panhandle_A(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Panhandle_A(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Panhandle_A(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Panhandle_A(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Panhandle_A(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Panhandle_A(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)

    # Sample problem from "Natural Gas Pipeline Flow Calculations" by "Harlan H. Bengtson"
    Q_panhandle = Panhandle_A(SG=0.65, Tavg=F2K(80), Ts=F2K(60), Ps=14.7*psi, L=500*foot, D=12*inch, P1=510*psi,
                          P2=490*psi, Zavg=0.919, E=0.92)
    mmscfd = Q_panhandle*day/foot**3/1e6
    assert_close(mmscfd, 401.3019451856126, rtol=1e-12)


def test_Panhandle_B():
    # Example 7-18 Gas of Crane TP 410M
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 42.35366178004172

    # Test all combinations of relevant missing inputs
    assert_close(Panhandle_B(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Panhandle_B(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Panhandle_B(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Panhandle_B(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Panhandle_B(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Panhandle_B(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)


def test_Weymouth():
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 32.07729055913029
    assert_close(Weymouth(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Weymouth(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Weymouth(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Weymouth(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Weymouth(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Weymouth(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)

    Q_Weymouth = Weymouth(SG=0.65, Tavg=F2K(80), Ts=F2K(60), Ps=14.7*psi, L=500*foot, D=12*inch, P1=510*psi,
                              P2=490*psi, Zavg=0.919, E=0.92)
    mmscfd = Q_Weymouth*day/foot**3/1e6
    assert_close(mmscfd, 272.5879686092862, rtol=1e-12)

def test_Spitzglass_high():

    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 29.42670246281681
    assert_close(Spitzglass_high(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Spitzglass_high(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Spitzglass_high(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Spitzglass_high(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Spitzglass_high(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Spitzglass_high(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)


def test_Spitzglass_low():
    D = 0.154051
    P1 = 6720.3199
    P2 = 0.0
    L = 54.864
    SG = 0.6
    Tavg = 288.7
    Q = 0.9488775242530617
    assert_close(Spitzglass_low(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Spitzglass_low(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Spitzglass_low(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2, atol=1E-10)
    assert_close(Spitzglass_low(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Spitzglass_low(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Spitzglass_low(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)


def test_Oliphant():
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 28.851535408143057
    assert_close(Oliphant(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Oliphant(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Oliphant(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Oliphant(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Oliphant(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Oliphant(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)


def test_Fritzsche():
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 39.421535157535565
    assert_close(Fritzsche(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_close(Fritzsche(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_close(Fritzsche(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_close(Fritzsche(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_close(Fritzsche(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Fritzsche(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)


def test_Muller():
    D = 0.340
    mu = 1E-5
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 60.45796698148663
    assert_close(Muller(D=D, P1=P1, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg), Q)
    assert_close(Muller(D=D, Q=Q, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg), P1)
    assert_close(Muller(D=D, Q=Q, P1=P1, L=L, SG=SG, mu=mu, Tavg=Tavg), P2)
    assert_close(Muller(D=D, Q=Q, P1=P1, P2=P2, SG=SG, mu=mu, Tavg=Tavg), L)
    assert_close(Muller(L=L, Q=Q, P1=P1, P2=P2, SG=SG, mu=mu, Tavg=Tavg), D)

    with pytest.raises(Exception):
        Muller(D=D, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg)

def test_IGT():
    D = 0.340
    mu = 1E-5
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 48.92351786788815
    assert_close(IGT(D=D, P1=P1, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg), Q)
    assert_close(IGT(D=D, Q=Q, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg), P1)
    assert_close(IGT(D=D, Q=Q, P1=P1, L=L, SG=SG, mu=mu, Tavg=Tavg), P2)
    assert_close(IGT(D=D, Q=Q, P1=P1, P2=P2, SG=SG, mu=mu, Tavg=Tavg), L)
    assert_close(IGT(L=L, Q=Q, P1=P1, P2=P2, SG=SG, mu=mu, Tavg=Tavg), D)

    with pytest.raises(Exception):
        IGT(D=D, P2=P2, L=L, SG=SG, mu=mu, Tavg=Tavg)


def test_isothermal_gas():
    mcalc = isothermal_gas(11.3, 0.00185, P1=1E6, P2=9E5, L=1000., D=0.5)
    assert_close(mcalc, 145.484757264)
    assert_close(isothermal_gas(11.3, 0.00185, P1=1E6, P2=9E5, m=145.484757264, D=0.5), 1000)
    assert_close(isothermal_gas(11.3, 0.00185, P2=9E5, m=145.484757264, L=1000., D=0.5), 1E6)
    assert_close(isothermal_gas(11.3, 0.00185, P1=1E6, m=145.484757264, L=1000., D=0.5), 9E5)
    assert_close(isothermal_gas(11.3, 0.00185, P1=1E6, P2=9E5, m=145.484757264, L=1000.), 0.5)

    with pytest.raises(Exception):
        isothermal_gas(11.3, 0.00185, P1=1E6, P2=9E5, L=1000.)
    with pytest.raises(Exception):
        isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=1E5, L=1000., D=0.5)
    with pytest.raises(Exception):
        isothermal_gas(rho=11.3, fd=0.00185, P2=1E6, P1=9E5, L=1000., D=0.5)

    # Newton can't converge, need a bounded solver
    P1 = isothermal_gas(rho=11.3, fd=0.00185, m=390., P2=9E5, L=1000., D=0.5)
    assert_close(P1, 2298973.786533209)

    # Case where the desired flow is greater than the choked flow's rate
    with pytest.raises(Exception):
        isothermal_gas(rho=11.3, fd=0.00185, m=400, P2=9E5, L=1000., D=0.5)

    # test the case where the ideal gas assumption is baked in:

    rho = 10.75342009105268 # Chemical('nitrogen', P=(1E6+9E5)/2).rho
    m1 = isothermal_gas(rho=rho, fd=0.00185, P1=1E6, P2=9E5, L=1000., D=0.5)
    assert_close(m1, 141.92260633059334)

    # They are fairly similar
    fd = 0.00185
    P1 = 1E6
    P2 = 9E5
    L = 1000
    D = 0.5
    T = 298.15
    # from scipy.constants import R
    # from thermo import property_molar_to_mass, Chemical, pi, log
    R = 296.8029514446658 # property_molar_to_mass(R, Chemical('nitrogen').MW)
    m2 = (pi**2/16*D**4/(R*T*(fd*L/D + 2*log(P1/P2)))*(P1**2-P2**2))**0.5
    assert_close(m2, 145.48786057477403)


def test_P_isothermal_critical_flow():
    P2_max = P_isothermal_critical_flow(P=1E6, fd=0.00185, L=1000., D=0.5)
    assert_close(P2_max, 389699.7317645518)