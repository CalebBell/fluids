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
from numpy.testing import assert_allclose
import pytest

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
    
    Su = Suratman(1E-4, 1000., 1E-3, 1E-1)
    assert_allclose(Su, 10000.0)
    

    BeL1 = Bejan_L(1E4, 1, 1E-3, 1E-6)
    assert_allclose(BeL1, 10000000000000)

    Bep1 = Bejan_p(1E4, 1, 1E-3, 1E-6)
    assert_allclose(Bep1, 10000000000000)
    
    Bo = Boiling(300, 3000, 800000)
    assert_allclose(Bo, 1.25e-05)

    e_D1 = relative_roughness(0.0254)
    e_D2 = relative_roughness(0.5, 1E-4)
    assert_allclose([e_D1, e_D2], [5.9842519685039374e-05, 0.0002])
    
    Co = Confinement(0.001, 1077, 76.5, 4.27E-3)
    assert_allclose(Co, 0.6596978265315191)
    
    De = Dean(10000, 0.1, 0.4)
    assert_allclose(De, 5000.0)

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

    K = K_from_f(fd=0.018, L=100., D=.3)
    assert_allclose(K, 6.0)

    K = K_from_L_equiv(240.)
    assert_allclose(K, 3.6)
    
    L_D = L_equiv_from_K(3.6)
    assert_allclose(L_D, 240.)

    dP = dP_from_K(K=10, rho=1000, V=3)
    assert_allclose(dP, 45000)

    head = head_from_K(K=10, V=1.5)
    assert_allclose(head, 1.1471807396001694)

    head = head_from_P(P=98066.5, rho=1000)
    assert_allclose(head, 10.0)

    P = P_from_head(head=5., rho=800.)
    assert_allclose(P, 39226.6)
