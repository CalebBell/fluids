# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import assert_close, assert_close1d
import pytest

def test_core_misc():
    alpha = thermal_diffusivity(0.02, 1., 1000.)
    assert_close(alpha, 2e-05)

    c = c_ideal_gas(1.4, 303., 28.96)
    assert_close(c, 348.9820361755092, rtol=1e-05)


def test_core_dimensionless():
    Re = Reynolds(2.5, 0.25, 1.1613, 1.9E-5)
    assert_close(Re, 38200.65789473684)
    Re = Reynolds(2.5, 0.25, nu=1.636e-05)
    assert_close(Re, 38202.93398533008)
    with pytest.raises(Exception):
        Reynolds(2.5, 0.25, 1.1613)

    PeH = Peclet_heat(1.5, 2, 1000., 4000., 0.6)
    assert_close(PeH, 20000000.0)
    PeH = Peclet_heat(1.5, 2., alpha=1E-7)
    assert_close(PeH, 30000000.0)
    with pytest.raises(Exception):
        Peclet_heat(1.5, 2, 1000., 4000.)

    PeM = Peclet_mass(1.5, 2., 1E-9)
    assert_close(PeM, 3000000000)

    FH1 = Fourier_heat(1.5, 2., 1000., 4000., 0.6)
    FH2 = Fourier_heat(1.5, 2, alpha=1E-7)
    assert_close1d([FH1, FH2], [5.625e-08, 3.75e-08])
    with pytest.raises(Exception):
        Fourier_heat(1.5, 2, 1000., 4000.)

    FHM = Fourier_mass(1.5, 2.0, 1E-9)
    assert_close(FHM,  3.7500000000000005e-10)

    GZh1 = Graetz_heat(1.5, 0.25, 5., 800., 2200., 0.6)
    GZh2 = Graetz_heat(1.5, 0.25, 5, alpha=1E-7)
    assert_close1d([GZh1, GZh2], [55000.0, 187500.0])
    with pytest.raises(Exception):
        Graetz_heat(1.5, 0.25, 5, 800., 2200.)

    Sc1 = Schmidt(D=2E-6, mu=4.61E-6, rho=800.)
    Sc2 = Schmidt(D=1E-9, nu=6E-7)
    assert_close1d([Sc1, Sc2], [0.00288125, 600.])
    with pytest.raises(Exception):
        Schmidt(D=2E-6, mu=4.61E-6)

    Le1 = Lewis(D=22.6E-6, alpha=19.1E-6)
    Le2 = Lewis(D=22.6E-6, rho=800., k=.2, Cp=2200.)
    assert_close1d([Le1, Le2], [0.8451327433628318, 0.00502815768302494])
    with pytest.raises(Exception):
        Lewis(D=22.6E-6, rho=800., k=.2)

    We = Weber(0.18, 0.001, 900., 0.01)
    assert_close(We, 2.916)

    Ma = Mach(33., 330.)
    assert_close(Ma, 0.1)

    Kn = Knudsen(1e-10, .001)
    assert_close(Kn, 1e-07)

    Pr1 = Prandtl(Cp=1637., k=0.010, mu=4.61E-6)
    Pr2 = Prandtl(Cp=1637., k=0.010, nu=6.4E-7, rho=7.1)
    Pr3 = Prandtl(nu=6.3E-7, alpha=9E-7)
    assert_close1d([Pr1, Pr2, Pr3], [0.754657, 0.7438528, 0.7])
    with pytest.raises(Exception):
        Prandtl(Cp=1637., k=0.010)

    Gr1 = Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613, mu=1.9E-5)
    Gr2 = Grashof(L=0.9144, beta=0.000933, T1=378.2, T2=200., nu=1.636e-05)
    assert_close1d([Gr1, Gr2], [4656936556.178915, 4657491516.530312])
    with pytest.raises(Exception):
        Grashof(L=0.9144, beta=0.000933, T1=178.2, rho=1.1613)

    Bo1 = Bond(1000., 1.2, .0589, 2.)
    assert_close(Bo1, 665187.2339558573)

    Ra1 = Rayleigh(1.2, 4.6E9)
    assert_close(Ra1, 5520000000)

    Fr1 = Froude(1.83, 2., 1.63)
    Fr2 = Froude(1.83, L=2., squared=True)
    assert_close1d([Fr1, Fr2], [1.0135432593877318, 0.17074638128208924])

    St = Strouhal(8., 2., 4.)
    assert_close(St, 4.0)

    Nu1 = Nusselt(1000., 1.2, 300.)
    Nu2 = Nusselt(10000., .01, 4000.)
    assert_close1d([Nu1, Nu2], [4.0, 0.025])

    Sh = Sherwood(1000., 1.2, 300.)
    assert_close(Sh, 4.0)

    Bi1 = Biot(1000., 1.2, 300.)
    Bi2 = Biot(10000., .01, 4000.)
    assert_close1d([Bi1, Bi2], [4.0, 0.025])

    St1 = Stanton(5000., 5., 800., 2000.)
    assert_close(St1, 0.000625)

    Eu1 = Euler(1E5, 1000., 4.)
    assert_close(Eu1, 6.25)

    Ca1 = Cavitation(2E5, 1E4, 1000., 10.)
    assert_close(Ca1, 3.8)

    Ec1 = Eckert(10., 2000., 25.)
    assert_close(Ec1, 0.002)

    Ja1 = Jakob(4000., 2E6, 10.)
    assert_close(Ja1, 0.02)

    Po1 = Power_number(P=180., L=0.01, N=2.5, rho=800.)
    assert_close(Po1, 144000000)

    Cd1 = Drag(1000., 0.0001, 5., 2000.)
    assert_close(Cd1, 400)

    Ca1 = Capillary(1.2, 0.01, .1)
    assert_close(Ca1, 0.12)

    Ar1 = Archimedes(0.002, 0.2804, 2699.37, 4E-5)
    Ar2 = Archimedes(0.002, 2., 3000., 1E-3)
    assert_close1d([Ar1, Ar2], [37109.575890227665, 470.4053872])

    Oh1 = Ohnesorge(1E-4, 1000., 1E-3, 1E-1)
    assert_close(Oh1, 0.01)

    Su = Suratman(1E-4, 1000., 1E-3, 1E-1)
    assert_close(Su, 10000.0)


    BeL1 = Bejan_L(1E4, 1., 1E-3, 1E-6)
    assert_close(BeL1, 10000000000000)

    Bep1 = Bejan_p(1E4, 1., 1E-3, 1E-6)
    assert_close(Bep1, 10000000000000)

    Bo = Boiling(300., 3000., 800000.)
    assert_close(Bo, 1.25e-05)

    e_D1 = relative_roughness(0.0254)
    e_D2 = relative_roughness(0.5, 1E-4)
    assert_close1d([e_D1, e_D2], [5.9842519685039374e-05, 0.0002])

    Co = Confinement(0.001, 1077, 76.5, 4.27E-3)
    assert_close(Co, 0.6596978265315191)

    De = Dean(10000., 0.1, 0.4)
    assert_close(De, 5000.0)

    Stk = Stokes_number(V=0.9, Dp=1E-5, D=1E-3, rhop=1000., mu=1E-5)
    assert_close(Stk, 0.5)

    Hg = Hagen(Re=2610., fd=1.935235)
    assert_close(Hg, 6591507.17175)
    # Where fd was obtained from:
    def Hagen2(rho, D, mu):
        return rho*D**3/mu**2

    correct = Hagen2(rho=992., mu=653E-6, D=6.568E-3)*10000

    guess = Hagen(Re=2610., fd=1.935235)
    assert_close(correct, guess)

    Fr = Froude_densimetric(1.83, L=2., rho2=1.2, rho1=800., g=9.81)
    assert_close(Fr, 0.4134543386272418)
    Fr = Froude_densimetric(1.83, L=2., rho2=1.2, rho1=800, g=9.81, heavy=False)
    assert_close(Fr, 0.016013017679205096)

    Mo = Morton(1077.0, 76.5, 4.27E-3, 0.023)
    assert_close(Mo, 2.311183104430743e-07)


def test_core_misc2():
    mu1 = nu_mu_converter(998., nu=1.0E-6)
    nu1 = nu_mu_converter(998., mu=0.000998)
    assert_close1d([mu1, nu1], [0.000998, 1E-6])
    with pytest.raises(Exception):
        nu_mu_converter(990)
    with pytest.raises(Exception):
        nu_mu_converter(990, 0.000998, 1E-6)

    g1 = gravity(55., 1E4)
    assert_close(g1, 9.784151976863571)

    K = K_from_f(fd=0.018, L=100., D=.3)
    assert_close(K, 6.0)

    K = K_from_L_equiv(240.)
    assert_close(K, 3.6)

    L_D = L_equiv_from_K(3.6)
    assert_close(L_D, 240.)

    L = L_from_K(K=6., fd=0.018, D=.3)
    assert_close(L, 100)

    dP = dP_from_K(K=10., rho=1000., V=3.)
    assert_close(dP, 45000)

    head = head_from_K(K=10., V=1.5)
    assert_close(head, 1.1471807396001694)

    head = head_from_P(P=98066.5, rho=1000.)
    assert_close(head, 10.0)

    P = P_from_head(head=5., rho=800.)
    assert_close(P, 39226.6)

    fd = f_from_K(K=0.6, L=100., D=.3)
    assert_close(fd, 0.0018, rtol=1e-13)



#from fluids.core import C2K, K2C, F2C, C2F, F2K, K2F, C2R, K2R, F2R, R2C, R2K, R2F

# The following are tests which were deprecated from scipy
# but are still desired to be here
# Taken from scipy/constants/constants.py as in commit
# https://github.com/scipy/scipy/commit/4b7d325cd50e8828b06d628e69426a18283dc5b5
# Also from https://github.com/scipy/scipy/pull/5292
# by Gillu13  (Gilles Aouizerate)
# Copyright individual contributors to SciPy


def test_fahrenheit_to_celcius():
    assert_close(F2C(32.), 0)
    assert_close1d([F2C(32)], [0])


def test_celcius_to_kelvin():
    assert_close1d([C2K(0.)], [273.15])


def test_kelvin_to_celcius():
    assert_close1d([K2C(0.)], [-273.15])


def test_fahrenheit_to_kelvin():
    assert_close1d([F2K(32.), F2K(32)], [273.15, 273.15])


def test_kelvin_to_fahrenheit():
    assert_close1d([K2F(273.15), K2F(273.15)], [32, 32])


def test_celcius_to_fahrenheit():
    assert_close1d([C2F(0.)]*2, [32, 32])


def test_celcius_to_rankine():
    assert_close1d([C2R(0.), C2R(0.)], [491.67, 491.67], rtol=0., atol=1e-13)


def test_kelvin_to_rankine():
    assert_close1d([K2R(273.15), K2R(273.15)], [491.67, 491.67], rtol=0.,
                    atol=1e-13)


def test_fahrenheit_to_rankine():
    assert_close1d([F2R(32.), F2R(32.)], [491.67, 491.67], rtol=0., atol=1e-13)


def test_rankine_to_fahrenheit():
    assert_close1d([R2F(491.67), R2F(491.67)], [32., 32.], rtol=0.,
                    atol=1e-13)


def test_rankine_to_celcius():
    assert_close1d([R2C(491.67), R2C(491.67)], [0., 0.], rtol=0., atol=1e-13)


def test_rankine_to_kelvin():
    assert_close1d([R2K(491.67), R2K(0.)], [273.15, 0.], rtol=0., atol=1e-13)

