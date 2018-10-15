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
from numpy.testing import assert_allclose, assert_equal
import pytest
import numpy as np

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
    
    Stk = Stokes_number(V=0.9, Dp=1E-5, D=1E-3, rhop=1000, mu=1E-5)
    assert_allclose(Stk, 0.5)
    
    Hg = Hagen(Re=2610, fd=1.935235)
    assert_allclose(Hg, 6591507.17175)
    # Where fd was obtained from:
    def Hagen2(rho, D, mu):
        return rho*D**3/mu**2

    correct = Hagen2(rho=992., mu=653E-6, D=6.568E-3)*10000
    
    guess = Hagen(Re=2610, fd=1.935235)
    assert_allclose(correct, guess)
    
    Fr = Froude_densimetric(1.83, L=2., rho2=1.2, rho1=800, g=9.81)
    assert_allclose(Fr, 0.4134543386272418)
    Fr = Froude_densimetric(1.83, L=2., rho2=1.2, rho1=800, g=9.81, heavy=False)
    assert_allclose(Fr, 0.016013017679205096)

    Mo = Morton(1077.0, 76.5, 4.27E-3, 0.023)
    assert_allclose(Mo, 2.311183104430743e-07)
    

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
    
    L = L_from_K(K=6, fd=0.018, D=.3)
    assert_allclose(L, 100)

    dP = dP_from_K(K=10, rho=1000, V=3)
    assert_allclose(dP, 45000)

    head = head_from_K(K=10, V=1.5)
    assert_allclose(head, 1.1471807396001694)

    head = head_from_P(P=98066.5, rho=1000)
    assert_allclose(head, 10.0)

    P = P_from_head(head=5., rho=800.)
    assert_allclose(P, 39226.6)



from fluids.core import C2K, K2C, F2C, C2F, F2K, K2F, C2R, K2R, F2R, R2C, R2K, R2F

# The following are tests which were deprecated from scipy
# but are still desired to be here
# Taken from scipy/constants/constants.py as in commit 
# https://github.com/scipy/scipy/commit/4b7d325cd50e8828b06d628e69426a18283dc5b5
# Also from https://github.com/scipy/scipy/pull/5292
# by Gillu13  (Gilles Aouizerate)
# Copyright individual contributors to SciPy


def test_fahrenheit_to_celcius():
    assert_equal(F2C(32), 0)
    assert_equal(F2C([32, 32]), [0, 0])


def test_celcius_to_kelvin():
    assert_equal(C2K([0, 0]), [273.15, 273.15])


def test_kelvin_to_celcius():
    assert_equal(K2C([0, 0]), [-273.15, -273.15])


def test_fahrenheit_to_kelvin():
    assert_equal(F2K([32, 32]), [273.15, 273.15])


def test_kelvin_to_fahrenheit():
    assert_equal(K2F([273.15, 273.15]), [32, 32])


def test_celcius_to_fahrenheit():
    assert_equal(C2F([0, 0]), [32, 32])


def test_celcius_to_rankine():
    assert_allclose(C2R([0, 0]), [491.67, 491.67], rtol=0., atol=1e-13)


def test_kelvin_to_rankine():
    assert_allclose(K2R([273.15, 273.15]), [491.67, 491.67], rtol=0., 
                    atol=1e-13)


def test_fahrenheit_to_rankine():
    assert_allclose(F2R([32, 32]), [491.67, 491.67], rtol=0., atol=1e-13)


def test_rankine_to_fahrenheit():
    assert_allclose(R2F([491.67, 491.67]), [32., 32.], rtol=0., 
                    atol=1e-13)


def test_rankine_to_celcius():
    assert_allclose(R2C([491.67, 491.67]), [0., 0.], rtol=0., atol=1e-13)


def test_rankine_to_kelvin():
    assert_allclose(R2K([491.67, 0.]), [273.15, 0.], rtol=0., atol=1e-13)
    
    
def test_horner():
    from fluids.core import horner
    assert_allclose(horner([1.0, 3.0], 2.0), 5.0)
    assert_allclose(horner([3.0], 2.0), 3.0)
    
    
def test_interp():
    from fluids.core import interp
    # Real world test data
    a = [0.29916, 0.29947, 0.31239, 0.31901, 0.32658, 0.33729, 0.34202, 0.34706,
         0.35903, 0.36596, 0.37258, 0.38487, 0.38581, 0.40125, 0.40535, 0.41574, 
         0.42425, 0.43401, 0.44788, 0.45259, 0.47181, 0.47309, 0.49354, 0.49924, 
         0.51653, 0.5238, 0.53763, 0.54806, 0.55684, 0.57389, 0.58235, 0.59782,
         0.60156, 0.62265, 0.62649, 0.64948, 0.65099, 0.6687, 0.67587, 0.68855,
         0.69318, 0.70618, 0.71333, 0.72351, 0.74954, 0.74965]
    b = [0.164534, 0.164504, 0.163591, 0.163508, 0.163439, 0.162652, 0.162224, 
         0.161866, 0.161238, 0.160786, 0.160295, 0.15928, 0.159193, 0.157776,
         0.157467, 0.156517, 0.155323, 0.153835, 0.151862, 0.151154, 0.14784, 
         0.147613, 0.144052, 0.14305, 0.140107, 0.138981, 0.136794, 0.134737, 
         0.132847, 0.129303, 0.127637, 0.124758, 0.124006, 0.119269, 0.118449,
         0.113605, 0.113269, 0.108995, 0.107109, 0.103688, 0.102529, 0.099567,
         0.097791, 0.095055, 0.087681, 0.087648]
    
    xs = np.linspace(0.29, 0.76, 100)
    ys = [interp(xi, a, b) for xi in xs.tolist()]
    ys_numpy = np.interp(xs, a, b)
    assert_allclose(ys, ys_numpy, atol=1e-12, rtol=1e-11)
    
    
def test_splev():
    from fluids.core import splev as my_splev
    from scipy.interpolate import splev
    # Originally Dukler_XA_tck
    tck = [np.array([-2.4791105294648372, -2.4791105294648372, -2.4791105294648372, 
                               -2.4791105294648372, 0.14360803483759585, 1.7199938263676038, 
                               1.7199938263676038, 1.7199938263676038, 1.7199938263676038]),
                     np.array([0.21299880246561081, 0.16299733301915248, -0.042340970712679615, 
                               -1.9967836909384598, -2.9917366639619414, 0.0, 0.0, 0.0, 0.0]),
                     3]
    my_tck = [tck[0].tolist(), tck[1].tolist(), tck[2]]
    
    xs = np.linspace(-3, 2, 100)
    
    # test extrapolation
    ys_scipy = splev(xs, tck, ext=0)
    ys = my_splev(xs, my_tck, ext=0)
    assert_allclose(ys, ys_scipy)
    
    # test truncating to side values
    ys_scipy = splev(xs, tck, ext=3)
    ys = my_splev(xs, my_tck, ext=3)
    assert_allclose(ys, ys_scipy)

    
    # Test returning zeros for bad values
    ys_scipy = splev(xs, tck, ext=1)
    ys = my_splev(xs, my_tck, ext=1)
    assert_allclose(ys, ys_scipy)
    
    # Test raising an error when extrapolating is not allowed
    with pytest.raises(ValueError):
        my_splev(xs, my_tck, ext=2)
    with pytest.raises(ValueError):
        splev(xs, my_tck, ext=2)
    

def test_bisplev():
    from fluids.core import bisplev as my_bisplev
    from scipy.interpolate import bisplev
    
    tck = [np.array([0.0, 0.0, 0.0, 0.0, 0.0213694, 0.0552542, 0.144818, 
                                     0.347109, 0.743614, 0.743614, 0.743614, 0.743614]), 
           np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]),
           np.array([1.0001228445490002, 0.9988161050974387, 0.9987070557919563, 0.9979385859402731, 
                     0.9970983069823832, 0.96602540121758, 0.955136014969614, 0.9476842472211648, 
                     0.9351143114374392, 0.9059649602818451, 0.9218915266550902, 0.9086000082864022, 
                     0.8934758292610783, 0.8737960765592091, 0.83185251064324, 0.8664296734965998, 
                     0.8349705397843921, 0.809133298969704, 0.7752206120745123, 0.7344035693011536,
                     0.817047920445813, 0.7694560150930563, 0.7250979336267909, 0.6766754605968431, 
                     0.629304180420512, 0.7137237030611423, 0.6408238328161417, 0.5772000233279148, 
                     0.504889627280836, 0.440579886434288, 0.6239736474980684, 0.5273646894226224, 
                     0.43995388722059986, 0.34359277007615313, 0.26986439252143746, 0.5640689738382749, 
                     0.4540959882735219, 0.35278120580740957, 0.24364672351604122, 0.1606942128340308]),
           3, 1]
    my_tck = [tck[0].tolist(), tck[1].tolist(), tck[2].tolist(), tck[3], tck[4]]
    
    xs = np.linspace(0, 1, 50)
    zs = np.linspace(0, 1, 50)
    
    ys_scipy = bisplev(xs, zs, tck)
    ys = my_bisplev(xs, zs, my_tck)
    assert_allclose(ys, ys_scipy)

    ys_scipy = bisplev(0.5, .7, tck)
    ys = my_bisplev(.5, .7, my_tck)
    assert_allclose(ys, ys_scipy)
