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

from __future__ import division
from fluids import *
from fluids.numerics import assert_close, assert_close1d, linspace
import pytest


def test_Thom():
#    >>> from sympy import *
#    >>> x, rhol, rhog, mug, mul = symbols('x, rhol, rhog, mug, mul')
#    >>> Z = (rhol/rhog)**Rational(555,1000)*(mug/mul)**Rational(111,1000)
#    >>> gamma = Z**1.6
#    >>> alpha = (gamma*x/(1 + x*(gamma-1)))
#    >>> alpha
#    x*((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6/(x*(((mug/mul)**(111/1000)*(rhol/rhog)**(111/200))**1.6 - 1) + 1)
#    >>> alpha.subs([(x, .4), (rhol, 800), (rhog, 2.5), (mul, 1E-3), (mug, 1E-5)])
#    0.980138792146901


    assert_close(Thom(.4, 800, 2.5, 1E-3, 1E-5), 0.9801482164042417)


def test_Zivi():
    assert_close(Zivi(.4, 800, 2.5), 0.9689339909056356)


def test_Smith():
    assert_close(Smith(.4, 800, 2.5), 0.959981235534199)

    # Quick test function, to ensure results are the same regardless of
    # the form of the expression
    def Smith2(x, rhol, rhog):
        K = 0.4
        first = 1 + rhog/rhol*K*(1/x-1)
        second = rhog/rhol*(1-K)*(1/x-1)
        third = ((rhol/rhog + K*(1/x-1))/(1 + K*(1/x -1)))**0.5
        return (first + second*third)**-1

    alpha_1 = [Smith(i, 800, 2.5) for i in linspace(1E-9,.99)]
    alpha_2 = [Smith2(i, 800, 2.5) for i in linspace(1E-9, .99)]
    assert_close1d(alpha_1, alpha_2)


def test_Fauske():
    assert_close(Fauske(.4, 800, 2.5), 0.9226347262627932)


def test_Chisholm_voidage():
    assert_close(Chisholm_voidage(.4, 800, 2.5), 0.949525900374774)


def test_Turner_Wallis():
    assert_close(Turner_Wallis(.4, 800, 2.5, 1E-3, 1E-5), 0.8384824581634625)


### Section 2

def test_homogeneous():
    assert_close(homogeneous(.4, 800, 2.5), 0.995334370139969)

    # 1./(1. + (1-x)/x*(rhog/rhol))

test_homogeneous()


def test_Chisholm_Armand():
    assert_close(Chisholm_Armand(.4, 800, 2.5), 0.9357814394262114)


def test_Armand():
    assert_close(Armand(.4, 800, 2.5), 0.8291135303265941)


def test_Nishino_Yamazaki():
    assert_close(Nishino_Yamazaki(.4, 800, 2.5), 0.931694583962682)


def test_Guzhov():
    assert_close(Guzhov(.4, 800, 2.5, 1, .3), 0.7626030108534588)


def test_Kawahara():
    alphas_calc = [Kawahara(.4, 800, 2.5, D) for D in [0.001, 100E-6, 1E-7]]
    alphas_exp = [0.8291135303265941, 0.9276148194410238, 0.8952146812696503]
    assert_close1d(alphas_calc, alphas_exp)


### Drift flux models


def test_Lockhart_Martinelli_Xtt():
    assert_close(Lockhart_Martinelli_Xtt(0.4, 800, 2.5, 1E-3, 1E-5), 0.12761659240532292)
    assert_close(Lockhart_Martinelli_Xtt(0.4, 800, 2.5, 1E-3, 1E-5, n=0.2), 0.12761659240532292)


def test_Baroczy():
    assert_close(Baroczy(.4, 800, 2.5, 1E-3, 1E-5), 0.9453544598460807)


def test_Tandon_Varma_Gupta():
    alphas_calc = [Tandon_Varma_Gupta(.4, 800, 2.5, 1E-3, 1E-5, m, 0.3) for m in [1, .1]]
    assert_close1d(alphas_calc, [0.9228265670341428, 0.8799794756817589])


def test_Harms():
    assert_close(Harms(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3), 0.9653289762907554)


def test_Domanski_Didion():
    assert_close(Domanski_Didion(.4, 800, 2.5, 1E-3, 1E-5), 0.9355795597059169)
    assert_close(Domanski_Didion(.002, 800, 2.5, 1E-3, 1E-5), 0.32567078492010837)


def test_Graham():
    assert_close(Graham(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3), 0.6403336287530644)
    assert 0 == Graham(.4, 800, 2.5, 1E-3, 1E-5, m=.001, D=0.3)


def test_Yashar():
    assert_close(Yashar(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3), 0.7934893185789146)


def test_Huq_Loth():
    assert_close(Huq_Loth(.4, 800, 2.5), 0.9593868838476147)


def test_Kopte_Newell_Chato():
    assert_close(Kopte_Newell_Chato(.4, 800, 2.5, 1E-3, 1E-5, m=1, D=0.3), 0.6864466770087425)
    assert_close(Kopte_Newell_Chato(.4, 800, 2.5, 1E-3, 1E-5, m=.01, D=0.3), 0.995334370139969)


def test_Steiner():
    assert_close(Steiner(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3), 0.895950181381335)


def test_Rouhani_1():
    assert_close(Rouhani_1(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3), 0.8588420244136714)


def test_Rouhani_2():
    assert_close(Rouhani_2(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3), 0.44819733138968865)


def test_Nicklin_Wilkes_Davidson():
    assert_close(Nicklin_Wilkes_Davidson(0.4, 800., 2.5, m=1, D=0.3), 0.6798826626721431)


def test_Gregory_Scott():
    assert_close(Gregory_Scott(0.4, 800., 2.5), 0.8364154370924108)


def test_Dix():
    assert_close(Dix(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3), 0.8268737961156514)


def test_Sun_Duffey_Peng():
    assert_close(Sun_Duffey_Peng(0.4, 800., 2.5, sigma=0.02, m=1, D=0.3, P=1E5, Pc=7E6), 0.7696546506515833)


def test_Woldesemayat_Ghajar():
    assert_close(Woldesemayat_Ghajar(0.4, 800., 2.5, sigma=0.2, m=1, D=0.3, P=1E6, angle=45.0), 0.7640815513429202)


def test_Xu_Fang_voidage():
    assert_close(Xu_Fang_voidage(0.4, 800., 2.5, m=1, D=0.3), 0.9414660089942093)


def test_liquid_gas_voidage():
    voidage = liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05)
    assert_close(voidage, 0.9744097632663492)

    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, P=1e5, Pc=1e7)
    for m in liquid_gas_voidage_methods(**kwargs):
        liquid_gas_voidage(Method=m, **kwargs)

    with pytest.raises(Exception):
        liquid_gas_voidage(Method='BADMETHOD', **kwargs)

    assert len(liquid_gas_voidage_methods(**kwargs)) == 29

def test_density_two_phase():
    assert_close(density_two_phase(.4, 800.0, 2.5), 481.0)


def test_two_phase_voidage_experimental():
    alpha = two_phase_voidage_experimental(481.0, 800.0, 2.5)
    assert_close(alpha, 0.4)


def test_Beattie_Whalley():
    mu = Beattie_Whalley(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2)
    assert_close(mu, 1.7363806909512365e-05)


def test_McAdams():
    mu = McAdams(x=0.4, mul=1E-3, mug=1E-5)
    assert_close(mu, 2.4630541871921184e-05)


def test_Cicchitti():
    mu = Cicchitti(x=0.4, mul=1E-3, mug=1E-5)
    assert_close(mu, 0.000604)


def test_Lin_Kwok():
    mu = Lin_Kwok(x=0.4, mul=1E-3, mug=1E-5)
    assert_close(mu, 3.515119398126066e-05)

def test_Fourar_Bories():
    mu = Fourar_Bories(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2)
    assert_close(mu, 2.127617150298565e-05)


def tets_Duckler():
    mu =  Duckler(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2)
    assert_close(mu, 1.2092040385066917e-05)


    def Duckler1(x, mul, mug, rhol, rhog):
        # Effective property models for homogeneous two-phase flows.
        # different formulation
        rhom = 1./(x/rhog + (1. - x)/rhol)
        return rhom*(x*(mug/rhog) + (1. - x)*mul/rhol)


    mu = Duckler1(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2)
    assert_close(mu, 1.2092040385066917e-05)


def test_gas_liquid_viscosity():
    mu = gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5)
    assert_close(2.4630541871921184e-05, mu)

    mu = gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2, Method='Duckler')
    assert_close(mu, 1.2092040385066917e-05)

    simple_methods = gas_liquid_viscosity_methods()
    assert list(sorted(simple_methods)) == list(sorted(['McAdams', 'Cicchitti', 'Lin Kwok']))

    all_methods = gas_liquid_viscosity_methods(rhol=1000.0, rhog=2.)
    all_methods_expect = ['Beattie Whalley', 'Fourar Bories', 'Duckler', 'McAdams', 'Cicchitti', 'Lin Kwok']
    assert list(sorted(all_methods)) == list(sorted(all_methods_expect))

    for m in all_methods_expect:
        gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850.0, rhog=1.2, Method=m)



    with pytest.raises(Exception):
        gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, Method='NOTAMETHOD')