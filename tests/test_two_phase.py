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
from math import log, log10
from random import uniform
from fluids import *
from fluids.numerics import assert_close, assert_close1d
import pytest

def log_uniform(low, high):
    return 10**uniform(log10(low), log10(high))


def test_Beggs_Brill():
    kwargs = dict(m=1.163125, x=0.30370768404083825, rhol=613.8,
                      rhog=141.3, sigma=0.028, D=0.077927, angle=90.0,
                      mul=0.0005, mug=2E-5, P=119E5, roughness=1.8E-6, L=100.0,
                      acceleration=True)
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 384066.2949427367)
    kwargs['angle'] = 45
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 289002.94186339306)
    kwargs['x'] = 0.6
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 220672.4414664162)
    kwargs['x'] = 0.9
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 240589.47045109692)
    kwargs['angle'] = 0
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 4310.718513863349)
    kwargs['x'] = 1e-7
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 1386.362401988662)
    kwargs['angle'] = -15
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, -154405.0395988586)

    kwargs['m'] = 100
    kwargs['x'] = 0.3
    kwargs['angle'] = 0
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 15382421.32990976)

    kwargs['angle'] = 10
    dP = Beggs_Brill(**kwargs)
    assert_close(dP, 15439041.350531114)

    kwargs = {'rhol': 2250.004745138356, 'rhog': 58.12314177331951, 'L': 111.74530635808999, 'sigma': 0.5871528902653206, 'P': 9587894383.375906, 'm': 0.005043652829299738, 'roughness': 0.07803567727862296, 'x': 0.529765332332195, 'mug': 1.134544741297285e-06, 'mul': 0.12943468582774414, 'D': 1.9772420342193617, 'angle': -77.18096944813536}
    dP = Beggs_Brill(**kwargs)
    # Check this calculation works - S gets too large, overflows in this region

@pytest.mark.fuzz
@pytest.mark.slow
def test_fuzz_Beggs_Brill():
    for i in range(250):
        m = log_uniform(1e-5, 100)
        x = uniform(0, 1)
        rhol = log_uniform(100, 4000)
        rhog = log_uniform(0.01, 200)
        sigma = log_uniform(1e-3, 1)
        D = log_uniform(1e-5, 5)
        angle = uniform(-90, 90)
        mul = log_uniform(1e-5, 1)
        mug = log_uniform(5e-7, 1e-3)
        P = log_uniform(1E8, 1e10)
        roughness = log_uniform(1e-5, D-1e-10)
        L = uniform(0, 1000)
        kwargs = dict(m=m, x=x, rhol=rhol, rhog=rhog, sigma=sigma, D=D, angle=angle, mul=mul, mug=mug, P=P, roughness=roughness, L=L)
        Beggs_Brill(**kwargs)


def test_Friedel():
    kwargs = dict(m=10, x=0.9, rhol=950., rhog=1.4, mul=1E-3, mug=1E-5, sigma=0.02, D=0.3, roughness=0., L=1.)
    dP = Friedel(**kwargs)
    dP_expect = 274.21322116878406
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Friedel(**kwargs)
    assert_close(dP, dP_expect*10)

    # Example 4 in [6]_:
    dP = Friedel(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, roughness=0., L=1.)
    assert_close(dP, 738.6500525002241)
    # 730 is the result in [1]_; they use the Blassius equation instead for friction
    # the multiplier was calculated to be 38.871 vs 38.64 in [6]_


def test_Gronnerud():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Gronnerud(**kwargs)
    dP_expect = 384.125411444741
    assert_close(dP, 384.125411444741)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Gronnerud(**kwargs)
    assert_close(dP, dP_expect*10)

    dP = Gronnerud(m=5, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 26650.676132410194)

def test_Chisholm():
    # Gamma < 28, G< 600
    dP = Chisholm(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 1084.1489922923736)

    # Gamma < 28, G > 600
    dP = Chisholm(m=2, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 7081.89630764668)

    # Gamma <= 9.5, G_tp <= 500
    dP = Chisholm(m=.6, x=0.1, rhol=915., rhog=30, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 222.36274920522493)

    # Gamma <= 9.5, G_tp < 1900:
    dP = Chisholm(m=2, x=0.1, rhol=915., rhog=30, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 1107.9944943816388)

    # Gamma <= 9.5, G_tp > 1900:
    dP = Chisholm(m=5, x=0.1, rhol=915., rhog=30, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 3414.1123536958203)

    dP = Chisholm(m=1, x=0.1, rhol=915., rhog=0.1, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 8743.742915625126)

    # Roughness correction
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=1E-4, L=1, rough_correction=True)
    dP = Chisholm(**kwargs)
    dP_expect = 846.6778299960783
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Chisholm(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Baroczy_Chisholm():
    # Gamma < 28, G< 600
    dP = Baroczy_Chisholm(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 1084.1489922923736)

    # Gamma <= 9.5, G_tp > 1900:
    dP = Baroczy_Chisholm(m=5, x=0.1, rhol=915., rhog=30, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 3414.1123536958203)

    kwargs = dict(m=1, x=0.1, rhol=915., rhog=0.1, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Baroczy_Chisholm(**kwargs)
    dP_expect = 8743.742915625126
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Baroczy_Chisholm(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Muller_Steinhagen_Heck():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Muller_Steinhagen_Heck(**kwargs)
    dP_expect = 793.4465457435081
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Muller_Steinhagen_Heck(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Lombardi_Pedrocchi():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, sigma=0.045, D=0.05, L=1.0)
    dP = Lombardi_Pedrocchi(**kwargs)
    dP_expect = 1567.328374498781
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Lombardi_Pedrocchi(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Theissing():
    dP = Theissing(m=0.6, x=.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 497.6156370699528)

    # Test x=1, x=0
    dP = Theissing(m=0.6, x=1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 4012.248776469056)

    kwargs = dict(m=0.6, x=0, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Theissing(**kwargs)
    dP_expect = 19.00276790390895
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Theissing(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Jung_Radermacher():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Jung_Radermacher(**kwargs)
    dP_expect = 552.068612372557
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Jung_Radermacher(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Tran():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    dP = Tran(**kwargs)
    dP_expect = 423.2563312951231
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Tran(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Chen_Friedel():
    dP = Chen_Friedel(m=.0005, x=0.9, rhol=950., rhog=1.4, mul=1E-3, mug=1E-5, sigma=0.02, D=0.003, roughness=0.0, L=1.0)
    assert_close(dP, 6249.247540588871)

    kwargs = dict(m=.1, x=0.9, rhol=950., rhog=1.4, mul=1E-3, mug=1E-5, sigma=0.02, D=0.03, roughness=0.0, L=1.0)
    dP = Chen_Friedel(**kwargs)
    dP_expect = 3541.7714973093725
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Chen_Friedel(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Zhang_Webb():
    kwargs = dict(m=0.6, x=0.1, rhol=915., mul=180E-6, P=2E5, Pc=4055000, D=0.05, roughness=0.0, L=1.0)
    dP = Zhang_Webb(**kwargs)
    dP_expect = 712.0999804205619
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Zhang_Webb(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Bankoff():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Bankoff(**kwargs)
    dP_expect = 4746.059442453398
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Bankoff(**kwargs)
    assert_close(dP, dP_expect*10)

def test_Xu_Fang():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    dP = Xu_Fang(**kwargs)
    dP_expect = 604.0595632116267
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Xu_Fang(**kwargs)
    assert_close(dP, dP_expect*10)

def test_Yu_France():
    kwargs = dict(m=0.6, x=.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Yu_France(**kwargs)
    dP_expect = 1146.983322553957
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Yu_France(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Wang_Chiang_Lu():
    dP = Wang_Chiang_Lu(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 448.29981978639154)

    kwargs = dict(m=0.1, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    dP = Wang_Chiang_Lu(**kwargs)
    dP_expect = 3.3087255464765417
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Wang_Chiang_Lu(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Hwang_Kim():
    kwargs = dict(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    dP = Hwang_Kim(**kwargs)
    dP_expect = 798.302774184557
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Hwang_Kim(**kwargs)
    assert_close(dP, dP_expect*10)



def test_Zhang_Hibiki_Mishima():
    dP = Zhang_Hibiki_Mishima(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    assert_close(dP, 444.9718476894804)

    dP = Zhang_Hibiki_Mishima(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1, flowtype='adiabatic gas')
    assert_close(dP, 1109.1976111277042)

    kwargs = dict(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1, flowtype='flow boiling')
    dP = Zhang_Hibiki_Mishima(**kwargs)
    dP_expect = 770.0975665928916
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Zhang_Hibiki_Mishima(**kwargs)
    assert_close(dP, dP_expect*10)

    with pytest.raises(Exception):
        Zhang_Hibiki_Mishima(m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1, flowtype='BADMETHOD')


def test_Kim_Mudawar():
    # turbulent-turbulent
    dP = Kim_Mudawar(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)
    assert_close(dP, 840.4137796786074)

    # Re_l >= Re_c and Re_g < Re_c
    dP = Kim_Mudawar(m=0.6, x=0.001, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)
    assert_close(dP, 68.61594310455612)

    # Re_l < Re_c and Re_g >= Re_c:
    dP = Kim_Mudawar(m=0.6, x=0.99, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)
    assert_close(dP, 5381.335846128011)

    # laminar-laminar
    dP = Kim_Mudawar(m=0.1, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.5, L=1.0)
    assert_close(dP, 0.005121833671658875)

    # Test friction Re < 20000
    kwargs = dict(m=0.1, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)
    dP = Kim_Mudawar(**kwargs)
    dP_expect = 33.74875494223592
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Kim_Mudawar(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Lockhart_Martinelli():
    dP = Lockhart_Martinelli(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, L=1.0)
    assert_close(dP, 716.4695654888484)

    # laminar-laminar
    dP = Lockhart_Martinelli(m=0.1, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=1, L=1.0)
    assert_close(dP, 9.06478815533121e-06)

    # Liquid laminar, gas turbulent
    dP = Lockhart_Martinelli(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=2, L=1.0)
    assert_close(dP, 8.654579552636214e-06)

    # Gas laminar, liquid turbulent
    kwargs = dict(m=0.6, x=0.05, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=2, L=1.0)
    dP = Lockhart_Martinelli(**kwargs)
    dP_expect = 4.56627076018814e-06
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Lockhart_Martinelli(**kwargs)
    assert_close(dP, dP_expect*10)


def test_Mishima_Hibiki():
    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    dP = Mishima_Hibiki(**kwargs)
    dP_expect = 732.4268200606265
    assert_close(dP, dP_expect)

    # Internal consistency for length dependence
    kwargs['L'] *= 10
    dP = Mishima_Hibiki(**kwargs)
    assert_close(dP, dP_expect*10)



def test_two_phase_dP():

    # TODO; Delete two_phase_dP method calls
    # Case 0
    assert ['Lombardi_Pedrocchi'] == two_phase_dP_methods(10, 0.7, 1000, 0.1, rhog=1.2, sigma=0.02)
    # Case 5
    assert ['Zhang_Webb'] == two_phase_dP_methods(10, 0.7, 1000, 0.1, mul=1E-3, P=1E5, Pc=1E6,)
    # Case 1,2

    expect = ['Jung_Radermacher', 'Muller_Steinhagen_Heck', 'Baroczy_Chisholm', 'Yu_France', 'Wang_Chiang_Lu', 'Theissing', 'Chisholm rough', 'Chisholm', 'Gronnerud', 'Lockhart_Martinelli', 'Bankoff']
    assert sorted(expect) == sorted(two_phase_dP_methods(10, 0.7, 1000, 0.1, rhog=1.2, mul=1E-3, mug=1E-6))

    # Case 3, 4; drags in 5, 1, 2
    expect = ['Zhang_Hibiki_Mishima adiabatic gas', 'Kim_Mudawar', 'Friedel', 'Jung_Radermacher', 'Hwang_Kim', 'Muller_Steinhagen_Heck', 'Baroczy_Chisholm', 'Tran', 'Yu_France', 'Zhang_Hibiki_Mishima flow boiling', 'Xu_Fang', 'Wang_Chiang_Lu', 'Theissing', 'Chisholm rough', 'Chisholm', 'Mishima_Hibiki', 'Gronnerud', 'Chen_Friedel', 'Lombardi_Pedrocchi', 'Zhang_Hibiki_Mishima', 'Lockhart_Martinelli', 'Bankoff']
    assert sorted(expect) == sorted(two_phase_dP_methods(10, 0.7, 1000, 0.1, rhog=1.2, mul=1E-3, mug=1E-6, sigma=0.014,))

    assert 24 == len(two_phase_dP_methods(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1, angle=30.0, roughness=1e-4, P=1e5, Pc=1e6))

    kwargs = dict(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1, angle=30.0, roughness=1e-4, P=1e5, Pc=1e6)
    for m in two_phase_dP_methods(**kwargs):
        two_phase_dP(Method=m, **kwargs)

    # Final method attempt Lombardi_Pedrocchi
    dP = two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, sigma=0.045, D=0.05, L=1.0)
    assert_close(dP, 1567.328374498781)

    # Second method attempt Zhang_Webb
    dP = two_phase_dP(m=0.6, x=0.1, rhol=915., mul=180E-6, P=2E5, Pc=4055000, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 712.0999804205619)

    # Second choice, for no sigma; Chisholm
    dP = two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 1084.1489922923736)

    # Preferred choice, Kim_Mudawar
    dP = two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)
    assert_close(dP, 840.4137796786074)

    # Case where i = 4
    dP = two_phase_dP(Method='Friedel', m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, roughness=0.0, L=1.0)
    assert_close(dP, 738.6500525002243)

    # Case where i = 1
    dP = two_phase_dP(Method='Lockhart_Martinelli', m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, L=1.0)
    assert_close(dP, 716.4695654888484)

    # Case where i = 101, 'Chisholm rough'
    dP = two_phase_dP(Method='Chisholm rough', m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=1E-4, L=1.0)
    assert_close(dP, 846.6778299960783)

    # Case where i = 102:
    dP = two_phase_dP(Method='Zhang_Hibiki_Mishima adiabatic gas', m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    assert_close(dP, 1109.1976111277042)

    # Case where i = 103:
    dP = two_phase_dP(Method='Zhang_Hibiki_Mishima flow boiling', m=0.0005, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.003, roughness=0.0, L=1.0)
    assert_close(dP, 770.0975665928916)

    # Don't give enough information:
    with pytest.raises(Exception):
        two_phase_dP(m=0.6, x=0.1, rhol=915., D=0.05, L=1.0)

    with pytest.raises(Exception):
        two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, sigma=0.045, D=0.05, L=1, Method='BADMETHOD')


def test_two_phase_dP_acceleration():
    m = 1.0
    D = 0.1
    xi = 0.37263067757947943
    xo = 0.5570214522041096
    rho_li = 827.1015716377739
    rho_lo = 827.05
    rho_gi = 3.9190921750559062
    rho_go = 3.811717994431281
    alpha_i = homogeneous(x=xi, rhol=rho_li, rhog=rho_gi)
    alpha_o = homogeneous(x=xo, rhol=rho_lo, rhog=rho_go)
    dP = two_phase_dP_acceleration(m=m, D=D, xi=xi, xo=xo, alpha_i=alpha_i,
                                   alpha_o=alpha_o, rho_li=rho_li, rho_gi=rho_gi,
                                   rho_go=rho_go, rho_lo=rho_lo)
    assert_close(dP, 824.0280564053887)


def test_two_phase_dP_dz_acceleration():
    dP_dz = two_phase_dP_dz_acceleration(m=1.0, D=0.1, x=0.372, rhol=827.1, rhog=3.919, dv_dP_l=-5e-12, dv_dP_g=-4e-7, dx_dP=-2e-7, dP_dL=120.0, dA_dL=0.0001)
    assert_close(dP_dz, 20.137876617489034)


def test_two_phase_dP_gravitational():
    dP = two_phase_dP_gravitational(angle=90.0, z=2.0, alpha_i=0.9685, rho_li=1518., rho_gi=2.6)
    assert_close(dP, 987.237416829999)

    dP = two_phase_dP_gravitational(angle=90, z=2, alpha_i=0.9685, rho_li=1518., rho_gi=2.6,  alpha_o=0.968, rho_lo=1517.9, rho_go=2.59)
    assert_close(dP, 994.5416058829999)


def test_two_phase_dP_dz_gravitational():
    dP_dz = two_phase_dP_dz_gravitational(angle=90.0, alpha=0.9685, rhol=1518., rhog=2.6)
    assert_close(dP_dz, 493.6187084149995)




def test_Taitel_Dukler_regime():
    from fluids.two_phase import Taitel_Dukler_regime

    regime = Taitel_Dukler_regime(m=1.0, x=0.05, rhol=600.12, rhog=80.67, mul=180E-6,
                                   mug=14E-6, D=0.02, roughness=0.0, angle=0.0)[0]

    assert regime == 'bubbly'
    regime = Taitel_Dukler_regime(m=1, x=0.05, rhol=600.12, rhog=80.67, mul=180E-6,
                                  mug=14E-6, D=0.021, roughness=0.0, angle=0)[0]
    assert regime == 'intermittent'

    regime = Taitel_Dukler_regime(m=.06, x=0.5, rhol=900.12, rhog=90.67, mul=180E-6,
                                  mug=14E-6, D=0.05, roughness=0.0, angle=0)[0]
    assert regime == 'stratified smooth'

    regime = Taitel_Dukler_regime(m=.07, x=0.5, rhol=900.12, rhog=90.67, mul=180E-6,
                                   mug=14E-6, D=0.05, roughness=0.0, angle=0)[0]
    assert regime == 'stratified wavy'

    regime, X, T, F, K = Taitel_Dukler_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6,
                                               mug=14E-6, D=0.05, roughness=0.0, angle=0)
    assert regime == 'annular'
    assert_close(F, 0.9902249725092789)
    assert_close(K, 271.86280111125365)
    assert_close(T, 0.04144054776101148)
    assert_close(X, 0.4505119305984412)


Dukler_XA_Xs = [0.0033181, 0.005498, 0.00911, 0.015096, 0.031528, 0.05224, 0.08476, 0.14045, 0.22788, 0.36203, 0.5515, 0.9332, 1.3919, 1.7179, 2.4055, 3.3683, 4.717, 7.185, 10.06, 13.507, 18.134, 23.839, 31.339, 40.341, 52.48]
Dukler_XA_As = [1.6956, 1.5942, 1.4677, 1.3799, 1.1936, 1.076, 0.9108, 0.771, 0.6258, 0.4973, 0.37894, 0.23909, 0.17105, 0.14167, 0.09515, 0.06391, 0.042921, 0.023869, 0.016031, 0.010323, 0.006788, 0.0041042, 0.0026427, 0.0016662, 0.0010396]
Dukler_XD_Xs = [1.7917, 2.9688, 4.919, 14.693, 24.346, 40.341, 131.07, 217.18, 352.37, 908.3, 1473.7, 3604]
Dukler_XD_Ds = [1.2318, 1.1581, 1.0662, 0.904, 0.8322, 0.7347, 0.5728, 0.4952, 0.41914, 0.30028, 0.25417, 0.16741]
Dukler_XC_Xs = [0.01471, 0.017582, 0.020794, 0.024853, 0.028483, 0.040271, 0.06734, 0.10247, 0.15111, 0.21596, 0.33933, 0.5006, 0.701, 1.149, 1.714, 2.5843, 4.0649, 6.065, 8.321, 11.534, 20.817, 28.865, 37.575, 50.48]
Dukler_XC_Cs = [1.9554, 2.1281, 2.3405, 2.5742, 2.8012, 3.2149, 4.0579, 4.8075, 5.403, 5.946, 6.21, 6.349, 6.224, 6.168, 5.618, 4.958, 4.1523, 3.6646, 3.0357, 2.6505, 1.8378, 1.5065, 1.2477, 1.0555]


@pytest.mark.scipy
def test_Taitel_Dukler_splines():
    from scipy.interpolate import splrep, splev
    import numpy as np
    from fluids.two_phase import Dukler_XA_tck, Dukler_XC_tck, Dukler_XD_tck
    Dukler_XA_tck2 = splrep(np.log10(Dukler_XA_Xs), np.log10(Dukler_XA_As), s=5e-3, k=3)
    [assert_close1d(i, j) for i, j in zip(Dukler_XA_tck[:-1], Dukler_XA_tck2[:-1])]
#     XA_interp = UnivariateSpline(np.log10(Dukler_XA_Xs), np.log10(Dukler_XA_As), s=5e-3, k=3) # , ext='const'
#    XA_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XA_tck))

    Dukler_XD_tck2 = splrep(np.log10(Dukler_XD_Xs), np.log10(Dukler_XD_Ds), s=1e-2, k=3)
    [assert_close1d(i, j) for i, j in zip(Dukler_XD_tck[:-1], Dukler_XD_tck[:-1])]
#     XD_interp = UnivariateSpline(np.log10(Dukler_XD_Xs), np.log10(Dukler_XD_Ds), s=1e-2, k=3) # , ext='const'
#    XD_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XD_tck))

    Dukler_XC_tck2 = splrep(np.log10(Dukler_XC_Xs), np.log10(Dukler_XC_Cs), s=1e-3, k=3)
    [assert_close1d(i, j) for i, j in zip(Dukler_XC_tck[:-1], Dukler_XC_tck2[:-1])]
#     XC_interp = UnivariateSpline(np.log10(Dukler_XC_Xs), np.log10(Dukler_XC_Cs), s=1e-3, k=3) # ext='const'
#    XC_interp_obj = lambda x: 10**float(splev(log10(x), Dukler_XC_tck))

    # Curves look great to 1E-4! Also to 1E4.

def plot_Taitel_Dukler_splines():
    import numpy as np
    import matplotlib.pyplot as plt

    from fluids.two_phase import XA_interp_obj, XC_interp_obj, XD_interp_obj
    Xs = np.logspace(np.log10(1e-5), np.log10(1e5), 1000)
    A_Xs = np.logspace(np.log10(1e-5), np.log10(1e5), 1000)
    C_Xs = np.logspace(np.log10(1e-5), np.log10(1e5), 1000)
    D_Xs = np.logspace(np.log10(1e-5), np.log10(1e5))
    A = [XA_interp_obj(X) for X in A_Xs]
    C = [XC_interp_obj(X) for X in C_Xs]
    D = [XD_interp_obj(X) for X in D_Xs]

    fig, ax1 = plt.subplots()

    ax1.loglog(C_Xs, C)
    ax1.set_ylim(1, 10**4)
    ax1.set_ylim(.01, 10**4)
    ax1.loglog(Dukler_XC_Xs, Dukler_XC_Cs, 'x')

    ax2 = ax1.twinx()

    ax2.loglog(A_Xs, A)
    ax2.loglog(D_Xs, D)
    ax2.loglog(Dukler_XD_Xs, Dukler_XD_Ds, '.')
    ax2.loglog(Dukler_XA_Xs, Dukler_XA_As, '+')

    ax2.set_ylim(1e-3, 10)

    fig.tight_layout()
    plt.show()
    return plt

def test_Mandhane_Gregory_Aziz_regime():
    from fluids.two_phase import Mandhane_Gregory_Aziz_regime
    regime = Mandhane_Gregory_Aziz_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67,  mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)[0]
    assert regime == 'slug'

    regime = Mandhane_Gregory_Aziz_regime(m=6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)[0]
    assert regime == 'annular mist'

    regime =  Mandhane_Gregory_Aziz_regime(m=.05, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)[0]
    assert regime == 'stratified'

    regime = Mandhane_Gregory_Aziz_regime(m=.005, x=0.95, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.065, D=0.01)[0]
    assert regime == 'wave'
