# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from numpy.testing import assert_allclose
import pytest
import numpy as np
from fluids.numerics import *
from scipy.integrate import quad
from math import *

def test_horner():
    from fluids.numerics import horner
    assert_allclose(horner([1.0, 3.0], 2.0), 5.0)
    assert_allclose(horner([3.0], 2.0), 3.0)
    
    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    assert_allclose(horner_and_der2(poly, 3.0), (14726.109396, 13747.040732, 8553.7884))
    assert_allclose(horner_and_der3(poly, 3.0), (14726.109396, 13747.040732, 8553.7884, 2674.56))
    
    
def test_interp():
    from fluids.numerics import interp
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
    from fluids.numerics import splev as my_splev
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
    from fluids.numerics import bisplev as my_bisplev
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
    
    xs = np.linspace(0, 1, 10)
    zs = np.linspace(0, 1, 10)
    
    ys_scipy = bisplev(xs, zs, tck)
    ys = my_bisplev(xs, zs, my_tck)
    assert_allclose(ys, ys_scipy)

    ys_scipy = bisplev(0.5, .7, tck)
    ys = my_bisplev(.5, .7, my_tck)
    assert_allclose(ys, ys_scipy)


def test_linspace():
    from fluids.numerics import linspace
    calc = linspace(-3,10, endpoint=True, num=8)
    expect = np.linspace(-3,10, endpoint=True, num=8)
    assert_allclose(calc, expect)
    
    calc = linspace(-3,10, endpoint=False, num=20)
    expect = np.linspace(-3,10, endpoint=False, num=20)
    assert_allclose(calc, expect)

    calc = linspace(0,1e-10, endpoint=False, num=3)
    expect = np.linspace(0,1e-10, endpoint=False, num=3)
    assert_allclose(calc, expect)
    
    calc = linspace(0,1e-10, endpoint=False, num=2)
    expect = np.linspace(0,1e-10, endpoint=False, num=2)
    assert_allclose(calc, expect)
    
    calc = linspace(0,1e-10, endpoint=False, num=1)
    expect = np.linspace(0,1e-10, endpoint=False, num=1)
    assert_allclose(calc, expect)
    
    calc, calc_step = linspace(0,1e-10, endpoint=False, num=2, retstep=True)
    expect, expect_step = np.linspace(0,1e-10, endpoint=False, num=2, retstep=True)
    assert_allclose(calc, expect)
    assert_allclose(calc_step, expect_step)

    calc, calc_step = linspace(0,1e-10, endpoint=False, num=1, retstep=True)
    expect, expect_step = np.linspace(0,1e-10, endpoint=False, num=1, retstep=True)
    assert_allclose(calc, expect)
    assert_allclose(calc_step, expect_step)

    calc, calc_step = linspace(100, 1000, endpoint=False, num=21, retstep=True)
    expect, expect_step = np.linspace(100, 1000, endpoint=False, num=21, retstep=True)
    assert_allclose(calc, expect)
    assert_allclose(calc_step, expect_step)


def test_logspace():
    from fluids.numerics import logspace
    calc = logspace(3,10, endpoint=True, num=8)
    expect = np.logspace(3,10, endpoint=True, num=8)
    assert_allclose(calc, expect)
    
    calc = logspace(3,10, endpoint=False, num=20)
    expect = np.logspace(3,10, endpoint=False, num=20)
    assert_allclose(calc, expect)

    calc = logspace(0,1e-10, endpoint=False, num=3)
    expect = np.logspace(0,1e-10, endpoint=False, num=3)
    assert_allclose(calc, expect)
    
    calc = logspace(0,1e-10, endpoint=False, num=2)
    expect = np.logspace(0,1e-10, endpoint=False, num=2)
    assert_allclose(calc, expect)
    
    calc = logspace(0,1e-10, endpoint=False, num=1)
    expect = np.logspace(0,1e-10, endpoint=False, num=1)
    assert_allclose(calc, expect)
    
    calc = logspace(0,1e-10, endpoint=False, num=2)
    expect = np.logspace(0,1e-10, endpoint=False, num=2)
    assert_allclose(calc, expect)

    calc = logspace(0,1e-10, endpoint=False, num=1)
    expect = np.logspace(0,1e-10, endpoint=False, num=1)
    assert_allclose(calc, expect)

    calc = logspace(100, 200, endpoint=False, num=21)
    expect = np.logspace(100, 200, endpoint=False, num=21)
    assert_allclose(calc, expect)


def test_diff():
    from fluids.numerics import diff
    
    test_arrs = [np.ones(10),
                 np.zeros(10), 
                 np.arange(1, 10),
                 np.arange(1, 10)*25.1241251,
                 (np.arange(1, 10)**1.2),
                 (10.1 + np.arange(1, 10)**20),
                 (10.1 + np.linspace(-100, -10, 9)),
                 (np.logspace(-10, -100, 19)**1.241),
                 (np.logspace(10, 100, 15)**1.241)
    ]
    for test_arr in test_arrs:
        arr = test_arr.tolist()
        for n in range(5):
            diff_np = np.diff(arr, n=n)
            diff_py = diff(arr, n=n)
            assert_allclose(diff_np, diff_py)

    assert tuple(diff([1,2,3], n=0)) == tuple([1,2,3])
    with pytest.raises(Exception):
        diff([1,2,3], n=-1)
        
        
def test_fit_integral_linear_extrapolation():
    coeffs = [-6.496329615255804e-23,2.1505678500404716e-19, -2.2204849352453665e-16,
              1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08,
              8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]
    
    Tmin, Tmax = 50.0, 1000.0
    Tmin_value, Tmax_value = 29.10061916353635, 32.697696220612684
    Tmin_slope, Tmax_slope = 9.512557609301246e-06, 0.005910807286115391
    
    int_coeffs = polyint(coeffs)
    T_int_T_coeffs, log_coeff = polyint_over_x(coeffs)
    def func(T):
        if T < Tmin:
            Cp = (T - Tmin)*Tmin_slope + Tmin_value
        elif T > Tmax:
            Cp = (T - Tmax)*Tmax_slope + Tmax_value
        else:
            Cp = horner(coeffs, T)
        return Cp
    
    assert_allclose(func(300), 29.12046448327871, rtol=1e-12)
    Ts = [0, 1, 25, 49, 50, 51, 500, 999, 1000, 1001, 2000, 50000]
    T_ends = [0, Tmin, Tmin*2.0, Tmax, Tmax*2.0]
    
    numericals = []
    analyticals = []
    analyticals2 = []
    for Tend in T_ends:
        for Tdiff in Ts:
            for (T1, T2) in zip([Tend, Tdiff], [Tdiff, Tend]):
                analytical = fit_integral_linear_extrapolation(T1, T2, int_coeffs, Tmin, Tmax, 
                                              Tmin_value, Tmax_value, 
                                              Tmin_slope, Tmax_slope)
                analytical2 = (best_fit_integral_value(T2, int_coeffs, Tmin, Tmax, 
                              Tmin_value, Tmax_value, 
                              Tmin_slope, Tmax_slope)
                              - best_fit_integral_value(T1, int_coeffs, Tmin, Tmax, 
                              Tmin_value, Tmax_value, 
                              Tmin_slope, Tmax_slope))
                
                
                numerical = quad(func, T1, T2, epsabs=1.49e-14, epsrel=1.49e-14)[0]
                numericals.append(numerical)
                analyticals.append(analytical)
                analyticals2.append(analytical2)
    assert_allclose(analyticals, numericals, rtol=1e-7)
    assert_allclose(analyticals2, numericals, rtol=1e-7)
    
    
    
    

    # Cannot have temperatures of 0 absolute for integrals over T cases
    Ts = [1e-9, 1, 25, 49, 50, 51, 500, 999, 1000, 1001, 2000, 50000]
    T_ends = [1e-9, Tmin, Tmin*2.0, Tmax, Tmax*2.0]
    numericals = []
    analyticals = []
    analyticals2 = []
    for Tend in T_ends:
        for Tdiff in Ts:
            for (T1, T2) in zip([Tend, Tdiff], [Tdiff, Tend]):
                analytical = fit_integral_over_T_linear_extrapolation(T1, T2, T_int_T_coeffs, log_coeff, 
                                                                      Tmin, Tmax, 
                                              Tmin_value, Tmax_value, 
                                              Tmin_slope, Tmax_slope)
                analytical2 = (best_fit_integral_over_T_value(T2, T_int_T_coeffs, log_coeff, 
                                                                      Tmin, Tmax, 
                                              Tmin_value, Tmax_value, 
                                              Tmin_slope, Tmax_slope) - best_fit_integral_over_T_value(T1, T_int_T_coeffs, log_coeff, 
                                                                      Tmin, Tmax, 
                                              Tmin_value, Tmax_value, 
                                              Tmin_slope, Tmax_slope))
                
                numerical = quad(lambda T: func(float(T))/T, T1, T2, epsabs=1.49e-12, epsrel=1.49e-14)[0]
                
                
                
                numericals.append(numerical)
                analyticals.append(analytical)
                analyticals2.append(analytical2)
    assert_allclose(analyticals, numericals, rtol=1e-7)
    assert_allclose(analyticals2, numericals, rtol=1e-7)



def test_best_bounding_bounds():
    def to_solve(x):
        return exp(x) - 100


    vals = best_bounding_bounds(0, 5, to_solve, xs_pos=[4.831, 4.6054], ys_pos= [25.38, 0.0288],
                        xs_neg=[4, 4.533, 4.6051690], ys_neg=[-45.40, -6.933, -0.0001139])
    assert_allclose(vals, (4.605169, 4.6054, -0.0001139, 0.0288), rtol=1e-12)
    
    vals = best_bounding_bounds(4.60517018598, 5, to_solve, xs_pos=[4.831, 4.6054], ys_pos= [25.38, 0.0288],
                        xs_neg=[4, 4.533, 4.6051690], ys_neg=[-45.40, -6.933, -0.0001139])
    assert_allclose(vals, (4.60517018598, 4.6054, -8.091802783383173e-10, 0.0288), rtol=1e-12)
    
    vals = best_bounding_bounds(0, 4.60517018599, to_solve, xs_pos=[4.831, 4.6054], ys_pos= [25.38, 0.0288],
                        xs_neg=[4, 4.533, 4.6051690], ys_neg=[-45.40, -6.933, -0.0001139])
    assert_allclose(vals, (4.605169, 4.60517018599, -0.0001139, 1.908233571157325e-10), rtol=1e-12)
    
    vals = best_bounding_bounds(0, 4.60517018599, fa=to_solve(0), fb=to_solve(4.60517018599),
                                xs_pos=[4.831, 4.6054], ys_pos= [25.38, 0.0288],
                        xs_neg=[4, 4.533, 4.6051690], ys_neg=[-45.40, -6.933, -0.0001139])
    assert_allclose(vals, (4.605169, 4.60517018599, -0.0001139, 1.908233571157325e-10), rtol=1e-12)
    
    
def test_is_poly_positive():
    assert not is_poly_positive([4, 3, 2, 1])
    for high in range(0, 100, 5):
        assert is_poly_positive([4, 3, 2, 1], domain=(0, 10**high))
    
    coeffs_4alpha = [2.1570803657937594e-10, 2.008831101045556e-06, -0.004656598178209313, 2.8575882247542514]
    assert not is_poly_positive(coeffs_4alpha)
    assert is_poly_positive(coeffs_4alpha, domain=(0, 511))
    assert is_poly_positive(coeffs_4alpha, domain=(-10000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-20000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-15000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-13000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-11500, 511))
    

def test_roots_quartic():
    coeffs = [1.0, -3.274390673429134, 0.3619541556604501, 2.4841800045762747, -0.49619076425603237]
    
    expect_roots = ((-0.8246324500888049+1.1609047395516947e-17j),
     (0.2041867922778502-3.6197168963943884e-17j),
     (1.027869356838059+2.910620457054182e-17j),
     (2.86696697440203-4.51808300211488e-18j))
    expect_mp_roots_real = [-0.824632450088805, 0.20418679227785, 1.0278693568380592, 2.86696697440203]
    roots_calc = roots_quartic(*coeffs)
    assert_allclose(expect_roots, roots_calc, rtol=1e-9)
    assert_allclose(expect_mp_roots_real, [i.real for i in roots_calc], rtol=1e-9)