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
from fluids.numerics import numpy as np
import pytest
import fluids.numerics
from fluids.numerics import *
from scipy.integrate import quad
from math import *
from random import random

assert_allclose = np.testing.assert_allclose



def test_sincos():
    N = 10**1
    for v in linspace(0.0, 2.0*pi, N):
        a, b = sincos(v)
        assert_close(a, sin(v), rtol=1e-14)
        assert_close(b, cos(v), rtol=1e-14)
    for v in linspace(-100.0, 100.0, N):
        a, b = sincos(v)
        assert_close(a, sin(v), rtol=1e-14)
        assert_close(b, cos(v), rtol=1e-14)


def test_bisect_log_exp_terminations():
    from math import exp, log
    from fluids.numerics import bisect
    def to_solve(x):
        try:
            return exp(x)
        except:
            return -1
    assert 709.782712893384 == bisect(to_solve, 600, 800, xtol=1e-16)

    def to_solve(x):
        x = 10**x
        try:
            return log(x)
        except:
            return 1.0
    assert -323.60724533877976 == bisect(to_solve, -300, -400, xtol=1e-16)


def test_horner():
    from fluids.numerics import horner
    assert_allclose(horner([1.0, 3.0], 2.0), 5.0)
    assert_allclose(horner([3.0], 2.0), 3.0)

    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    assert_allclose(horner_and_der2(poly, 3.0), (14726.109396, 13747.040732, 8553.7884))
    assert_allclose(horner_and_der3(poly, 3.0), (14726.109396, 13747.040732, 8553.7884, 2674.56))

    poly = [1.12, 432.32, 325.5342, .235532, 32.235, 1.01]
    assert_allclose(horner_and_der4(poly, 3.0), [np.polyval(np.polyder(poly,o), 3) for o in range(5)])

def test_quadratic_from_f_ders():
    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    p = 3.0
    v, d1, d2 = horner_and_der2(poly, p)
    quadratic = quadratic_from_f_ders(p, v, d1, d2)
    v_new, d1_new, d2_new = horner_and_der2(quadratic, p)

    assert_close(v_new, v)
    assert_close(d1_new, d1)
    assert_close(d2_new, d2)

    p = 2.9
    v, d1, d2 = horner_and_der2(poly, p)
    v_new, d1_new, d2_new = horner_and_der2(quadratic, p)
    v_new, d1_new, d2_new, v, d1, d2
    assert_close(v_new, v, rtol=1e-4)
    assert_close(d1_new, d1, rtol=5e-3)



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


    # Test custom extrapolation method
    xs = [1,2,3]
    ys = [.1, .2, .3]
    assert_close(interp(3.5, xs, ys, extrapolate=True), .35, rtol=1e-15)
    assert_close(interp(0, xs, ys, extrapolate=True), 0, rtol=1e-15)
    assert_close(interp(-1, xs, ys, extrapolate=True), -.1, rtol=1e-15)
    assert_close(interp(-100, xs, ys, extrapolate=True), -10, rtol=1e-15)
    assert_close(interp(10, xs, ys, extrapolate=True), 1, rtol=1e-15)
    assert_close(interp(10.0**30, xs, ys, extrapolate=True), 10.0**29, rtol=1e-15)




def test_splev():
    from fluids.numerics import py_splev
    from scipy.interpolate import splev
    # Originally Dukler_XA_tck
    tck = [np.array([-2.4791105294648372, -2.4791105294648372, -2.4791105294648372,
                               -2.4791105294648372, 0.14360803483759585, 1.7199938263676038,
                               1.7199938263676038, 1.7199938263676038, 1.7199938263676038]),
                     np.array([0.21299880246561081, 0.16299733301915248, -0.042340970712679615,
                               -1.9967836909384598, -2.9917366639619414, 0.0, 0.0, 0.0, 0.0]),
                     3]
    my_tck = [tck[0].tolist(), tck[1].tolist(), tck[2]]

    xs = np.linspace(-3, 2, 100).tolist()

    # test extrapolation
    ys_scipy = splev(xs, tck, ext=0)
    ys = [py_splev(xi, my_tck, ext=0) for xi in xs]
    assert_allclose(ys, ys_scipy)

    # test truncating to side values
    ys_scipy = splev(xs, tck, ext=3)
    ys = [py_splev(xi, my_tck, ext=3) for xi in xs]
    assert_allclose(ys, ys_scipy)


    # Test returning zeros for bad values
    ys_scipy = splev(xs, tck, ext=1)
    ys = [py_splev(xi, my_tck, ext=1) for xi in xs]
    assert_allclose(ys, ys_scipy)

    # Test raising an error when extrapolating is not allowed
    with pytest.raises(ValueError):
        py_splev(xs[0], my_tck, ext=2)
    with pytest.raises(ValueError):
        splev(xs[0], my_tck, ext=2)


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
    assert isnan(calc_step)
    # Cannot compare against numpy expect_step - it did not use to give nan in older versions

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

    expect_values = [0.0, 0.0, 29.10014829193469, -29.10014829193469, 727.50656106565, -727.50656106565, 1425.9184530725483, -1425.9184530725483, 1455.0190674798057, -1455.0190674798057, 1484.119654811151, -1484.119654811151, 14588.023573849947, -14588.023573849947, 30106.09421115182, -30106.09421115182, 30138.789058157658, -30138.789058157658, 30171.489709781912, -30171.489709781912, 65791.88892182804, -65791.88892182804, 8728250.050849704, -8728250.050849704, -1455.0190674798057, 1455.0190674798057, -1425.918919187871, 1425.918919187871, -727.5125064141557, 727.5125064141557, -29.100614407257353, 29.100614407257353, 0.0, 0.0, 29.100587331345196, -29.100587331345196, 13133.004506370142, -13133.004506370142, 28651.075143672013, -28651.075143672013, 28683.76999067785, -28683.76999067785, 28716.470642302105, -28716.470642302105, 64336.869854348224, -64336.869854348224, 8726795.031782223, -8726795.031782223, -2910.0427925248396, 2910.0427925248396, -2880.942644232905, 2880.942644232905, -2182.53623145919, 2182.53623145919, -1484.1243394522915, 1484.1243394522915, -1455.023725045034, 1455.023725045034, -1425.923137713689, 1425.923137713689, 11677.980781325108, -11677.980781325108, 27196.05141862698, -27196.05141862698, 27228.746265632813, -27228.746265632813, 27261.446917257068, -27261.446917257068, 62881.84612930319, -62881.84612930319, 8725340.008057179, -8725340.008057179, -30138.789058157658, 30138.789058157658, -30109.688909865723, 30109.688909865723, -29411.282497092005, 29411.282497092005, -28712.870605085107, 28712.870605085107, -28683.76999067785, 28683.76999067785, -28654.669403346503, 28654.669403346503, -15550.765484307707, 15550.765484307707, -32.6948470058378, 32.6948470058378, 0.0, 0.0, 32.70065162425453, -32.70065162425453, 35653.09986367037, -35653.09986367037, 8698111.261791546, -8698111.261791546, -65791.88892182804, 65791.88892182804, -65762.7887735361, 65762.7887735361, -65064.38236076238, 65064.38236076238, -64365.97046875548, 64365.97046875548, -64336.869854348224, 64336.869854348224, -64307.769267016876, 64307.769267016876, -51203.86534797808, 51203.86534797808, -35685.794710676215, 35685.794710676215, -35653.09986367037, 35653.09986367037, -35620.39921204612, 35620.39921204612, 0.0, 0.0, 8662458.161927875, -8662458.161927875]


    numericals = []
    analyticals = []
    analyticals2 = []
    for Tend in T_ends:
        for Tdiff in Ts:
            for (T1, T2) in zip([Tend, Tdiff], [Tdiff, Tend]):
                analytical = fit_integral_linear_extrapolation(T1, T2, int_coeffs, Tmin, Tmax,
                                              Tmin_value, Tmax_value,
                                              Tmin_slope, Tmax_slope)
                analytical2 = (poly_fit_integral_value(T2, int_coeffs, Tmin, Tmax,
                              Tmin_value, Tmax_value,
                              Tmin_slope, Tmax_slope)
                              - poly_fit_integral_value(T1, int_coeffs, Tmin, Tmax,
                              Tmin_value, Tmax_value,
                              Tmin_slope, Tmax_slope))


#                numerical = quad(func, T1, T2, epsabs=1.49e-14, epsrel=1.49e-14)[0]
#                numericals.append(numerical)
                analyticals.append(analytical)
                analyticals2.append(analytical2)
#    print(analyticals)
#    assert_allclose(analyticals, numericals, rtol=1e-7)
#    assert_allclose(analyticals2, numericals, rtol=1e-7)
    assert_allclose(analyticals, expect_values, rtol=1e-12)
    assert_allclose(analyticals2, expect_values, rtol=1e-12)




    # Cannot have temperatures of 0 absolute for integrals over T cases
    Ts = [1e-9, 1, 25, 49, 50, 51, 500, 999, 1000, 1001, 2000, 50000]
    T_ends = [1e-9, Tmin, Tmin*2.0, Tmax, Tmax*2.0]
    expect_values = [0.0, 0.0, 603.0500198952521, -603.0500198952521, 696.719996723752, -696.719996723752, 716.3030057880156, -716.3030057880156, 716.890916983322, -716.890916983322, 717.467185070396, -717.467185070396, 783.9851859794122, -783.9851859794122, 805.3820356163964, -805.3820356163964, 805.4147468212567, -805.4147468212567, 805.4474311329551, -805.4474311329551, 829.8928106482913, -829.8928106482913, 1199.8352295965128, -1199.8352295965128, -716.890916983322, 716.890916983322, -113.84089708806982, 113.84089708806982, -20.170920259569826, 20.170920259569826, -0.5879111953062761, 0.5879111953062761, 0.0, 0.0, 0.5762680870740127, -0.5762680870740127, 67.09426899609028, -67.09426899609028, 88.4911186330744, -88.4911186330744, 88.5238298379347, -88.5238298379347, 88.5565141496331, -88.5565141496331, 113.00189366496929, -113.00189366496929, 482.94431261319096, -482.94431261319096, -737.0618021522312, 737.0618021522312, -134.01178225697902, 134.01178225697902, -40.34180542847902, 40.34180542847902, -20.75879636421547, 20.75879636421547, -20.170885168909194, 20.170885168909194, -19.59461708183518, 19.59461708183518, 46.923383827181084, -46.923383827181084, 68.3202334641652, -68.3202334641652, 68.3529446690255, -68.3529446690255, 68.38562898072391, -68.38562898072391, 92.8310084960601, -92.8310084960601, 462.77342744428177, -462.77342744428177, -805.4147468212567, 805.4147468212567, -202.36472692600452, 202.36472692600452, -108.69475009750452, 108.69475009750452, -89.11174103324097, 89.11174103324097, -88.5238298379347, 88.5238298379347, -87.94756175086069, 87.94756175086069, -21.42956084184442, 21.42956084184442, -0.03271120486030554, 0.03271120486030554, 0.0, 0.0, 0.03268431169840369, -0.03268431169840369, 24.47806382703459, -24.47806382703459, 394.42048277525623, -394.42048277525623, -829.8928106482913, 829.8928106482913, -226.8427907530391, 226.8427907530391, -133.17281392453913, 133.17281392453913, -113.58980486027556, 113.58980486027556, -113.00189366496929, 113.00189366496929, -112.42562557789527, 112.42562557789527, -45.90762466887901, 45.90762466887901, -24.510775031894894, 24.510775031894894, -24.47806382703459, 24.47806382703459, -24.445379515336185, 24.445379515336185, 0.0, 0.0, 369.94241894822164, -369.94241894822164]

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
                analytical2 = (poly_fit_integral_over_T_value(T2, T_int_T_coeffs, log_coeff,
                                                                      Tmin, Tmax,
                                              Tmin_value, Tmax_value,
                                              Tmin_slope, Tmax_slope) - poly_fit_integral_over_T_value(T1, T_int_T_coeffs, log_coeff,
                                                                      Tmin, Tmax,
                                              Tmin_value, Tmax_value,
                                              Tmin_slope, Tmax_slope))

#                numerical = quad(lambda T: func(float(T))/T, T1, T2, epsabs=1.49e-12, epsrel=1.49e-14)[0]



#                numericals.append(numerical)
                analyticals.append(analytical)
                analyticals2.append(analytical2)
#
#    assert_allclose(analyticals, numericals, rtol=1e-7)
#    assert_allclose(analyticals2, numericals, rtol=1e-7)
    assert_allclose(analyticals, expect_values, rtol=1e-11)
    assert_allclose(analyticals2, expect_values, rtol=1e-11)


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


def test_array_as_tridiagonals():
    A = [[10.0, 2.0, 0.0, 0.0],
     [3.0, 10.0, 4.0, 0.0],
     [0.0, 1.0, 7.0, 5.0],
     [0.0, 0.0, 3.0, 4.0]]

    tridiagonals = array_as_tridiagonals(A)
    expect_diags = [[3.0, 1.0, 3.0], [10.0, 10.0, 7.0, 4.0], [2.0, 4.0, 5.0]]

    assert_allclose(tridiagonals[0], expect_diags[0], rtol=0, atol=0)
    assert_allclose(tridiagonals[1], expect_diags[1], rtol=0, atol=0)
    assert_allclose(tridiagonals[2], expect_diags[2], rtol=0, atol=0)
    A = np.array(A)
    tridiagonals = array_as_tridiagonals(A)
    assert_allclose(tridiagonals[0], expect_diags[0], rtol=0, atol=0)
    assert_allclose(tridiagonals[1], expect_diags[1], rtol=0, atol=0)
    assert_allclose(tridiagonals[2], expect_diags[2], rtol=0, atol=0)


    a, b, c = [3.0, 1.0, 3.0], [10.0, 10.0, 7.0, 4.0], [2.0, 4.0, 5.0]
    expect_mat = tridiagonals_as_array(a, b, c)
    assert_allclose(expect_mat, A, rtol=0, atol=0)

    d = [3.0, 4.0, 5.0, 6.0]

    solved_expect = [0.1487758945386064, 0.756120527306968, -1.001883239171375, 2.2514124293785316]
    assert_allclose(solve_tridiagonal(a, b, c, d), solved_expect, rtol=1e-12)


def test_subset_matrix():
    kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]

    expect = [[0, 0.00061], [0.00061, 0]]
    got = subset_matrix(kijs, [1,2])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(1, 3, 1))
    assert_allclose(expect, got, atol=0, rtol=0)

    expect = [[0, 0.00171], [0.00171, 0]]
    got = subset_matrix(kijs, [0,2])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 3, 2))
    assert_allclose(expect, got, atol=0, rtol=0)

    expect = [[0, 0.00076], [0.00076, 0]]
    got = subset_matrix(kijs, [0,1])
    assert_allclose(expect, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 2, 1))
    assert_allclose(expect, got, atol=0, rtol=0)

    got = subset_matrix(kijs, [0,1, 2])
    assert_allclose(kijs, got, atol=0, rtol=0)
    got = subset_matrix(kijs, slice(0, 3, 1))
    assert_allclose(kijs, got, atol=0, rtol=0)


def test_translate_bound_func():
    def rosen_test(x):
        x, y = x
        return (1.0 - x)**2 + 100.0*(y - x**2)**2

    f, into, outof = translate_bound_func(rosen_test, low=[-2, -.2], high=[3.0, 4])

    point =  [.6, .7]
    in_exp = [0.0800427076735365, -1.2992829841302609]
    assert_allclose(into(point), in_exp, rtol=1e-12)

    assert_allclose(outof(in_exp), point, rtol=1e-12)
    assert f(into([1, 1])) < 1e-20
    assert_allclose(outof(into([1, 1])), [1, 1], rtol=1e-12)


    f, into, outof = translate_bound_func(rosen_test, bounds=[[-2, 3], [-.2, 4]])

    point =  [.6, .7]
    in_exp = [0.0800427076735365, -1.2992829841302609]
    assert_allclose(into(point), in_exp, rtol=1e-12)

    assert_allclose(outof(in_exp), point, rtol=1e-12)
    assert f(into([1, 1])) < 1e-20
    assert_allclose(outof(into([1, 1])), [1, 1], rtol=1e-12)

def test_translate_bound_jac():
    from scipy.optimize import rosen_der
    def rosen_test(x):
        x, y = x
        return (1.0 - x)**2 + 100.0*(y - x**2)**2
    j, into, outof = translate_bound_jac(rosen_der, low=[-2, -.2], high=[3.0, 4])
    f, into, outof = translate_bound_func(rosen_test, low=[-2, -.2], high=[3.0, 4])

    point = [3, -2]
    jac_num = jacobian(f, point, perturbation=1e-8)
    jac_anal = j(point)
    assert_allclose(jac_num, jac_anal, rtol=1e-6)


def test_translate_bound_f_jac():
    from scipy.optimize import rosen_der
    def rosen_test(x):
        x, y = x
        return (1.0 - x)**2 + 100.0*(y - x**2)**2

    low, high = [-2, -.2], [3.0, 4]
    f_j, into, outof = translate_bound_f_jac(rosen_test, rosen_der, low=low, high=high)

    point = [3, -2]
    f0, j0 = f_j(point)
    f0_check = translate_bound_func(rosen_test, low=low, high=high)[0](point)
    assert_allclose(f0_check, f0, rtol=1e-13)

    j0_check = translate_bound_jac(rosen_der, low=low, high=high)[0](point)
    assert_allclose(j0_check, j0, rtol=1e-13)

def test_translate_bound_f_jac_multivariable():
    def cons_f(x):
        return [x[0]**2 + x[1], x[0]**2 - x[1]]
    def cons_J(x):
        return[[2*x[0], 1], [2*x[0], -1]]

    new_f_j, translate_into, translate_outof = translate_bound_f_jac(cons_f, cons_J,
                                                                     bounds=[(-10, 10), (-5, 5)])
    assert_close2d(new_f_j([-2.23, 3.45])[1], jacobian(lambda g: new_f_j(g, )[0], [-2.23, 3.45], scalar=False),
                  rtol=3e-6)



def test_solve_direct():
    A = [[1.0,2.53252], [34.34, .5342]]
    B = [1.1241, .54354]
    assert_close1d(np.linalg.solve(A, B), solve_2_direct(A, B), rtol=1e-14)
    assert type(solve_2_direct(A, B)) is list


    A = [[1.0,2.53252, 54.54], [34.34, .5342, .545], [12.43, .545, .55555]]
    B = [1.1241, .54354, 1.22333]
    assert_close1d(np.linalg.solve(A, B), solve_3_direct(A, B), rtol=5e-14)
    assert type(solve_3_direct(A, B)) is list

    A = [[1.0,2.53252, 54.54, .235], [34.34, .5342, .545, .223], [12.43, .545, .55555, 33.33], [1.11, 2.2, 3.33, 4.44]]
    B = [1.1241, .54354, 1.22333, 9.009]
    ans = solve_4_direct(A, B)
    assert_close1d(np.linalg.solve(A, B), ans, rtol=5e-14)
    assert type(ans) is list


def test_polylog2():
    x = polylog2(0.5)
    assert_close(x, 0.5822405264516294)

    xs = linspace(0,0.99999, 50)
#    ys_act = [float(polylog(2, x)) for x in xs]
#    from sympy import polylog
    ys_act = [0.0, 0.020513035768572635, 0.0412401364927588, 0.06218738124039796, 0.08336114665629184,
              0.10476812813791354, 0.12641536301777412, 0.1483102559926201, 0.1704606070746889,
              0.19287464238138674, 0.21556104812821067, 0.23852900824703252, 0.261788246119884,
              0.2853490709994786, 0.309222429784819, 0.33341996493707843, 0.3579540794622072,
              0.38283801005840257, 0.4080859097363812, 0.43371294147827794, 0.45973538481992837,
              0.4861707576383267, 0.5130379559237574, 0.540357414944675, 0.5681512960135646,
              0.596443704089335, 0.6252609427828415, 0.6546318150738004, 0.6845879803506546,
              0.7151643814663312, 0.7463997596771124, 0.7783372810645774, 0.811025306032588,
              0.8445183447984893, 0.878878258148156, 0.9141757868110273, 0.9504925291123206,
              0.9879235426949309, 1.026580835488432, 1.0665981582977615, 1.108137763432647,
              1.1514002456979586, 1.1966394380910048, 1.244186068718536, 1.2944877067946645,

              1.3481819162579485, 1.4062463083287482, 1.4703641942000052, 1.5441353484206717, 1.644808936992927]

    ys = [polylog2(x) for x in xs]
    assert_close1d(ys, ys_act, rtol=1E-7, atol=1E-10)
