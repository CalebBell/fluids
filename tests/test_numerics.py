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
SOFTWARE.
'''

from math import cos, erf, exp, isnan, log, pi, sin, sqrt

import pytest

from fluids.numerics import (
    SolverInterface,
    array_as_tridiagonals,
    assert_close,
    assert_close1d,
    assert_close2d,
    best_bounding_bounds,
    chebder,
    chebint,
    chebval,
    chebval_ln_tau,
    chebval_ln_tau_and_der,
    chebval_ln_tau_and_der2,
    chebval_ln_tau_and_der3,
    cumsum,
    derivative,
    hessian,
    assert_close3d,
    exp_cheb,
    exp_cheb_and_der,
    exp_cheb_and_der2,
    exp_cheb_and_der3,
    exp_cheb_ln_tau,
    exp_cheb_ln_tau_and_der,
    exp_cheb_ln_tau_and_der2,
    fit_integral_linear_extrapolation,
    fit_integral_over_T_linear_extrapolation,
    full,
    horner,
    is_monotonic,
    is_poly_positive,
    isclose,
    jacobian,
    linspace,
    max_abs_error,
    max_abs_rel_error,
    max_squared_error,
    max_squared_rel_error,
    mean_abs_error,
    mean_abs_rel_error,
    mean_squared_error,
    mean_squared_rel_error,
    min_max_ratios,
    newton_system,
    poly_fit_integral_over_T_value,
    poly_fit_integral_value,
    polyint,
    polyint_over_x,
    polylog2,
    polynomial_offset_scale,
    secant,
    sincos,
    fixed_point,
    solve_2_direct,
    solve_3_direct,
    solve_4_direct,
    solve_tridiagonal,
    std,
    subset_matrix,
    translate_bound_f_jac,
    translate_bound_func,
    translate_bound_jac,
    tridiagonals_as_array,
    trunc_exp_numpy,
    trunc_log_numpy,
    zeros,
    is_increasing,
    argsort1d,
    fixed_point_to_residual,
    residual_to_fixed_point,
    broyden2,
    fixed_point_aitken, 
    fixed_point_gdem,
    fixed_point_anderson,
    cumulative_trapezoid,
)
from fluids.numerics import numpy as np

assert_allclose = np.testing.assert_allclose


def test_error_functions():
    data = [1.0, 2.0, 3.0]
    calc = [.99, 2.01, 3.2]
    assert_close(max_abs_error(data, calc), 0.2, rtol=1e-13)
    assert_close(max_abs_rel_error(data, calc), 0.06666666666666672, rtol=1e-13)
    assert_close(max_squared_error(data, calc), 0.04000000000000007, rtol=1e-13)
    assert_close(max_squared_rel_error(data, calc), 0.004444444444444451, rtol=1e-13)

    assert_close(mean_abs_error(data, calc), 0.07333333333333332, rtol=1e-13)
    assert_close(mean_abs_rel_error(data, calc), 0.027222222222222207, rtol=1e-13)
    assert_close(mean_squared_error(data, calc), 0.013400000000000023, rtol=1e-13)
    assert_close(mean_squared_rel_error(data, calc), 0.0015231481481481502, rtol=1e-13)



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











def test_cumsum():
    assert_close1d(cumsum([1,2,3,4,5]), [1, 3, 6, 10, 15])
    assert_close1d(cumsum([1]), [1])

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


def test_zeros():
    # Only support up to 4d so far
    assert_allclose(zeros(2),
                    np.zeros(2), atol=0)
    assert_allclose(zeros((2, )),
                    np.zeros((2, )), atol=0)
    assert_allclose(zeros((2, 3)),
                    np.zeros((2, 3)), atol=0)
    assert_allclose(zeros((2, 3, 4)),
                    np.zeros((2, 3, 4)), atol=0)
    assert_allclose(zeros((2, 3, 4, 5)),
                    np.zeros((2, 3, 4, 5)), atol=0)
def test_full():

    assert_allclose(full(2, 1),
                    np.full(2, 1), atol=0)
    assert_allclose(full((2, ), 1),
                    np.full((2, ), 1), atol=0)
    assert_allclose(full((2, 3), 1),
                    np.full((2, 3), 1), atol=0)
    assert_allclose(full((2, 3, 4), 1),
                    np.full((2, 3, 4), 1), atol=0)
    assert_allclose(full((2, 3, 4, 5), 1),
                    np.full((2, 3, 4, 5), 1), atol=0)

def test_splev():
    from scipy.interpolate import splev

    from fluids.numerics import py_splev
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
    from scipy.interpolate import bisplev

    from fluids.numerics import py_bisplev as my_bisplev

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
    ys = [[my_bisplev(xi, zi, my_tck) for zi in zs] for xi in xs]
    assert_allclose(ys, ys_scipy)

    ys_scipy = bisplev(0.5, .7, tck)
    ys = my_bisplev(.5, .7, my_tck)
    assert_allclose(ys, ys_scipy)

def test_cumulative_trapezoid():
    """Test the cumulative_trapezoid implementation against scipy's version
    and known analytical solutions."""
    from scipy.integrate import cumulative_trapezoid as scipy_cumtrapz
    
    # Compare against scipy for linear function
    x = np.linspace(-2, 2, num=20).tolist()
    y = x.copy()  # f(x) = x, integral should be x^2/2
    y_scipy = scipy_cumtrapz(y, x).tolist()
    y_py = cumulative_trapezoid(y, x)
    assert_allclose(y_py, y_scipy)

    # Unit spacing (x=1)
    y = [1, 2, 3, 4]  # Integral should be [1.5, 4.0, 7.5]
    y_scipy = scipy_cumtrapz(y, dx=1).tolist()
    y_py = cumulative_trapezoid(y, dx=1)
    assert_allclose(y_py, y_scipy)
    
    # Error handling
    # Test mismatched lengths
    with pytest.raises(ValueError):
        cumulative_trapezoid([1, 2, 3], [1, 2])
    
    # Non-uniform spacing
    x = [0, 1, 4, 10]  # Non-uniform grid
    y = [0, 1, 4, 10]  # f(x) = x
    y_scipy = scipy_cumtrapz(y, x).tolist()
    y_py = cumulative_trapezoid(y, x)
    assert_allclose(y_py, y_scipy)
    
    # Constant function
    x = np.linspace(0, 1, 5).tolist()
    y = [2.0] * 5  # f(x) = 2
    y_scipy = scipy_cumtrapz(y, x).tolist()
    y_py = cumulative_trapezoid(y, x)
    assert_allclose(y_py, y_scipy)
    
    # Negative values
    x = [-2, -1, 0, 1, 2]
    y = [-4, -1, 0, 1, 4]
    y_scipy = scipy_cumtrapz(y, x).tolist()
    y_py = cumulative_trapezoid(y, x)
    assert_allclose(y_py, y_scipy)

    # Test empty inputs
    assert cumulative_trapezoid([]) == []
    assert cumulative_trapezoid([1]) == []
    


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

    assert tuple(diff([1,2,3], n=0)) == (1,2,3)
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
    for high in range(0, 30, 5):
        assert is_poly_positive([4, 3, 2, 1], domain=(0, 10**high))

    coeffs_4alpha = [2.1570803657937594e-10, 2.008831101045556e-06, -0.004656598178209313, 2.8575882247542514]
    assert not is_poly_positive(coeffs_4alpha)
    assert is_poly_positive(coeffs_4alpha, domain=(0, 511))
    assert is_poly_positive(coeffs_4alpha, domain=(-10000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-20000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-15000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-13000, 511))
    assert not is_poly_positive(coeffs_4alpha, domain=(-11500, 511))

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

def test_std():
    inputs = ([1.0,5.0,11.0,4.0],
              [1.0,-5.0,11.0,-4.0],
              [1e12,-5.e13,11e14,-4e13],
             [1, 2, 3, 4],
             [-1, -2, -3, 4],
             [14, 8, 11, 10, 7, 9, 10, 11, 10, 15, 5, 10]
             )
    for thing in inputs:
        assert_close(std(thing), np.std(thing), rtol=1e-14)

def test_min_max_ratios():
    actual = [1,2,3,4,5]
    calculated = [.9, 2.1, 3.05, 3.8, 5.5]

    min_ratio_np, max_ratio_np = np.min(np.array(calculated)/actual), np.max(np.array(calculated)/actual)
    assert_close1d([min_ratio_np, max_ratio_np], min_max_ratios(actual, calculated), rtol=1e-14)

    # Case with a zero match
    actual = [1,2,3,0,5]
    calculated = [.9, 2.1, 3.05, 0.0, 5.5]
    assert_close1d(min_max_ratios(actual, calculated), (0.9, 1.1), rtol=0)

    # Case with a zero mismatch
    actual = [1,2,3,0,5]
    calculated = [.9, 2.1, 3.05, 1, 5.5]
    assert_close1d(min_max_ratios(actual, calculated), (0.9, 10.0), rtol=0)



def test_exp_cheb_fit_ln_tau():
    coeffs = [-5.922664830406188, -3.6003367212635444, -0.0989717205896406, 0.05343895281736921, -0.02476759166597864, 0.010447569392539213, -0.004240542036664352, 0.0017273355647560718, -0.0007199858491173661, 0.00030714447101984343, -0.00013315510546685339, 5.832551964424226e-05, -2.5742454514671165e-05, 1.143577875153956e-05, -5.110008470393668e-06, 2.295229193177706e-06, -1.0355920205401548e-06, 4.690917226601865e-07, -2.1322112805921556e-07, 9.721709759435981e-08, -4.4448656630335925e-08, 2.0373327115630335e-08, -9.359475430792408e-09, 4.308620855930645e-09, -1.9872392620357004e-09, 9.181429297400179e-10, -4.2489342599871804e-10, 1.969051449668413e-10, -9.139573819982871e-11, 4.2452263926406886e-11, -1.9768853221080462e-11, 9.190537220149508e-12, -4.2949394041258415e-12, 1.9981863386142606e-12, -9.396025624219817e-13, 4.335282133283158e-13, -2.0410756418343112e-13, 1.0455525334407412e-13, -4.748978987834107e-14, 2.7630675525358583e-14]
    Tmin, Tmax, Tc = 233.22, 646.15, 647.096

    expect = 0.031264474019763455
    expect_d, expect_d2 = -0.00023379922039411865, -1.0453010755999069e-07
    xmin, xmax = log(1-Tmin/Tc), log(1-Tmax/Tc)
    offset, scale = polynomial_offset_scale(xmin, xmax)
    coeffs_d = chebder(coeffs, m=1, scl=scale)
    coeffs_d2 = chebder(coeffs_d, m=1, scl=scale)
    coeffs_d3 = chebder(coeffs_d2, m=1, scl=scale)

    T = 500
    calc = exp_cheb_ln_tau(T, Tc, coeffs, offset, scale)
    assert 0 == exp_cheb_ln_tau(700, Tc, coeffs, offset, scale)
    assert_close(expect, calc)

    calc2 = exp_cheb_ln_tau_and_der(T, Tc, coeffs, coeffs_d, offset, scale)
    assert (0,0) == exp_cheb_ln_tau_and_der(700, Tc, coeffs, coeffs_d, offset, scale)
    assert_close(expect, calc2[0])
    assert_close(expect_d, calc2[1])


    calc3 = exp_cheb_ln_tau_and_der2(T, Tc, coeffs, coeffs_d, coeffs_d2, offset, scale)
    assert (0,0,0) == exp_cheb_ln_tau_and_der2(700, Tc, coeffs, coeffs_d, coeffs_d2, offset, scale)
    assert_close(expect, calc3[0])
    assert_close(expect_d, calc3[1])
    assert_close(expect_d2, calc3[2])





def test_chebval_ln_tau():
    Tmin, Tmax = 178.18, 591.0
    Tc = 591.75

    coeffs = [18231.740838720892, -18598.514785409734, 5237.841944302821, -1010.5549489362293, 147.88312821848922, -17.412144225239444, 1.7141064359038864, -0.14493639179363527, 0.01073811633477817, -0.0007078634084791702, 4.202655964036239e-05, -2.274648068123497e-06, 1.1239490049774759e-07]
    T = 500

    xmin, xmax = log(1-Tmin/Tc), log(1-Tmax/Tc)

    offset, scale = polynomial_offset_scale(xmin, xmax)
    coeffs_d = chebder(coeffs, m=1, scl=scale)
    coeffs_d2 = chebder(coeffs_d, m=1, scl=scale)
    coeffs_d3 = chebder(coeffs_d2, m=1, scl=scale)


    calc = chebval_ln_tau(T, Tc, coeffs, offset, scale)
    assert 0 == chebval_ln_tau(600, Tc, coeffs, offset, scale)
    expect = 24498.131947622023
    expect_d, expect_d2, expect_d3 = -100.77476795241955, -0.6838185834436981, -0.012093191904152178
    assert_close(expect, calc)

    calc2 = chebval_ln_tau_and_der(T, Tc, coeffs, coeffs_d, offset, scale)
    assert (0,0) == chebval_ln_tau_and_der(600, Tc, coeffs, coeffs_d, offset, scale)
    assert_close(expect, calc2[0])
    assert_close(expect_d, calc2[1])


    calc3 = chebval_ln_tau_and_der2(T, Tc, coeffs, coeffs_d, coeffs_d2, offset, scale)
    assert (0,0,0) == chebval_ln_tau_and_der2(600, Tc, coeffs, coeffs_d, coeffs_d2, offset, scale)
    assert_close(expect, calc3[0])
    assert_close(expect_d, calc3[1])
    assert_close(expect_d2, calc3[2])

    calc4 = chebval_ln_tau_and_der3(T, Tc, coeffs, coeffs_d, coeffs_d2, coeffs_d3, offset, scale)
    assert (0,0,0,0) == chebval_ln_tau_and_der3(600, Tc, coeffs, coeffs_d, coeffs_d2, coeffs_d3, offset, scale)
    assert_close(expect, calc4[0])
    assert_close(expect_d, calc4[1])
    assert_close(expect_d2, calc4[2])
    assert_close(expect_d3, calc4[3])

def test_exp_cheb():
    xmin, xmax = (309.0, 591.72)
    coeffs = [12.570668791524573, 3.1092695610681673, -0.5485217707981505, 0.11115875762247596, -0.01809803938553478, 0.003674911307077089, -0.00037626163070525465, 0.0001962813915017403, 6.120764548889213e-05, 3.602752453735203e-05]
    x = 400.0
    offset, scale = polynomial_offset_scale(xmin, xmax)
    expect = 157186.81766860923
    calc = exp_cheb(x, coeffs, offset, scale)
    assert_close(calc, expect, rtol=1e-14)

    coeffs_d = chebder(coeffs, m=1, scl=scale)
    coeffs_d2 = chebder(coeffs_d, m=1, scl=scale)
    coeffs_d3 = chebder(coeffs_d2, m=1, scl=scale)

    der_num = derivative(exp_cheb, x, args=(coeffs, offset, scale), dx=x*1e-7)
    der_analytical = exp_cheb_and_der(x, coeffs, coeffs_d, offset, scale)[1]
    assert_close(der_num, der_analytical, rtol=1e-7)
    assert_close(der_analytical, 4056.277312107932, rtol=1e-14)


    der_num = derivative(lambda *args: exp_cheb_and_der(*args)[1], x,
                         args=(coeffs, coeffs_d, offset, scale), dx=x*1e-7)
    der_analytical = exp_cheb_and_der2(x, coeffs, coeffs_d, coeffs_d2, offset, scale)[-1]
    assert_close(der_analytical, 81.34302144188977, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)


    der_num = derivative(lambda *args: exp_cheb_and_der2(*args)[-1], x,
                         args=(coeffs, coeffs_d, coeffs_d2, offset, scale), dx=x*1e-7)
    der_analytical = exp_cheb_and_der3(x, coeffs, coeffs_d, coeffs_d2, coeffs_d3, offset, scale)[-1]
    assert_close(der_num, der_analytical, rtol=1e-7)
    assert_close(der_analytical, 1.105438780935656, rtol=1e-14)

    vals = exp_cheb_and_der3(x, coeffs, coeffs_d, coeffs_d2, coeffs_d3, offset, scale)
    assert_close1d(vals, (157186.81766860923, 4056.277312107932, 81.34302144188977, 1.105438780935656), rtol=1e-14)

    vals = exp_cheb_and_der2(x, coeffs, coeffs_d, coeffs_d2, offset, scale)
    assert_close1d(vals, (157186.81766860923, 4056.277312107932, 81.34302144188977), rtol=1e-14)


    vals = exp_cheb_and_der(x, coeffs, coeffs_d, offset, scale)
    assert_close1d(vals, (157186.81766860923, 4056.277312107932), rtol=1e-14)


def test_cheb():
    Tmin, Tmax = 50, 1500.0
    toluene_TRC_cheb_fit = [194.9993931442641, 135.143566535142, -31.391834328585, -0.03951841213554952, 5.633110876073714, -3.686554783541794, 1.3108038668007862, -0.09053861376310801, -0.2614279887767278, 0.24832452742026911, -0.15919652548841812, 0.09374295717647019, -0.06233192560577938, 0.050814520356653126, -0.046331125185531064, 0.0424579816955023, -0.03739513702085129, 0.031402017733109244, -0.025212485578021915, 0.01939423141593144, -0.014231480849538403, 0.009801281575488097, -0.006075456686871594, 0.0029909809015365996, -0.0004841890018462136, -0.0014991199985455728, 0.0030051480117581075, -0.004076901418829215, 0.004758297389532928, -0.005096275567543218, 0.00514099984344718, -0.004944736724873944, 0.004560044671604424, -0.004037777783658769, 0.0034252408915679267, -0.002764690626354871, 0.0020922734527478726, -0.0014374230267101273, 0.0008226963858916081, -0.00026400260413972365, -0.0002288377348015347, 0.0006512726893767029, -0.0010030137199867895, 0.0012869214641443305, -0.001507857723972772, 0.001671575150882565, -0.0017837100581746812, 0.001848935469520696, -0.0009351605848800237]
    toluene_TRC_cheb_fit_copy = [v for v in toluene_TRC_cheb_fit]
    offset, scale = polynomial_offset_scale(Tmin, Tmax)

    val = chebval(300, toluene_TRC_cheb_fit, offset, scale)
    assert_close(val, 104.46956642594124, rtol=1e-14)

    d1_coeffs = chebder(toluene_TRC_cheb_fit, m=1, scl=scale)
    d2_coeffs = chebder(toluene_TRC_cheb_fit, m=2, scl=scale)
    d2_2_coeffs = chebder(d1_coeffs, m=1, scl=scale)

    val = chebval(300, d1_coeffs, offset, scale)
    assert_close(val, 0.36241217517888635, rtol=1e-14)

    val = chebval(300, d2_coeffs, offset, scale)
    assert_close(val, -6.445511348110282e-06, rtol=1e-14)

    val = chebval(300, d2_2_coeffs, offset, scale)
    assert_close(val, -6.445511348110282e-06, rtol=1e-14)
    assert d2_2_coeffs == d2_coeffs


    int_coeffs = chebint(toluene_TRC_cheb_fit, m=1, lbnd=0, scl=1/scale)
    assert_close(chebval(300, int_coeffs, offset, scale), -83708.18079449862, rtol=1e-10)

    assert toluene_TRC_cheb_fit == toluene_TRC_cheb_fit_copy

def test_cheb_more():
    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 0.5
    assert_close(result, expected, rtol=1e-13)

    c = [1]
    x = 1.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 1.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2]
    x = -1.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = -1.0
    assert_close(result, expected, rtol=1e-13)

    c = [0, 0, 1]
    x = 2.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 7.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, -1, 1, -1, 1]
    x = 0.1
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 1.1368
    assert_close(result, expected, rtol=1e-13)

    c = [0.1, 0.2, 0.3, 0.4, 0.5]
    x = 3.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 333.8999999999999
    assert_close(result, expected, rtol=1e-13)

    c = [10, 20, 30, 40, 50]
    x = -5.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 222030.0
    assert_close(result, expected, rtol=1e-13)

    c = [1e-05, 2e-05, 3e-05, 4e-05]
    x = 100.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 160.58998000000003
    assert_close(result, expected, rtol=1e-13)

    c = [1, 0, 0, 0, 1]
    x = 3.141592653589793
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 702.3158930633044
    assert_close(result, expected, rtol=1e-13)

    c = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = 0.75
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 6.9560546875
    assert_close(result, expected, rtol=1e-13)

    c = [1, -1, 1, -1, 1, -1, 1, -1]
    x = -0.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 1.5
    assert_close(result, expected, rtol=1e-13)

    c = [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666]
    x = 1.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 20.116666666666664
    assert_close(result, expected, rtol=1e-13)

    c = [2.718281828459045, 3.141592653589793, 2.718, 3.142]
    x = 0.25
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = -1.0346950081435065
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = -1
    scale = 2
    result = chebval(x, c, offset, scale)
    expected = -2.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    x = 1.0
    offset = 10
    scale = 100
    result = chebval(x, c, offset, scale)
    expected = 5877283503.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0.001
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 0.508006
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1e-06
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 0.5000080000060003
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1000.0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 6008000.5
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1000000.0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 6000008000000.5
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1000000000.0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 6.000000008e+18
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 0.001
    result = chebval(x, c, offset, scale)
    expected = -1.9989985
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = -1.9999989999985
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1000.0
    result = chebval(x, c, offset, scale)
    expected = 1500998.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1000000.0
    result = chebval(x, c, offset, scale)
    expected = 1500000999998.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1000000000.0
    result = chebval(x, c, offset, scale)
    expected = 1.500000001e+18
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1e-06
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = -1.9999969999865
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 1000000.0
    scale = 1000000.0
    result = chebval(x, c, offset, scale)
    expected = 13500002999998.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = -1000000000.0
    scale = 2000000000.0
    result = chebval(x, c, offset, scale)
    expected = -2.0
    assert_close(result, expected, rtol=1e-13)

    c = [1e-09, 2e-09, 3e-09]
    x = 1000000000.0
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 6000000002.0
    assert_close(result, expected, rtol=1e-13)

    c = [1000000000.0, 2000000000.0, 3000000000.0]
    x = 1e-09
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = -1999999998.0
    assert_close(result, expected, rtol=1e-13)

    c = [42]
    x = 0.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 42.0
    assert_close(result, expected, rtol=1e-13)

    c = [42]
    x = 1000
    offset = 1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = 42.0
    assert_close(result, expected, rtol=1e-13)

    c = [42]
    x = -1000
    offset = -1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = 42.0
    assert_close(result, expected, rtol=1e-13)

    c = [1e-09]
    x = 1000000000.0
    offset = 1e-09
    scale = 1000000000.0
    result = chebval(x, c, offset, scale)
    expected = 1.0e-9
    assert_close(result, expected, rtol=1e-13)

    c = [1000000000.0]
    x = 1e-09
    offset = 1000000000.0
    scale = 1e-09
    result = chebval(x, c, offset, scale)
    expected = 1000000000.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2]
    x = 0.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 2.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2]
    x = 1000
    offset = 1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = 2000001.002
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2]
    x = -1000
    offset = -1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = -1999999.002
    assert_close(result, expected, rtol=1e-13)

    c = [1e-09, 2e-09]
    x = 1000000000.0
    offset = 1e-09
    scale = 1000000000.0
    result = chebval(x, c, offset, scale)
    expected = 2000000000.0
    assert_close(result, expected, rtol=1e-13)

    c = [1000000000.0, 2000000000.0]
    x = 1e-09
    offset = 1000000000.0
    scale = 1e-09
    result = chebval(x, c, offset, scale)
    expected = 2.000000001e+18
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 0.5
    offset = 0
    scale = 1
    result = chebval(x, c, offset, scale)
    expected = 0.5
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = 1000
    offset = 1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = 6000002011998.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    x = -1000
    offset = -1000000.0
    scale = 1e-06
    result = chebval(x, c, offset, scale)
    expected = 5999998011998.0
    assert_close(result, expected, rtol=1e-13)

    c = [1e-09, 2e-09, 3e-09]
    x = 1000000000.0
    offset = 1e-09
    scale = 1000000000.0
    result = chebval(x, c, offset, scale)
    expected = 6.0e+27
    assert_close(result, expected, rtol=1e-13)

    c = [1000000000.0, 2000000000.0, 3000000000.0]
    x = 1e-09
    offset = 1000000000.0
    scale = 1e-09
    result = chebval(x, c, offset, scale)
    expected = 6.000000002e+27
    assert_close(result, expected, rtol=1e-13)

    c = [0]
    x = 3.141592653589793
    offset = -1000000.0
    scale = 1000000.0
    result = chebval(x, c, offset, scale)
    expected = 0.0
    assert_close(result, expected, rtol=1e-13)

    c = [1, 0]
    x = 1e-09
    offset = 1000000000.0
    scale = 1e-09
    result = chebval(x, c, offset, scale)
    expected = 1.0
    assert_close(result, expected, rtol=1e-13)

    c = [0, 0, 1]
    x = 1000000000.0
    offset = -1000000000.0
    scale = 2000000000.0
    result = chebval(x, c, offset, scale)
    expected = 7.999999992e+36
    assert_close(result, expected, rtol=1e-13)

    c = [1e-15]
    x = 1000000000000000.0
    offset = 1000000000000000.0
    scale = 1e-15
    result = chebval(x, c, offset, scale)
    expected = 1.0e-15
    assert_close(result, expected, rtol=1e-13)

    c = [1000000000000000.0, 1e-15]
    x = 1e-15
    offset = 1e-15
    scale = 1000000000000000.0
    result = chebval(x, c, offset, scale)
    expected = 1.0e+15
    assert_close(result, expected, rtol=1e-13)

def test_chebder_more():
    c = [1, 2, 3, 4, 5]
    m = 0
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [1, 2, 3, 4, 5]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [14.0, 52.0, 24, 40]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    scl = 2.0
    result = chebder(c, m, scl)
    expected = [28.0, 104.0, 48.0, 80.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    scl = 0.5
    result = chebder(c, m, scl)
    expected = [7.0, 26.0, 12.0, 20.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [172.0, 96.0, 240]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    scl = 2.0
    result = chebder(c, m, scl)
    expected = [688.0, 384.0, 960.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    scl = 0.5
    result = chebder(c, m, scl)
    expected = [43.0, 24.0, 60.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [96.0, 960.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    scl = 2.0
    result = chebder(c, m, scl)
    expected = [768.0, 7680.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    scl = 0.5
    result = chebder(c, m, scl)
    expected = [12.0, 120.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 4
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [960.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 4
    scl = 2.0
    result = chebder(c, m, scl)
    expected = [15360.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 4
    scl = 0.5
    result = chebder(c, m, scl)
    expected = [60.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1]
    m = 1
    scl = 1.0
    result = chebder(c, m, scl)
    expected = []
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2]
    m = 1
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [2]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    m = 3
    scl = 1.0
    result = chebder(c, m, scl)
    expected = []
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    m = 4
    scl = 1.0
    result = chebder(c, m, scl)
    expected = []
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 2, 3]
    m = 5
    scl = 1.0
    result = chebder(c, m, scl)
    expected = []
    assert_close1d(result, expected, rtol=1e-13)

    c = [1, 0, 1, 0, 1]
    m = 2
    scl = 1.0
    result = chebder(c, m, scl)
    expected = [36.0, 0.0, 48]
    assert_close1d(result, expected, rtol=1e-13)

    c = [1e-05, 2e-05, 3e-05, 4e-05]
    m = 1
    scl = 100000.0
    result = chebder(c, m, scl)
    expected = [14.0, 12.0, 24.0]
    assert_close1d(result, expected, rtol=1e-13)

    c = [100000.0, 200000.0, 300000.0, 400000.0]
    m = 1
    scl = 1e-05
    result = chebder(c, m, scl)
    expected = [14.0, 12.000000000000002, 24.0]
    assert_close1d(result, expected, rtol=1e-13)

def test_chebint_more():
    c = [1, 2, 3, 4, 5]
    m = 0
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-1.0, -0.5, -0.5, -0.33333333333333337, 0.5, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    lbnd = 1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.3333333333333335, -0.5, -0.5, -0.33333333333333337, 0.5, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    lbnd = -1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.33333333333333337, -0.5, -0.5, -0.33333333333333337, 0.5, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    lbnd = 0
    scl = 2.0
    result = chebint(c, m, lbnd, scl)
    expected = [-2.0, -1.0, -1.0, -0.6666666666666667, 1.0, 1.0]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 1
    lbnd = 0
    scl = 0.5
    result = chebint(c, m, lbnd, scl)
    expected = [-0.5, -0.25, -0.25, -0.16666666666666669, 0.25, 0.25]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.10416666666666669, -0.75, -0.04166666666666666, -0.16666666666666666, -0.10416666666666667, 0.05, 0.041666666666666664]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    lbnd = 1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.36250000000000016, 0.5833333333333335, -0.04166666666666666, -0.16666666666666666, -0.10416666666666667, 0.05, 0.041666666666666664]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    lbnd = -1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.09583333333333335, -0.08333333333333337, -0.04166666666666666, -0.16666666666666666, -0.10416666666666667, 0.05, 0.041666666666666664]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 2
    lbnd = 0
    scl = 2.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.41666666666666674, -3.0, -0.16666666666666663, -0.6666666666666666, -0.4166666666666667, 0.2, 0.16666666666666666]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.11458333333333334, 0.125, -0.14583333333333334, 0.010416666666666668, -0.027083333333333334, -0.014583333333333334, 0.004166666666666667, 0.002976190476190476]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    lbnd = 1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.17827380952380967, -0.34166666666666684, 0.18750000000000003, 0.010416666666666668, -0.027083333333333334, -0.014583333333333334, 0.004166666666666667, 0.002976190476190476]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 3
    lbnd = -1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.07410714285714287, -0.07500000000000002, 0.020833333333333322, 0.010416666666666668, -0.027083333333333334, -0.014583333333333334, 0.004166666666666667, 0.002976190476190476]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 4
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.023871527777777776, -0.04166666666666667, 0.028645833333333332, -0.019791666666666666, 0.003125, -0.003125, -0.0014632936507936508, 0.0002976190476190476, 0.00018601190476190475]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4, 5]
    m = 4
    lbnd = 1
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [-0.03128720238095246, 0.08452380952380965, -0.08802083333333338, 0.035763888888888894, 0.003125, -0.003125, -0.0014632936507936508, 0.0002976190476190476, 0.00018601190476190475]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1]
    m = 1
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.0, 1.0]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2]
    m = 1
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.5, 1.0, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [0]
    m = 1
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.0]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3]
    m = 5
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.0008680555555555565, -0.009114583333333336, 0.0013020833333333343, -0.004427083333333334, 0.0005208333333333334, -0.0007812499999999999, 8.680555555555556e-05, 3.720238095238095e-05]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 0, 1, 0, 1]
    m = 2
    lbnd = 0
    scl = 1.0
    result = chebint(c, m, lbnd, scl)
    expected = [0.14583333333333334, 0.0, 0.125, 0.0, -0.0125, 0.0, 0.008333333333333333]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1e-05, 2e-05, 3e-05, 4e-05]
    m = 1
    lbnd = 0
    scl = 100000.0
    result = chebint(c, m, lbnd, scl)
    expected = [-1.0, -0.5, -0.5, 0.5, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [100000.0, 200000.0, 300000.0, 400000.0]
    m = 1
    lbnd = 0
    scl = 1e-05
    result = chebint(c, m, lbnd, scl)
    expected = [-1.0, -0.5000000000000002, -0.5, 0.5000000000000001, 0.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3]
    m = 1
    lbnd = 2
    scl = 3
    result = chebint(c, m, lbnd, scl)
    expected = [-46.5, -1.5, 1.5, 1.5]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

    c = [1, 2, 3, 4]
    m = 2
    lbnd = -1
    scl = 0.5
    result = chebint(c, m, lbnd, scl)
    expected = [0.08020833333333333, 0.0625, -0.0625, -0.041666666666666664, 0.015625, 0.0125]
    assert_close1d(result, expected, rtol=1e-13, atol=1e-13)

def test_is_monotonic():
    assert is_monotonic([1,2,3])
    assert is_monotonic([3, 2, 1])
    assert is_monotonic([1,1,2,3])
    assert is_monotonic([3,3, 2, 1])
    assert is_monotonic([1])
    assert not is_monotonic([3,3,5, 2, 1])
    assert not is_monotonic([1,2,3,1])

    assert is_monotonic([-1, -2])
    assert not is_monotonic([-1, -2, -3, -2])
    assert is_monotonic([-2, -1, 0])
    assert not is_monotonic([-2, -1, 0, -1])



def test_secant_cases_tough():
    def to_solve(x):
        err =  -0.1 if x < 0.5 else x - 0.6  # Bad case, completely flat, cannot convergewhen x < 0.5 but difficulties elsewhere too
        return err
    assert_close(secant(to_solve, 1e20, xtol=1e-12, ytol=1e-20, require_xtol=False, bisection=True, maxiter=200), 0.6)

    def to_solve(x):
        # Have to try multiple initial guesses to get a working point
        err = x**20.0 - 1.0
        return err
    assert_close(secant(to_solve, .104, xtol=1e-12, ytol=1e-20, require_xtol=False, bisection=True, maxiter=1000, additional_guesses=True),
                 1)


    def to_solve(x):
        err = 10.0**14*(1.0*x**7 - 7.0*x**6 + 21.0*x**5 - 35.0*x**4 + 35.0*x**3 - 21.0*x**2 + 7.0*x - 1.0)
        return err
    assert_close(secant(to_solve, 1.013, xtol=1e-12, ytol=1e-20, require_xtol=False, bisection=True, maxiter=1000, additional_guesses=True),
                 0.9926199643387525)

# Go through a set of cases collected online
def newton_baffler(x):
    y = x - 6.25
    if y < -0.25:
        return 0.75*y- 0.3125
    elif y < 0.25:
        return 2.0*y
    else:
        return 0.75*y + 0.3125

def newton_baffler_d(x):
    # others are zero
    y = x - 6.25
    if y < -0.25:
        return 0.75
    elif y < 0.25:
        return 2.0
    else:
        return 0.75


def flat_stanley(x):
    if x == 1:
        return 0
    else:
        if x < 1:
            return -exp(log(1000) + log(1.0 - x) - 1.0/(x - 1.0)**2)
        return exp(log(1000) + log(x - 1.0) - 1.0/(x - 1.0)**2)
        # factor = (-1.0 if x < 1.0 else 1.0)
        # return factor*exp(log(1000) + log(abs(x - 1.0)) - 1.0/(x - 1.0)**2)

def flat_stanley_d(x):
    if x == 1:
        return 0
    else:
        if x < 1:
            return 1000*exp(-1.0/(x - 1.0)**2) + 2.0*(1000*x - 1000.0)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**3
        return 1000*exp(-1.0/(x - 1.0)**2) + 2.0*(1000*x - 1000.0)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**3

def flat_stanley_d2(x):
    if x == 1:
        return 0
    else:
        if x < 1:
            return (-(6.0 - 4.0/(x - 1.0)**2)*(1000*x - 1000.0)/(x - 1.0) + 4000.0)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**3
        return (-(6.0 - 4.0/(x - 1.0)**2)*(1000*x - 1000.0)/(x - 1.0) + 4000.0)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**3

def flat_stanley_d3(x):
    if x == 1:
        return 0
    else:
        if x < 1:
            return (-18000.0 + (1000*x - 1000.0)*(24.0 - 36.0/(x - 1.0)**2 + 8.0/(x - 1.0)**4)/(x - 1.0) + 12000.0/(x - 1.0)**2)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**4
        return (-18000.0 + (1000*x - 1000.0)*(24.0 - 36.0/(x - 1.0)**2 + 8.0/(x - 1.0)**4)/(x - 1.0) + 12000.0/(x - 1.0)**2)*exp(-1.0/(x - 1.0)**2)/(x - 1.0)**4

def newton_pathological(x):
    if x == 0.0:
        return 0.0
    else:
        factor = (-1.0 if x < 0.0 else 1.0)
        return factor*abs(x)**(1.0/3.0)*exp(-x**2)


solver_1d_test_cases = [(lambda x: sin(x) - x/2, lambda x: cos(x) - 1/2, lambda x: -sin(x), lambda x: -cos(x), [0.1], None, None, 'TEST_ZERO #1'),
        (lambda x: 2*x - exp(-x), lambda x: 2 + exp(-x), lambda x: -exp(-x), lambda x: exp(-x), [1.], None, None, 'TEST_ZERO #2'),
        (lambda x: x*exp(-x), lambda x: -x*exp(-x) + exp(-x), lambda x: (x - 2)*exp(-x), lambda x: (3 - x)*exp(-x), [1.], None, None, 'TEST_ZERO #3'),
        (lambda x: exp(x) - 1.0/(10.0*x)**2, lambda x: exp(x) + 0.02/x**3, lambda x: exp(x) - 0.06/x**4, lambda x: exp(x) + 0.24/x**5, [1.], None, None, 'TEST_ZERO #4'),
        (lambda x: (x+3)*(x-1)**2, lambda x: (x - 1)**2 + (x + 3)*(2*x - 2), lambda x: 2*(3*x + 1), lambda x: 6, [1.], None, None, 'TEST_ZERO #5'),
        (lambda x: exp(x) - 2 - 1/(10*x)**2 + 2/(100*x)**3, lambda x: exp(x) + 1/(50*x**3) - 3/(500000*x**4), lambda x: exp(x) - 3/(50*x**4) + 3/(125000*x**5), lambda x: exp(x) + 6/(25*x**5) - 3/(25000*x**6), [1.], None, None, 'TEST_ZERO #6'),
        (lambda x: x**3, lambda x: 3*x**2, lambda x: 6*x, lambda x: 6.0, [1.], None, None, 'TEST_ZERO #7'),
        (lambda x: cos(x) - x, lambda x: -sin(x) - 1, lambda x: -cos(x), lambda x: sin(x), [1.], None, None, 'TEST_ZERO #8'),
        (newton_baffler, newton_baffler_d, lambda x: 0, lambda x: 0, [6.25 + 5.0, 6.25 - 1.0, 6.25 + 0.1], None, None, 'TEST_ZERO #9/Newton Baffler'),
        (lambda x: 20*x/(100*x**2 + 1), lambda x: -4000*x**2/(100*x**2 + 1)**2 + 20/(100*x**2 + 1), lambda x: 4000*x*(400*x**2/(100*x**2 + 1) - 3)/(100*x**2 + 1)**2, lambda x: 12000*(-400*x**2*(200*x**2/(100*x**2 + 1) - 1)/(100*x**2 + 1) + 400*x**2/(100*x**2 + 1) - 1)/(100*x**2 + 1)**2, [1., -0.14, 0.041], None, None, 'TEST_ZERO #10/Repeller'),
        (lambda x: (16 - x**4)/(16*x**4 + 0.00001), lambda x: -x**3*(16 - x**4)/(4*(x**4 + 6.25e-7)**2) - 4*x**3/(16*x**4 + 1.0e-5), lambda x: x**2*(2*x**4/(x**4 + 6.25e-7)**2 - (x**4 - 16)*(8*x**4/(x**4 + 6.25e-7) - 3)/(4*(x**4 + 6.25e-7)**2) - 12/(16*x**4 + 1.0e-5)), lambda x: 3*x*(-x**4*(8*x**4/(x**4 + 6.25e-7) - 3)/(x**4 + 6.25e-7)**2 + 3*x**4/(x**4 + 6.25e-7)**2 + (x**4 - 16)*(16*x**8/(x**4 + 6.25e-7)**2 - 12*x**4/(x**4 + 6.25e-7) + 1)/(2*(x**4 + 6.25e-7)**2) - 8/(16*x**4 + 1.0e-5)), [.25, 5., 1.1], None, None, 'TEST_ZERO #11/Pinhead'),
        (flat_stanley, flat_stanley_d, flat_stanley_d2, flat_stanley_d3, [2, 0.5, 4], None, None, 'TEST_ZERO #12/Flat Stanley'),
        (lambda x: 0.00000000001*(x - 100),lambda x: 1.00000000000000e-11, lambda x: 0, lambda x: 0, [100., 1.], None, None, 'TEST_ZERO #13/Lazy Boy'),
        (lambda x: 1.0/((x - 0.3)**2 + 0.01) + 1.0/((x - 0.9)**2 + 0.04) + 2.0*x - 5.2, lambda x: 1.0*(0.6 - 2*x)/((x - 0.3)**2 + 0.01)**2 + 1.0*(1.8 - 2*x)/((x - 0.9)**2 + 0.04)**2 + 2.0, lambda x: 1.0*(2*x - 1.8)*(4*x - 3.6)/((x - 0.9)**2 + 0.04)**3 + 1.0*(2*x - 0.6)*(4*x - 1.2)/((x - 0.3)**2 + 0.01)**3 - 2.0/((x - 0.3)**2 + 0.01)**2 - 2.0/((x - 0.9)**2 + 0.04)**2, lambda x: -1.0*(2*x - 1.8)*(4*x - 3.6)*(6*x - 5.4)/((x - 0.9)**2 + 0.04)**4 - 1.0*(2*x - 0.6)*(4*x - 1.2)*(6*x - 1.8)/((x - 0.3)**2 + 0.01)**4 + (8.0*x - 7.2)/((x - 0.9)**2 + 0.04)**3 + (8.0*x - 2.4)/((x - 0.3)**2 + 0.01)**3 + (16.0*x - 14.4)/((x - 0.9)**2 + 0.04)**3 + (16.0*x - 4.8)/((x - 0.3)**2 + 0.01)**3, [3., -0.5, 0, 2.12742], None, None, 'TEST_ZERO #14/Camel'),
        # (newton_pathological, [0.01, -0.25], None, None, 'TEST_ZERO #15/Newton Pathological'),
        (lambda x: pi*(x - 5.0)/180.0 - 0.8*sin(pi*x/180), lambda x: -0.00444444444444444*pi*cos(pi*x/180) + 0.00555555555555556*pi, lambda x: 2.46913580246914e-5*pi**2*sin(pi*x/180), lambda x: 1.37174211248285e-7*pi**3*cos(pi*x/180), [0., 5+180., 5.], None, None, 'TEST_ZERO #16/Kepler'),
        (lambda x: x**3 - 2*x - 5,lambda x: 3*x**2 - 2, lambda x: 6*x, lambda x: 6,  [2., 3.], None, None, 'TEST_ZERO #17/Wallis example'),
        (lambda x: 10.0**14*(1.0*x**7 - 7.0*x**6 + 21.0*x**5 - 35.0*x**4 + 35.0*x**3 - 21.0*x**2 + 7.0*x - 1.0), lambda x: 700000000000000.0*x**6 - 4.2e+15*x**5 + 1.05e+16*x**4 - 1.4e+16*x**3 + 1.05e+16*x**2 - 4.2e+15*x + 700000000000000.0, lambda x: 4.2e+15*x**5 - 2.1e+16*x**4 + 4.2e+16*x**3 - 4.2e+16*x**2 + 2.1e+16*x - 4.2e+15, lambda x: 2.1e+16*x**4 - 8.4e+16*x**3 + 1.26e+17*x**2 - 8.4e+16*x + 2.1e+16, [.99, 1.013], None, None, 'TEST_ZERO #18'),
        (lambda x: cos(100.0*x) - 4.0*erf(30*x - 10), lambda x: -100.0*sin(100.0*x) - 240.0*exp(-(30*x - 10)**2)/sqrt(pi), lambda x: (432000.0*x - 144000.0)*exp(-100*(3*x - 1)**2)/sqrt(pi) - 10000.0*cos(100.0*x), lambda x: -86400000.0*(3*x - 1)**2*exp(-100*(3*x - 1)**2)/sqrt(pi) + 1000000.0*sin(100.0*x) + 432000.0*exp(-100*(3*x - 1)**2)/sqrt(pi), [0., 1., 0.5], None, None, 'TEST_ZERO #19'),

#         # From scipy
        (lambda x: x**2 - 2*x - 1, lambda x: 2*x - 2, lambda x: 2, lambda x: 0, [3,], None, None, 'Scipy/x**2 - 2*x - 1'),
        (lambda x: exp(x) - cos(x), lambda x: exp(x) + sin(x), lambda x: exp(x) + cos(x), lambda x: exp(x) - sin(x), [3,], None, None, 'Scipy/exp(x) - cos(x)'),
        (lambda x: x - 0.1, lambda x: 1, lambda x: 0, lambda x: 0, [-1e8, 1e7], None, None, 'Scipy/GH5555'),
        (lambda x: -0.1 if x < 0.5 else x - 0.6, lambda x: 0.0 if x < 0.5 else 1.0, lambda x: 0, lambda x: 0, [1e20, 1.], None, None, 'Scipy/GH5557'), # Fail at 0 is expected - 0 slope
        (lambda x: (x - 100.0)**2, lambda x: 2.0*x - 200.0, lambda x: 2., lambda x: 0, [10*(200.0 - 6.828499381469512e-06) / (2.0 + 6.828499381469512e-06)], None, None, 'Scipy/zero_der_nz_dp'),
        (lambda x: x**3 - x**2, lambda x: 3*x**2 - 2*x, lambda x: 2*(3*x - 1), lambda x: 6, [0., 0.5], None, None, 'Scipy/GH8904'),
        (lambda x: x**(1.00/9.0) - 9**(1.0/9), lambda x: 1/(9*x**(8/9)), lambda x: -8/(81*x**(17/9)), lambda x: 136/(729*x**(26/9)), [0.1], None, None, 'Scipy/GH8881'),

#         # From scipy optimization suite - root functions only
        (lambda x: sin(x) + sin(10.0 / 3.0 * x), lambda x: cos(x) + 10*cos(10*x/3)/3, lambda x: -(sin(x) + 100*sin(10*x/3)/9), lambda x: -(cos(x) + 1000*cos(10*x/3)/27),  [2.7048, 3.18, 4.14, 4.62, 5.1, 5.58, 6.06, 7.02, 7.4952], 2.7, 7.5, 'Scipy/Problem02'),
        (lambda x: -sum(k * sin((k + 1) * x + k) for k in range(1, 6)), lambda x: -2*cos(2*x + 1) - 6*cos(3*x + 2) - 12*cos(4*x + 3) - 20*cos(5*x + 4) - 30*cos(6*x + 5), lambda x: 2*(2*sin(2*x + 1) + 9*sin(3*x + 2) + 24*sin(4*x + 3) + 50*sin(5*x + 4) + 90*sin(6*x + 5)), lambda x: 2*(4*cos(2*x + 1) + 27*cos(3*x + 2) + 96*cos(4*x + 3) + 250*cos(5*x + 4) + 540*cos(6*x + 5)), [-9.98, -8.0, -4.0, -2.0, 0.0, 2.0, 4, 8.0, 9.98], -10, 10, 'Scipy/Problem03'),
        (lambda x: -(1.4 - 3 * x) * sin(18.0 * x), lambda x: 18.0*(3*x - 1.4)*cos(18.0*x) + 3*sin(18.0*x), lambda x: -324.0*(3*x - 1.4)*sin(18.0*x) + 108.0*cos(18.0*x), lambda x: -(5832.0*(3*x - 1.4)*cos(18.0*x) + 2916.0*sin(18.0*x)), [0.0012, 0.12, 0.36, 0.48, 0.6, 0.72, 0.84, 1.08, 1.1988], 0, 1.2, 'Scipy/Problem05'),
        (lambda x: -(x + sin(x)) * exp(-x ** 2.0), lambda x: -2*x*(-x - sin(x))*exp(-x**2) + (-cos(x) - 1)*exp(-x**2), lambda x: (4*x*(cos(x) + 1) - 2*(x + sin(x))*(2*x**2 - 1) + sin(x))*exp(-x**2), lambda x: (4*x*(x + sin(x))*(2*x**2 - 3) - 6*x*sin(x) - 6*(2*x**2 - 1)*(cos(x) + 1) + cos(x))*exp(-x**2), [-9.98, -8.0, -4.0, -2.0, 0.0, 2.0, 4, 8.0, 9.98], -10, 10, 'Scipy/Problem06'),
        (lambda x: sin(x) + sin(10.0 / 3.0 * x) + log(x) - 0.84 * x + 3, lambda x: cos(x) + 10*cos(10*x/3)/3 - 21/25 + 1/x, lambda x: -(sin(x) + 100*sin(10*x/3)/9 + x**(-2)), lambda x: -cos(x) - 1000*cos(10*x/3)/27 + 2/x**3, [2.7048, 3.18, 4.14, 4.62, 5.1, 5.58, 6.06, 7.02, 7.4952], 2.7, 7.5, 'Scipy/Problem07'),
        (lambda x: -sum(k * cos((k + 1) * x + k) for k in range(1, 6)), lambda x: 2*sin(2*x + 1) + 6*sin(3*x + 2) + 12*sin(4*x + 3) + 20*sin(5*x + 4) + 30*sin(6*x + 5), lambda x: 2*(2*cos(2*x + 1) + 9*cos(3*x + 2) + 24*cos(4*x + 3) + 50*cos(5*x + 4) + 90*cos(6*x + 5)), lambda x: -2*(4*sin(2*x + 1) + 27*sin(3*x + 2) + 96*sin(4*x + 3) + 250*sin(5*x + 4) + 540*sin(6*x + 5)), [-9.98, -8.0, -4.0, -2.0, 0.0, 2.0, 4, 8.0, 9.98], -10, 10, 'Scipy/Problem08'),
        (lambda x: sin(x) + sin(2.0 / 3.0 * x), lambda x: 2*cos(2*x/3)/3 + cos(x), lambda x: -(4*sin(2*x/3)/9 + sin(x)), lambda x: -(8*cos(2*x/3)/27 + cos(x)), [3.1173, 4.83, 8.29, 10.02, 11.75, 13.48, 15.21, 18.67, 20.3827], 3.1, 20.4, 'Scipy/Problem09'),
        (lambda x: -x * sin(x), lambda x: -x*cos(x) - sin(x), lambda x: x*sin(x) - 2*cos(x), lambda x: x*cos(x) + 3*sin(x), [0.01, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 9.99], 0, 10, 'Scipy/Problem10'),
        (lambda x: 2 * cos(x) + cos(2 * x), lambda x: -2*sin(x) - 2*sin(2*x), lambda x: -2*(cos(x) + 2*cos(2*x)), lambda x: 2*(sin(x) + 4*sin(2*x)), [-1.5629423451609221, -0.7853981633974483, 0.7853981633974483, 1.5707963267948966, 2.356194490192345,
                                                3.141592653589793, 3.9269908169872414, 5.497787143782138, 6.275331325545612],
            -pi / 2, 2 * pi, 'Scipy/Problem11'),
        (lambda x: (sin(x))**3.0 + (cos(x))**3.0, lambda x: 3*sin(x)**2*cos(x) - 3*sin(x)*cos(x)**2, lambda x: 3*(-sin(x)**3 + 2*sin(x)**2*cos(x) + 2*sin(x)*cos(x)**2 - cos(x)**3), lambda x: 3*(-2*sin(x)**3 - 7*sin(x)**2*cos(x) + 7*sin(x)*cos(x)**2 + 2*cos(x)**3), [0.006283185307179587, 0.6283185307179586, 1.8849555921538759, 2.5132741228718345,
                                                    3.141592653589793, 3.7699111843077517, 4.39822971502571, 5.654866776461628, 6.276902121872407],
            0, 2*pi, 'Scipy/Problem12'),
        (lambda x: -exp(-x) * sin(2.0 * pi * x), lambda x: exp(-x)*sin(2*pi*x) - 2*pi*exp(-x)*cos(2*pi*x), lambda x: (-sin(2*pi*x) + 4*pi**2*sin(2*pi*x) + 4*pi*cos(2*pi*x))*exp(-x), lambda x: (-12*pi**2*sin(2*pi*x) + sin(2*pi*x) - 6*pi*cos(2*pi*x) + 8*pi**3*cos(2*pi*x))*exp(-x), [0.004, 0.4, 1.2, 1.6, 2.0, 2.4, 2.8, 3.6, 3.996], 0, 4, 'Scipy/Problem14'),
        (lambda x: -(x - sin(x)) * exp(-x ** 2.0), lambda x: -2*x*(-x + sin(x))*exp(-x**2) + (cos(x) - 1)*exp(-x**2), lambda x: -(4*x*(cos(x) - 1) + 2*(x - sin(x))*(2*x**2 - 1) + sin(x))*exp(-x**2), lambda x: (4*x*(x - sin(x))*(2*x**2 - 3) + 6*x*sin(x) + 6*(2*x**2 - 1)*(cos(x) - 1) - cos(x))*exp(-x**2), [-9.98, -8.0, -4.0, -2.0, 0.0, 2.0, 4, 8.0, 9.98], -10, 10, 'Scipy/Problem20'),
        (lambda x: x * sin(x) + x * cos(2.0 * x), lambda x: -2.0*x*sin(2.0*x) + x*cos(x) + sin(x) + cos(2.0*x), lambda x: -x*sin(x) - 4.0*x*cos(2.0*x) - 4.0*sin(2.0*x) + 2*cos(x), lambda x: 8.0*x*sin(2.0*x) - x*cos(x) - 3*sin(x) - 12.0*cos(2.0*x), [0.01, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 9.99], 0, 10, 'Scipy/Problem21'),
        (lambda x: exp(-3.0 * x) - (sin(x)) ** 3.0, lambda x: -3*sin(x)**2*cos(x) - 3*exp(-3*x), lambda x: 3*(sin(x)**3 - 2*sin(x)*cos(x)**2 + 3*exp(-3*x)), lambda x: 3*(7*sin(x)**2*cos(x) - 2*cos(x)**3 - 9*exp(-3*x)), [0.02, 2.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 19.98], 0, 20, 'Scipy/Problem22'),


#         # from gsl
        (lambda x: sin(x), lambda x: cos(x), lambda x: -sin(x), lambda x: -cos(x), [3, 4, -4, -3, -1/3., 1], None, None, 'sin(x)'),
        (lambda x: cos(x), lambda x: -sin(x), lambda x: -cos(x), lambda x: sin(x), [0, 3, -3, 0], None, None, 'cos(x)'),
        (lambda x: x**20 - 1,lambda x: 20*x**19, lambda x: 380*x**18, lambda x: 6840*x**17, [0.1, 2], None, None, 'x^20 - 1'), # Numerical derivative at .1 is 0, failing secant
        # (lambda x: np.sign(x)*abs(x)**0.5, [-1.0/3, 1], None, None, 'sign(x)*sqrt(abs(x))'),
        (lambda x: x**2 - 1e-8, lambda x: 2*x, lambda x: 2, lambda x: 0, [0, 1], None, None, 'x^2 - 1e-8'),
        (lambda x: (x-1.0)**7, lambda x: 7*(x - 1.0)**6, lambda x: 42*(x - 1.0)**5, lambda x: 210*(x - 1.0)**4, [0.9995, 1.0002], None, None, '(x-1.0)**7'),

        # From Roots.jl
        # (lambda x: abs(x - 0.0), [0, 1], None, None, 'abs(x - 0.0)'),
        (lambda x: 1024*x**11 - 2816*x**9 + 2816*x**7 - 1232*x**5 + 220*x**3 - 11*x, lambda x: 11264*x**10 - 25344*x**8 + 19712*x**6 - 6160*x**4 + 660*x**2 - 11, lambda x: 88*x*(1280*x**8 - 2304*x**6 + 1344*x**4 - 280*x**2 + 15), lambda x: 264*(3840*x**8 - 5376*x**6 + 2240*x**4 - 280*x**2 + 5), [0, -1, 1, 21], None, None, '1024*x**11 - 2816*x**9 + 2816*x**7 - 1232*x**5 + 220*x**3 - 11*x'),
        (lambda x: 512*x**9 - 1024*x**7 + 672*x**5 - 160*x**3 +10*x, lambda x: 4608*x**8 - 7168*x**6 + 3360*x**4 - 480*x**2 + 10, lambda x: 192*x*(192*x**6 - 224*x**4 + 70*x**2 - 5), lambda x: 192*(1344*x**6 - 1120*x**4 + 210*x**2 - 5), [0, -1, 1, 7], None, None, '512*x**9 - 1024*x**7 + 672*x**5 - 160*x**3 +10*x')
        ]

def test_secant_cases_internet():

    solvers = [secant]
    solver_kwargs = [{'xtol': 1e-14, 'bisection': True, 'require_xtol': False, 'ytol': 1e-12,
                      'additional_guesses': True, 'require_eval': True},
                     ]
    names = ['fluids secant']
    def print_err(fun):
        def f(x):
            err = fun(x)
    #         print([err, x])
            return err
        return f

    for solver, kwargs, name in zip(solvers, solver_kwargs, names):
        passes = 0
        fails = 0
        failed_fs = set()
        for f, fder, fder2, fder3, xs, low, high, note in solver_1d_test_cases:
            for x0 in xs:
                v = solver(print_err(f), x0, maxiter=3000, low=low, high=high, **kwargs)
                try:
                    v = solver(print_err(f), x0, maxiter=3000, low=low, high=high, **kwargs)
                    assert abs(f(v)) < 1e-10
                    passes += 1
                except Exception as e:
                    failed_fs.add(f)
                    fails += 1
        assert fails == 0

@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
@pytest.mark.filterwarnings("ignore:invalid value encountered in power")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar power")
def test_secant_cases_nan_inf():
    import numpy as np
    from fluids.numerics import UnconvergedError
    def div_by_zero(x):
        err = 1 / (np.array(x) - 1)
        return err

    def sqrt_of_negative(x):
        x = np.array(x)
        return (x - 2) ** 0.5

    with pytest.raises((ValueError, UnconvergedError)):
        secant(div_by_zero, 1)

    with pytest.raises((ValueError, UnconvergedError)):
        secant(div_by_zero, x0=2, x1=1)

    with pytest.raises((ValueError, UnconvergedError)):
        secant(sqrt_of_negative, 1)

    with pytest.raises((ValueError, UnconvergedError)):
        secant(sqrt_of_negative, x0=4, x1=1)

    # test we can solve :)
    assert secant(sqrt_of_negative, 2, x1=2.1) == 2

    # Force the solver to go into the nan regime via a high damping factor
    with pytest.raises((ValueError, UnconvergedError)):
        secant(sqrt_of_negative, 3, x1=2.1, damping=4)

def test_newton_halley_cases_internet():
    debug = False
    fcalls = 0
    from math import inf, isinf, isnan

    from fluids.numerics import newton
    def print_err(fun):
        def f(x):
            nonlocal fcalls
            fcalls += 1
            try:
                err = fun(x)
            except:
                return inf
            if err.imag != 0.0:
                return inf
            # if debug:
            #     print(f"x={x}, err={err} fder={fder(x)}")
            return err
        return f
    def print_fder(fun):
        def f(x):
            try:
                fder = fun(x)
            except:
                return inf
            if fder.imag != 0.0:
                return inf
            return fder
        return f

    solvers = [newton]
    solver_kwargs = [{'xtol': 1e-14, 'bisection': True, 'ytol': 1e-11, 'require_xtol': False,
                    'require_eval': True, 'additional_guesses': True},
                    ]
    names = ['fluids newton']
    for solver, kwargs, name in zip(solvers, solver_kwargs, names):
        passes = 0
        fails = 0
        failed_fs = set()
        for f, fder, fder2, fder3, xs, low, high, note in solver_1d_test_cases:
            for x0 in xs:
                # if debug:
                #     print(f'Solving with x0={x0}')
                try:
                    # low=low, high=high
                    v = solver(print_err(f), x0, print_fder(fder), low=low, high=high, maxiter=3000, **kwargs)
                    v = solver(print_err(f), x0, print_fder(fder), low=low, high=high, fprime2=fder2, maxiter=3000, **kwargs)
                    v = solver(print_err(f), x0, print_fder(fder), fprime2=fder2, maxiter=3000, **kwargs)
                    assert not isinf(v)
                    assert not isnan(v)
                    assert abs(f(v)) < 1e-10
                    passes += 1
                except Exception as e:
                    # print('Failed', e, note, x0, low, high)
                    failed_fs.add(f)
                    fails += 1
        assert fails == 0




def test_basic_newton_system():
    # Define system of equations
    def system(inputs):
        x, y = inputs
        # Example system:
        # f1 = x^2 + y^2 - 4 = 0
        # f2 = exp(x) - y = 0
        return [x**2 + y**2 - 4, exp(x) - y]
    
    # Define Jacobian matrix
    def jacobian(inputs):
        x, y = inputs
        return [
            [2*x, 2*y],
            [exp(x), -1]
        ]
    # Initial guess
    x0 = [1.0, 1.0]
    # Solve system
    solution, iterations = newton_system(
        f=system,
        x0=x0,
        jac=jacobian,
        xtol=1e-10,
        maxiter=100
    )
    
    # Check results
    assert iterations > 0 
    assert iterations < 10  # Should converge in about 6 iters
    # Verify solution satisfies equations
    final_residuals = system(solution)
    assert_close1d(final_residuals, [0, 0], atol=1e-6)

def test_newton_system_no_iterations():
    def test_objf_direct(inputs):
        x, T = inputs
        k = 0.12*exp(12581*(T-298.)/(298.*T))
        return [120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)]
    def test_jac_not_called(inputs):
        raise ValueError("Should not be called")

    expect = [0.05995136780143791, 296.85996516970505]
    found, iterations = newton_system(test_objf_direct, expect, jac=test_jac_not_called, ytol=1e-7, xtol=None)
    assert iterations == 0
    assert found == expect

def test_newton_system_with_damping_function():
    # can get up to all sorts of fun with damping, useful for very weird cases
    def damping_func_max_one_second(x, step, damping, *args):
        new_step = []
        for v in step:
            if abs(v) > damping:
                if v < 0:
                    new_step.append(-damping)
                else:
                    new_step.append(damping)
            else:
                new_step.append(v)
        xnew = [xi+s for xi, s in zip(x, new_step)]
        return xnew

    Ts = []
    def test_objf_with_damping(inputs):
        x, T = inputs
        Ts.append(T)
        k = 0.12*exp(12581*(T-298.)/(298.*T))
        return [120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)]
    def test_jac_with_damping(inputs):
        x, T = inputs
        ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
                (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
               [T - 873,
                x - 11.0]]
        return ans

    near_solution = [0.05995136780143791, 300]
    ans, iterations = newton_system(test_objf_with_damping, near_solution, jac=test_jac_with_damping,
                                    line_search=True, damping=1,
                                    ytol=1e-7, xtol=None, damping_func=damping_func_max_one_second)

    assert Ts[0:4] == [300, 299, 298, 297]

    Ts = []
    ans, iterations = newton_system(test_objf_with_damping, near_solution, jac=test_jac_with_damping,
                                    line_search=True, damping=2,
                                    ytol=1e-7, xtol=None, damping_func=damping_func_max_one_second)
    assert [300, 298] == Ts[0:2]

def test_newton_system_nan_inf_inputs_outputs():
    for bad in (float("nan"), float("inf")):
        Ts = []
        def test_objf_with_nan_iterations(inputs):
            x, T = inputs
            Ts.append(T)
            k = 0.12*exp(12581*(T-298.)/(298.*T))
            if len(Ts) == 3:
                return [bad, bad]
            return [120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)]
        def test_jac_with_nan_point(inputs):
            x, T = inputs
            ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
                    (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
                   [T - 873,
                    x - 11.0]]
            return ans

        near_solution = [0.05995136780143791, 300]
        with pytest.raises(ValueError):
            newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                        line_search=False, check_numbers=True,
                                        ytol=1e-7, xtol=None)
        Ts = []

        # The line search will allow the solver to try again
        ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                        line_search=True, check_numbers=True, xtol=1e-12)
        assert_close1d(ans, [0.059951367801437914, 296.85996516970505])

        # the lack of line search causes a failure
        for progress in (True, False):
            with pytest.raises(ValueError):
                Ts = []
                ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                                line_search=False, check_numbers=True, xtol=1e-12, require_progress=progress)


def test_newton_jacobian_has_nan_inf():
    Ts = []
    def test_objf_with_nan_iterations(inputs):
        x, T = inputs
        Ts.append(T)
        k = 0.12*exp(12581*(T-298.)/(298.*T))
        return [120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)]
    def test_jac_with_nan_point(inputs):
        x, T = inputs
        ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
                (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
               [T - 873,
                x - 11.0]]
    #     if len(Ts) == 3:
        ans[0][0] = float('inf')
        return ans

    near_solution = [0.05995136780143791, 300]
    with pytest.raises(ValueError):
        newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                    line_search=False, check_numbers=True,
                                    ytol=1e-7, xtol=None)
    Ts = []

    # # The line search will allow the solver to try again
    ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                    line_search=True, check_numbers=True, xtol=1e-12, jac_error_allowed=True)
    assert_close1d(ans, [0.059951367801437914, 296.85996516970505])

    # Should also work without a linesearch
    Ts = []
    ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point,
                                    line_search=False, check_numbers=True, xtol=1e-12, jac_error_allowed=True)

    def test_jac_with_nan_point_first_only(inputs):
        x, T = inputs
        ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
                (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
               [T - 873,
                x - 11.0]]
        if len(Ts) == 1:
            ans[0][0] = float('inf')
        return ans

    # Check that if the initial jacobian is bad, it gets caught
    Ts = []
    with pytest.raises(ValueError):
        ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point_first_only,
                                        line_search=True, check_numbers=True, xtol=1e-12, jac_error_allowed=False)

    Ts = []
    # check that the initial jacobian won't stop the problem if we say yes
    ans, iterations = newton_system(test_objf_with_nan_iterations, near_solution, jac=test_jac_with_nan_point_first_only,
                                    line_search=False, check_numbers=True, xtol=1e-12, jac_error_allowed=True)
    assert_close1d(ans, [0.059951367801437914, 296.85996516970505])




def to_solve_newton_python(inputs):
    x, T = inputs
    if not isinstance(inputs, list):
        raise ValueError("Bad input type")
    k = 0.12*exp(12581*(T-298.)/(298.*T))
    return [120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)]

def to_solve_newton_numpy(inputs):
    if not isinstance(inputs, np.ndarray):
        raise ValueError("Bad input type")
    x, T = inputs
    k = 0.12*exp(12581*(T-298.)/(298.*T))
    return np.array([120*x-75*k*(1-x), -x*(873-T)-11.0*(T-300)])

def to_solve_jac_newton_python(inputs):
    # derived with sympy
    if not isinstance(inputs, list):
        raise ValueError("Bad input type")
    x, T = inputs
    ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
            (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
           [T - 873,
            x - 11.0]]
    return ans

def to_solve_jac_newton_numpy(inputs):
    # derived with sympy
    if not isinstance(inputs, np.ndarray):
        raise ValueError("Bad input type")
    x, T = inputs
    ans = [[9.0*exp(0.00335570469798658*(12581*T - 3749138.0)/T) + 120,
            (42.2181208053691/T - 0.00335570469798658*(12581*T - 3749138.0)/T**2)*(9.0*x - 9.0)*exp(0.00335570469798658*(12581*T - 3749138.0)/T),],
           [T - 873,
            x - 11.0]]
    return np.array(ans)

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import numdifftools as nd
            jacob_methods = ['analytical', 'python', 'numdifftools_forward']
        except:
            jacob_methods = ['analytical', 'python']

        try:
            import jacobi
            jacob_methods += ['jacobi_forward']
        except:
            pass
except:
    pass

@pytest.mark.parametrize("jacob_method", jacob_methods)
@pytest.mark.filterwarnings("ignore:Method")
def test_SolverInterface_basics(jacob_method):
    solver = SolverInterface(method='newton_system', objf=to_solve_newton_python,
                             jacobian_method=jacob_method, jac=to_solve_jac_newton_python)
    # Not testing convergence so start quite close to the point
    x0 = [.06, 297.]
    res = solver.solve(x0)
    expect = [0.05995136780143791, 296.85996516970505]
    assert_close1d(res, expect)
    res = solver.solve(np.array(x0))
    assert isinstance(res, np.ndarray)

    j0 = jacobian(to_solve_newton_python, x0, scalar=False)
    assert_close2d(to_solve_jac_newton_python(x0), j0, rtol=1e-5)
    assert_close2d(to_solve_jac_newton_numpy(np.array(x0)), j0, rtol=1e-5)

    if jacob_method == 'python':
        working_methods = ('newton_system_line_search', 'homotopy_solver', 'hybr', 'lm',
                       'krylov', 'df-sane', 'Nelder-Mead', 'Powell', 'BFGS', 'Newton-CG',
                       'TNC', 'SLSQP')
    else:
        working_methods = ('newton_system_line_search', 'hybr', 'Powell', 'BFGS')

    for method in working_methods:
        solver = SolverInterface(method=method, objf=to_solve_newton_python, jacobian_method=jacob_method,
                                 jac=to_solve_jac_newton_python)
        res = solver.solve(x0)
        assert_close1d(res, expect, rtol=1e-5)
        assert type(res) is list
        if method in ('newton_system_line_search', 'hybr', 'Powell', 'BFGS'):
            res = solver.solve(np.array(x0))
            assert isinstance(res, np.ndarray)

            j = solver.jacobian(np.array(x0))
            assert isinstance(j, np.ndarray)


    for method in working_methods:

        solver = SolverInterface(method=method, objf=to_solve_newton_numpy, jacobian_method=jacob_method, objf_numpy=True,
                                 jac=to_solve_jac_newton_numpy)
        res = solver.solve(np.array(x0))
        assert_close1d(res, expect, rtol=1e-5)
        assert isinstance(res, np.ndarray)
        if method in ('newton_system_line_search', 'hybr', 'Powell', 'BFGS'):
            res = solver.solve([1, 400.])
            assert type(res) is list

def fixed_point_1_func(x):
    conversion = 0.8
    stoichiometry = [-1, 0.5, 2, 1, 0.1, 0.001, 0, 0]
    fixed_point_1_feed = [20, 10, 0, 0, 0, 0, 30, 15]
    recycle = [x_i for x_i in x]
    reactor_feed = [recycle_i + feed_i for recycle_i, feed_i in zip(recycle, fixed_point_1_feed)]
    effluent = [reactor_feed_i + (reactor_feed[0] * stoich_i * conversion)
                for reactor_feed_i, stoich_i in zip(reactor_feed, stoichiometry)]
    product = [effluent_i * 0.1 for effluent_i in effluent]
    thing = [effluent_i - product_i for effluent_i, product_i in zip(effluent, product)]
    return [xi- ti for xi, ti in zip(x, thing)]

fixed_point_1_guess = [20, 10, 0, 0, 0, 0, 30, 15]
fixed_point_1_expect = [4.390243902439022, 177.80487804878027, 351.2195121951214, 175.6097560975607, 17.560975609756067, 0.17560975609756074, 269.9999999999998, 134.9999999999999]



def test_fixed_point_process():
    result, iterations = fixed_point(fixed_point_1_func, fixed_point_1_guess, xtol=1e-9, maxiter=1000)
    assert iterations < 250 # 235 last time
    assert_close1d(result, fixed_point_1_expect, rtol=1e-9)

    # Test the function with an adapter with broyden
    result, iterations = broyden2(fixed_point_1_guess, fixed_point_to_residual(fixed_point_1_func), jac=None, skip_J=True, xtol=1e-9, maxiter=1000)
    assert iterations < 50 # 38 last time
    assert_close1d(result, fixed_point_1_expect, rtol=1e-9)

    # Test it with standard newton forms:
    def basic_system(inputs):
        x, y = inputs
        return [x**2 + y**2 - 4, exp(x) - y]

    # Initial guess, note it does not converge with damping of 1. It seems many functions will not converge no matter the function.
    x0 = [1.0, 1.0]
    solution, iterations = fixed_point(
        f=residual_to_fixed_point(basic_system),
        x0=x0,
        xtol=1e-10,
        maxiter=1000,
        damping=.3, 
    )
    assert iterations < 80 # 74 last check
    assert_close1d(basic_system(solution), [0, 0], atol=1e-6)

    # Check the other methods converge
    solution, iterations = fixed_point_aitken(
        f=residual_to_fixed_point(basic_system),
        x0=x0,
        xtol=1e-10,
        maxiter=1000,
        damping=0.3, 
        acc_damping=0.4,# 44 last run iterations
    )
    assert iterations < 100
    assert_close1d(basic_system(solution), [0, 0], atol=1e-6)

    # Check the other methods converge
    solution, iterations = fixed_point_gdem(
        f=residual_to_fixed_point(basic_system),
        x0=x0,
        xtol=1e-10,
        maxiter=1000,
        damping=0.3, 
        acc_damping=0.3, # 81 last run
    )
    assert iterations < 100
    assert_close1d(basic_system(solution), [0, 0], atol=1e-6)

    # # Anderson acceleration does something different, always solves for the same set of residuals
    solution, iterations = fixed_point_anderson(
        f=(basic_system),
        x0=x0,
        xtol=1e-6,
        maxiter=1000,
        window_size=2, reg=0, mixing_param=0.5,
        damping=0.3,
        initial_iterations=0
    )
    assert iterations < 15 # 25 last time
    # assert_close1d(basic_system(solution), [0, 0], atol=1e-6)

    # def to_fixed_point(inputs):
    #     x, y = inputs
    #     new_y = exp(x)
    #     new_x = (abs(4 - y*y))**0.5
    #     return [new_x, new_y]

    # # Initial guess
    # x0 = [1.0, 1]

    # solution, iterations = fixed_point_anderson(
    #     f=(to_fixed_point),
    #     x0=x0,
    #     xtol=1e-6,
    #     maxiter=1000,
    #     window_size=2, reg=0, mixing_param=0.5,
    #     damping=0.3,
    #     initial_iterations=0
    # )
    # assert iterations < 15 # 25 last time
    # # assert_close1d(basic_system(solution), [0, 0], atol=1e-6)




def test_SolverInterface_fixed_point():
    # Largely from Yoel's Flexsolve which is excellent, and wikipedia sample code https://en.wikipedia.org/wiki/Aitken%27s_delta-squared_process
    # Works really nice, few parameters not exposed still though
    solver = SolverInterface(method='fixed_point', objf=fixed_point_to_residual(fixed_point_1_func), maxiter=1000, xtol=1e-9)
    ans = solver.solve(fixed_point_1_guess)
    assert_close1d(ans, fixed_point_1_expect, rtol=1e-6)

    solver = SolverInterface(method='fixed_point_aitken', objf=fixed_point_to_residual(fixed_point_1_func), maxiter=1000, xtol=1e-12)
    ans = solver.solve(fixed_point_1_guess)
    assert solver.fval_iter == 79
    assert_close1d(ans, fixed_point_1_expect, rtol=1e-6)

    solver = SolverInterface(method='fixed_point_gdem', objf=fixed_point_to_residual(fixed_point_1_func), maxiter=1000, xtol=1e-12)
    ans = solver.solve(fixed_point_1_guess)
    assert solver.fval_iter == 37
    assert_close1d(ans, fixed_point_1_expect, rtol=1e-12)
    

def test_isclose():
    assert not isclose(1.0, 2.0, rel_tol=0.0, abs_tol=0.0)
    assert not isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=0.0)
    assert isclose(1.0, 1+1e-14, rel_tol=1e-13, abs_tol=0.0)
    assert isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=1e-14)
    assert not isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=9e-15)

    assert isclose(1e300, 1e300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(1e-300, 1e-300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(-1e300, -1e300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(-1e-300, -1e-300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)

    assert not isclose(1.0, 2.0, rel_tol=0.0, abs_tol=0.0)
    assert not isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=0.0)
    assert isclose(1.0, 1+1e-14, rel_tol=1e-13, abs_tol=0.0)
    assert isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=1e-14)
    assert not isclose(1.0, 1+1e-14, rel_tol=0.0, abs_tol=9e-15)

    assert isclose(1e300, 1e300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(1e-300, 1e-300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(-1e300, -1e300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)
    assert isclose(-1e-300, -1e-300*(1+1e-14), rel_tol=2e-14, abs_tol=0.0)

def test_trunc_exp_numpy():
    dat = np.array([-1000, 1e4, -1.2, 0.0, 1.0, 1000.0])
    expect = [0.0, 1.7976931348622732e+308, 0.30119421191220214, 1.0, 2.718281828459045, 1.7976931348622732e+308]
    calc = trunc_exp_numpy(dat)
    assert_close1d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))

    out_array = np.zeros(6)
    calc = trunc_exp_numpy(dat, out=out_array)
    assert_close1d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))
    assert out_array is calc

def test_trunc_log_numpy():
    dat = np.array([0.0, 1e4, 0.0, 1.0, 1000.0])
    expect = [-744.4400719213812, 9.210340371976184, -744.4400719213812, 0.0, 6.907755278982137]
    calc = trunc_log_numpy(dat)
    assert_close1d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))

    out_array = np.zeros(5)
    calc = trunc_log_numpy(dat, out=out_array)
    assert_close1d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))
    assert out_array is calc

    dat = np.array([[0.0, 1e4, 1.0], [2.0, 0.0, 1000.0]])
    expect = [[-744.4400719213812, 9.210340371976184, 0.0], [0.6931471805599453, -744.4400719213812, 6.907755278982137]]
    calc = trunc_log_numpy(dat)
    assert_close2d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))

    out_array = np.zeros(6).reshape((2, 3))
    calc = trunc_log_numpy(dat, out=out_array)
    assert_close2d(calc, expect, rtol=1e-13)
    assert not np.any(np.isnan(expect))
    assert not np.any(np.isinf(expect))
    assert out_array is calc

def test_transpose():
    from fluids.numerics import transpose
    l=[[1,2,3],[4,5,6],[7,8,9]]
    out = transpose(l)
    assert_close2d(out, [[1, 4, 7], [2, 5, 8], [3, 6, 9]])


def test_is_increasing():
    assert is_increasing([1, 2, 3, 4])
    assert not is_increasing([1, 1, 2, 3])
    assert not is_increasing([4, 3, 2, 1])
    assert is_increasing([1000000, 1000001, 1000002, 1000003])
    assert not is_increasing([1000003, 1000002, 1000001, 1000000])
    assert is_increasing([1.1, 2.2, 3.3, 4.4])
    assert not is_increasing([4.4, 3.3, 2.2, 1.1])
    assert is_increasing([1, 2.5, 3, 4.5])
    assert not is_increasing([4.5, 3, 2.5, 1])
    assert is_increasing([1000000])
    assert is_increasing([3.14159])
    assert is_increasing([-4, -3, -2, -1])
    assert not is_increasing([-1, -2, -3, -4])
    assert is_increasing([])
    assert is_increasing([1, 2])
    assert not is_increasing([2, 1])

def test_py_lambertw():
    from fluids.numerics import py_lambertw
    # from scipy.special import lambertw as py_lambertw
    """ Test the basic functionality of the Lambert W function """
    # Check a simple positive value
    assert abs(py_lambertw(0).real - 0) < 0.22
    # Check a value greater than 1
    result = py_lambertw(exp(1) - 1).real  # LambertW(e-1) should be 1
    assert_close(abs(result - 1), 0.21570504021558445)

    # Test the boundary at -1/e
    minus_one_over_e = -1/exp(1)
    res = abs(py_lambertw(minus_one_over_e, k=-1).real + 1)


def test_hessian():
    def f(x):
        return x[0]**2 + x[1]**2

    x0 = [1.0, 2.0]
    result = hessian(f, x0, perturbation=1e-4)
    expected = [[2.0, 0.0], [0.0, 2.0]]
    assert_close2d(result, expected)


    def f(x):
        return 1000*sin(x[0]) + 4000*cos(x[1])

    x0 = [.3, .7]
    result = hessian(f, x0, perturbation=1e-5)
    expected = [[-296.1312838005886, 0.0], [0.0, -3055.5183310366906]]
    assert_close2d(result, expected, rtol=0.1)




    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

    x0 = [1.0, 1.0]
    result = hessian(f, x0, full=False, perturbation=1e-4)
    expected = [[2.0], [1.0, 4.0]]
    assert_close1d(result[0], expected[0])
    assert_close1d(result[1], expected[1])


    def f(x):
        return [x[0]**2 - x[1], x[0]*x[1]]

    x0 = [1.0, 1.0]
    result = hessian(f, x0, scalar=False, perturbation=1e-4)
    expected = [[[2.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]]
    assert_close3d(result, expected, rtol=1e-5)


