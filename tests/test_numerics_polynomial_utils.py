'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import pytest
from fluids.numerics import (
    assert_close,
    assert_close1d,
    deflate_cubic_real_roots,
    exp_poly_ln_tau_coeffs2,
    exp_poly_ln_tau_coeffs3,
    horner_and_der2,
    polyder,
    polyint,
    polyint_over_x,
    polynomial_offset_scale,
    quadratic_from_f_ders,
    quadratic_from_points,
    stable_poly_to_unstable,
    polyint_stable,
    polyint_over_x_stable,
    poly_convert,
)

try:
    import mpmath
    has_mpmath = True
except:
    has_mpmath = False


def test_polyint():
    coeffs = [-6.496329615255804e-23,2.1505678500404716e-19, -2.2204849352453665e-16,
            1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08,
            8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]

    int_coeffs_expect = [-7.218144016950893e-24, 2.6882098125505894e-20, -3.1721213360648096e-17, 2.9091262394195675e-15, 1.9592992970538825e-11, -1.1917794632375709e-08, 2.794975451876413e-06, -0.00029777396580599517, 29.114778709934264, 0.0]
    int_coeffs = polyint(coeffs)
    assert_close1d(int_coeffs, int_coeffs_expect, rtol=1e-15)

    one_coeff = polyint([1.0])
    one_coeff_expect = [1.0, 0.0]
    assert_close1d(one_coeff, one_coeff_expect, rtol=1e-15)

    zero_coeff = polyint([])
    zero_coeff_expect = [0.0]
    assert_close1d(zero_coeff, zero_coeff_expect, rtol=1e-15)

def test_polyint_over_x():
    coeffs = [-6.496329615255804e-23,2.1505678500404716e-19, -2.2204849352453665e-16,
          1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08,
          8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]

    int_coeffs_expect = [-8.120412019069755e-24, 3.072239785772102e-20, -3.700808225408944e-17, 3.490951487303481e-15, 2.449124121317353e-11, -1.5890392843167613e-08, 4.192463177814619e-06, -0.0005955479316119903, 0.0]

    int_coeffs, int_const = polyint_over_x(coeffs)
    assert_close1d(int_coeffs, int_coeffs_expect, rtol=1e-15)
    assert_close(int_const, 29.114778709934264)

    one_coeff, one_const = polyint_over_x([1.0])
    one_coeff_expect = [0.0]
    assert_close1d(one_coeff, one_coeff_expect, rtol=1e-15)
    assert_close(one_const, 1)

    zero_coeff, zero_const = polyint_over_x([])
    zero_coeff_expect = []
    assert_close1d(zero_coeff, zero_coeff_expect, rtol=1e-15)
    assert_close(zero_const, 0.0, rtol=1e-15)

def test_polyder():
    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    coeffs_calc = polyder(coeffs)
    coeffs_expect = [2.0, 6.0, 12.0, 20.0]
    assert_close1d(coeffs_calc, coeffs_expect)

    coeffs_calc = polyder(coeffs, m=3)
    coeffs_expect = [24.0, 120.0]
    assert_close1d(coeffs_calc, coeffs_expect)

    coeffs_calc = polyder(coeffs, m=5)
    coeffs_expect = []
    assert_close1d(coeffs_calc, coeffs_expect)


    coeffs = [1.0]
    coeffs_calc = polyder(coeffs, m=0)
    coeffs_expect = [1.0]
    assert_close1d(coeffs_calc, coeffs_expect)

    coeffs_calc = polyder(coeffs, m=1)
    coeffs_expect = []
    assert_close1d(coeffs_calc, coeffs_expect)

    # Check that we can go beyond the bounds
    coeffs_calc = polyder(coeffs, m=2)
    coeffs_expect = []
    assert_close1d(coeffs_calc, coeffs_expect)


def test_quadratic_from_points():
    a, b, c = quadratic_from_points(1.0, 2.0, 3.0, 3.0, 5.0, 33.0)
    assert_close(a, 13.0, rtol=1e-13)
    assert_close(b, -37.0, rtol=1e-13)
    assert_close(c, 27.0, rtol=1e-13)

def test_quadratic_from_f_ders():
    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    p = 3.0
    v, d1, d2 = horner_and_der2(poly, p)
    quadratic = list(quadratic_from_f_ders(p, v, d1, d2))
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


def test_deflate_cubic_real_roots():
    assert_close1d(deflate_cubic_real_roots(2.0, 4.5, 1.1, 43.0), (0.0005684682709485855, -45.00056846827095), rtol=1e-14)

    assert_close1d(deflate_cubic_real_roots(2.0, 4.5, -1e5, 43.0),  (0.0, 0.0), atol=0.0)

def test_exp_poly_ln_tau_coeffs2():

    args = (300, 647.096, 0.06576090309133853, -0.0002202298609576884)
    a, b = exp_poly_ln_tau_coeffs2(*args)
    assert_close1d([a, b], [1.1624065398371628, -1.9976745939643825 ], rtol=1e-9)

def test_exp_poly_ln_tau_coeffs3():
    args = (300, 647.096, 0.06730658226743809, -0.00020056690242827797, -5.155567532930362e-09)
    a, b, c = exp_poly_ln_tau_coeffs3(*args)
    assert_close1d([a, b, c], [-0.022358482008994165, 1.0064575672832698, -2.062906603289078])

def test_polynomial_offset_scale():
    offset, scale = polynomial_offset_scale(-0.4469282485663427, -6.528007370549513)
    assert_close(offset, -1.1469897824386766, rtol=1e-14)
    assert_close(scale, -0.32888899484467765, rtol=1e-14)

def test_stable_poly_to_unstable():
    # Lots of test cases because needed to rewrite test to not use numpy for speed
    stuff = [1,2,3,4]
    out = stable_poly_to_unstable(stuff, 10, 100)
    expect = [1.0973936899862826e-05, -0.0008230452674897121, 0.05761316872427985, 1.4951989026063095]
    assert_close1d(out, expect, rtol=1e-12)

    out = stable_poly_to_unstable(stuff, 10, 10)
    assert_close1d(out, stuff)

    coeffs = [1, 2, 3, 4]
    low, high = 10, 100
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0973936899862826e-05, -0.0008230452674897121, 0.05761316872427985, 1.4951989026063095]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 2, 3, 4]
    low, high = 10, 10
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1, 2, 3, 4]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, -2, 3, -4, 5]
    low, high = -50, 50
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.6000000000000003e-07, -1.6000000000000003e-05, 0.0012, -0.08, 5.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [0.1, 0.2, 0.3]
    low, high = 0, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [0.4, 0.0, 0.19999999999999998]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [10000, 20000, 30000]
    low, high = -1000, 1000
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [0.01, 20.0, 30000.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [-1, -2, -3, -4]
    low, high = -100, -10
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [-1.0973936899862826e-05, -0.0027983539094650206, -0.2748971193415638, -12.480109739369]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1e-05, 2e-05, 3e-05]
    low, high = -1000000.0, 1000000.0
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1e-17, 2.0000000000000002e-11, 3e-05]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 0, 0, 0, 1]
    low, high = -1, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0, 0.0, 0.0, 0.0, 1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 1, 1, 1, 1]
    low, high = 0, 100000
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.6000000000000006e-19, -2.4000000000000005e-14, 1.6000000000000003e-09, -4e-05, 1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [0, 0, 0, 1]
    low, high = -10, 10
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 2, 3, 4, 5]
    low, high = 0.1, 0.2
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [160000.0, -80000.00000000001, 15600.000000000004, -1360.0000000000005, 47.00000000000003]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, -1, 1, -1, 1]
    low, high = -1000000, 1000000
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1e-24, -9.999999999999999e-19, 1e-12, -1e-06, 1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [0.5, 1.5, 2.5]
    low, high = -0.5, 0.5
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [2.0, 3.0, 2.5]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [100, 200, 300, 400]
    low, high = 99, 101
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [100.0, -29800.0, 2960300.0, -98029600.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 10, 100, 1000]
    low, high = -1e-06, 1e-06
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1e+18, 10000000000000.0, 100000000.0, 1000.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [42]
    low, high = -1, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [42.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [3, 7]
    low, high = 0, 10
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [0.6000000000000001, 4.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = []
    low, high = 5, 15
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = []
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    low, high = -100, 100
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0000000000000003e-18, 2.0000000000000005e-16, 3.0000000000000005e-14, 4.000000000000001e-12, 5e-10, 6e-08, 7.000000000000001e-06, 0.0008, 0.09, 10.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    low, high = -1000, 1000
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0000000000000007e-57, 2.0000000000000012e-54, 3e-51, 4.0000000000000023e-48, 5.000000000000003e-45, 6e-42, 7e-39, 8.000000000000004e-36, 9.000000000000005e-33, 1.0000000000000004e-29, 1.0999999999999999e-26, 1.2000000000000001e-23, 1.3e-20, 1.4e-17, 1.5000000000000002e-14, 1.6000000000000003e-11, 1.7000000000000003e-08, 1.8000000000000004e-05, 0.019, 20.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
    low, high = -5, 5
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [7.040000000000002e-05, 0.0007040000000000002, 0.005280000000000001, 0.03520000000000001, 0.22000000000000003, 1.32, 7.7]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1]
    low, high = 1000, 1000
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [0, 1]
    low, high = -1000000000.0, 1000000000.0
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [-1, 0, 1]
    low, high = -3.141592653589793, 3.141592653589793
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [-0.10132118364233779, 0.0, 1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    low, high = 0, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [16384.0, -106496.0, 323584.0, -608256.0, 789504.0, -748032.0, 533248.0, -290432.0, 121408.0, -38752.0, 9296.0, -1624.0, 196.0, -14.0, 1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]
    low, high = -1, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0, 10000000000.0, 100000000000.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125]
    low, high = 0.1, 0.2
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1280000000.0, -1312000000.0000002, 577066666.666667, -141160000.0000001, 20737600.000000015, -1829453.333333335, 89730.85714285723, -1887.4535714285735]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    low, high = -100, 100
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0000000000000004e-34, -1.0000000000000003e-32, 1.0000000000000003e-30, -1.0000000000000003e-28, 1.0000000000000003e-26, -1.0000000000000003e-24, 1.0000000000000003e-22, -1.0000000000000002e-20, 1.0000000000000003e-18, -1.0000000000000002e-16, 1.0000000000000002e-14, -1.0000000000000002e-12, 1.0000000000000002e-10, -1.0000000000000002e-08, 1.0000000000000002e-06, -0.0001, 0.01, -1.0]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [2.718281828459045, 3.141592653589793, 2.718, 3.142]
    low, high = -10, 10
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [0.0027182818284590456, 0.031415926535897934, 0.2718, 3.142]
    assert_close1d(result, expected, rtol=1e-12)

    coeffs = [0, 0, 0, 0, 0, 1]
    low, high = -1, 1
    result = stable_poly_to_unstable(coeffs, low, high)
    expected = [1.0]
    assert_close1d(result, expected, rtol=1e-12)


def test_polyint_stable():
    coeffs = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0
    expect = [-41.298949195476155, 39.0117740732049, 236.44351075433477, -231.5592751637343, -561.2796493247411, 548.0939357018894, 769.1662981674165, -719.2308222415658, -711.9191528399639, 642.3457574352843, 409.2699514702697, -363.22931752942026, -134.6328547344325, 67.11476327717565, 129.41107925589787, -91.88923909769476, -24.648457402294206, 124.10916590172992, -169.47997914514303, 49.77167860871583, 280.687547727181, -494.09522423312563, -473.27655194287644, 4754.741468163904, 37473.448792536445, 0.0]


    out = polyint_stable(coeffs, Tmin, Tmax)
    assert_close1d(out, expect)

def test_polyint_over_x_stable_simple():
    int_over_x_coeffs_expect = [3.9123328133529204e-16, -1.8970302735986266e-12, 4.622646341408311e-09, -5.905414280620714e-06, 0.0071812547510319395, 0.0]
    log_coeff_expect = 1.749905045767787
    coeffs = [1,2,3, 4, 5, 6]
    Tmin, Tmax = 251.165, 2000.0

    int_over_x_coeffs, log_coeff_expect = polyint_over_x_stable(coeffs, Tmin, Tmax)
    assert_close(log_coeff_expect, log_coeff_expect, rtol=1e-13)
    assert_close1d(int_over_x_coeffs, int_over_x_coeffs_expect, rtol=1e-13)

    coeffs = [1, 2, 3, 4, 5, 6]
    xmin, xmax = 251.165, 2000.0
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [3.9123328133529204e-16, -1.8970302735986266e-12, 4.622646341408311e-09, -5.905414280620714e-06, 0.0071812547510319395, 0.0]
    expected_log_coeff = 1.749905045767787
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1]
    xmin, xmax = 0, 1
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0]
    expected_log_coeff = 1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1, 2]
    xmin, xmax = -1, 1
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [1.0, 0.0]
    expected_log_coeff = 2.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [0, 0, 1]
    xmin, xmax = 0, 100
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0, 0.0, 0.0]
    expected_log_coeff = 1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1, -1, 1, -1, 1]
    xmin, xmax = -10, 10
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [2.5000000000000008e-05, -0.00033333333333333343, 0.005000000000000001, -0.1, 0.0]
    expected_log_coeff = 1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    xmin, xmax = 0, 5
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0006400000000000003, -0.004266666666666668, 0.024000000000000004, 4.4408920985006264e-17, 0.0]
    expected_log_coeff = 0.29999999999999993
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [10, 20, 30, 40, 50]
    xmin, xmax = 100, 1000
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [6.096631611034904e-11, -1.05674947924605e-07, 0.00011431184270690446, -0.03718945282731293, 0.0]
    expected_log_coeff = 31.72534674592288
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1e-05, 2e-05, 3e-05, 4e-05]
    xmin, xmax = -1000, 1000
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [3.333333333333334e-15, 1.0000000000000001e-11, 3.0000000000000004e-08, 0.0]
    expected_log_coeff = 4e-05
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1, 0, 0, 0, 1]
    xmin, xmax = -3.141592653589793, 3.141592653589793
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0025664955636710844, 0.0, 0.0, 0.0, 0.0]
    expected_log_coeff = 1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xmin, xmax = 0, 10
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0, 3.2000000000000017e-07, -1.0971428571428577e-05, 0.0001813333333333334, -0.0017920000000000008, 0.012000000000000004, -0.053333333333333344, 0.20000000000000004, 0.0, 0.0]
    expected_log_coeff = 5.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1, -1, 1, -1, 1, -1, 1, -1]
    xmin, xmax = -100, 100
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [1.4285714285714287e-15, -1.666666666666667e-13, 2.0000000000000002e-11, -2.5000000000000005e-09, 3.333333333333334e-07, -5e-05, 0.01, 0.0]
    expected_log_coeff = -1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666]
    xmin, xmax = 0.1, 1.0
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [10.838456197395386, -34.20887737302919, 46.92148328789087, -36.41636606885808, 19.086081051330257, 0.0]
    expected_log_coeff = -1.9245702721468616
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [2.718281828459045, 3.141592653589793, 2.718, 3.142]
    xmin, xmax = -10, 10
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0009060939428196819, 0.015707963267948967, 0.2718, 0.0]
    expected_log_coeff = 3.142
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [0, 0, 0, 0, 0, 1]
    xmin, xmax = -1, 1
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected_log_coeff = 1.0
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

    coeffs = [1, 10, 100, 1000, 10000]
    xmin, xmax = 1, 1000
    int_over_x_coeffs, log_coeff = polyint_over_x_stable(coeffs, xmin, xmax)
    expected_int_over_x_coeffs = [4.016040080140224e-12, 1.6026677279812922e-08, 0.00015223228836852077, 1.6530445801958276, 0.0]
    expected_log_coeff = 9089.346650907131
    assert_close1d(int_over_x_coeffs, expected_int_over_x_coeffs, rtol=1e-13)
    assert_close(log_coeff, expected_log_coeff, rtol=1e-13)

def test_polyint_over_x_stable_real():
    int_over_x_coeffs_expect = [-1.2322441102026994e-72, 3.575470736932928e-68, -4.9223708612288747e-64, 4.277576156314128e-60, -2.6333674644719005e-56, 1.2218089960807912e-52, -4.437607286601807e-49, 1.2935905647823088e-45, -3.0786333426941215e-42, 6.052509795894643e-39, -9.907217102698286e-36, 1.3567568865208235e-32, -1.5579127525009192e-29, 1.499553966471978e-26, -1.2065502962399921e-23, 8.069449660788723e-21, -4.445415598884699e-18, 1.9901372278914136e-15, -7.100819961746428e-13, 1.9642027689940838e-10, -4.050907926020504e-08, 5.890729117069776e-06, -0.0005458412005135644, 0.022571673162274146, 0.0]
    log_coeff_expect = 33.94307866530403
    coeffs = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0

    int_over_x_coeffs, log_coeff_expect = polyint_over_x_stable(coeffs, Tmin, Tmax)
    assert_close(log_coeff_expect, log_coeff_expect, rtol=1e-13)
    assert_close1d(int_over_x_coeffs, int_over_x_coeffs_expect, rtol=1e-13)


@pytest.mark.mpmath
@pytest.mark.skipif(not has_mpmath, reason='mpmath is not installed')
def test_polyint_over_x_stable_real_precise():
    import mpmath as mp
    from numpy.polynomial.polynomial import Polynomial
    coeffs_num = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0
    int_T_coeffs_expect = [-0.04919816763192372, 0.11263751330617996, 0.13111045246317343, -0.45423261676182597, -0.060044010117414455, 0.7411591719418316, -0.07854882531076747, -0.7638508595972384, 0.17966146925345727, 0.5368870307001614, -0.238984435878899, -0.1160558677189392, -0.004958654675811305, 0.09069452297160363, 0.034376716486728694, -0.16593023302511933, 0.20857847967437174, -0.1446358723821105, -0.008913096668590397, 0.08207169354184629, 0.26919218469883643, -1.215427392470841, 1.5349428351610082, 6.923550076265145, 3.2254691726042486]

    mp.mp.dps = 50
    coeffs_mp = [mp.mpf(v) for v in coeffs_num]
    int_T_coeffs, log_term = polyint_over_x_stable(coeffs_mp, mp.mpf(Tmin), mp.mpf(Tmax))


    to_change_poly = Polynomial(int_T_coeffs[::-1])
    domain_new = (mp.mpf(Tmin), mp.mpf(Tmax))
    fixed =  to_change_poly.convert(domain=domain_new).coef.tolist()[::-1]
    coeffs_fixed = [float(v) for v in fixed]

    assert_close1d(coeffs_fixed, int_T_coeffs_expect, rtol=3e-15)


def test_poly_convert():
    coeffs = [1, 2, 3, 4]
    Tmin, Tmax = 0, 1
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [3.25, 4.0, 2.25, 0.5]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, -1, 1]
    Tmin, Tmax = -1, 1
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1.0, -1.0, 1.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, 0, 0, 1]
    Tmin, Tmax = 0, 100
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [125001.0, 375000.0, 375000.0, 125000.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, 2, 3, 4, 5]
    Tmin, Tmax = -10, 10
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1.0, 20.0, 300.0, 4000.0, 50000.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [0.1, 0.2, 0.3]
    Tmin, Tmax = 0, 5
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [2.475, 4.25, 1.875]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [10, 20, 30]
    Tmin, Tmax = 100, 1000
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [9086010.0, 14859000.0, 6075000.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1e-05, 2e-05, 3e-05]
    Tmin, Tmax = -1000, 1000
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1e-05, 0.02, 30.000000000000004]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1]
    Tmin, Tmax = -3.141592653589793, 3.141592653589793
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Tmin, Tmax = 0, 10
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [21362305.0, 187683105.0, 733375550.0, 1672744750.0, 2454221250.0, 2401906250.0, 1567984375.0, 658359375.0, 161328125.0, 17578125.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, -1, 1, -1, 1]
    Tmin, Tmax = -100, 100
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1.0, -100.0, 10000.0, -1000000.0, 100000000.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1.0, 0.5, 0.3333333333333333, 0.25]
    Tmin, Tmax = 0.1, 1.0
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [1.4174270833333333, 0.4920937500000001, 0.15103125, 0.022781250000000003]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [2.718281828459045, 3.141592653589793]
    Tmin, Tmax = -10, 10
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [2.718281828459045, 31.41592653589793]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [0, 0, 0, 1]
    Tmin, Tmax = -1, 1
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [0.0, 0.0, 0.0, 1.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, 10, 100, 1000]
    Tmin, Tmax = 1, 1000
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [125400430156.0, 375424629570.0, 374649575400.0, 124625374875.0]
    assert_close1d(result, expected, rtol=1e-13)

    coeffs = [1, 2, 3]
    Tmin, Tmax = 251.165, 2000.0
    result = poly_convert(coeffs, Tmin, Tmax)
    expected = [3803060.0579187497, 5907123.0491625, 2293817.89291875]
    assert_close1d(result, expected, rtol=1e-13)

