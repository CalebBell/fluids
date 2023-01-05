# -*- coding: utf-8 -*-
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
SOFTWARE.'''

from __future__ import division
from fluids.numerics import numpy as np
import pytest
import fluids.numerics
from fluids.numerics import ( assert_close, assert_close1d, horner, horner_and_der2)
from fluids.numerics import (polyint, polyint_over_x, polyder, quadratic_from_points, quadratic_from_f_ders, deflate_cubic_real_roots,
exp_poly_ln_tau_coeffs2, exp_poly_ln_tau_coeffs3, polynomial_offset_scale, horner_domain, stable_poly_to_unstable)

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
    stuff = [1,2,3,4]
    out = stable_poly_to_unstable(stuff, 10, 100)
    expect = [1.0973936899862826e-05, -0.0008230452674897121, 0.05761316872427985, 1.4951989026063095]
    assert_close1d(out, expect, rtol=1e-12)
    
    out = stable_poly_to_unstable(stuff, 10, 10)
    assert_close1d(out, stuff)

