# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics.doubledouble import *
from fluids.numerics import assert_close, assert_close1d
from math import *
import pytest

try:
    import mpmath as mp
    has_mpmath = True
    mp.mp.dps=100

except:
    has_mpmath = False

def mark_mpmath(f):
    f = pytest.mark.mpmath(f)
    f = pytest.mark.skipif(not has_mpmath, reason='missing mpmath')(f)
    return f

def test_cube_dd():
    ans = cube_dd(2.718281828459045, 1.4456468917292502e-16)
    assert_close1d(ans, (20.085536923187668, -1.8275625525512858e-16), rtol=1e-14)

def test_div_imag_dd():
    mp_ans = (-6221711.561975023 - 2.5074521914349666e-10j)/(1247.1686953729213 +2160.157819988352j)
    ans = div_imag_dd(-6221711.561975023, 0.0, - 2.5074521914349666e-10, 0.0,
                      1247.1686953729213, 0.0, 2160.157819988352, 0.0)
    assert_close1d(ans, (-1247.168695372921, -7.085386715133776e-14, 2160.1578199883515, -1.7404258749820416e-14), rtol=1e-13)

def test_add_dd():
    ans = add_dd(2.718281828459045, 1.4456468917292502e-16, -3.141592653589793, -1.2246467991473532e-16)
    assert_close1d(ans, (-0.423310825130748, 2.2100009258189695e-17), rtol=1e-13)

@mark_mpmath
def test_add_dd_mp():
    ans = add_dd(2.718281828459045, 1.4456468917292502e-16, -3.141592653589793, -1.2246467991473532e-16)
    assert abs(mp.e-mp.pi - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30

@mark_mpmath
def test_mul_dd_mp():
    ans = mul_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (8.539734222673568, -6.773815290502423e-16), rtol=1e-13)
    assert abs(mp.e*mp.pi - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30

def test_div_dd():
    ans = div_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (0.8652559794322651, 2.1741459631779752e-17), rtol=1e-13)

@mark_mpmath
def test_div_dd_mp():
    ans = div_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert abs(mp.e/mp.pi - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30

def test_sqrt_dd():
    ans = sqrt_dd(2.718281828459045, 1.4456468917292502e-16)
    assert_close1d(ans, (1.6487212707001282, -4.731568479435833e-17), rtol=1e-13)

@mark_mpmath
def test_sqrt_dd_mp():
    ans = sqrt_dd(2.718281828459045, 1.4456468917292502e-16)
    assert abs(mp.sqrt(mp.e) - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30

def test_mul_noerrors_dd():
    ans = mul_noerrors_dd(2.718281828, 3.1415926)
    assert_close1d(ans, (8.539734075559272, 6.111502792870851e-16), rtol=1e-13)
    assert_close1d(ans, mul_dd(2.718281828, 0.0, 3.1415926, 0.0), rtol=1e-13)

def test_square_dd():
    ans = square_dd(2.718281828459045, 1.4456468917292502e-16)
    assert_close1d(ans, (7.38905609893065, -1.7971139497839148e-16), rtol=1e-13)

@mark_mpmath
def test_square_dd_mp():
    ans = square_dd(2.718281828459045, 1.4456468917292502e-16)
    assert abs(mp.e**2 - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30


def test_mul_imag_dd():
    ans = mul_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (-2.4805483021587085, 8.193747384776261e-17, 17.079468445347135, -1.3547630581004847e-15), rtol=1e-14)

    ans = mul_imag_dd(1.11, 0.00, .027, 0.0, -1.2, 0.0, .2995, 0.0)
    assert_close1d(ans, (-1.3400865, -1.0926552718171222e-16, 0.300045, 1.1863843241144421e-17), rtol=1e-14)

@mark_mpmath
def test_mul_imag_dd_mp():
    ans = mul_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    mp_ans = (mp.e + mp.pi*1j)*(mp.e + mp.pi*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30


def test_mul_imag_noerrors_dd():
    ans = mul_imag_noerrors_dd(1.11, .027, -1.2, .2995)
    assert_close1d(ans, (-1.3400865, -1.0926552718171222e-16, 0.300045, 1.1863843241144421e-17), rtol=1e-14)

def test_sqrt_imag_dd():
    ans = sqrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (1.853730863795006, 1.8197550334075816e-17, 0.8473702183385572, 4.942630438677856e-17))

    ans = sqrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 0, 0)
    assert_close1d(ans, (1.6487212707001282, -4.731568479435833e-17, 0.0, 0.0))

@mark_mpmath
def test_sqrt_imag_dd():
    ans = sqrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    mp_ans = mp.sqrt(mp.e + mp.pi*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

    ans = sqrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 0, 0)
    mp_ans = mp.sqrt(mp.e)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

def test_add_imag_dd():
    ans = add_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                      3.141592653589793, 1.2246467991473532e-16, 2.718281828459045, 1.4456468917292502e-16)
    assert_close1d(ans, (5.859874482048839, -1.770598407624023e-16, 5.859874482048839, -1.770598407624023e-16))

@mark_mpmath
def test_add_imag_dd_mp():
    ans = add_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                      3.141592653589793, 1.2246467991473532e-16, 2.718281828459045, 1.4456468917292502e-16)
    mp_ans = (mp.e + mp.pi*1j) + (mp.pi + mp.e*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

def test_div_imag_dd():
    ans = div_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                      3.141592653589793, 1.2246467991473532e-16, 2.718281828459045, 1.4456468917292502e-16)
    assert_close1d(ans, (0.9896172675351794, -2.57625629153772e-17, 0.14372774191576643, 1.174077497180584e-17), rtol=1e-13)

@mark_mpmath
def test_div_imag_dd_mp():
    ans = div_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16,
                      3.141592653589793, 1.2246467991473532e-16, 2.718281828459045, 1.4456468917292502e-16)
    mp_ans = (mp.e + mp.pi*1j)/(mp.pi + mp.e*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

def test_imag_inv_dd():
    ans = imag_inv_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (0.15750247989731844, -3.9415361951446925e-18, -0.18202992367722576, -5.408691854671257e-18), rtol=1e-13)

@mark_mpmath
def test_imag_inv_dd_mp():
    ans = imag_inv_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    mp_ans = 1/(mp.e + mp.pi*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

def test_cbrt_imag_dd():
    ans = cbrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    assert_close1d(ans, (1.5423370526780396, 8.764992916723807e-17, 0.45326965450036827, 2.105896617565284e-17), rtol=1e-14)

@mark_mpmath
def test_cbrt_imag_dd_mp():
    ans = cbrt_imag_dd(2.718281828459045, 1.4456468917292502e-16, 3.141592653589793, 1.2246467991473532e-16)
    mp_ans = mp.cbrt(mp.e + mp.pi*1j)
    assert abs(mp_ans.real - (mp.mpf(ans[0]) + mp.mpf(ans[1]) )) < 1e-30
    assert abs(mp_ans.imag - (mp.mpf(ans[2]) + mp.mpf(ans[3]) )) < 1e-30

def test_cbrt_dd():
    ans = cbrt_dd(3, 0.0)
    assert_close1d(ans, (1.4422495703074083, 8.054912676113696e-17), rtol=1e-13)

    ans = cbrt_dd(1.4422495703074083, 8.054912676113696e-17)
    assert_close1d(ans, (1.129830963909753, 1.601807086137469e-17), rtol=1e-13)

@mark_mpmath
def test_cbrt_dd_mp():
    ans = cbrt_dd(3, 0.0)
    assert abs(mp.mpf(3)**(mp.mpf(1)/mp.mpf(3)) - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30

    ans = cbrt_dd(1.4422495703074083, 8.054912676113696e-17)
    assert abs((mp.mpf(3)**(mp.mpf(1)/mp.mpf(3)))**(mp.mpf(1)/mp.mpf(3)) - (mp.mpf(ans[0]) + mp.mpf(ans[1]))) < 1e-30