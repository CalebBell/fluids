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

from fluids.numerics import assert_close, assert_close1d
from fluids.numerics.polynomial_roots import roots_cubic, roots_cubic_a1, roots_cubic_a2, roots_quadratic, roots_quartic


def test_roots_quartic():
    coeffs = [1.0, -3.274390673429134, 0.3619541556604501, 2.4841800045762747, -0.49619076425603237]

    expect_roots = ((-0.8246324500888049+1.1609047395516947e-17j),
     (0.2041867922778502-3.6197168963943884e-17j),
     (1.027869356838059+2.910620457054182e-17j),
     (2.86696697440203-4.51808300211488e-18j))
    expect_mp_roots_real = [-0.824632450088805, 0.20418679227785, 1.0278693568380592, 2.86696697440203]
    roots_calc = roots_quartic(*coeffs)
    assert_close1d(expect_roots, roots_calc, rtol=1e-9)
    assert_close1d(expect_mp_roots_real, [i.real for i in roots_calc], rtol=1e-9)

def test_roots_quadratic():
    a, b, c = 1,2,3
    v0, v1 = roots_quadratic(a, b, c)
    if v0.imag < v1.imag:
        v1, v0 = v0, v1
    assert_close(v0, -1+1.4142135623730951j, rtol=1e-12)
    assert_close(v1, -1-1.4142135623730951j, rtol=1e-12)

    a, b, c = .1,2,3
    v0, v1 = roots_quadratic(a, b, c)
    if v0.real < v1.real:
        v1, v0 = v0, v1
    assert_close(v0, -1.6333997346592444, rtol=1e-12)
    assert_close(v1, -18.366600265340757, rtol=1e-12)

    a, b, c = 0,2,3
    v0, v1 = roots_quadratic(a, b, c)
    assert_close(v0, -1.5, rtol=1e-12)
    assert_close(v1, -1.5, rtol=1e-12)
    assert v0 == v1

def test_roots_cubic_a1():
    v0, v1, v2 = roots_cubic_a1(123.4,-23.34,1.234)
    assert_close(v0.real, 0.09446632563251711, rtol=1e-12)
    assert_close(v0.imag, -0.03257032605677779, rtol=1e-12)
    assert_close(v1.real, -123.588932651265, rtol=1e-12)
    assert_close(v1.imag, 1.5855372570428017e-15, atol=1e-12)
    assert_close(v2.real, 0.09446632563248869, rtol=1e-12)
    assert_close(v2.imag, +0.03257032605677779, rtol=1e-12)

    v0, v1, v2 = roots_cubic_a1(100.0, 1000.0, 10.0)

    assert_close(v0.real, -0.010010019045111562, rtol=1e-12)
    assert_close(v0.imag, -2.1316282072803006e-14, atol=1e-12)
    assert_close(v1.real, -88.73128838313305, rtol=1e-12)
    assert_close(v1.imag, 1.5855372570428017e-15, atol=1e-12)
    assert_close(v2.real, -11.258701597821826, rtol=1e-12)
    assert_close(v2.imag, 2.296510222925631e-14, atol=1e-12)

def test_roots_cubic_a2():
    v0, v1, v2 = roots_cubic_a2(1.0, 123.4,-23.34,1.234)
    assert_close(v0.real, 0.09446632563251711, rtol=1e-12)
    assert_close(v0.imag, -0.03257032605677779, rtol=1e-12)
    assert_close(v1.real, -123.588932651265, rtol=1e-12)
    assert_close(v1.imag, 1.5855372570428017e-15, atol=1e-12)
    assert_close(v2.real, 0.09446632563248869, rtol=1e-12)
    assert_close(v2.imag, +0.03257032605677779, rtol=1e-12)

    v0, v1, v2 = roots_cubic_a2(1.0, 100.0, 1000.0, 10.0)

    assert_close(v0.real, -0.010010019045111562, rtol=1e-12)
    assert_close(v0.imag, -2.1316282072803006e-14, atol=1e-12)
    assert_close(v1.real, -88.73128838313305, rtol=1e-12)
    assert_close(v1.imag, 1.5855372570428017e-15, atol=1e-12)
    assert_close(v2.real, -11.258701597821826, rtol=1e-12)
    assert_close(v2.imag, 2.296510222925631e-14, atol=1e-12)

    v0, v1, v2 = roots_cubic_a2(33.0, 100.0, 1000.0, 10.0)
    assert_close(v0.real, -0.010009986884777167, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -1.5101465217091263, rtol=1e-12)
    assert_close(v1.imag, 5.2907707087197915, atol=1e-12)
    assert_close(v2.real, -1.5101465217091263, rtol=1e-12)
    assert_close(v2.imag, -5.2907707087197915, atol=1e-12)


def test_roots_cubic():
    v0, v1, v2 = roots_cubic(1.0, 123.4,-23.34,1.234)
    assert_close(v0.real, -123.588932651265, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, 0.0944663256325029, rtol=1e-12)
    assert_close(v1.imag, 0.03257032605695922, atol=1e-12)
    assert_close(v2.real, 0.0944663256325029, rtol=1e-12)
    assert_close(v2.imag, -0.03257032605695922, rtol=1e-12)

    v0, v1, v2 = roots_cubic(1.0, 100.0, 1000.0, 10.0)

    assert_close(v0.real, -0.010010019045111562, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -88.73128838313305, rtol=1e-12)
    assert_close(v1.imag, 0, atol=0)
    assert_close(v2.real, -11.258701597821826, rtol=1e-12)
    assert_close(v2.imag, 0, atol=0)

    v0, v1, v2 = roots_cubic(33.0, 100.0, 1000.0, 10.0)

    assert_close(v0.real, -0.010009986884774058, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -1.5101465217091279, rtol=1e-12)
    assert_close(v1.imag, 5.290770708719792, atol=1e-12)
    assert_close(v2.real, -1.5101465217091279, rtol=1e-12)
    assert_close(v2.imag, -5.290770708719792, atol=1e-12)

    # a=0 case
    # arbitrary choice on which root to repeat
    v0, v1, v2 = roots_cubic(0.0, 1.0, 2.0, 3.0)
    assert_close(v0.real, -1.0, rtol=1e-12)
    assert_close(v0.imag, 1.4142135623730951, rtol=1e-12)
    assert_close(v1.real, -1.0, rtol=1e-12)
    assert_close(v1.imag, 1.4142135623730951, rtol=1e-12)
    assert_close(v2.real, -1.0, rtol=1e-12)
    assert_close(v2.imag, -1.4142135623730951, rtol=1e-12)

    # case with different branch
    v0, v1, v2 = roots_cubic(0.0, .1, 2.0, 3.0)
    assert_close(v0.real, -1.6333997346592444, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -1.6333997346592444, rtol=1e-12)
    assert_close(v1.imag, 0, atol=0)
    assert_close(v2.real, -18.366600265340757, rtol=1e-12)
    assert_close(v2.imag, 0, atol=0)


    # case with repeating quadratic root
    v0, v1, v2 = roots_cubic(0.0, 0.0, 2.0, 3.0)
    assert_close(v0.real, -1.5, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -1.5, rtol=1e-12)
    assert_close(v1.imag, 0, atol=0)
    assert_close(v2.real, -1.5, rtol=1e-12)
    assert_close(v2.imag, 0, atol=0)

    # Nasty case T >= 0.0
    v0, v1, v2 = roots_cubic(1.0, -0.9317082987361708, -0.061964336199539824, -0.001069098250830773)
    assert_close(v0.real, 0.9950599982590833, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -0.031675849761456265, rtol=1e-12)
    assert_close(v1.imag, 0.008428900244340058, rtol=1e-12)
    assert_close(v2.real, -0.031675849761456265, rtol=1e-12)
    assert_close(v2.imag, -0.008428900244340058, rtol=1e-12)

    # sincos case
    v0, v1, v2 = roots_cubic(1.0, -0.9999998288102078, -1.5499734058215601e-07, -2.77199246547356e-15)
    assert_close(v0.real, 0.9999999838075536, rtol=1e-12)
    assert_close(v0.imag, 0, atol=0)
    assert_close(v1.real, -1.341318024428162e-07, rtol=1e-12)
    assert_close(v1.imag, 0.0, atol=1e-12)
    assert_close(v2.real, -2.0865543459702707e-08, rtol=1e-12)
    assert_close(v2.imag, -0.0, atol=1e-12)



