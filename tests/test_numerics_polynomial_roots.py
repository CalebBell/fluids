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
from fluids.numerics import numpy as np
import pytest
import fluids.numerics
from fluids.numerics import ( assert_close, assert_close1d, roots_quartic, roots_quadratic)
assert_allclose = np.testing.assert_allclose

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

def test_roots_quadratic():
    a, b, c = 1,2,3
    v0, v1 = roots_quadratic(a, b, c)
    if v0.imag < v1.imag:
        v1, v0 = v0, v1
    assert_close(v0, -1+1.4142135623730951j, rtol=1e-14)
    assert_close(v1, -1-1.4142135623730951j, rtol=1e-14)
    
    a, b, c = .1,2,3
    v0, v1 = roots_quadratic(a, b, c)
    if v0.real < v1.real:
        v1, v0 = v0, v1
    assert_close(v0, -1.6333997346592444, rtol=1e-14)
    assert_close(v1, -18.366600265340757, rtol=1e-14)

    a, b, c = 0,2,3
    v0, v1 = roots_quadratic(a, b, c)
    assert_close(v0, -1.5, rtol=1e-14)
    assert_close(v1, -1.5, rtol=1e-14)
    assert v0 == v1
    
