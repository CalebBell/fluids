# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import assert_close, assert_close1d

def test_horner():
    from fluids.numerics import horner, horner_backwards, horner_and_der2, horner_and_der3, horner_and_der4
    assert_close(horner([1.0, 3.0], 2.0), 5.0, rtol=1e-15)
    assert_close(horner_backwards(2.0, [1.0, 3.0]), 5.0, rtol=1e-15)
    assert_close(horner([3.0], 2.0), 3.0, rtol=1e-15)

    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    assert_close1d(horner_and_der2(poly, 3.0), (14726.109396, 13747.040732, 8553.7884), rtol=1e-15)
    assert_close1d(horner_and_der3(poly, 3.0), (14726.109396, 13747.040732, 8553.7884, 2674.56), rtol=1e-15)

    poly = [1.12, 432.32, 325.5342, .235532, 32.235, 1.01]
    assert_close1d(horner_and_der4(poly, 3.0), (44179.338188, 55967.231592, 53155.446664, 33685.04519999999, 10778.880000000001), rtol=1e-15)
    assert_close1d(horner_and_der4(poly, 3.0), [np.polyval(np.polyder(poly,o), 3) for o in range(5)])

