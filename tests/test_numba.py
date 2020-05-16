# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import assert_close
import pytest
try:
    import numba
except:
    numba = None
import fluids.numba
import numpy as np

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Clamond_numba():
    assert_close(fluids.numba.Clamond(10000.0, 2.0), 
                 fluids.Clamond(10000.0, 2.0), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, True),
                 fluids.Clamond(10000.0, 2.0, True), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, False),
                 fluids.Clamond(10000.0, 2.0, False), rtol=5e-15)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_string():
    fluids.numba.entrance_sharp('Miller')
    fluids.numba.entrance_sharp()


