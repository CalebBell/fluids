# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.vectorized import *
from fluids import *
from numpy.testing import assert_allclose
import pytest
import fluids.vectorized
import numpy as np


def test_a_complicated_function():
    orig = [fluids.integrate_drag_sphere(D=D, rhop=rhop, rho=1.2, mu=1.78E-5, t=0.5, V=30, distance=True) for D, rhop in zip([0.002, 0.001], [2200., 2300])]

    ans_vect = fluids.vectorized.integrate_drag_sphere(D=[0.002, 0.001], rhop=[2200., 2300], rho=1.2, mu=1.78E-5, t=0.5, V=30, distance=True)
    # Note the transpose requirement to match!
    ans_vect = np.array(ans_vect).T
    assert_allclose(ans_vect, orig)



def test_Morsi_Alexander():
    Cds = [Morsi_Alexander(i) for i in [500, 5000, 5000]]
    Cds_vect = fluids.vectorized.Morsi_Alexander([500, 5000, 5000])
    assert_allclose(Cds, Cds_vect)


