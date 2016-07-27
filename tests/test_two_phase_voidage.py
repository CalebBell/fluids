# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from __future__ import division
from fluids import *
import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_Thom():
    assert_allclose(Thom(.4, 800, 2.5, 1E-3, 1E-5), 0.9801482164042417)


def test_Zivi():
    assert_allclose(Zivi(.4, 800, 2.5), 0.9689339909056356)


def test_Smith():
    assert_allclose(Smith(.4, 800, 2.5), 0.959981235534199)
    
    # Quick test function, to ensure results are the same regardless of 
    # the form of the expression
    def Smith2(x, rhol, rhog):
        K = 0.4
        x_ratio = (1-x)/x
        first = 1 + rhog/rhol*K*(1/x-1)
        second = rhog/rhol*(1-K)*(1/x-1)
        third = ((rhol/rhog + K*(1/x-1))/(1 + K*(1/x -1)))**0.5
        return (first + second*third)**-1

    alpha_1 = [Smith(i, 800, 2.5) for i in np.linspace(0,1)]
    alpha_2 = [Smith2(i, 800, 2.5) for i in np.linspace(0,1)]
    assert_allclose(alpha_1, alpha_2)


def test_Fauske():
    assert_allclose(Fauske(.4, 800, 2.5), 0.9226347262627932)


def test_Chisholm():
    assert_allclose(Chisholm(.4, 800, 2.5), 0.949525900374774)


def test_Turner_Wallis():
    assert_allclose(Turner_Wallis(.4, 800, 2.5, 1E-3, 1E-5), 0.8384824581634625)


### Section 2

def test_homogeneous():
    assert_allclose(homogeneous(.4, 800, 2.5), 0.995334370139969)