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

from fluids import *
from numpy.testing import assert_allclose
import pytest

def test_orifice_discharge():
    m = orifice_discharge(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1646, C=0.5988, expansibility=0.9975)
    assert_allclose(m, 0.01120390943807026)
    
def test_orifice_expansibility():
    epsilon = orifice_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    assert_allclose(epsilon, 0.9974739057343425)
    # Tested against a value in the standard
    
def test_C_Reader_Harris_Gallagher():
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, k=1.4, taps='corner' )
    assert_allclose(C, 0.6000085121444034)
    
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, k=1.4, taps='D' )
    assert_allclose(C, 0.5988219225153976)
    
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, P1=1E5, P2=9.9E4, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, k=1.4, taps='flange' )
    assert_allclose(C, 0.5990042535666878)

def test_Reader_Harris_Gallagher_discharge():
    m = Reader_Harris_Gallagher_discharge(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, taps='D')
    assert_allclose(m, 7.702338035732167)