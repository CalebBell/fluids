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
import numpy as np
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
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='corner' )
    assert_allclose(C, 0.6000085121444034)
    
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='D' )
    assert_allclose(C, 0.5988219225153976)
    
    C = C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.1645909036, mu=0.0000185861753095, m=0.124431876, taps='flange' )
    assert_allclose(C, 0.5990042535666878)

def test_Reader_Harris_Gallagher_discharge():
    m = Reader_Harris_Gallagher_discharge(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, taps='D')
    assert_allclose(m, 7.702338035732167)
    
    
def test_K_to_discharge_coefficient():
    C = K_to_discharge_coefficient(D=0.07366, Do=0.05, K=5.2314291729754)
    assert_allclose(C, 0.6151200000000001)
    
def test_discharge_coefficient_to_K():
    K = discharge_coefficient_to_K(D=0.07366, Do=0.05, C=0.61512)
    assert_allclose(K, 5.2314291729754)
    
def test_dP_orifice():
    dP = dP_orifice(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0, C=0.61512)
    assert_allclose(dP, 9069.474705745388)
    
def test_velocity_of_approach_factor():
    factor = velocity_of_approach_factor(D=0.0739, Do=0.0222)
    assert_allclose(factor, 1.0040970074165514)

def test_orifice_flow_coefficient():
    factor = orifice_flow_coefficient(D=0.0739, Do=0.0222, C=0.6)
    assert_allclose(factor, 0.6024582044499308)
    
def test_nozzle_expansibility():
    epsilon = nozzle_expansibility(D=0.0739, Do=0.0222, P1=1E5, P2=9.9E4, k=1.4)
    assert_allclose(epsilon, 0.991617725452954)

def test_C_long_radius_nozzle():
    C = C_long_radius_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    assert_allclose(C, 0.9805503704679863)
    
def test_C_ISA_1932_nozzle():
    C = C_ISA_1932_nozzle(D=0.07391, Do=0.0422, rho=1.2, mu=1.8E-5, m=0.1)
    assert_allclose(C, 0.9635849973250495)


@pytest.mark.slow
def test_fuzz_K_to_discharge_coefficient():
    '''
    # Testing the different formulas
    from sympy import *
    C, beta, K = symbols('C, beta, K')

    expr = Eq(K, (sqrt(1 - beta**4*(1 - C*C))/(C*beta**2) - 1)**2)
    solns = solve(expr, C)
    [i.subs({'K': 5.2314291729754, 'beta': 0.05/0.07366}) for i in solns]
    
    [-sqrt(-beta**4/(-2*sqrt(K)*beta**4 + K*beta**4) + 1/(-2*sqrt(K)*beta**4 + K*beta**4)),
 sqrt(-beta**4/(-2*sqrt(K)*beta**4 + K*beta**4) + 1/(-2*sqrt(K)*beta**4 + K*beta**4)),
 -sqrt(-beta**4/(2*sqrt(K)*beta**4 + K*beta**4) + 1/(2*sqrt(K)*beta**4 + K*beta**4)),
 sqrt(-beta**4/(2*sqrt(K)*beta**4 + K*beta**4) + 1/(2*sqrt(K)*beta**4 + K*beta**4))]
    
    # Getting the formula
    from sympy import *
    C, beta, K = symbols('C, beta, K')
    
    expr = Eq(K, (sqrt(1 - beta**4*(1 - C*C))/(C*beta**2) - 1)**2)
    print(latex(solve(expr, C)[3]))
    '''
    
    Ds = np.logspace(np.log10(1-1E-9), np.log10(1E-9))
    for D_ratio in Ds:
        Ks = np.logspace(np.log10(1E-9), np.log10(50000))
        Ks_recalc = []
        for K in Ks:
            C = K_to_discharge_coefficient(D=1, Do=D_ratio, K=K)
            K_calc = discharge_coefficient_to_K(D=1, Do=D_ratio, C=C)
            Ks_recalc.append(K_calc)
        assert_allclose(Ks, Ks_recalc)