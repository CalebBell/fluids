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

from fluids import *
from numpy.testing import assert_allclose
import pytest


def test_K_separator_Watkins():
    calc = [[K_separator_Watkins(0.88, 985.4, 1.3, horizontal, method) for
    method in ['spline', 'branan', 'blackwell']] for horizontal in [False, True]]
    
    expect = [[0.06355763251223817, 0.06108986837654085, 0.06994527471072351],
    [0.07944704064029771, 0.07636233547067607, 0.0874315933884044]]
    
    assert_allclose(calc, expect)

    with pytest.raises(Exception):
        K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True, method='BADMETHOD')
        
        
def test_K_separator_demister_York():
    from scipy.constants import  psi
    Ks_expect = [0.056387999999999994, 0.056387999999999994, 0.09662736507185091,
                 0.10667999999999998, 0.10520347947487964, 0.1036391539227465, 0.07068690636639535]
    Ks = []
    for P in [.1, 1, 10, 20, 40, 50, 5600]:
        Ks.append(K_separator_demister_York(P*psi))
        
    assert_allclose(Ks, Ks_expect)
    
    K  = K_separator_demister_York(25*psi, horizontal=True)
    assert_allclose(K, 0.13334999999999997)
    
    
def test_v_Sounders_Brown():
    v = v_Sounders_Brown(K=0.08, rhol=985.4, rhog=1.3)
    assert_allclose(v, 2.2010906387516167)
    
    
def test_K_Sounders_Brown_theoretical():
    K = K_Sounders_Brown_theoretical(D=150E-6, Cd=0.5)
    assert_allclose(K, 0.06263114241333939)