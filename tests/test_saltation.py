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
from fluids.numerics import assert_close, assert_close1d
import pytest



def test_Rizk():
    V1 = Rizk(0.25, 100E-6, 1.2, 0.078)
    assert_close(V1, 9.8833092829357)

def test_Matsumoto_1974():
    V2 = Matsumoto_1974(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    assert_close(V2, 19.583617317317895)

def test_Matsumoto_1975():
    V3 = Matsumoto_1975(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    assert_close(V3, 18.04523091703009)

def test_Matsumoto_1977():
    V1 = Matsumoto_1977(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    V2 = Matsumoto_1977(mp=1., rhop=600., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    assert_close1d([V1, V2], [16.64284834446686, 10.586175424073561])

def test_Schade():
    V1 = Schade(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1)
    assert_close(V1, 13.697415809497912)

def test_Weber_saltation():
    V1 = Weber_saltation(mp=1.0, rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=4.0)
    V2 = Weber_saltation(mp=1.0, rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=2.0)
    assert_close1d([V1, V2], [15.227445436331474, 13.020222930460088])

def test_Geldart_Ling():
    V1 = Geldart_Ling(1., 1.2, 0.1, 2E-5)
    V2 = Geldart_Ling(50., 1.2, 0.1, 2E-5)
    assert_close1d([V1, V2], [7.467495862402707, 44.01407469835619])