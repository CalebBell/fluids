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
from fluids import *
from fluids.numerics import assert_close, assert_close1d
import pytest



def test_piping():
    P1 = nearest_pipe(Di=0.021)
    assert_close1d(P1, (1, 0.02664, 0.0334, 0.0033799999999999998))
    P2 = nearest_pipe(Do=.273, schedule='5S')
    assert_close1d(P2, (10, 0.26630000000000004, 0.2731, 0.0034))
    
    ans_str = nearest_pipe(Do=0.5, schedule='80')
    ans_int = nearest_pipe(Do=0.5, schedule=80)
    ans_float = nearest_pipe(Do=0.5, schedule=80.0)
    ans_expect = (20, 0.45562, 0.508, 0.02619)
    assert_close1d(ans_str, ans_expect)
    assert_close1d(ans_str, ans_int)
    assert_close1d(ans_str, ans_float)
    
    
def test_gauge():

    g1s = gauge_from_t(.5, False, 'BWG'), gauge_from_t(0.005588, True)
    assert_close1d(g1s, (0.2, 5))
    g2s = gauge_from_t(0.5165, False, 'AWG'), gauge_from_t(0.00462026, True, 'AWG')
    assert_close1d(g2s, (0.2, 5))
    g3s = gauge_from_t(.4305, False, 'SWG'), gauge_from_t(0.0052578, True, 'SWG')
    assert_close1d(g3s, (0.2, 5))
    g4s = gauge_from_t(.005, False, 'MWG'), gauge_from_t(0.0003556, True, 'MWG')
    assert_close1d(g4s, (0.2, 5))
    g5s = gauge_from_t(.432, False, 'BSWG'), gauge_from_t(0.0053848, True, 'BSWG')
    assert_close1d(g5s, (0.2, 5))
    g6s = gauge_from_t(0.227, False, 'SSWG'), gauge_from_t(0.0051816, True, 'SSWG')
    assert_close1d(g6s, (1, 5))

    with pytest.raises(Exception):
        gauge_from_t(.5, False, 'FAIL') # Not in schedule
    with pytest.raises(Exception):
        gauge_from_t(0.02) # Too large

    g1 = gauge_from_t(0.002) # not in index; gauge 14, 2 mm
    g2 = gauge_from_t(0.00185) # not in index, gauge 15, within tol (10% default)
    # Limits between them are 0.0018288 and 0.0021082 m.
    g3 = gauge_from_t(0.00002)
    assert_close1d([g1, g2, g3], [14, 15, 0.004])


    t1s = t_from_gauge(.2, False, 'BWG'), t_from_gauge(5, True)
    assert_close1d(t1s, (0.5, 0.005588))

    t2s = t_from_gauge(.2, False, 'AWG'), t_from_gauge(5, True, 'AWG')
    assert_close1d(t2s, (0.5165, 0.00462026))

    t3s = t_from_gauge(.2, False, 'SWG'), t_from_gauge(5, True, 'SWG')
    assert_close1d(t3s, (0.4305, 0.0052578))

    t4s = t_from_gauge(.2, False, 'MWG'), t_from_gauge(5, True, 'MWG')
    assert_close1d(t4s, (0.005, 0.0003556))

    t5s = t_from_gauge(.2, False, 'BSWG'), t_from_gauge(5, True, 'BSWG')
    assert_close1d(t5s, (0.432, 0.0053848))

    t6s = t_from_gauge(1, False, 'SSWG'), t_from_gauge(5, True, 'SSWG')
    assert_close1d(t6s, (0.227, 0.0051816))

    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='FAIL')
    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='MWG')
