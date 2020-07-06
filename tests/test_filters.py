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


def test_filters():
    K1 = round_edge_screen(0.5, 100.0)
    K2 = round_edge_screen(0.5, 100, 45.0)
    K3 = round_edge_screen(0.5, 100, 85)

    assert_close1d([K1, K2, K3], [2.0999999999999996, 1.05, 0.18899999999999997])

    Ks =  [round_edge_open_mesh(0.88, i) for i in ['round bar screen', 'diamond pattern wire', 'knotted net', 'knotless net']]
    K_values = [0.11687999999999998, 0.09912, 0.15455999999999998, 0.11664]
    assert_close1d(Ks, K_values)

    K1 = round_edge_open_mesh(0.96, angle=33.)
    K2 = round_edge_open_mesh(0.96, angle=50)
    assert_close1d([K1, K2], [0.02031327712601458, 0.012996000000000014])

    with pytest.raises(Exception):
        round_edge_open_mesh(0.96, subtype='not_filter', angle=33.)

    K = square_edge_screen(0.99)
    assert_close(K, 0.008000000000000009)

    K1 = square_edge_grill(.45)
    K2 = square_edge_grill(.45, l=.15, Dh=.002, fd=.0185)
    assert_close1d([K1, K2], [5.296296296296296, 12.148148148148147])

    K1 = round_edge_grill(.4)
    K2 = round_edge_grill(.4, l=.15, Dh=.002, fd=.0185)
    assert_close1d([K1, K2], [1.0, 2.3874999999999997])

@pytest.mark.scipy
def test_grills_rounded():
    from scipy.interpolate import splrep
    from fluids.filters import grills_rounded_tck, grills_rounded_alphas, grills_rounded_Ks
    tck_recalc = splrep(grills_rounded_alphas, grills_rounded_Ks, s=0, k=2)
    [assert_close1d(i, j) for i, j in zip(grills_rounded_tck[:-1], tck_recalc[:-1])]

