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


def test_open_flow():
    Q1 = Q_weir_V_Shen(0.6, angle=45)
    Q2 = Q_weir_V_Shen(1.2)

    assert_close1d([Q1, Q2], [0.21071725775478228, 2.8587083148501078])

    Q1 = Q_weir_rectangular_Kindsvater_Carter(0.2, 0.5, 1.0)
    assert_close(Q1, 0.15545928949179422)

    Q1 = Q_weir_rectangular_SIA(0.2, 0.5, 1.0, 2.0)
    assert_close(Q1, 1.0408858453811165)

    Q1 = Q_weir_rectangular_full_Ackers(h1=0.9, h2=0.6, b=5.0)
    Q2 = Q_weir_rectangular_full_Ackers(h1=0.3, h2=0.4, b=2.0)
    assert_close1d([Q1, Q2], [9.251938159899948, 0.6489618999846898])

    Q1 = Q_weir_rectangular_full_SIA(h1=0.3, h2=0.4, b=2.0)
    assert_close(Q1, 1.1875825055400384)

    Q1 = Q_weir_rectangular_full_Rehbock(h1=0.3, h2=0.4, b=2.0)
    assert_close(Q1, 0.6486856330601333)

    Q1 = Q_weir_rectangular_full_Kindsvater_Carter(h1=0.3, h2=0.4, b=2.0)
    assert_close(Q1, 0.641560300081563)

    V1 = V_Manning(0.2859, 0.005236, 0.03)*0.5721
    V2 = V_Manning(0.2859, 0.005236, 0.03)
    V3 = V_Manning(Rh=5, S=0.001, n=0.05)
    assert_close1d([V1, V2, V3], [0.5988618058239864, 1.0467781958118971, 1.8493111942973235])

    C = n_Manning_to_C_Chezy(0.05, Rh=5.0)
    assert_close(C, 26.15320972023661)

    n = C_Chezy_to_n_Manning(26.15, Rh=5.0)
    assert_close(n, 0.05000613713238358)

    V = V_Chezy(Rh=5.0, S=0.001, C=26.153)
    assert_close(V, 1.8492963648371776)

    n_tot = 0.0
    for thing in n_dicts:
        for val in thing.values():
            for vals in val.values():
                n_tot += abs(sum(vals))

    assert_close(n_tot, 11.115999999999984)