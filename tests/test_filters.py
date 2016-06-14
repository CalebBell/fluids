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
from numpy.testing import assert_allclose
import pytest


def test_filters():
    K1 = round_edge_screen(0.5, 100)
    K2 = round_edge_screen(0.5, 100, 45)
    K3 = round_edge_screen(0.5, 100, 85)

    assert_allclose([K1, K2, K3], [2.0999999999999996, 1.05, 0.18899999999999997])

    Ks =  [round_edge_open_mesh(0.88, i) for i in ['round bar screen', 'diamond pattern wire', 'knotted net', 'knotless net']]
    K_values = [0.11687999999999998, 0.09912, 0.15455999999999998, 0.11664]
    assert_allclose(Ks, K_values)

    K1 = round_edge_open_mesh(0.96, angle=33.)
    K2 = round_edge_open_mesh(0.96, angle=50)
    assert_allclose([K1, K2], [0.02031327712601458, 0.012996000000000014])

    with pytest.raises(Exception):
        round_edge_open_mesh(0.96, subtype='not_filter', angle=33.)

    K = square_edge_screen(0.99)
    assert_allclose(K, 0.008000000000000009)

    K1 = square_edge_grill(.45)
    K2 = square_edge_grill(.45, l=.15, Dh=.002, fd=.0185)
    assert_allclose([K1, K2], [5.296296296296296, 12.148148148148147])

    K1 = round_edge_grill(.4)
    K2 = round_edge_grill(.4, l=.15, Dh=.002, fd=.0185)
    assert_allclose([K1, K2], [1.0, 2.3874999999999997])
