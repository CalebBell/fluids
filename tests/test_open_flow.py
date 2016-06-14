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


def test_open_flow():
    Q1 = Q_weir_V_Shen(0.6, angle=45)
    Q2 = Q_weir_V_Shen(1.2)

    assert_allclose([Q1, Q2], [0.21071725775478228, 2.8587083148501078])

    Q1 = Q_weir_rectangular_Kindsvater_Carter(0.2, 0.5, 1)
    assert_allclose(Q1, 0.15545928949179422)

    Q1 = Q_weir_rectangular_SIA(0.2, 0.5, 1, 2)
    assert_allclose(Q1, 1.0408858453811165)

    Q1 = Q_weir_rectangular_full_Ackers(h1=0.9, h2=0.6, b=5)
    Q2 = Q_weir_rectangular_full_Ackers(h1=0.3, h2=0.4, b=2)
    assert_allclose([Q1, Q2], [9.251938159899948, 0.6489618999846898])

    Q1 = Q_weir_rectangular_full_SIA(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 1.1875825055400384)

    Q1 = Q_weir_rectangular_full_Rehbock(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 0.6486856330601333)

    Q1 = Q_weir_rectangular_full_Kindsvater_Carter(h1=0.3, h2=0.4, b=2)
    assert_allclose(Q1, 0.641560300081563)

    V1 = V_Manning(0.2859, 0.005236, 0.03)*0.5721
    V2 = V_Manning(0.2859, 0.005236, 0.03)
    V3 = V_Manning(Rh=5, S=0.001, n=0.05)
    assert_allclose([V1, V2, V3], [0.5988618058239864, 1.0467781958118971, 1.8493111942973235])

    C = n_Manning_to_C_Chezy(0.05, Rh=5)
    assert_allclose(C, 26.15320972023661)

    n = C_Chezy_to_n_Manning(26.15, Rh=5)
    assert_allclose(n, 0.05000613713238358)

    V = V_Chezy(Rh=5, S=0.001, C=26.153)
    assert_allclose(V, 1.8492963648371776)

    n_tot = np.sum(np.concatenate(np.array([list(val.values()) for thing in n_dicts for val in thing.values()])))
    assert_allclose(n_tot, 11.115999999999984)