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

from fluids import *
from numpy.testing import assert_allclose


def test_compressible():
    T = T_critical_flow(473, 1.289)
    assert_allclose(T, 413.2809086937528)

    P = P_critical_flow(1400000, 1.289)
    assert_allclose(P, 766812.9022792266)

    TF = [is_critical_flow(670E3, 532E3, 1.11), is_critical_flow(670E3, 101E3, 1.11)]
    assert_allclose(TF, [False, True])

    SE = stagnation_energy(125)
    assert_allclose(SE, 7812.5)

    PST = P_stagnation(54050., 255.7, 286.8, 1.4)
    assert_allclose(PST, 80772.80495900588)

    Tst = T_stagnation(286.8, 54050, 54050*8, 1.4)
    assert_allclose(Tst, 519.5230938217768)

    Tstid = T_stagnation_ideal(255.7, 250, 1005.)
    assert_allclose(Tstid, 286.79452736318405)
