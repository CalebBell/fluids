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
from scipy.constants import hp


def test_pump():
    eta = Corripio_pump_efficiency(461./15850.323)
    assert_allclose(eta, 0.7058888670951621)
    eta = Corripio_motor_efficiency(137*745.7)
    assert_allclose(eta, 0.9128920875679222)

    eta = VFD_efficiency(10*hp)
    assert_allclose(eta, 0.96)

    eta = VFD_efficiency(100*hp, load=0.5)
    assert_allclose(eta, 0.96)

    # Lower bound, 3 hp; upper bound, 400 hp; 0.016 load bound
    etas = VFD_efficiency(1*hp), VFD_efficiency(500*hp), VFD_efficiency(8*hp, load=0.01)
    assert_allclose(etas, [0.94, 0.97, 0.386])

    hp_sum = sum(nema_sizes_hp)
    assert_allclose(hp_sum, 3356.333333333333)
    W_sum = sum(nema_sizes)
    assert_allclose(W_sum, 2502817.33565396)

    sizes = [motor_round_size(i) for i in [.1*hp, .25*hp, 1E5, 3E5]]
    sizes_calc = [186.42496789556753, 186.42496789556753, 111854.98073734052, 335564.94221202156]
    assert_allclose(sizes, sizes_calc)

    with pytest.raises(Exception):
        motor_round_size(1E100)

    nema_high_P_calcs = [CSA_motor_efficiency(k*hp, high_efficiency=True, closed=i, poles=j) for i in [True, False] for j in [2, 4, 6] for k in nema_high_P]
    nema_high_Ps = [0.77, 0.84, 0.855, 0.865, 0.885, 0.885, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.95, 0.95, 0.954, 0.954, 0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.945, 0.95, 0.954, 0.954, 0.954, 0.958, 0.962, 0.962, 0.825, 0.875, 0.885, 0.895, 0.895, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.958, 0.958, 0.958, 0.77, 0.84, 0.855, 0.855, 0.865, 0.865, 0.865, 0.885, 0.895, 0.902, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.91, 0.917, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.954, 0.954, 0.958, 0.958, 0.958, 0.825, 0.865, 0.875, 0.885, 0.895, 0.895, 0.895, 0.902, 0.917, 0.917, 0.924, 0.93, 0.936, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954]
    assert_allclose(nema_high_P_calcs, nema_high_Ps)

    nema_min_P_calcs = [CSA_motor_efficiency(k*hp, high_efficiency=False, closed=i, poles=j) for i in [True, False] for j in [2, 4, 6, 8] for k in nema_min_P]
    nema_min_Ps = [0.755, 0.825, 0.84, 0.855, 0.855, 0.875, 0.875, 0.885, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.825, 0.84, 0.84, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.91, 0.91, 0.924, 0.924, 0.93, 0.93, 0.936, 0.941, 0.945, 0.945, 0.95, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.8, 0.855, 0.865, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.902, 0.902, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.74, 0.77, 0.825, 0.84, 0.84, 0.855, 0.855, 0.855, 0.885, 0.885, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.755, 0.825, 0.84, 0.84, 0.84, 0.855, 0.855, 0.875, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.93, 0.936, 0.936, 0.945, 0.945, 0.945, 0.95, 0.95, 0.954, 0.958, 0.958, 0.825, 0.84, 0.84, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.958, 0.8, 0.84, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.902, 0.902, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.74, 0.755, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]

    full_efficiencies = [motor_efficiency_underloaded(P*hp,  .99) for P in (0.5, 2.5, 7, 12, 42, 90)]
    assert_allclose(full_efficiencies, [1, 1, 1, 1, 1, 1])

    low_efficiencies = [motor_efficiency_underloaded(P*hp,  .25) for P in (0.5, 2.5, 7, 12, 42, 90)]
    low_ans = [0.6761088414400706, 0.7581996772085579, 0.8679397648030529, 0.9163243775499996, 0.9522559064662419, 0.9798906308690559]
    assert_allclose(low_efficiencies, low_ans)

    nS = specific_speed(0.0402, 100, 3550)
    assert_allclose(nS, 22.50823182748925)

    Ds = specific_diameter(Q=0.1, H=10., D=0.1)
    assert_allclose(Ds, 0.5623413251903491)

    s1, s2 = speed_synchronous(50, poles=12), speed_synchronous(60, phase=1)
    assert_allclose([s1, s2], [1500, 3600])


def test_current_ideal():
    I = current_ideal(V=120, P=1E4, PF=1, phase=1)
    assert_allclose(I, 83.33333333333333)

    I = current_ideal(V=208, P=1E4, PF=1, phase=3)
    assert_allclose(I, 27.757224480270473)

    I = current_ideal(V=208, P=1E4, PF=0.95, phase=3)
    assert_allclose(I,29.218131031863656)

    with pytest.raises(Exception):
        current_ideal(V=208, P=1E4, PF=0.95, phase=5)


def test_power_sources():
    assert sum(map(ord, plug_types)) == 1001
    assert len(plug_types) == 14

    assert sum(voltages_1_phase_residential) == 1262
    assert len(voltages_1_phase_residential) == 8

    assert sum(voltages_3_phase) == 3800
    assert len(voltages_3_phase) == 13

    assert frequencies == [50, 60]

    assert sum([i.voltage for i in residential_power.values()]) == 42071
    assert sum([i.freq for i in residential_power.values()]) == 10530
    assert len(residential_power) == 203

    ca = residential_power['ca']
    assert (ca.voltage, ca.freq, ca.plugs) == (120, 60, ['A', 'B'])

    assert sum([sum(i.voltage) for i in industrial_power.values()]) == 82144
    assert sum([i.freq for i in industrial_power.values()]) == 10210
    assert len(industrial_power) == 197

    ca = industrial_power['ca']
    assert (ca.voltage, ca.freq) == ([120, 208, 240, 480, 347, 600], 60)
