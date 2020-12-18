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
from fluids.constants import hp
import pytest


def test_Corripio_pump_efficiency():
    eta = Corripio_pump_efficiency(461./15850.323)
    assert_close(eta, 0.7058888670951621)

def test_Corripio_motor_efficiency():
    eta = Corripio_motor_efficiency(137*745.7)
    assert_close(eta, 0.9128920875679222)


def test_VFD_efficiency():
    eta = VFD_efficiency(10*hp)
    assert_close(eta, 0.96)

    eta = VFD_efficiency(100*hp, load=0.5)
    assert_close(eta, 0.96)

    # Lower bound, 3 hp; upper bound, 400 hp; 0.016 load bound
    etas = VFD_efficiency(1*hp), VFD_efficiency(500*hp), VFD_efficiency(8*hp, load=0.01)
    assert_close1d(etas, [0.94, 0.97, 0.386])

    hp_sum = sum(nema_sizes_hp)
    assert_close(hp_sum, 3356.333333333333)
    W_sum = sum(nema_sizes)
    assert_close(W_sum, 2502817.33565396)

def test_motor_round_size():
    sizes = [motor_round_size(i) for i in [.1*hp, .25*hp, 1E5, 3E5]]
    sizes_calc = [186.42496789556753, 186.42496789556753, 111854.98073734052, 335564.94221202156]
    assert_close1d(sizes, sizes_calc)

    with pytest.raises(Exception):
        motor_round_size(1E100)


def test_CSA_motor_efficiency():
    nema_high_P_calcs = [CSA_motor_efficiency(k*hp, high_efficiency=True, closed=i, poles=j) for i in [True, False] for j in [2, 4, 6] for k in nema_high_P]
    nema_high_Ps = [0.77, 0.84, 0.855, 0.865, 0.885, 0.885, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.95, 0.95, 0.954, 0.954, 0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.941, 0.945, 0.95, 0.954, 0.954, 0.954, 0.958, 0.962, 0.962, 0.825, 0.875, 0.885, 0.895, 0.895, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.958, 0.958, 0.958, 0.77, 0.84, 0.855, 0.855, 0.865, 0.865, 0.865, 0.885, 0.895, 0.902, 0.91, 0.917, 0.917, 0.924, 0.93, 0.936, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.855, 0.865, 0.865, 0.895, 0.895, 0.895, 0.895, 0.91, 0.917, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.954, 0.954, 0.958, 0.958, 0.958, 0.825, 0.865, 0.875, 0.885, 0.895, 0.895, 0.895, 0.902, 0.917, 0.917, 0.924, 0.93, 0.936, 0.941, 0.941, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954]
    assert_close1d(nema_high_P_calcs, nema_high_Ps)

    nema_min_P_calcs = [CSA_motor_efficiency(k*hp, high_efficiency=False, closed=i, poles=j) for i in [True, False] for j in [2, 4, 6, 8] for k in nema_min_P]
    nema_min_Ps = [0.755, 0.825, 0.84, 0.855, 0.855, 0.875, 0.875, 0.885, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.945, 0.945, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.825, 0.84, 0.84, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.91, 0.91, 0.924, 0.924, 0.93, 0.93, 0.936, 0.941, 0.945, 0.945, 0.95, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.8, 0.855, 0.865, 0.875, 0.875, 0.875, 0.875, 0.895, 0.895, 0.902, 0.902, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.74, 0.77, 0.825, 0.84, 0.84, 0.855, 0.855, 0.855, 0.885, 0.885, 0.895, 0.895, 0.91, 0.91, 0.917, 0.917, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945, 0.755, 0.825, 0.84, 0.84, 0.84, 0.855, 0.855, 0.875, 0.885, 0.895, 0.902, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.93, 0.936, 0.936, 0.945, 0.945, 0.945, 0.95, 0.95, 0.954, 0.958, 0.958, 0.825, 0.84, 0.84, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.91, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.941, 0.941, 0.945, 0.95, 0.95, 0.95, 0.954, 0.954, 0.954, 0.954, 0.958, 0.958, 0.8, 0.84, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.902, 0.902, 0.91, 0.917, 0.924, 0.93, 0.93, 0.936, 0.936, 0.941, 0.941, 0.945, 0.945, 0.945, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.74, 0.755, 0.855, 0.865, 0.865, 0.875, 0.875, 0.885, 0.895, 0.895, 0.902, 0.902, 0.91, 0.91, 0.917, 0.924, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.945, 0.945, 0.945, 0.945, 0.945, 0.945]
    assert_close1d(nema_min_P_calcs, nema_min_Ps)

def test_motor_efficiency_underloaded():
    full_efficiencies = [motor_efficiency_underloaded(P*hp,  .99) for P in (0.5, 2.5, 7, 12, 42, 90)]
    assert_close1d(full_efficiencies, [1, 1, 1, 1, 1, 1])

    low_efficiencies = [motor_efficiency_underloaded(P*hp,  .25) for P in (0.5, 2.5, 7, 12, 42, 90)]
    low_ans = [0.6761088414400706, 0.7581996772085579, 0.8679397648030529, 0.9163243775499996, 0.9522559064662419, 0.9798906308690559]
    assert_close1d(low_efficiencies, low_ans)


def test_specific_speed():
    nS = specific_speed(0.0402, 100.0, 3550.0)
    assert_close(nS, 22.50823182748925)


def test_specific_diameter():
    Ds = specific_diameter(Q=0.1, H=10., D=0.1)
    assert_close(Ds, 0.5623413251903491)


def test_speed_synchronous():
    s1, s2 = speed_synchronous(50.0, poles=12), speed_synchronous(60.0, phase=1)
    assert_close1d([s1, s2], [1500, 3600])


def test_current_ideal():
    I = current_ideal(V=120.0, P=1E4, PF=1.0, phase=1)
    assert_close(I, 83.33333333333333)

    I = current_ideal(V=208, P=1E4, PF=1, phase=3)
    assert_close(I, 27.757224480270473)

    I = current_ideal(V=208, P=1E4, PF=0.95, phase=3)
    assert_close(I,29.218131031863656)

    with pytest.raises(Exception):
        current_ideal(V=208, P=1E4, PF=0.95, phase=5)


def test_power_sources():
    assert sum(map(ord, electrical_plug_types)) == 1001
    assert len(electrical_plug_types) == 14

    assert sum(voltages_1_phase_residential) == 1262
    assert len(voltages_1_phase_residential) == 8

    assert sum(voltages_3_phase) == 3800
    assert len(voltages_3_phase) == 13

    assert residential_power_frequencies == [50, 60]

    assert sum([i.voltage for i in residential_power.values()]) == 42071
    assert sum([i.freq for i in residential_power.values()]) == 10530
    assert len(residential_power) == 203

    ca = residential_power['ca']
    assert (ca.voltage, ca.freq, ca.plugs) == (120, 60, ('A', 'B'))

    assert sum([sum(i.voltage) for i in industrial_power.values()]) == 82144
    assert sum([i.freq for i in industrial_power.values()]) == 10210
    assert len(industrial_power) == 197

    ca = industrial_power['ca']
    assert (ca.voltage, ca.freq) == ((120, 208, 240, 480, 347, 600), 60)


def test_CountryPower():
    a = CountryPower(plugs=('C', 'F', 'M', 'N'), voltage=230.0, freq=50.0, country="South Africa")
    assert type(a) is CountryPower
    assert type(a.voltage) is float
    assert type(a.freq) is float

    CountryPower(plugs=('G',), voltage=240, freq=50, country="Seychelles")
    CountryPower(plugs=('C', 'F'), voltage=230, freq=50, country="Armenia")
    CountryPower(plugs=('D', 'G', 'J', 'K', 'L'), voltage=230, freq=50, country="Maldives")
