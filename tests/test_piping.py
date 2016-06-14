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



def test_piping():
    P1 = nearest_pipe(Di=0.021)
    assert_allclose(P1, (1, 0.02664, 0.0334, 0.0033799999999999998))
    P2 = nearest_pipe(Do=.273, schedule='5S')
    assert_allclose(P2, (10, 0.26630000000000004, 0.2731, 0.0034))

    g1s = gauge_from_t(.5, False, 'BWG'), gauge_from_t(0.005588, True)
    assert_allclose(g1s, (0.2, 5))
    g2s = gauge_from_t(0.5165, False, 'AWG'), gauge_from_t(0.00462026, True, 'AWG')
    assert_allclose(g2s, (0.2, 5))
    g3s = gauge_from_t(.4305, False, 'SWG'), gauge_from_t(0.0052578, True, 'SWG')
    assert_allclose(g3s, (0.2, 5))
    g4s = gauge_from_t(.005, False, 'MWG'), gauge_from_t(0.0003556, True, 'MWG')
    assert_allclose(g4s, (0.2, 5))
    g5s = gauge_from_t(.432, False, 'BSWG'), gauge_from_t(0.0053848, True, 'BSWG')
    assert_allclose(g5s, (0.2, 5))
    g6s = gauge_from_t(0.227, False, 'SSWG'), gauge_from_t(0.0051816, True, 'SSWG')
    assert_allclose(g6s, (1, 5))

    with pytest.raises(Exception):
        gauge_from_t(.5, False, 'FAIL') # Not in schedule
    with pytest.raises(Exception):
        gauge_from_t(0.02) # Too large

    g1 = gauge_from_t(0.002) # not in index; gauge 14, 2 mm
    g2 = gauge_from_t(0.00185) # not in index, gauge 15, within tol (10% default)
    # Limits between them are 0.0018288 and 0.0021082 m.
    g3 = gauge_from_t(0.00002)
    assert_allclose([g1, g2, g3], [14, 15, 0.004])


    t1s = t_from_gauge(.2, False, 'BWG'), t_from_gauge(5, True)
    assert_allclose(t1s, (0.5, 0.005588))

    t2s = t_from_gauge(.2, False, 'AWG'), t_from_gauge(5, True, 'AWG')
    assert_allclose(t2s, (0.5165, 0.00462026))

    t3s = t_from_gauge(.2, False, 'SWG'), t_from_gauge(5, True, 'SWG')
    assert_allclose(t3s, (0.4305, 0.0052578))

    t4s = t_from_gauge(.2, False, 'MWG'), t_from_gauge(5, True, 'MWG')
    assert_allclose(t4s, (0.005, 0.0003556))

    t5s = t_from_gauge(.2, False, 'BSWG'), t_from_gauge(5, True, 'BSWG')
    assert_allclose(t5s, (0.432, 0.0053848))

    t6s = t_from_gauge(1, False, 'SSWG'), t_from_gauge(5, True, 'SSWG')
    assert_allclose(t6s, (0.227, 0.0051816))

    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='FAIL')
    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='MWG')
