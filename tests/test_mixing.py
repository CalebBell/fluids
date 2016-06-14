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


def test_mixing():
    t1 = agitator_time_homogeneous(D=36*.0254, N=56/60., P=957., T=1.83, H=1.83, mu=0.018, rho=1020, homogeneity=.995)
    t2 = agitator_time_homogeneous(D=1, N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)
    t3 = agitator_time_homogeneous(N=125/60., P=298., T=3, H=2.5, mu=.5, rho=980, homogeneity=.95)

    assert_allclose([t1, t2, t3], [15.143198226374668, 67.7575069865228, 51.70865552491966])

    Kp = Kp_helical_ribbon_Rieger(D=1.9, h=1.9, nb=2, pitch=1.9, width=.19, T=2)
    assert_allclose(Kp, 357.39749163259256)

    t = time_helical_ribbon_Grenville(357.4, 4/60.)
    assert_allclose(t, 650.980654028894)

    CoV = size_tee(Q1=11.7, Q2=2.74, D=0.762, D2=None, n=1, pipe_diameters=5)
    assert_allclose(CoV, 0.2940930233038544)

    CoV = COV_motionless_mixer(Ki=.33, Q1=11.7, Q2=2.74, pipe_diameters=4.74/.762)
    assert_allclose(CoV, 0.0020900028665727685)

    K = K_motionless_mixer(K=150, L=.762*5, D=.762, fd=.01)
    assert_allclose(K, 7.5)
