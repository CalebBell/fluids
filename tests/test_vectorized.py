# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import fluids.vectorized
import numpy as np


#def test_a_complicated_function():
#    # Disturbingly, the following test does not pass even if the new arguments
#    # are converted to floats; odeint is just behaving differently
#    orig = [fluids.integrate_drag_sphere(D=D, rhop=rhop, rho=1.2, mu=1.78E-5, t=0.5, V=30, distance=True) for D, rhop in zip([0.002, 0.001], [2200., 2300])]
#    
#    ans_vect = fluids.vectorized.integrate_drag_sphere(D=[0.002, 0.001], rhop=[2200., 2300], rho=1.2, mu=1.78E-5, t=0.5, V=30, distance=True)
#    assert_allclose(ans_vect, orig)
    


def test_Morsi_Alexander():
    Cds = [Morsi_Alexander(i) for i in [500, 5000, 5000]]
    Cds_vect = fluids.vectorized.Morsi_Alexander([500, 5000, 5000])
    assert_allclose(Cds, Cds_vect)

    
test_Morsi_Alexander()