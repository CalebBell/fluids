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
import pytest


def test_K_separator_Watkins():
    calc = [[K_separator_Watkins(0.88, 985.4, 1.3, horizontal, method) for
    method in ['spline', 'branan', 'blackwell']] for horizontal in [False, True]]
    
    expect = [[0.06355763251223817, 0.06108986837654085, 0.06994527471072351],
    [0.07944704064029771, 0.07636233547067607, 0.0874315933884044]]
    
    assert_allclose(calc, expect)

    with pytest.raises(Exception):
        K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True, method='BADMETHOD')