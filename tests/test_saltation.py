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
import numpy as np
from numpy.testing import assert_allclose
import pytest

close = assert_allclose


def test_saltation():
    V1 = Rizk(0.25, 100E-6, 1.2, 0.078)
    close(V1, 9.8833092829357)

    V2 = Matsumoto_1974(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    close(V2, 19.583617317317895)

    V3 = Matsumoto_1975(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    close(V3, 18.04523091703009)

    V1 = Matsumoto_1977(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    V2 = Matsumoto_1977(mp=1., rhop=600., dp=1E-3, rhog=1.2, D=0.1, Vterminal=5.24)
    close([V1, V2], [16.64284834446686, 10.586175424073561])

    V1 = Schade(mp=1., rhop=1000., dp=1E-3, rhog=1.2, D=0.1)
    close(V1, 13.697415809497912)

    V1 = Weber_saltation(mp=1, rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=4)
    V2 = Weber_saltation(mp=1, rhop=1000., dp=1E-3, rhog=1.2, D=0.1, Vterminal=2)
    close([V1, V2], [15.227445436331474, 13.020222930460088])

    V1 = Geldart_Ling(1., 1.2, 0.1, 2E-5)
    V2 = Geldart_Ling(50., 1.2, 0.1, 2E-5)
    close([V1, V2], [7.467495862402707, 44.01407469835619])