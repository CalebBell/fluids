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


def test_drag():
    Cd = Stokes(0.1)
    close(Cd, 240.0)

    # All examples from [1]_, in a table of calculated values, matches.
    Cds = [Barati(200.), Barati(0.002)]
    close(Cds, [0.7682237950389874, 12008.864343802072])

    # All examples from [1]_, in a table of calculated values, matches.
    Cds = [Barati_high(i) for i in [200, 0.002, 1E6]]
    Cd_values = [0.7730544082789523, 12034.714777630921, 0.21254574397767056]
    close(Cds, Cd_values)

    Cds = [Rouse(i) for i in [200, 0.002]]
    Cd_values = [0.6721320343559642, 12067.422039324994]
    close(Cds, Cd_values)

    Cds = [Engelund_Hansen(i) for i in [200, 0.002]]
    Cd_values = [1.62, 12001.5]
    close(Cds, Cd_values)

    Cds = [Clift_Gauvin(i) for i in [200, 0.002]]
    Cd_values = [0.7905400398000133, 12027.153270425813]
    close(Cds, Cd_values)

    Cds = [Morsi_Alexander(i) for i in [0.002, 0.5, 5., 50., 500., 25E2, 7.5E3, 1E5]]
    Cd_values = [12000.0, 49.511199999999995, 6.899784, 1.500032, 0.549948, 0.408848, 0.4048818666666667, 0.50301667]
    close(Cds, Cd_values)

    Cds = [Graf(i) for i in [200, 0.002]]
    Cd_values = [0.8520984424785725, 12007.237509093471]
    close(Cds, Cd_values)

    Cds = [Flemmer_Banks(i) for i in [200, 0.002]]
    Cd_values = [0.7849169609270039, 12194.582998088363]
    close(Cds, Cd_values)

    Cds = [Khan_Richardson(i) for i in [200, 0.002]]
    Cd_values = [0.7747572379211097, 12335.279663284822]
    close(Cds, Cd_values)

    Cds = [Swamee_Ojha(i) for i in [200, 0.002]]
    Cd_values = [0.8490012397545713, 12006.510258198376]
    close(Cds, Cd_values)

    Cds = [Yen(i) for i in [200, 0.002]]
    Cd_values = [0.7822647002187014, 12080.906446259793]
    close(Cds, Cd_values)

    Cds = [Haider_Levenspiel(i) for i in [200, 0.002]]
    Cd_values = [0.7959551680251666, 12039.14121183969]
    close(Cds, Cd_values)

    Cds = [Cheng(i) for i in [200, 0.002]]
    Cd_values = [0.7939143028294227, 12002.787740305668]
    close(Cds, Cd_values)

    Cds = [Terfous(i) for i in [200]]
    Cd_values = [0.7814651149769638]
    close(Cds, Cd_values)

    Cds = [Mikhailov_Freire(i) for i in [200, 0.002]]
    Cd_values = [0.7514111388018659, 12132.189886046555]
    close(Cds, Cd_values)

    Cds = [Clift(i) for i in [0.002, 0.5, 50., 500., 2500., 40000, 75000., 340000, 5E5]]
    Cd_values = [12000.1875, 51.538273834491875, 1.5742657203722197, 0.5549240285782678, 0.40817983162668914, 0.4639066546786017, 0.49399935325210037, 0.4631617396760497, 0.5928043008238435]
    close(Cds, Cd_values)

    Cds = [Ceylan(i) for i in [200]]
    Cd_values = [0.7816735980280175]
    close(Cds, Cd_values)

    Cds = [Almedeij(i) for i in [200, 0.002]]
    Cd_values = [0.7114768646813396, 12000.000000391443]
    close(Cds, Cd_values)

    Cds = [Morrison(i) for i in [200, 0.002]]
    Cd_values = [0.767731559965325, 12000.134917101897]
    close(Cds, Cd_values)


