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

if __name__ == '__main__':
    import doctest
    doctest.testmod(compressible)
    doctest.testmod(core)
    doctest.testmod(filters)
    doctest.testmod(fittings)
    doctest.testmod(friction)
    doctest.testmod(mixing)
    doctest.testmod(packed_bed)
    doctest.testmod(piping)
    doctest.testmod(pump)
    doctest.testmod(open_flow)
    doctest.testmod(geometry)
    doctest.testmod(control_valve)
    doctest.testmod(safety_valve)
    doctest.testmod(packed_tower)
    doctest.testmod(saltation)
    doctest.testmod(two_phase_voidage)
