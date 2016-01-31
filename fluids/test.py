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
from fluids import filters, fittings, friction_factor, mixing, piping, core
from fluids import compressible
from fluids import pump, packed_bed, open_flow, geometry

import warnings

#warnings.simplefilter("always") #error for error

if __name__ == '__main__':
    import doctest
    doctest.testmod(compressible)
    doctest.testmod(core)
    doctest.testmod(filters)
    doctest.testmod(fittings)
    doctest.testmod(friction_factor)
    doctest.testmod(mixing)
    doctest.testmod(packed_bed)
    doctest.testmod(piping)
    doctest.testmod(pump)
    doctest.testmod(open_flow)
    doctest.testmod(geometry)
