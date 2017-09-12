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
