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

from numpy.testing import assert_allclose
import pytest
import numpy as np
from fluids.design_climate import *
from fluids.design_climate import _latlongs, stations

def test_IntegratedSurfaceDatabaseStation():
    
    # Information confirmed elsewhere i.e. https://geographic.org/global_weather/not_specified_canada/calgary_intl_cs_713930_99999.html
    values = [713930.0, 99999.0, 'CALGARY INTL CS', 'CA', None, None, 51.1, -114.0, 1081.0, 20040921.0, 20150831.0]
    test_station = IntegratedSurfaceDatabaseStation(*values)
    for value, attr in zip(values, test_station.__slots__):
        assert value == getattr(test_station, attr)
        
def test_data():
    assert _latlongs.shape[0] >= 27591
    for station in stations:
        assert abs(station.LAT) <= 90
        assert abs(station.LON) <= 180

