# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.atmosphere import ATMOSPHERE_NRLMSISE00
import numpy as np
import os

def helper_test_match(f, atms):
    indexes = [1, 2, 3, 7, 8, 10]
    keys = ['O_density', 'N2_density', 'O2_density',
            'He_density', 'Ar_density', 'N_density']
    
    for i, k in zip(indexes, keys):
        calcs = [getattr(a, k) for a in atms]
        assert_allclose(calcs, f[:, i]*1E6, rtol=1E-3)
    
    calcs = [a.rho for a in atms]
    assert_allclose(calcs, f[:, 4]*1E3, rtol=1E-3)
    calcs = [a.T for a in atms]
    assert_allclose(calcs, f[:, 5], rtol=1E-3)
    calcs = [a.T_exospheric for a in atms]
    assert_allclose(calcs, f[:, 6], rtol=1E-3)


def test_ATMOSPHERE_NRLMSISE00():
    name = os.path.join(os.path.dirname(__file__), 'known_data_height.txt')
    f = np.loadtxt(name, delimiter=' ')
    heights = f[:,0]
    
    atms = [ATMOSPHERE_NRLMSISE00(float(h)*1000, latitude=45, longitude=45, day=1, seconds=0, geomagnetic_disturbance_indices=[4]*7) for h in heights]
    helper_test_match(f, atms)
    
    name = os.path.join(os.path.dirname(__file__), 'known_data_high_height.txt')
    f = np.loadtxt(name, delimiter=' ')
    heights = f[:,0]
    atms = [ATMOSPHERE_NRLMSISE00(float(h)*1000, latitude=45, longitude=45, day=1, seconds=0, geomagnetic_disturbance_indices=[4]*7) for h in heights]
    helper_test_match(f, atms)
    
    name = os.path.join(os.path.dirname(__file__), 'known_data_day_of_year.txt')
    f = np.loadtxt(name, delimiter=' ')    
    atms = [ATMOSPHERE_NRLMSISE00(100000., latitude=45, longitude=45, day=d, seconds=0, geomagnetic_disturbance_indices=[4]*7) for d in range(1, 367)]
    helper_test_match(f, atms)
    
    name = os.path.join(os.path.dirname(__file__), 'known_data_hours.txt')
    f = np.loadtxt(name, delimiter=' ')    
    atms = [ATMOSPHERE_NRLMSISE00(100000., latitude=45, longitude=45, day=1, seconds=3600.*h, geomagnetic_disturbance_indices=[4]*7) for h in range(1, 25)]
    helper_test_match(f, atms)
    
    name = os.path.join(os.path.dirname(__file__), 'known_data_latitudes.txt')
    f = np.loadtxt(name, delimiter=' ')    
    atms = [ATMOSPHERE_NRLMSISE00(100000., latitude=l, longitude=45, day=1, seconds=0, geomagnetic_disturbance_indices=[4]*7) for l in range(-90, 91)]
    helper_test_match(f, atms)
    
    name = os.path.join(os.path.dirname(__file__), 'known_data_longitudes.txt')
    f = np.loadtxt(name, delimiter=' ')    
    atms = [ATMOSPHERE_NRLMSISE00(100000., latitude=45, longitude=l, day=1, seconds=0, geomagnetic_disturbance_indices=[4]*7) for l in range(0, 361)]
    helper_test_match(f, atms)
