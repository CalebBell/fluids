# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
from fluids import *
from math import *
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d
import pytest
try:
    import numba
    import fluids.numba
except:
    numba = None
import numpy as np

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Clamond_numba():
    assert_close(fluids.numba.Clamond(10000.0, 2.0), 
                 fluids.Clamond(10000.0, 2.0), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, True),
                 fluids.Clamond(10000.0, 2.0, True), rtol=5e-15)
    assert_close(fluids.numba.Clamond(10000.0, 2.0, False),
                 fluids.Clamond(10000.0, 2.0, False), rtol=5e-15)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_string_error_message_outside_function():
    fluids.numba.entrance_sharp('Miller')
    fluids.numba.entrance_sharp()
    
    fluids.numba.entrance_angled(30, 'Idelchik')
    fluids.numba.entrance_angled(30, None)
    fluids.numba.entrance_angled(30.0)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_interp():

    assert_close(fluids.numba.CSA_motor_efficiency(100*hp, closed=True, poles=6, high_efficiency=True), 0.95)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_constants():
    assert_close(fluids.numba.K_separator_demister_York(975000), 0.09635076944244816)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_calling_function_in_other_module():
    assert_close(fluids.numba.ft_Crane(.5), 0.011782458726227104, rtol=1e-4)
    
    
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_None_is_not_multiplied_add_check_on_is_None():
    assert_close(fluids.numba.polytropic_exponent(1.4, eta_p=0.78), 1.5780346820809246, rtol=1e-5)
    
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_core_from_other_module():
    assert_close(fluids.numba.helical_turbulent_fd_Srinivasan(1E4, 0.01, .02), 0.0570745212117107)
    
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_string_branches():
    # Currently slower
    assert_close(fluids.numba.C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange'),  0.5990326277163659)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_interp_with_own_list():
    assert_close(fluids.numba.dP_venturi_tube(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0), 1788.5717754177406)
    
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_C_Reader_Harris_Gallagher_wet_venturi_tube_numba():
    assert_close(fluids.numba.C_Reader_Harris_Gallagher_wet_venturi_tube(mg=5.31926, ml=5.31926/2,  rhog=50.0, rhol=800., D=.1, Do=.06, H=1), 0.9754210845876333)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_rename_constant():
    assert_close(fluids.numba.friction_plate_Martin_1999(Re=20000, plate_enlargement_factor=1.15), 2.284018089834135)

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_list_in_list_constant_converted():
    assert_close(fluids.numba.friction_plate_Kumar(Re=2000, chevron_angle=30),
                 friction_plate_Kumar(Re=2000, chevron_angle=30))

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_have_to_make_zero_division_a_check():
    # Manually requires changes, and is unpythonic
    assert_close(fluids.numba.SA_ellipsoidal_head(2, 1.5), 
                 SA_ellipsoidal_head(2, 1.5))
    
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_functions_used_to_return_different_return_value_signatures_changed():
    assert_close1d(fluids.numba.SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',sideB_a=0.5), 
                    SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',sideB_a=0.5))

@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_secant_runs():
    # Really feel like the kwargs should work in object mode, but it doesn't
    # Just gets slower
    @numba.jit
    def to_solve(x):
        return sin(x*.3) - .5
    fluids.numba.secant(to_solve, .3, ytol=1e-10)



'''
Functions not working:
    
# splev needs to be working for this - very challenging!
fluids.numba.entrance_rounded(Di=0.1, rc=0.0235)

# lambertw won't work because optimizers won't work
fluids.numba.P_isothermal_critical_flow(P=1E6, fd=0.00185, L=1000., D=0.5)
fluids.numba.lambertw(.5)

# Set lookup missing in numba
fluids.numba.differential_pressure_meter_beta(D=0.2575, D2=0.184,  meter_type='cone meter')
fluids.numba.differential_pressure_meter_C_epsilon(D=0.07366, D2=0.05, P1=200000.0,  P2=183000.0, rho=999.1, mu=0.0011, k=1.33, m=7.702338035732168, meter_type='ISO 5167 orifice', taps='D')

# newton_system not working
fluids.numba.Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)

# Using dictionaries outside is broken
# Also, nopython is broken for this case - https://github.com/numba/numba/issues/5377
fluids.numba.roughness_Farshad('Cr13, bare', 0.05)

# Obviously not going to work
# nearest_material_roughness('condensate pipes', clean=False)

# Solver won't work because of function-in-function
fluids.numba.differential_pressure_meter_solver(D=0.07366, m=7.702338, P1=200000.0, 
P2=183000.0, rho=999.1, mu=0.0011, k=1.33, 
meter_type='ISO 5167 orifice', taps='D')

'''


'''
numba is not up to speeding up the various solvers!

I was able to contruct a secant version which numba would optimize, mostly.
However, it took 30x the time.

Trying to improve this, it was found reducing the number of arguments to secant
imroves things ~20%. Removing ytol or the exceptions did not improve things at all.

Eventually it was discovered, the rtol and xtol arguments should be fixed values inside the function.
This makes little sense, but it is what happened.
Slighyly better performance was found than in pure-python that way, although definitely not vs. pypy.


from math import sin
import inspect
source = inspect.getsource(secant)
source = source.replace(', kwargs={}', '').replace(', **kwargs', '')
source = source.replace('iterations=i, point=p, err=q1', '')
source = source.replace(', q1=q1, p1=p1, q0=q0, p0=p0', '')
exec(source)
import fluids.numba
@numba.njit
def to_solve(x):
    return sin(x*.3) - .5

new_secant = numba.njit(secant)
new_secant(to_solve, .3, ytol=1e-10)


'''