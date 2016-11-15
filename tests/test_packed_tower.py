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
from fluids import *
from numpy.testing import assert_allclose
import pytest


def test_packed_tower():
    dP = dP_demister_dry_Setekleiv_Svendsen(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
    assert_allclose(dP, 320.3280788941329)
    dP = dP_demister_dry_Setekleiv_Svendsen_lit(S=250, voidage=.983, vs=1.2, rho=10, mu=3E-5, L=1)
    assert_allclose(dP, 209.083848658307)
    dP = voidage_experimental(m=126, rho=8000, D=1, H=1)
    assert_allclose(dP, 0.9799464771704212)

    S = specific_area_mesh(voidage=.934, d=3e-4)
    assert_allclose(S, 879.9999999999994)

    dP_dry = Stichlmair_dry(Vg=0.4, rhog=5., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7, C3=1)
    assert_allclose(dP_dry, 236.80904286559885)

    dP_wet = Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    assert_allclose(dP_wet, 539.8768237253518)

    Vg = Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
    assert_allclose(Vg, 0.6394323542687361)
    

def test_dP_demister_wet_ElDessouky():
    # Point from their figure 8
    rho = 176.35
    V = 6.
    dw = 0.32
    dP_orig = 3.88178*rho**0.375798*V**0.81317*dw**-1.56114147
    # 689.4685604448499, compares with maybe 690 Pa/m from figure
    
    voidage = 1-rho/7999.
    dP = dP_demister_wet_ElDessouky(V, voidage, dw/1000.)
    assert_allclose(dP_orig, dP)
    assert_allclose(dP, 689.4685604448499)
    
    # Test length multiplier
    assert_allclose(dP*10, dP_demister_wet_ElDessouky(V, voidage, dw/1000., 10))
    
    
def test_separation_demister_ElDessouky():
    # Point from their figure 6
    dw = 0.2
    rho = 208.16
    d_p = 5
    V = 1.35
    eta1 = 17.5047*dw**-0.28264*rho**0.099625*V**0.106878*d_p**0.383197
    eta1 /=100. # Convert to a 0-1 basis.
    voidage = 1-rho/7999.

    eta = separation_demister_ElDessouky(V, voidage, dw/1000., d_p/1000.)
    assert_allclose(eta1, eta)
    assert_allclose(eta, 0.8983693041263305)

    assert 1 == separation_demister_ElDessouky(1.35, 0.92, 0.0002, 0.005)   


def test_Robbins():
    dP = Robbins(Fpd=24, L=12.2, G=2.03, rhol=1000., rhog=1.1853, mul=0.001, H=2)
    assert_allclose(dP, 619.6624593438099)


