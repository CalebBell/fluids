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

from __future__ import division
from fluids import *
from fluids.numerics import assert_close, assert_close1d
import pytest

def test_packed_bed():
    dP = Ergun(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1338.8671874999995)

    dP = Kuo_Nydegger(dp=8E-1, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 0.025651460973648624)

    dP = Jones_Krier(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1362.2719449873746)

    dP = Carman(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1614.721678121775)

    dP = Hicks(dp=0.01, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 3.631703956680737)

    dP = Brauer(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1441.5479196020563)

    dP = KTA(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1440.409277034248)

    dP = Erdim_Akgiray_Demir(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1438.2826958844414)

    dP = Tallmadge(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1365.2739144209424)

    dP = Fahien_Schriver(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1470.6175541844711)

    dP = Idelchik(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_close(dP, 1571.909125999067)

    dP1 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2)
    assert_close1d([dP1, dP2], [1104.6473821473724, 1255.1625662548427])

    dP1 = Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    dP2 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.4, L=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    dP3 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.3, L=0.5, vs=0.05, rho=1000., mu=1.00E-003, Dt=1.0)
    assert_close1d([dP1, dP2, dP3], [1148.1905244077548, 212.67409611116554, 540.501305905986])

    dP1 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01)
    dP3 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP4 = dP_packed_bed(dp=8E-4, voidage=0.4, sphericity=0.6, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP5 = dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3)
    assert_close1d([dP1, dP2, dP3, dP4, dP5], [1438.2826958844414, 1255.1625662548427, 1338.8671874999995, 3696.2890624999986, 1438.2826958844414])

    # REMOVE ONCE DEPRECATED
    methods_dP_val = ['Harrison, Brunner & Hecker', 'Carman', 'Guo, Sun, Zhang, Ding & Liu', 'Hicks', 'Montillet, Akkari & Comiti', 'Idelchik', 'Erdim, Akgiray & Demir', 'KTA', 'Kuo & Nydegger', 'Ergun', 'Brauer', 'Fahien & Schriver', 'Jones & Krier', 'Tallmadge']
    methods_dP_val.sort()

    for m in methods_dP_val:
        dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method=m)

    all_methods = dP_packed_bed_methods(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=1, Dt=1e-2)
    assert 'Erdim, Akgiray & Demir' == dP_packed_bed_methods(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=1.0)[0]
    assert 'Harrison, Brunner & Hecker' == all_methods[0]
    all_methods.sort()
    assert all_methods == methods_dP_val



    with pytest.raises(Exception):
        dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3, Method='Fail')

    v = voidage_Benyahia_Oneil(1E-3, 1E-2, .8)
    assert_close(v, 0.41395363849210065)
    v = voidage_Benyahia_Oneil_spherical(.001, .05)
    assert_close(v, 0.3906653157443224)
    v = voidage_Benyahia_Oneil_cylindrical(.01, .1, .6)
    assert_close(v, 0.38812523109607894)


def test_Guo_Sun():
    dP = Guo_Sun(dp=14.2E-3, voidage=0.492, vs=0.6, rho=1E3, mu=1E-3, Dt=40.9E-3)
    assert_close(dP, 42019.529911473706)
    # Confirmed to be 42 kPa from a graph they provided