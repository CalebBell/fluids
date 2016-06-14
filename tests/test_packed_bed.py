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

def test_packed_bed():
    dP = Ergun(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1338.8671874999995)

    dP = Kuo_Nydegger(dp=8E-1, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 0.025651460973648624)

    dP = Jones_Krier(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1362.2719449873746)

    dP = Carman(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1614.721678121775)

    dP = Hicks(dp=0.01, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 3.631703956680737)

    dP = Brauer(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1441.5479196020563)

    dP = KTA(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1440.409277034248)

    dP = Erdim_Akgiray_Demir(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1438.2826958844414)

    dP = Fahien_Schriver(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1470.6175541844711)

    dP = Idelchik(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    assert_allclose(dP, 1571.909125999067)

    dP1 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2)
    assert_allclose([dP1, dP2], [1104.6473821473724, 1255.1625662548427])

    dP1 = Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    dP2 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.4, L=0.5, vs=0.05, rho=1000., mu=1.00E-003)
    dP3 = Montillet_Akkari_Comiti(dp=0.08, voidage=0.3, L=0.5, vs=0.05, rho=1000., mu=1.00E-003, Dt=1)
    assert_allclose([dP1, dP2, dP3], [1148.1905244077548, 212.67409611116554, 540.501305905986])

    dP1 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    dP2 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01)
    dP3 = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP4 = dP_packed_bed(dp=8E-4, voidage=0.4, sphericity=0.6, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, Method='Ergun')
    dP5 = dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3)
    assert_allclose([dP1, dP2, dP3, dP4, dP5], [1438.2826958844414, 1255.1625662548427, 1338.8671874999995, 3696.2890624999986, 1438.2826958844414])

    methods_dP = dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01, AvailableMethods=True)
    methods_dP.sort()
    methods_dP_val = ['Harrison, Brunner & Hecker', 'Carman', 'Hicks', 'Montillet, Akkari & Comiti', 'Idelchik', 'Erdim, Akgiray & Demir', 'KTA', 'Kuo & Nydegger', 'Ergun', 'Brauer', 'Fahien & Schriver', 'Jones & Krier']
    methods_dP_val.sort()
    assert methods_dP == methods_dP_val

    with pytest.raises(Exception):
        dP_packed_bed(8E-4, 0.4, 1E-3, 1E3, 1E-3, Method='Fail')

    v = voidage_Benyahia_Oneil(1E-3, 1E-2, .8)
    assert_allclose(v, 0.41395363849210065)
    v = voidage_Benyahia_Oneil_spherical(.001, .05)
    assert_allclose(v, 0.3906653157443224)
    v = voidage_Benyahia_Oneil_cylindrical(.01, .1, .6)
    assert_allclose(v, 0.38812523109607894)
