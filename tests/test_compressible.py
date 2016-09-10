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
from numpy.testing import assert_allclose
import pytest


def test_compressible():
    T = T_critical_flow(473, 1.289)
    assert_allclose(T, 413.2809086937528)

    P = P_critical_flow(1400000, 1.289)
    assert_allclose(P, 766812.9022792266)

    TF = [is_critical_flow(670E3, 532E3, 1.11), is_critical_flow(670E3, 101E3, 1.11)]
    assert_allclose(TF, [False, True])

    SE = stagnation_energy(125)
    assert_allclose(SE, 7812.5)

    PST = P_stagnation(54050., 255.7, 286.8, 1.4)
    assert_allclose(PST, 80772.80495900588)

    Tst = T_stagnation(286.8, 54050, 54050*8, 1.4)
    assert_allclose(Tst, 519.5230938217768)

    Tstid = T_stagnation_ideal(255.7, 250, 1005.)
    assert_allclose(Tstid, 286.79452736318405)


def test_Panhandle_A():
    # Example 7-18 Gas of Crane TP 410M
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 42.56082051195928
    
    # Test all combinations of relevant missing inputs
    assert_allclose(Panhandle_A(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_allclose(Panhandle_A(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_allclose(Panhandle_A(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_allclose(Panhandle_A(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_allclose(Panhandle_A(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)
    
    with pytest.raises(Exception):
        Panhandle_A(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)
        
def test_Panhandle_B():
    # Example 7-18 Gas of Crane TP 410M
    D = 0.340
    P1 = 90E5
    P2 = 20E5
    L = 160E3
    SG=0.693
    Tavg = 277.15
    Q = 42.35366178004172
    
    # Test all combinations of relevant missing inputs
    assert_allclose(Panhandle_B(D=D, P1=P1, P2=P2, L=L, SG=SG, Tavg=Tavg), Q)
    assert_allclose(Panhandle_B(D=D, Q=Q, P2=P2, L=L, SG=SG, Tavg=Tavg), P1)
    assert_allclose(Panhandle_B(D=D, Q=Q, P1=P1, L=L, SG=SG, Tavg=Tavg), P2)
    assert_allclose(Panhandle_B(D=D, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), L)
    assert_allclose(Panhandle_B(L=L, Q=Q, P1=P1, P2=P2, SG=SG, Tavg=Tavg), D)
    
    with pytest.raises(Exception):
        Panhandle_B(D=0.340, P1=90E5, L=160E3, SG=0.693, Tavg=277.15)
    

