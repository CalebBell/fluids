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


def test_safety_valve():
    A = API520_round_size(1E-4)
    assert_close(A, 0.00012645136)
    assert 'E' == API526_letters[API526_A.index(API520_round_size(1E-4))]
    with pytest.raises(Exception):
        API520_round_size(1)

    C1, C2 = API520_C(1.35), API520_C(1.)
    Cs = [0.02669419967057233, 0.023945830445454768]
    assert_close1d([C1, C2], Cs)

    F2 = API520_F2(1.8, 1E6, 7E5)
    assert_close(F2, 0.8600724121105563)

    Kv_calcs = [API520_Kv(100), API520_Kv(4525), API520_Kv(1E5)]
    Kvs = [0.6157445891444229, 0.9639390032437682, 0.9973949303006829]
    assert_close1d(Kv_calcs, Kvs)

    KN = API520_N(1774700)
    assert_close(KN, 0.9490406958152466)

    with pytest.raises(Exception):
        API520_SH(593+273.15, 21E6)
    with pytest.raises(Exception):
        API520_SH(1000, 1066E3)
    # Test under 15 psig sat case
    assert API520_SH(320, 5E4) == 1

    from fluids.safety_valve import _KSH_Pa, _KSH_tempKs
    KSH_tot =  sum([API520_SH(T, P) for P in _KSH_Pa[:-1] for T in _KSH_tempKs])
    assert_close(229.93, KSH_tot)

    KW = [API520_W(1E6, 3E5), API520_W(1E6, 1E5)]
    assert_close1d(KW, [0.9511471848008564, 1])

    B_calc = [API520_B(1E6, 3E5), API520_B(1E6, 5E5), API520_B(1E6, 5E5, overpressure=.16), API520_B(1E6, 5E5, overpressure=.21)]
    Bs = [1, 0.7929945420944432, 0.94825439189912, 1]
    assert_close1d(B_calc, Bs)

    with pytest.raises(Exception):
        API520_B(1E6, 5E5, overpressure=.17)
    with pytest.raises(Exception):
        API520_B(1E6, 7E5, overpressure=.16)

    A1 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1.0, Kc=1.0)
    A2 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1.0, Kc=1.0)
    As = [0.0036990460646834414, 0.004248358775943481]
    assert_close1d([A1, A2], As)

    A = API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1.0, Kc=1.0)
    assert_close(A, 0.0011034712423692733)

