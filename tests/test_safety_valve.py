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
SOFTWARE.
'''

import pytest

from fluids.constants import atm
from fluids.numerics import assert_close, assert_close1d, linspace
from fluids.safety_valve import (
    API520_B,
    API520_C,
    API520_F2,
    API520_N,
    API520_SH,
    API520_W,
    API526_A,
    API520_A_g,
    API520_A_steam,
    API520_Kv,
    API520_round_size,
    API521_noise,
    API521_noise_graph,
    API526_letters,
    VDI_3732_noise_elevated_flare,
    VDI_3732_noise_ground_flare,
)


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

    KN = API520_N(1774700)
    assert_close(KN, 1)


    KW = [API520_W(1E6, 3E5), API520_W(1E6, 1E5)]
    assert_close1d(KW, [0.9511471848008564, 1])

    B_calc = [API520_B(1E6, 3E5), API520_B(1E6, 5E5), API520_B(1E6, 5E5, overpressure=.16), API520_B(1E6, 5E5, overpressure=.21)]
    Bs = [1, 0.7929945420944432, 0.94825439189912, 1]
    assert_close1d(B_calc, Bs)

    # Issue # 45
    assert 1 == API520_B(2*atm, 1.5*atm, overpressure=.21)

    with pytest.raises(Exception):
        API520_B(1E6, 5E5, overpressure=.17)
    with pytest.raises(Exception):
        API520_B(1E6, 7E5, overpressure=.16)

    A1 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1.0, Kc=1.0)
    A2 = API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, P2=532E3, Kd=0.975, Kb=1.0, Kc=1.0)
    As = [0.0036990460646834414, 0.004248358775943481]
    assert_close1d([A1, A2], As)

def test_API520_Kv():
    Kv_calcs = [API520_Kv(100, edition='7E'), API520_Kv(4525, edition='7E'), API520_Kv(1E5, edition='7E')]
    Kvs = [0.6157445891444229, 0.9639390032437682, 0.9973949303006829]
    assert_close1d(Kv_calcs, Kvs)

    assert API520_Kv(1e9, edition='7E') == 1

    assert_close(API520_Kv(4525, edition='10E'), 0.9817287137013179)


def test_API520_SH():
    with pytest.raises(Exception):
        API520_SH(593+273.15, 21E6, '7E')
    with pytest.raises(Exception):
        API520_SH(1000, 1066E3, '7E')
    # Test under 15 psig sat case
    assert API520_SH(320, 5E4, '7E') == 1

    from fluids.safety_valve import _KSH_Pa_7E, _KSH_tempKs_7E
    KSH_tot =  sum([API520_SH(T, P, '7E') for P in _KSH_Pa_7E[:-1] for T in _KSH_tempKs_7E])
    assert_close(229.93, KSH_tot)

    # 10E
    with pytest.raises(Exception):
        # too high temp
        API520_SH(593+273.15, 23E6, '10E')
    with pytest.raises(Exception):
        API520_SH(1000, 1066E3, '10E')

    assert API520_SH(470, 1066E3, '10E') == 1.0
    from fluids.safety_valve import _KSH_K_10E, _KSH_Pa_10E
    KSH_10E_tot =  sum([API520_SH(T, P, '10E') for P in _KSH_Pa_10E for T in _KSH_K_10E])
    assert_close(1336.4789999999996, KSH_10E_tot)

    for P in linspace(_KSH_Pa_10E[0], _KSH_Pa_10E[-1], 30):
        for T in linspace(_KSH_K_10E[0], _KSH_K_10E[-1], 30):
            val = API520_SH(T, P, '10E')
            assert val <= (1+1e-15)
            assert val >= 0.627



def test_API520_A_steam():
    A = API520_A_steam(m=69615/3600., T=592.5, P1=12236E3, Kd=0.975, Kb=1.0, Kc=1.0, edition='7E')
    assert_close(A, 0.0011034712423692733)

    A = API520_A_steam(m=69615/3600., T=707.0389, P1=12236E3, Kd=0.975, Kb=1, Kc=1, edition='10E')
    assert_close(A, 0.00128518893191)


def test_API521_noise_graph():
    assert_close(API521_noise_graph(1.5), 28.25, atol=.01)
    assert_close(API521_noise_graph(2.92), 53.675762)
    assert_close(API521_noise_graph(10), 56.4456)
    assert_close(API521_noise_graph(2.925), 53.80566202669078)
    assert_close(API521_noise_graph(4),54.525977192166955)
    assert_close(API521_noise_graph(1.), 12.7647)
    assert_close(API521_noise_graph(0), API521_noise_graph(1))

def test_API521_noise():
    assert_close(API521_noise(m=14.6, P1=330E3, P2=101325, c=353.0, r=30), 113.68410573691534)


def test_VDI_3732():
    assert_close(VDI_3732_noise_elevated_flare(3.0), 163.56820384327)
    assert_close(VDI_3732_noise_ground_flare(3.0), 145.501356332)

