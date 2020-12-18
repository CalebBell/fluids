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



def test_piping():
    P1 = nearest_pipe(Di=0.021)
    assert_close1d(P1, (1, 0.02664, 0.0334, 0.0033799999999999998))
    P2 = nearest_pipe(Do=.273, schedule='5S')
    assert_close1d(P2, (10, 0.26630000000000004, 0.2731, 0.0034))

    ans_str = nearest_pipe(Do=0.5, schedule='80')
    ans_int = nearest_pipe(Do=0.5, schedule=80)
    ans_float = nearest_pipe(Do=0.5, schedule=80.0)
    ans_expect = (20, 0.45562, 0.508, 0.02619)
    assert_close1d(ans_str, ans_expect)
    assert_close1d(ans_str, ans_int)
    assert_close1d(ans_str, ans_float)


def test_gauge():

    g1s = gauge_from_t(.5, False, 'BWG'), gauge_from_t(0.005588, True)
    assert_close1d(g1s, (0.2, 5))
    g2s = gauge_from_t(0.5165, False, 'AWG'), gauge_from_t(0.00462026, True, 'AWG')
    assert_close1d(g2s, (0.2, 5))
    g3s = gauge_from_t(.4305, False, 'SWG'), gauge_from_t(0.0052578, True, 'SWG')
    assert_close1d(g3s, (0.2, 5))
    g4s = gauge_from_t(.005, False, 'MWG'), gauge_from_t(0.0003556, True, 'MWG')
    assert_close1d(g4s, (0.2, 5))
    g5s = gauge_from_t(.432, False, 'BSWG'), gauge_from_t(0.0053848, True, 'BSWG')
    assert_close1d(g5s, (0.2, 5))
    g6s = gauge_from_t(0.227, False, 'SSWG'), gauge_from_t(0.0051816, True, 'SSWG')
    assert_close1d(g6s, (1, 5))

    with pytest.raises(Exception):
        gauge_from_t(.5, False, 'FAIL') # Not in schedule
    with pytest.raises(Exception):
        gauge_from_t(0.02) # Too large

    g1 = gauge_from_t(0.002) # not in index; gauge 14, 2 mm
    g2 = gauge_from_t(0.00185) # not in index, gauge 15, within tol (10% default)
    # Limits between them are 0.0018288 and 0.0021082 m.
    g3 = gauge_from_t(0.00002)
    assert_close1d([g1, g2, g3], [14, 15, 0.004])


    t1s = t_from_gauge(.2, False, 'BWG'), t_from_gauge(5, True)
    assert_close1d(t1s, (0.5, 0.005588))

    t2s = t_from_gauge(.2, False, 'AWG'), t_from_gauge(5, True, 'AWG')
    assert_close1d(t2s, (0.5165, 0.00462026))

    t3s = t_from_gauge(.2, False, 'SWG'), t_from_gauge(5, True, 'SWG')
    assert_close1d(t3s, (0.4305, 0.0052578))

    t4s = t_from_gauge(.2, False, 'MWG'), t_from_gauge(5, True, 'MWG')
    assert_close1d(t4s, (0.005, 0.0003556))

    t5s = t_from_gauge(.2, False, 'BSWG'), t_from_gauge(5, True, 'BSWG')
    assert_close1d(t5s, (0.432, 0.0053848))

    t6s = t_from_gauge(1, False, 'SSWG'), t_from_gauge(5, True, 'SSWG')
    assert_close1d(t6s, (0.227, 0.0051816))

    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='FAIL')
    with pytest.raises(Exception):
        t_from_gauge(17.5, schedule='MWG')

    # Test schedule is implemented
    NPS, Di, Do, t = nearest_pipe(Do=.273, schedule='80D1527')
    assert NPS == 10
    assert_close1d((Di, Do, t), (0.2429256, 0.27305, 0.015062200000000001))

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=8, schedule='ABSD2680')
    assert NPS == 8
    assert_close1d((0.19685, 0.239014, 0.021082), (Di, Do, t), rtol=1e-12)


    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=27, schedule='PS115F679')
    assert NPS == 27
    assert_close1d((0.6591046, 0.7100062, 0.025450800000000003), (Di, Do, t), rtol=1e-12)

    NPS, Di, Do, t = nearest_pipe(NPS=27, schedule='PS75F679')
    assert NPS == 27
    assert_close1d((0.665607, 0.7100062, 0.0221996), (Di, Do, t), rtol=1e-12)

    NPS, Di, Do, t = nearest_pipe(NPS=27, schedule='PS46F679')
    assert NPS == 27
    assert_close1d((0.6721602, 0.7100062, 0.018923), (Di, Do, t), rtol=1e-12)

    from fluids.piping import NPS120_D1785
    assert_close(NPS120_D1785[0], 0.5)
    assert_close(NPS120_D1785[-1], 12)

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=6, schedule='PVCD2665')
    assert_close1d((0.154051, 0.168275, 0.007112), (Di, Do, t), rtol=1e-12)

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=6, schedule='80D1785')
    assert_close1d((0.1463294, 0.168275, 0.0109728), (Di, Do, t), rtol=1e-12)

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=6, schedule='DR21D2241')
    assert_close1d((0.15222219999999997, 0.168275, 0.008026400000000001), (Di, Do, t), rtol=1e-12)

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=1, schedule='DR21D2241CTS')
    assert_close1d((0.025527, 0.028575, 0.001524), (Di, Do, t), rtol=1e-12)

    # initially accidentally implemented this using the mm's given in the standard
    # however the IPS ones are authoritative.
    NPS, Di, Do, t = nearest_pipe(NPS=10, schedule='DR325D2241PIP')
    assert_close1d((0.2431288, 0.25908, 0.0079756), (Di, Do, t), rtol=1e-12)


    # Test schedule with DN
    NPS, Di, Do, t = nearest_pipe(NPS=100, schedule='S40F441SI')
    assert_close1d((0.10226, 0.1143, 0.006019999999999999), (Di, Do, t), rtol=1e-12)


def test_piping_schedule_basics():
    from fluids.piping import schedule_lookup

    for k, (NPSs, Dis, Dos, ts) in schedule_lookup.items():
        assert len(NPSs) == len(Dis)
        assert len(Dis) == len(Dos)
        assert len(Dos) == len(ts)

        for i in range(len(NPSs)-1):
            assert NPSs[i+1] >= NPSs[i]
        for i in range(len(Dis)-1):
            assert Dis[i+1] >= Dis[i]
        for i in range(len(Dos)-1):
            assert Dos[i+1] >= Dos[i]
        for i in range(len(ts)-1):
            assert ts[i+1] >= ts[i]

        for i in range(len(NPSs)):
            err = abs((Dis[i] + ts[i]*2)/Dos[i] -1)
            assert err < 1e-14
