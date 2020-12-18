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

from fluids import *
from fluids.numerics import assert_close1d, assert_close
import pytest

close = assert_close1d


def test_drag():
    Cd = Stokes(0.1)
    assert_close(Cd, 240.0)

    # All examples from [1]_, in a table of calculated values, matches.
    Cds = [Barati(200.), Barati(0.002)]
    close(Cds, [0.7682237950389874, 12008.864343802072])

    # All examples from [1]_, in a table of calculated values, matches.
    Cds = [Barati_high(i) for i in [200, 0.002, 1E6]]
    Cd_values = [0.7730544082789523, 12034.714777630921, 0.21254574397767056]
    close(Cds, Cd_values)

    Cds = [Rouse(i) for i in [200, 0.002]]
    Cd_values = [0.6721320343559642, 12067.422039324994]
    close(Cds, Cd_values)

    Cds = [Engelund_Hansen(i) for i in [200, 0.002]]
    Cd_values = [1.62, 12001.5]
    close(Cds, Cd_values)

    Cds = [Clift_Gauvin(i) for i in [200, 0.002]]
    Cd_values = [0.7905400398000133, 12027.153270425813]
    close(Cds, Cd_values)

    Cds = [Morsi_Alexander(i) for i in [0.002, 0.5, 5., 50., 500., 25E2, 7.5E3, 1E5]]
    Cd_values = [12000.0, 49.511199999999995, 6.899784, 1.500032, 0.549948, 0.408848, 0.4048818666666667, 0.50301667]
    close(Cds, Cd_values)

    Cds = [Graf(i) for i in [200, 0.002]]
    Cd_values = [0.8520984424785725, 12007.237509093471]
    close(Cds, Cd_values)

    Cds = [Flemmer_Banks(i) for i in [200, 0.002]]
    Cd_values = [0.7849169609270039, 12194.582998088363]
    close(Cds, Cd_values)

    Cds = [Khan_Richardson(i) for i in [200, 0.002]]
    Cd_values = [0.7747572379211097, 12335.279663284822]
    close(Cds, Cd_values)

    Cds = [Swamee_Ojha(i) for i in [200, 0.002]]
    Cd_values = [0.8490012397545713, 12006.510258198376]
    close(Cds, Cd_values)

    Cds = [Yen(i) for i in [200, 0.002]]
    Cd_values = [0.7822647002187014, 12080.906446259793]
    close(Cds, Cd_values)

    Cds = [Haider_Levenspiel(i) for i in [200, 0.002]]
    Cd_values = [0.7959551680251666, 12039.14121183969]
    close(Cds, Cd_values)

    Cds = [Cheng(i) for i in [200, 0.002]]
    Cd_values = [0.7939143028294227, 12002.787740305668]
    close(Cds, Cd_values)

    Cds = [Terfous(i) for i in [200]]
    Cd_values = [0.7814651149769638]
    close(Cds, Cd_values)

    Cds = [Mikhailov_Freire(i) for i in [200, 0.002]]
    Cd_values = [0.7514111388018659, 12132.189886046555]
    close(Cds, Cd_values)

    Cds = [Clift(i) for i in [0.002, 0.5, 50., 500., 2500., 40000, 75000., 340000, 5E5]]
    Cd_values = [12000.1875, 51.538273834491875, 1.5742657203722197, 0.5549240285782678, 0.40817983162668914, 0.4639066546786017, 0.49399935325210037, 0.4631617396760497, 0.5928043008238435]
    close(Cds, Cd_values)

    Cds = [Ceylan(i) for i in [200]]
    Cd_values = [0.7816735980280175]
    close(Cds, Cd_values)

    Cds = [Almedeij(i) for i in [200, 0.002]]
    Cd_values = [0.7114768646813396, 12000.000000391443]
    close(Cds, Cd_values)

    Cds = [Morrison(i) for i in [200, 0.002]]
    Cd_values = [0.767731559965325, 12000.134917101897]
    close(Cds, Cd_values)

    Cd = Song_Xu(1.72525554724508000000)
    assert_close(Cd, 17.1249219416881000000)

    Cd = Song_Xu(1.24798925062065, sphericity=0.64, S=0.55325984525397)
    assert_close(Cd, 36.00464629658840)

    from fluids.drag import drag_sphere_correlations
    for k in drag_sphere_correlations.keys():
        drag_sphere(1e6, Method=k)


def test_drag_sphere():
    Cd = drag_sphere(200)
    assert_close(Cd, 0.7682237950389874)

    Cd = drag_sphere(1E6)
    assert_close(Cd, 0.21254574397767056)

    Cd = drag_sphere(1E6, Method='Barati_high')
    assert_close(Cd, 0.21254574397767056)

    Cd = drag_sphere(0.001)
    assert_close(Cd, 24000.0)

    Cd = drag_sphere(0.05)
    assert_close(Cd, 481.23769162684573)

    with pytest.raises(Exception):
        drag_sphere(200, Method='BADMETHOD')

    with pytest.raises(Exception):
        drag_sphere(1E7)


    methods = drag_sphere_methods(3E5, True)
    method_known = ['Barati_high', 'Ceylan', 'Morrison', 'Clift', 'Almedeij']
    assert sorted(method_known) == sorted(methods)
    assert 20 == len(drag_sphere_methods(200))
    assert 21 == len(drag_sphere_methods(200000, check_ranges=False))
    assert 5 == len(drag_sphere_methods(200000, check_ranges=True))

def test_v_terminal():
    v_t = v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3)
    assert_close(v_t, 0.00414249724453)

    v_t = v_terminal(D=70E-9, rhop=2600., rho=1000., mu=1E-3)
    assert_close(v_t, 4.271340888888889e-09)

    # [2] has a good example
    v_t = v_terminal(D=70E-6, rhop=2.6E3, rho=1000., mu=1E-3)
    assert_close(v_t, 0.004142497244531304)
    # vs 0.00406 by [2], with the Oseen correlation not implemented here
    # It also has another example
    v_t = v_terminal(D=50E-6, rhop=2.8E3, rho=1000., mu=1E-3)
    assert_close(v_t, 0.0024195143465496655)
    # vs 0.002453 in [2]

    # Laminar example
    v_t = v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-1)
    assert_close(v_t, 4.271340888888888e-05)

    v_t = v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3, Method='Rouse')
    assert_close(v_t, 0.003991779430745852)

@pytest.mark.scipy
def test_integrate_drag_sphere():
    ans = integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.5, V=30.0, distance=True)
    assert_close1d(ans, (9.686465044063436, 7.829454643649386))

    ans = integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.5, V=30.0)
    assert_close(ans, 9.686465044063436)

    # Check no error when V is zero


    ans = integrate_drag_sphere(D=0.001, rhop=1.20001, rho=1.2, mu=1.78E-5, t=0.5, V=0.0)
    assert_close(ans, 3.0607521920092645e-07)

    # Stokes law regime integration
    ans = integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.1, V=0, distance=True, Method='Stokes')
    assert_close1d(ans, [0.9730274844308592, 0.04876946395795378])

    ans = integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.1, V=10, distance=True, Method='Stokes')
    assert_close1d(ans, [10.828446488771524, 1.041522867361668])

    ans = integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.1, V=-10, distance=True, Method='Stokes')
    assert_close1d(ans, [-8.882391519909806, -0.9439839394457605])

    # Stokes law regime - test case where particle is ensured to be laminar before and after the simulation
    for m in (None, 'Stokes'):
        ans = integrate_drag_sphere(D=0.000001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.1, V=0, distance=True, Method=m)
        assert_close1d(ans, [6.729981897140177e-05, 6.729519788530099e-06], rtol=1e-11)

def test_time_v_terminal_Stokes():
    t = time_v_terminal_Stokes(D=1e-7, rhop=2200., rho=1.2, mu=1.78E-5, V0=1.0)
    assert_close(t, 3.188003113787154e-06)

    # Very slow - many iterations
    t = time_v_terminal_Stokes(D=1e-2, rhop=2200., rho=1.2, mu=1.78E-5, V0=1.0, tol=1e-30)
    assert_close(t, 24800.636391802)