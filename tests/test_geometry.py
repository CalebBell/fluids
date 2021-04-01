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
from fluids.numerics import assert_close, assert_close1d, assert_close2d, isclose, linspace
from math import *
from fluids.constants import *
import pytest


def test_SA_partial_sphere():
    SA1 = SA_partial_sphere(1., 0.7)
    assert_close(SA1, 2.199114857512855)
    SA2 = SA_partial_sphere(2, 1) # One spherical head's surface area:
    assert_close(SA2, 6.283185307179586)

def test_V_partial_sphere():
    V1 = V_partial_sphere(1., 0.7)
    assert_close(V1, 0.4105014400690663)
    assert 0.0 == V_partial_sphere(1., 0.0)

def test_V_horiz_conical():
    # Two examples from [1]_, and at midway, full, and empty.
    Vs_horiz_conical1 = [V_horiz_conical(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs_horiz_conical1s = [2041.1923581273443, 6180.540773905826, 3648.490668241736, 7296.981336483472, 0.0]
    assert_close1d(Vs_horiz_conical1, Vs_horiz_conical1s)

    with pytest.raises(Exception):
        V_horiz_conical(D=108., L=156., a=42., h=109)


    # Head only custom example:
    V_head1 = V_horiz_conical(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 = V_horiz_conical(108., 156., 42., 84., headonly=True)/231.
    assert_close1d([V_head1, V_head2], [508.8239000645628]*2)

    assert V_horiz_conical(D=108., L=156., a=42., h=0, headonly=True) == 0.0

def test_V_horiz_ellipsoidal():
    # Two examples from [1]_, and at midway, full, and empty.
    Vs_horiz_ellipsoidal = [V_horiz_ellipsoidal(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs_horiz_ellipsoidals = [2380.9565415578145, 7103.445235921378, 4203.695769930696, 8407.391539861392, 0.0]
    assert_close1d(Vs_horiz_ellipsoidal, Vs_horiz_ellipsoidals)

    #Head only custom example:
    V_head1 = V_horiz_ellipsoidal(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 = V_horiz_ellipsoidal(108., 156., 42., 84., headonly=True)/231.
    assert_close1d([V_head1, V_head2], [970.2761310723387]*2)
    assert 0.0 == V_horiz_ellipsoidal(108., 156., 42., 0., headonly=True)

def test_V_horiz_guppy():

    # Two examples from [1]_, and at midway, full, and empty.
    V_calc = [V_horiz_guppy(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs = [1931.7208029476762, 5954.110515329029, 3412.8543046053724, 7296.981336483472, 0.0]
    assert_close1d(V_calc, Vs)

    # Head only custom example:
    V_head1 = V_horiz_guppy(D=108., L=156., a=42., h=36, headonly=True)/231.
    V_head2 = V_horiz_guppy(108., 156., 42., 36, headonly=True)/231.
    assert_close1d([V_head1, V_head2], [63.266257496613804]*2)
    assert 0.0 == V_horiz_guppy(108., 156., 42., 0.0, headonly=True)

def test_V_horiz_spherical():
    # Two examples from [1]_, and at midway, full, and empty.
    V_calc = [V_horiz_spherical(D=108., L=156., a=42., h=i)/231. for i in (36, 84, 54, 108, 0)]
    Vs = [2303.9615116986183, 6935.163365275476, 4094.025626387197, 8188.051252774394, 0.0]
    assert_close1d(V_calc, Vs)
    assert 0.0 == V_horiz_spherical(D=108., L=156., a=42., h=0)

    # Test z = 0 zero division error
    base = V_horiz_spherical(D=108., L=156., a=54, h=36, headonly=True)
    perturbed = V_horiz_spherical(D=108., L=156., a=53.999999999, h=36, headonly=True)
    assert_close(base, perturbed, rtol=1e-10)

    # Test while z is very very slow
    perturbed = V_horiz_spherical(D=108., L=156., a=53.99999999, h=36, headonly=True)
    assert_close(base, perturbed, rtol=1e-7)


    # Test when the integration function is called, on its limits:
    # The integral can be done analytically, but there's a zero to the power of negative integer error
    # the expression is
    # -cmath.atan(cmath.sqrt((R ** 2 - x ** 2) / (-R ** 2 + r ** 2))) * x ** 3 / 3 + cmath.atan(cmath.sqrt((R ** 2 - x ** 2) / (-R ** 2 + r ** 2))) * r ** 2 * x + x * (R ** 2 - r ** 2) * cmath.sqrt(0.1e1 / (R ** 2 - r ** 2) * x ** 2 - R ** 2 / (R ** 2 - r ** 2)) / 6 + R ** 2 * cmath.log(x / (R ** 2 - r ** 2) * (0.1e1 / (R ** 2 - r ** 2)) ** (-0.1e1 / 0.2e1) + cmath.sqrt(0.1e1 / (R ** 2 - r ** 2) * x ** 2 - R ** 2 / (R ** 2 - r ** 2))) * (0.1e1 / (R ** 2 - r ** 2)) ** (-0.1e1 / 0.2e1) / 6 - 0.2e1 / 0.3e1 * r ** 2 * cmath.log(x / (R ** 2 - r ** 2) * (0.1e1 / (R ** 2 - r ** 2)) ** (-0.1e1 / 0.2e1) + cmath.sqrt(0.1e1 / (R ** 2 - r ** 2) * x ** 2 - R ** 2 / (R ** 2 - r ** 2))) * (0.1e1 / (R ** 2 - r ** 2)) ** (-0.1e1 / 0.2e1) - r ** 3 * cmath.atan((-2 + 2 / (R ** 2 - r ** 2) * r * (x - r)) * (0.1e1 / (R ** 2 - r ** 2) * (x - r) ** 2 + 2 / (R ** 2 - r ** 2) * r * (x - r) - 1) ** (-0.1e1 / 0.2e1) / 2) / 3 + r ** 3 * cmath.atan((-2 - 2 / (R ** 2 - r ** 2) * r * (x + r)) * (0.1e1 / (R ** 2 - r ** 2) * (x + r) ** 2 - 2 / (R ** 2 - r ** 2) * r * (x + r) - 1) ** (-0.1e1 / 0.2e1) / 2) / 3

    Vs = [V_horiz_spherical(D=108., L=156., a=i, h=84.)/231. for i in (108*.009999999, 108*.01000001)]
    V_calc = [5201.54341872961, 5201.543461255985]
    assert_close1d(Vs, V_calc)

    # Head only custom example:
    V_head1 =  V_horiz_spherical(D=108., L=156., a=42., h=84., headonly=True)/231.
    V_head2 =  V_horiz_spherical(108., 156., 42., 84., headonly=True)/231.
    assert_close1d([V_head1, V_head2], [886.1351957493874]*2)

def test_V_horiz_torispherical():

    # Two examples from [1]_, and at midway, full, empty, and 1 inch; covering
    # all code cases.
    V_calc  = [V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=i)/231. for i in [36, 84, 54, 108, 0, 1]]
    Vs = [2028.626670842139, 5939.897910157917, 3534.9973314622794, 7069.994662924554, 0.0, 9.580013820942611]
    assert_close1d(V_calc, Vs)

    # Head only custom example:
    V_head1 = V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=36, headonly=True)/231.
    V_head2 = V_horiz_torispherical(108., 156., 1., 0.06, 36, headonly=True)/231.
    assert_close1d([V_head1, V_head2], [111.71919144384525]*2)
    assert 0.0 == V_horiz_torispherical(108., 156., 1., 0.06, 0.0)

def test_V_vertical_conical():
    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_conical(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [250.67461381371024, 2251.175535772343, 0.0, 6516.560761446257]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_conical(132., 33., 0.0)

def test_V_vertical_ellipsoidal():

    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_ellipsoidal(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [783.3581681678445, 2902.831611916969, 0.0, 7168.216837590883]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_ellipsoidal(132., 33., 0.0)

def test_V_vertical_spherical():
    # Two examples from [1]_, and at empty and h=D.
    Vs_calc = [V_vertical_spherical(132., 33., i)/231. for i in [24, 60, 0, 132]]
    Vs = [583.6018352850442, 2658.4605833627343, 0.0, 6923.845809036648]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_spherical(132., 33., 0.0)

def test_V_vertical_torispherical():
    # Two examples from [1]_, and at empty, 1, 22, and h=D.
    Vs_calc = [V_vertical_torispherical(132., 1.0, 0.06, i)/231. for i in [24, 60, 0, 1, 22, 132]]
    Vs = [904.0688283793511, 3036.7614412163075, 0.0, 1.7906624793188568, 785.587561468186, 7302.146666890221]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_torispherical(132., 1.0, 0.06, 0.0)

def test_V_vertical_conical_concave():
    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_conical_concave(113., -33.0, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [251.15825565795188, 614.6068425492208, 1693.1654406426783, 0.0, 4428.278844757774]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_conical_concave(113., -33.0, 0.0)

def test_V_vertical_ellipsoidal_concave():
    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_ellipsoidal_concave(113., -33.0, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [44.84968851034856, 207.6374468071692, 1215.605957384487, 0.0, 3950.7193614995826]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_ellipsoidal_concave(113., -33, 0.0)

def test_V_vertical_spherical_concave():
    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_spherical_concave(113., -33.0, i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [112.81405437348528, 341.7056403375114, 1372.9286894955042, 0.0, 4108.042093610599]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_spherical_concave(113., -33, 0.0)


def test_V_vertical_torispherical_concave():
    # Three examples from [1]_, and at empty and with h=D.
    Vs_calc = [V_vertical_torispherical_concave(D=113., f=0.71, k=0.081, h=i)/231 for i in [15., 25., 50., 0, 113]]
    Vs = [103.88569287163769, 388.72142877582087, 1468.762358198084, 0.0, 4203.87576231318]
    assert_close1d(Vs_calc, Vs)
    assert 0.0 == V_vertical_torispherical_concave(D=113., f=0.71, k=0.081, h=0.0)

    # Does not use 0 <= h < a2; and then does use it; should be the same
    base = V_vertical_torispherical_concave(D=113., f=0.71, k=0.16794375443150927, h=15)
    perturbed = V_vertical_torispherical_concave(D=113., f=0.71, k=0.16794375443151, h=15)
    assert_close(base, perturbed, rtol=1e-14)


def test_geometry():

    SA1 = SA_ellipsoidal_head(2, 1)
    SA2 = SA_ellipsoidal_head(2, 0.999)
    SAs = [6.283185307179586, 6.278996936093318]
    assert_close1d([SA1, SA2], SAs)
    SA = SA_ellipsoidal_head(2, 1.5)
    assert_close(SA, 8.459109081729984, rtol=1e-12)

    # Check code avoids zero division error
    assert_close(SA_ellipsoidal_head(2, 1e-8), pi)

    SA1 = SA_conical_head(2, 1)
    SAs = 4.442882938158366
    assert_close(SA1, SAs)

    SA1 = SA_guppy_head(2, 1)
    assert_close(SA1, 6.654000019110157)

    SA1 = SA_torispheroidal(D=2.54, f=1.039370079, k=0.062362205)
    assert_close(SA1, 6.00394283477063, rtol=1e-12)

    SA1 = SA_tank(D=2, L=2)[0]
    SA2 = SA_tank(D=1., L=0, sideA='ellipsoidal', sideA_a=2, sideB='ellipsoidal', sideB_a=2)[0]
    SA3 = SA_tank(D=1., L=5, sideA='conical', sideA_a=2, sideB='conical', sideB_a=2)[0]
    SA4 = SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical', sideB_a=0.5)[0]
    SAs = [18.84955592153876, 10.124375616183064, 22.18452243965656, 18.84955592153876]
    assert_close1d([SA1, SA2, SA3, SA4], SAs)

    SA1, SA2, SA3, SA4 = SA_tank(D=2.54, L=5, sideA='torispherical', sideB='torispherical', sideA_f=1.039370079, sideA_k=0.062362205, sideB_f=1.039370079, sideB_k=0.062362205)
    SAs = [51.90611237013163, 6.00394283477063, 6.00394283477063, 39.89822670059037]
    assert_close1d([SA1, SA2, SA3, SA4], SAs)

    SA1 = SA_tank(D=1., L=5, sideA='guppy', sideA_a=0.5, sideB='guppy', sideB_a=0.5)[0]
    assert_close(SA1, 19.034963277504044)

    a1 = a_torispherical(D=96., f=0.9, k=0.2)
    a2 = a_torispherical(D=108., f=1., k=0.06)
    ais = [25.684268924767125, 18.288462280484797]
    assert_close1d([a1, a2], ais)

    # Horizontal configurations, compared with TankCalc - Ellipsoidal*2,
    # Ellipsoidal/None, spherical/conical, None/None. Final test is guppy/torispherical,
    # no checks available.

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=2, sideB_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [108.05249928250362, 416.5904542901302, 1086.4674593664702, 1756.34446444281, 2172.9349187329403]
    assert_close1d(Vs_calc, Vs)
    Vs_calc =[V_from_h(h=h, D=10., L=25., horizontal=True, sideA='ellipsoidal', sideA_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [105.12034613915314, 400.22799255268336, 1034.1075818066402, 1667.9871710605971, 2068.2151636132803]
    assert_close1d(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='spherical', sideB='conical', sideA_a=2, sideB_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [104.20408244287965, 400.47607362329063, 1049.291946298991, 1698.107818974691, 2098.583892597982]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideB='spherical', sideA='conical', sideB_a=2, sideA_a=2) for h in [1, 2.5, 5, 7.5, 10]]
    assert_close1d(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=True) for h in [0, 0.75, 1.5]]
    Vs = [0.0, 4.417864669110647, 8.835729338221293]
    assert_close1d(Vs_calc, Vs)

    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideA='guppy', sideB='torispherical', sideA_a=2, sideB_f=1., sideB_k=0.06) for h in [1, 2.5, 5, 7.5, 10]]
    Vs = [104.68706323659293, 399.0285611453449, 1037.3160340613756, 1683.391972469731, 2096.854290344973]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=10., L=25., horizontal=True, sideB='guppy', sideA='torispherical', sideB_a=2, sideA_f=1., sideA_k=0.06) for h in [1, 2.5, 5, 7.5, 10]]
    assert_close1d(Vs_calc, Vs)

    with pytest.raises(Exception):
        V_from_h(h=7, D=1.5, L=5)

    # bad head cases
    with pytest.raises(Exception):
        V_from_h(h=2.6, D=10., L=25., horizontal=True, sideA='BADHEAD', sideB='torispherical', sideA_a=2, sideB_f=1., sideB_k=0.06)
    with pytest.raises(Exception):
        V_from_h(h=2.6, D=10., L=25., horizontal=True, sideA='torispherical', sideB='BADHEAD', sideA_a=2, sideB_f=1., sideB_k=0.06)

    # Vertical configurations, compared with TankCalc - conical*2, spherical*2,
    # ellipsoidal*2. Torispherical*2 has no check. None*2 checks.

    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.) for h in [0, 1, 2, 5., 7, 7.2, 8]]
    Vs = [0.0, 0.14726215563702155, 1.1780972450961726, 6.4795348480289485, 10.013826583317465, 10.301282311120932, 10.602875205865551]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='spherical', sideB='spherical', sideA_a=3., sideB_a=4.) for h in [0, 1.5, 3, 8.5, 13., 15., 16.2, 17]]
    Vs = [0.0, 25.91813939211579, 89.5353906273091, 365.99554414321085, 592.190215201676, 684.3435997069765, 718.7251897078633, 726.2315017548405]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=3., sideB_a=4.) for h in [0, 1.5, 3, 8.5, 13., 15., 16.2, 17]]
    Vs = [0.0, 31.41592653589793, 100.53096491487338, 376.99111843077515, 603.1857894892403, 695.3391739945409, 729.7207639954277, 737.2270760424049]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=8., L=10., horizontal=False, sideA='torispherical', sideB='torispherical', sideA_a=1.3547, sideB_a=1.3547, sideA_f=1.,  sideA_k=0.06, sideB_f=1., sideB_k=0.06) for h in [0, 1.3, 9.3, 10.1, 10.7094, 12]]
    Vs = [0.0, 38.723353379954276, 440.84578224136413, 481.0581682073135, 511.68995321687544, 573.323556832692]
    assert_close1d(Vs_calc, Vs)
    Vs_calc = [V_from_h(h=h, D=1.5, L=5., horizontal=False) for h in [0, 2.5, 5]]
    Vs = [0, 4.417864669110647, 8.835729338221293]
    assert_close1d(Vs_calc, Vs)

    with pytest.raises(Exception):
        V_from_h(h=7, D=1.5, L=5., horizontal=False)

def test_TANK_cross_sectional_area():
    T1 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='torispherical' ,sideB='same')

    assert_close(T1.A_cross_sectional(0.5*T1.h_max), 0.25*pi*T1.D**2)

def test_from_two_specs():
    # Takes about 1 ms
    T0 = TANK(horizontal=True, L=15, D=3)
    A_cross = T0.A_cross_sectional(1.5)
    T1 = TANK.from_two_specs(T0.V_total, A_cross, spec0_name='V', spec1_name='A_cross',
                           h=1e-10, horizontal=False)
    assert_close(T1.V_total, T0.V_total)
    assert_close(T0.A_cross_sectional(1.5), T1.A_cross_sectional(1e-10))


def test_SA_partial():
    # Checked with
    # https://www.aqua-calc.com/calculate/volume-in-a-horizontal-cylinder
    SA = SA_partial_cylindrical_body(L=120*inch, D=72*inch, h=24*inch) + 2*A_partial_circle(D=72*inch, h=24*inch)
    assert_close(SA/(foot**2), (8.250207631*2+73.85756504))

    # partial area of one circle
    # Checked at https://www.aqua-calc.com/calculate/volume-in-a-horizontal-cylinder
    assert_close(A_partial_circle(D=72, h=24), 1188.02989891)
    assert_close(A_partial_circle(D=72, h=72), 0.25*pi*72**2)
    assert_close(A_partial_circle(D=72, h=0), 0)

    # hard checks
    assert A_partial_circle(D=72, h=72*(1+1e-15)) == A_partial_circle(D=72, h=72)
    assert 0 == A_partial_circle(D=72, h=1e-9)
    assert 0 == A_partial_circle(D=72, h=-1e-20)

    assert_close(SA_partial_cylindrical_body(L=200.0, D=96., h=22.0), 19168.852890279868, rtol=1e-12)
    assert_close(SA_partial_cylindrical_body(L=200.0, D=96., h=96), pi*96*200.0, rtol=1e-15) # Pi D L check for full

    assert 0 == SA_partial_cylindrical_body(L=200.0, D=96., h=0)
    assert 0 == SA_partial_cylindrical_body(L=200.0, D=96., h=-1e-14)

    SA_higher = SA_partial_cylindrical_body(L=200.0, D=1., h=1+1e-15)
    assert_close(SA_higher, pi*200.0, rtol=1e-15)


def test_SA_partial_horiz_conical_head():
    # Conical heads
    As_expect = [101.35826, 141.37167, 181.38508]
    hs = [24*inch, 36*inch, 48*inch]
    for h, A_expect in zip(hs, As_expect):
        SA = (2*SA_partial_horiz_conical_head(D=72*inch, a=48*inch, h=h)
              + SA_partial_cylindrical_body(D=72*inch, L=120*inch, h=h))
        A_calc = SA/(foot**2)
        assert_close(A_calc, A_expect, rtol=4e-8)

    assert 0 == SA_partial_horiz_conical_head(D=72., a=48.0, h=0)
    assert 0 == SA_partial_horiz_conical_head(D=72., a=48.0, h=-1e-16)
    assert SA_partial_horiz_conical_head(D=72., a=48.0, h=72) == SA_partial_horiz_conical_head(D=72., a=48.0, h=72+1e-5)

    assert_close(SA_partial_horiz_conical_head(D=72., a=0, h=35),
                 A_partial_circle(D=72, h=35), rtol=1e-12)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='conical', sideA_a=48*inch, sideB='same')
    assert_close(T1.SA_from_h(24*inch)/foot**2, 101.35826, rtol=1e-7)
    assert_close(T1.SA_from_h(36*inch)/foot**2, 141.37167, rtol=1e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 181.38508, rtol=1e-7)
    assert T1.SA_from_h(0) == 0.0
    assert_close(T1.SA_from_h(T1.h_max), T1.A, rtol=1e-14)


def test_SA_partial_horiz_spherical_head():

    L = 120*inch
    D = 72*inch

    a_values = [24*inch]*3 + [36*inch]*3
    h_values = [24*inch, 36*inch, 48*inch]*2
    SA_expect = [99.49977, 135.08848, 170.67720, 111.55668, 150.79645, 190.03622]
    SA_expect = [i*foot**2 for i in SA_expect]

    for i in range(6):
        SA = (2*SA_partial_horiz_spherical_head(D=D, a=a_values[i], h=h_values[i])
              + SA_partial_cylindrical_body(D=D, L=L, h=h_values[i]))
        assert_close(SA, SA_expect[i], rtol=4e-8)

    # numerical integral, expect 1e-7 tol from the code
    SA_calc = SA_partial_horiz_spherical_head(D=72., a=48.0, h=24.0)
    assert_close(SA_calc, 2027.2672091672684, rtol=1e-7)

    assert 0 == SA_partial_horiz_spherical_head(D=72., a=48.0, h=1e-20)
    assert 0 == SA_partial_horiz_spherical_head(D=72., a=48.0, h=-1e-12)

    assert SA_partial_horiz_spherical_head(D=72., a=48.0, h=7200) == SA_partial_horiz_spherical_head(D=72., a=48.0, h=72)

    assert_close(SA_partial_horiz_spherical_head(D=72., a=36+1e-11, h=22),
                 SA_partial_horiz_spherical_head(D=72., a=36, h=22), rtol=1e-8)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='spherical', sideA_a=24*inch, sideB='same')
    assert_close(T1.SA_from_h(24*inch)/foot**2, 99.49977, rtol=1e-7)
    assert_close(T1.SA_from_h(36*inch)/foot**2, 135.08848, rtol=1e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 170.67720, rtol=1e-7)
    assert 0.0 == T1.SA_from_h(0)

    # Numerical integral here too
    assert_close(T1.SA_from_h(T1.h_max), T1.A, rtol=1e-7)


    T2 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='spherical', sideA_a=36*inch, sideB='same')
    assert_close(T2.SA_from_h(24*inch)/foot**2, 111.55668, rtol=1e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 150.79645, rtol=1e-7)
    assert_close(T2.SA_from_h(48*inch)/foot**2, 190.03622, rtol=1e-7)
    assert 0.0 == T2.SA_from_h(0)
    assert_close(T2.SA_from_h(T2.h_max), T2.A, rtol=2e-12)



def test_SA_partial_horiz_guppy_head():
    L = 120*inch
    D = 72*inch
    h_values = [24*inch, 36*inch, 48*inch]
    SA_expect = [94.24500, 129.98330, 167.06207]
    SA_expect = [i*foot**2 for i in SA_expect]

    for i in range(3):
        SA = (2*SA_partial_horiz_guppy_head(D=D, a=48*inch, h=h_values[i])
              + SA_partial_cylindrical_body(D=D, L=L, h=h_values[i]))
        assert_close(SA, SA_expect[i], rtol=5e-8)

    assert 0 == SA_partial_horiz_guppy_head(D=72., a=48.0, h=1e-20)
    assert 0 == SA_partial_horiz_guppy_head(D=72., a=48.0, h=-1e-12)

    assert SA_partial_horiz_guppy_head(D=72., a=48.0, h=7200) == SA_partial_horiz_guppy_head(D=72., a=48.0, h=72)

    assert_close(SA_partial_horiz_guppy_head(D=72., a=48.0, h=24.0), 1467.8949780037, rtol=1e-8)
    assert pi*72*inch/2*72*inch == SA_partial_horiz_guppy_head(D=72*inch, a=36*inch, h=72*inch)
    # Area is NOT CONSISTENT!
    T1 = TANK(L=120*inch, D=72*inch, horizontal=True,
          sideA='guppy', sideA_a=48*inch, sideB='same')
    assert_close(T1.SA_from_h(24*inch)/foot**2, 94.24500, rtol=1e-7)
    assert_close(T1.SA_from_h(36*inch)/foot**2, 129.98330, rtol=1e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 167.06207, rtol=1e-7)
    assert 0.0 == T1.SA_from_h(0)
    # assert_close(T1.SA_from_h(T1.h_max), T1.A, rtol=1e-12)


def test_SA_partial_horiz_ellipsoidal_head():
    L = 120*inch
    D = 72*inch
    h_values = [24*inch, 36*inch, 48*inch]*3
    SA_expect = [102.59905, 138.74815, 174.89725,
                111.55668, 150.79645, 190.03622,
                121.09692, 163.71486, 206.33279]
    SA_expect = [i*foot**2 for i in SA_expect]
    a_values = [24*inch]*3 + [36*inch]*3  + [48*inch]*3

    for i in range(9):
        SA = (2*SA_partial_horiz_ellipsoidal_head(D=D, a=a_values[i], h=h_values[i])
              + SA_partial_cylindrical_body(D=D, L=L, h=h_values[i]))
        assert_close(SA, SA_expect[i], rtol=5e-8)

    assert 0 == SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=1e-20)
    assert 0 == SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=-1e-12)

    assert SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=7200) == SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=72)

    assert_close(SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=24.0), 3401.2336225352738, rtol=1e-11)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='ellipsoidal', sideA_a=24*inch, sideB='same')
    assert_close(T1.SA_from_h(24*inch)/foot**2, 102.59905, rtol=1e-7)
    assert_close(T1.SA_from_h(36*inch)/foot**2, 138.74815, rtol=1e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 174.89725, rtol=1e-7)
    assert_close(T1.SA_from_h(T1.h_max), T1.A, rtol=1e-12)
    assert 0.0 == T1.SA_from_h(0)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='ellipsoidal', sideA_a=36*inch, sideB='same')
    assert_close(T2.SA_from_h(24*inch)/foot**2, 111.55668, rtol=1e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 150.79645, rtol=1e-7)
    assert_close(T2.SA_from_h(48*inch)/foot**2, 190.03622, rtol=1e-7)
    assert 0.0 == T2.SA_from_h(0)
    assert_close(T2.SA_from_h(T2.h_max), T2.A, rtol=1e-12)

    T3 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='ellipsoidal', sideA_a=48*inch, sideB='same')
    assert_close(T3.SA_from_h(24*inch)/foot**2, 121.09692, rtol=1e-7)
    assert_close(T3.SA_from_h(36*inch)/foot**2, 163.71486, rtol=1e-7)
    assert_close(T3.SA_from_h(48*inch)/foot**2, 206.33279, rtol=1e-7)
    assert 0.0 == T3.SA_from_h(0)
    assert_close(T3.SA_from_h(T3.h_max), T3.A, rtol=1e-12)



def test_SA_partial_horiz_torispherical_head():

    # Nasty Python-2 only numerical issue in _SA_partial_horiz_torispherical_head_int_1 ; fixed
    # by ensuring numbers were complex
    assert_close(SA_partial_horiz_torispherical_head(D=1.8288, f=1.0, k=0.06, h=0.6095999999999999), 0.9491605631461236)

    # Python 2 issue with trig due to my own mistake
    assert_close(SA_partial_horiz_torispherical_head(D=1.8288, f=0.9, k=0.1, h=0.6095999999999999),
             1.037030313486593, rtol=1e-6)


    L = 120*inch
    D = 72*inch
    h_values = [2.28*inch, 24*inch, 36*inch, 48*inch, 69.72*inch]
    h_values += [3*inch, 24*inch, 36*inch, 48*inch, 69*inch]
    SA_expect = [22.74924, 94.29092, 127.74876,
                 161.20660, 232.74828,
                 26.82339, 96.18257, 130.22802,
                 164.27347, 233.63265]
    SA_expect = [i*foot**2 for i in SA_expect]
    k_values = [.06]*5 + [.1]*5
    f_values = [1.0]*5 + [.9]*5

    for i in range(9):
        SA = (2*SA_partial_horiz_torispherical_head(D=D, f=f_values[i], k=k_values[i], h=h_values[i])
              + SA_partial_cylindrical_body(D=D, L=L, h=h_values[i]))
        assert_close(SA, SA_expect[i], rtol=2e-7)


    # Precision points for the three regimes
    SA = SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=2)
    assert_close(SA, 80.54614956735351, rtol=1e-7)

    # Only have 1e-7 tolerance here due to numerical itnegration
    SA = SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=20)
    assert_close(SA, 1171.9138610357936, rtol=1e-7)

    SA = SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=71)
    assert_close(SA, 4784.441787378645, rtol=1e-7)

    # Error handling
    # Was a bug computing this
    SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=1e-20)

    assert 0 == SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=0)
    assert 0 == SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=-1e-12)

    assert SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=7200) == SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=72)

    # Check G returns a real number
    assert_close(SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=1e-13), 3.859157404406146e-12, rtol=.1)

    # Torispherical tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='torispherical', sideA_f=1, sideA_k=.06, sideB='same')
    assert_close(T1.SA_from_h(2.28*inch)/foot**2, 22.74924, rtol=1e-7)
    assert_close(T1.SA_from_h(24*inch)/foot**2, 94.29092, rtol=1e-7)
    assert_close(T1.SA_from_h(36*inch)/foot**2, 127.74876, rtol=1e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 161.20660, rtol=1e-7)
    assert_close(T1.SA_from_h(69.72*inch)/foot**2, 232.74828, rtol=1e-7)
    assert 0.0 == T1.SA_from_h(0)
    assert_close(T1.SA_from_h(T1.h_max), T1.A, rtol=1e-12)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=True,
              sideA='torispherical', sideA_f=.9, sideA_k=.1, sideB='same')
    assert_close(T2.SA_from_h(3*inch)/foot**2, 26.82339, rtol=2e-7)
    assert_close(T2.SA_from_h(24*inch)/foot**2, 96.18257, rtol=1e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 130.22802, rtol=1e-7)
    assert_close(T2.SA_from_h(48*inch)/foot**2, 164.27347, rtol=1e-7)
    assert_close(T2.SA_from_h(69*inch)/foot**2, 233.63265, rtol=1e-7)
    assert 0.0 == T2.SA_from_h(0)
    assert_close(T2.SA_from_h(T2.h_max), T2.A, rtol=1e-12)




def test_SA_vert_flat_got_area():
    T_actually_flat = TANK(L=120, D=72, horizontal=False)
    assert_close(T_actually_flat.SA_from_h(0), 4071.5040790523717, rtol=1e-15)
    assert_close(T_actually_flat.SA_from_h(T_actually_flat.h_max), 35286.36868512056, rtol=1e-15)

    T_actually_flat2 = TANK(L=1e-100, D=72, horizontal=False)
    assert_close(T_actually_flat2.SA_from_h(0), 4071.5040790523717, rtol=1e-15)
    assert_close(T_actually_flat2.SA_from_h(T_actually_flat2.h_max), 2*4071.5040790523717, rtol=1e-15)


def test_SA_partial_vertical_conical_head():
    SA = SA_partial_vertical_conical_head(D=72., a=48.0, h=24.0)
    assert_close(SA, 1696.4600329384882)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='conical', sideA_a=36*inch, sideB='same')
    assert_close(T1.SA_from_h(36*inch)/foot**2, 39.98595)
    assert_close(T1.SA_from_h(0)/foot**2, 0)
    assert_close(T1.SA_from_h(-1e-14)/foot**2, 0)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='conical', sideA_a=48*inch, sideB='same')
    assert_close(T2.SA_from_h(24*inch)/foot**2, 11.78097, rtol=3e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 26.50719, rtol=1e-7)
    assert_close(T2.SA_from_h(48*inch)/foot**2, 47.12389, rtol=1e-7)
    assert_close(T2.SA_from_h(60*inch)/foot**2, 65.97345, rtol=1e-7)
    assert_close(T2.SA_from_h(72*inch)/foot**2, 84.82300, rtol=1e-7)
    assert_close(T2.SA_from_h(0)/foot**2,0, rtol=1e-7)
    assert_close(T2.SA_from_h(-1e-14)/foot**2,0, rtol=1e-7)
    assert_close(T2.SA_from_h(T2.h_max), 26.26771571641428, rtol=1e-12)
    assert_close(T2.SA_from_h(T2.h_max*.95), 26.046081865057033, rtol=1e-12)

    T_flat = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='conical', sideA_a=0, sideB='same')
    T_actually_flat = TANK(L=120*inch, D=72*inch, horizontal=False)
    for h in (0, T_flat.h_max*.1, T_flat.h_max*.4, T_flat.h_max*.9, T_flat.h_max):
        assert_close(T_flat.SA_from_h(h), T_actually_flat.SA_from_h(h), rtol=1e-11)



def test_SA_partial_vertical_spherical_head():
    SA = SA_partial_vertical_spherical_head(72, a=24, h=12)
    assert_close(SA, 2940.5307237600464)

    # Make sure we cover zeros, avoid the zero division
    assert_close(SA_partial_vertical_spherical_head(D=1, a=0.0, h=1e-100), 0.7853981633974483, rtol=1e-12)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='spherical', sideA_a=24*inch, sideB='same')
    assert_close(T1.SA_from_h(12*inch)/foot**2, 20.42035, rtol=2e-7)
    assert_close(T1.SA_from_h(24*inch)/foot**2, 40.84070, rtol=2e-7)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 78.53982, rtol=2e-7)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='spherical', sideA_a=36*inch, sideB='same')
    assert_close(T2.SA_from_h(18*inch)/foot**2, 28.27433, rtol=2e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 56.54867, rtol=2e-7)
    assert_close(T2.SA_from_h(60*inch)/foot**2, 94.24778, rtol=2e-7)
    assert_close(T2.SA_from_h(T2.h_max), 28.01889676417523, rtol=1e-10)
    assert 0 == T2.SA_from_h(0)
    assert 0 == T2.SA_from_h(-1e-12)
    # T2.SA_from_h(1e-320) # works :)

    T_flat = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='spherical', sideA_a=0, sideB='same')
    T_actually_flat = TANK(L=120*inch, D=72*inch, horizontal=False)
    for h in (0, T_flat.h_max*.1, T_flat.h_max*.4, T_flat.h_max*.9, T_flat.h_max):
        assert_close(T_flat.SA_from_h(h), T_actually_flat.SA_from_h(h), rtol=1e-15)



def test_SA_partial_vertical_torispherical_head():
    assert_close(SA_partial_vertical_torispherical_head(D=1.8288, f=1, k=.06, h=0.2127198169675985*(1-1e-12)),
             SA_partial_vertical_torispherical_head(D=1.8288, f=1, k=.06, h=0.2127198169675985*(1+1e-12)), rtol=1e-9)

    assert_close(SA_partial_vertical_torispherical_head(D=1.8288, f=1, k=.06, h=.2), 2.2981378579540053, rtol=1e-12)
    assert_close(SA_partial_vertical_torispherical_head(D=1.8288, f=1, k=.06, h=.3), 3.056637737809865, rtol=1e-12)

    assert 0 == SA_partial_vertical_torispherical_head(D=72*inch, f=1, k=.06, h=0)
    assert 0 == SA_partial_vertical_torispherical_head(D=72*inch, f=1, k=.06, h=-1e-16)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='torispherical', sideA_f=1, sideA_k=.06, sideB='same')
    assert_close(T1.SA_from_h(2*inch)/foot**2, 6.28319, rtol=1e-6)
    assert_close(T1.SA_from_h(6*inch)/foot**2, 18.84956, rtol=2.5e-7)
    assert_close(T1.SA_from_h(12*inch)/foot**2, 33.19882, rtol=1e-7)
    assert_close(T1.SA_from_h(T1.sideA_a)/foot**2, 33.50098, rtol=1e-7)
    assert_close(T1.SA_from_h(36.19231*inch)/foot**2, 71.20010, rtol=1e-7)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='torispherical', sideA_f=.9, sideA_k=.1, sideB='same')
    assert_close(T2.SA_from_h(6*inch)/foot**2, 16.96460, rtol=2e-7)
    assert_close(T2.SA_from_h(8*inch)/foot**2, 22.61947, rtol=2e-7)
    assert_close(T2.SA_from_h(12*inch)/foot**2, 31.28983, rtol=2e-7)
    assert_close(T2.SA_from_h(14.91694*inch)/foot**2, 35.98024, rtol=2e-7)
    assert_close(T2.SA_from_h(38.91694*inch)/foot**2, 73.67936, rtol=2e-7)
    assert_close(T2.SA_from_h(1), 6.911163458435757, rtol=1e-12)
    assert_close(T2.SA_from_h(T2.h_max), 24.197157590255674, rtol=1e-12)
    assert_close(T2.SA_from_h(T2.h_max*.9642342), 22.789489525238952, rtol=1e-12)
    assert T2.SA_from_h(0) == 0
    assert T2.SA_from_h(-1e-12) == 0

    # Cannot do flat tests with torispherical, it does not support it

def test_SA_partial_vertical_ellipsoidal_head():
    SA = SA_partial_vertical_ellipsoidal_head(D=72., a=48.0, h=24.0)
    assert_close(SA, 4675.237891376319, rtol=1e-12)

    # Integration tests
    T1 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='ellipsoidal', sideA_a=24*inch, sideB='same')
    assert_close(T1.SA_from_h(12*inch)/foot**2, 24.71061)
    assert_close(T1.SA_from_h(24*inch)/foot**2, 44.50037)
    assert_close(T1.SA_from_h(48*inch)/foot**2, 82.19948)

    T2 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='ellipsoidal', sideA_a=36*inch, sideB='same')
    assert_close(T2.SA_from_h(18*inch)/foot**2, 28.27433, rtol=1.5e-7)
    assert_close(T2.SA_from_h(36*inch)/foot**2, 56.54867, rtol=1e-7)
    assert_close(T2.SA_from_h(60*inch)/foot**2, 94.24778, rtol=1e-7)

    T3 = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='ellipsoidal', sideA_a=48*inch, sideB='same')
    assert_close(T3.SA_from_h(24*inch)/foot**2, 32.46693, rtol=1e-7)
    assert_close(T3.SA_from_h(48*inch)/foot**2, 69.46708, rtol=1e-7)
    assert_close(T3.SA_from_h(72*inch)/foot**2, 107.16619, rtol=1e-7)
    assert_close(T3.SA_from_h(T3.h_max), 30.419215944509517, rtol=1e-12)
    assert T3.SA_from_h(0) == 0
    assert T3.SA_from_h(-1e-13) == 0

    # Flat test case
    T_flat = TANK(L=120*inch, D=72*inch, horizontal=False, sideA='ellipsoidal', sideA_a=0, sideB='same')
    T_actually_flat = TANK(L=120*inch, D=72*inch, horizontal=False)
    for h in (0, T_flat.h_max*.1, T_flat.h_max*.4, T_flat.h_max*.9, T_flat.h_max):
        assert_close(T_flat.SA_from_h(h), T_actually_flat.SA_from_h(h), rtol=1e-11)

    # Numerical issue - unsolved, when `a` approaches `R`. The switch is smooth if we zoom out but not if we are close
    low = SA_partial_vertical_ellipsoidal_head(72*inch, a=36.*inch*(1+1e-9), h=18*inch)
    high = SA_partial_vertical_ellipsoidal_head(72*inch, a=36.*inch*(1-1e-9), h=18*inch)
    assert_close(low, high, rtol=1e-8)

    # The issue has been identified to be occurring in the just-above 36 code
    with pytest.raises(Exception):
        low = SA_partial_vertical_ellipsoidal_head(72*inch, a=36.*inch*(1+1e-12), h=18*inch)
        high = SA_partial_vertical_ellipsoidal_head(72*inch, a=36.*inch*(1-1e-12), h=18*inch)
        assert_close(low, high, rtol=1e-8)

    assert_close(0.25*pi*72**2*inch**2, SA_partial_vertical_ellipsoidal_head(72*inch, a=0.0, h=1e-20))

def test_SA_from_h_basics():
    # Bad side names
    with pytest.raises(ValueError):
        SA_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='NOTASIDE', sideA_a=2., sideB_a=1.)
    with pytest.raises(ValueError):
        SA_from_h(h=7, D=1.5, L=5., horizontal=False, sideB='conical', sideA='NOTASIDE', sideA_a=2., sideB_a=1.)

    # height above tank height
    with pytest.raises(ValueError):
        SA_from_h(h=70, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)

    # height above tank height
    with pytest.raises(ValueError):
        SA_from_h(h=15, D=1.5, L=5., horizontal=True, sideA=None, sideB=None)


    assert_close(SA_from_h(h=1.5, D=1.5, L=5., horizontal=True, sideA=None, sideB=None),
             0.25*pi*1.5**2*2 + 1.5*5*pi, rtol=1e-13)


def test_pitch_angle_solver():
    ans = [{'angle': 30.0, 'pitch': 2., 'pitch_parallel': 1.7320508075688774, 'pitch_normal': 1.},
           {'angle': 60.0, 'pitch': 2., 'pitch_parallel': 1., 'pitch_normal': 1.7320508075688774},
           {'angle': 45.0, 'pitch': 2., 'pitch_parallel': 1.414213562373095, 'pitch_normal': 1.414213562373095},
           {'angle': 90.0, 'pitch': 1., 'pitch_parallel': 0., 'pitch_normal': 1.},
           {'angle': 0.0, 'pitch': 1., 'pitch_parallel': 1., 'pitch_normal': 0.},
           ]
    for ans_set in ans:
        for k1, v1 in ans_set.items():
            for k2, v2 in ans_set.items():
                if k1 != k2 and v1 != 0 and v2 != 0:
                    angle, pitch, pitch_parallel, pitch_normal = pitch_angle_solver(**{k1:v1, k2:v2})
                    assert_close(ans_set['angle'], angle, atol=1e-16)
                    assert_close(ans_set['pitch'], pitch, atol=1e-16)
                    assert_close(ans_set['pitch_parallel'], pitch_parallel, atol=1e-16)
                    assert_close(ans_set['pitch_normal'], pitch_normal, atol=1e-16)

    with pytest.raises(Exception):
        pitch_angle_solver(30)


def test_AirCooledExchanger():
    # Full solution, exchanger in Serth
    AC = AirCooledExchanger(tube_rows=1, tube_passes=1, tubes_per_row=56, tube_length=10.9728,
                              tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                              angle=30, pitch=2.5*inch, fin_height=0.625*inch, tube_thickness=0.00338)

    assert_close(AC.A_fin_per_tube, 18.041542744557212)


    # Minimal solution
    AC = AirCooledExchanger(tube_rows=1, tube_passes=1, tubes_per_row=56, tube_length=10.9728,
                              tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                              angle=30, pitch=2.5*inch, fin_height=0.625*inch)

    with pytest.raises(Exception):
        AirCooledExchanger(tube_rows=1, tube_passes=1, tubes_per_row=56, tube_length=10.9728,
                              tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                              angle=30, pitch=2.5*inch)

    # test AC with geometry whose minimum area is lower on the diagonal plane
    AC = AirCooledExchanger(tube_rows=1, tube_passes=1, tubes_per_row=56, tube_length=10.9728,
                              tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                              angle=60, pitch=2.2*inch, fin_height=0.625*inch, tube_thickness=0.00338)

    assert_close(AC.A_diagonal_per_bundle, AC.A_min_per_bundle)


def test_AirCooledExchangerFull():
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=10.9728,
                            tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                            angle=30, pitch=2.5*inch, fin_height=0.625*inch, tube_thickness=0.00338,
                            bundles_per_bay=2, parallel_bays=3)
    assert_close(AC.bare_length, 0.0022097999999999996)
    assert AC.tubes_per_bundle == 224
    assert AC.tubes_per_bay == 224*2
    assert AC.tubes == 224*2*3

    assert_close(AC.pitch_diagonal, 0.057238126497990836)

    assert_close(AC.A_bare_tube_per_tube, 0.875590523880476)
    assert_close(AC.A_bare_tube_per_row, AC.A_bare_tube_per_tube*AC.tubes_per_row)
    assert_close(AC.A_bare_tube_per_bundle, AC.A_bare_tube_per_tube*AC.tubes_per_bundle)
    assert_close(AC.A_bare_tube_per_bay, AC.A_bare_tube_per_tube*AC.tubes_per_bay)
    assert_close(AC.A_bare_tube, AC.A_bare_tube_per_tube*AC.tubes)

    assert_close(AC.A_tube_showing_per_tube, 0.7617637557760141)
    assert_close(AC.A_tube_showing_per_row, AC.A_tube_showing_per_tube*AC.tubes_per_row)
    assert_close(AC.A_tube_showing_per_bundle, AC.A_tube_showing_per_tube*AC.tubes_per_bundle)
    assert_close(AC.A_tube_showing_per_bay, AC.A_tube_showing_per_tube*AC.tubes_per_bay)
    assert_close(AC.A_tube_showing, AC.A_tube_showing_per_tube*AC.tubes)

    assert_close(AC.A_per_fin, 0.0041762830427215765)
    assert_close(AC.A_fin_per_tube, 18.041542744557212)
    assert_close(AC.A_fin_per_row, AC.A_fin_per_tube*AC.tubes_per_row)
    assert_close(AC.A_fin_per_bundle, AC.A_fin_per_tube*AC.tubes_per_bundle)
    assert_close(AC.A_fin_per_bay, AC.A_fin_per_tube*AC.tubes_per_bay)
    assert_close(AC.A_fin, AC.A_fin_per_tube*AC.tubes)

    assert_close(AC.A_per_tube, 18.803306500333225)
    assert_close(AC.A_per_row, AC.A_per_tube*AC.tubes_per_row)
    assert_close(AC.A_per_bundle, AC.A_per_tube*AC.tubes_per_bundle)
    assert_close(AC.A_per_bay, AC.A_per_tube*AC.tubes_per_bay)
    assert_close(AC.A, AC.A_per_tube*AC.tubes)
    assert_close(AC.A_increase, 21.47500000000001)

    assert_close(AC.A_diagonal_per_bundle, 34.05507419296123)
    assert_close(AC.A_normal_per_bundle, 1.365674687999997)
    assert_close(AC.A_normal_per_bundle, AC.A_normal_per_bundle)
    assert_close(AC.A_min_per_bay, AC.A_min_per_bundle*AC.bundles_per_bay)
    assert_close(AC.A_min, AC.A_min_per_bay*AC.parallel_bays)

    assert_close(AC.A_face_per_bundle, 19.858025)
    assert_close(AC.A_face_per_bay, AC.A_face_per_bundle*AC.bundles_per_bay)
    assert_close(AC.A_face, AC.A_face_per_bay*AC.parallel_bays)
    assert_close(AC.flow_area_contraction_ratio, 0.06877192982456128)

    assert_close(AC.Di, 0.018639999999999997)
    assert_close(AC.A_tube_flow, 0.00027288627771317794)
    assert_close(AC.tube_volume_per_tube, 0.0029943265480911587)
    assert_close(AC.tube_volume_per_row, AC.tube_volume_per_tube*AC.tubes_per_row)
    assert_close(AC.tube_volume_per_bundle, AC.tube_volume_per_tube*AC.tubes_per_bundle)
    assert_close(AC.tube_volume, AC.tube_volume_per_tube*AC.tubes)

    assert AC.channels == 56
    assert AC.pitch_str == 'triangular'
    assert AC.pitch_class == 'staggered'

    # test with corbels
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=10.9728,
                            tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                            angle=30, pitch=2.5*inch, fin_height=0.625*inch, tube_thickness=0.00338,
                            bundles_per_bay=2, parallel_bays=3, corbels=True)
    assert_close(AC.A_face_per_bundle, 19.683831599999998)


def test_geometry_tank():
    V1 = TANK(D=1.2, L=4, horizontal=False).V_total
    assert_close(V1, 4.523893421169302)

    V2 = TANK(D=1.2, L=4, horizontal=False).V_from_h(.5)
    assert_close(V2, 0.5654866776461628)

    V3 = TANK(D=1.2, L=4, horizontal=False).h_from_V(.5)
    assert_close(V3, 0.44209706414415373)

    T1 = TANK(V=10, L_over_D=0.7, sideB='conical', sideB_a=0.5)
#    T1.set_table(dx=0.001)
    things_calc = T1.A, T1.A_sideA, T1.A_sideB, T1.A_lateral
    things = (24.94775907657148, 5.118555935958284, 5.497246519930003, 14.331956620683194)
    assert_close1d(things_calc, things)

    L1 = TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L
    D1 = TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D
    L2 = TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L
    assert_close1d([L1, D1, L2], [4.699531057009146, 9.999999999999407, 4.69953105700979])

    L1 = TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L
    D1 = TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D
    L2 = TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L
    assert_close1d([L1, D1, L2], [4.699531057009146, 9.999999999999407, 4.69953105700979])

    # Test L_over_D setting simple cases
    L1 = TANK(D=1.2, L_over_D=3.5, horizontal=False).L
    D1 = TANK(L=1.2, L_over_D=3.5, horizontal=False).D
    assert_close1d([L1, D1], [4.2, 0.342857142857])
    # Test toripsherical a calculation
    V = TANK(L=1.2, L_over_D=3.5, sideA='torispherical', sideB='torispherical', sideA_f=1.,  sideA_k=0.06, sideB_f=1., sideB_k=0.06).V_total
    assert_close(V, 0.117318265914)

    # Test default a_ratio
    assert_close(0.25, TANK(V=10, L=10, sideA='conical', sideA_a_ratio=None).sideA_a_ratio)

    with pytest.raises(Exception):
        # Test overdefinition case
        TANK(V=10, L=10, D=10)
    with pytest.raises(Exception):
        # Test sides specified with V solving
        TANK(V=10, L=10, sideA='conical', sideB_a=0.5)

    # Couple points that needed some polishing
    base = TANK(D=10., horizontal=True, sideA_a_ratio=.25, sideB_f=1., sideB_k=0.06,
         sideA='conical', sideB='torispherical', V=500)

    forward = TANK(D=10.0, horizontal=True, sideA_a_ratio=.25, sideB_f=1., sideB_k=0.06,
         sideA='conical', sideB='torispherical', L=base.L)
    assert_close(base.V, forward.V_total, rtol=1e-11)

    base = TANK(D=10., horizontal=True, sideB_a_ratio=.25, sideA_f=1., sideA_k=0.06,
         sideB='conical', sideA='torispherical', V=500)

    forward = TANK(D=10.0, horizontal=True, sideB_a_ratio=.25, sideA_f=1., sideA_k=0.06,
         sideB='conical', sideA='torispherical', L=base.L)
    assert_close(base.V, forward.V_total, rtol=1e-11)

    # Same tank keyword
    T1 = TANK(V=10, L_over_D=0.7, sideB='conical', sideB_a=0.5, sideA='same')
    assert T1.sideB == T1.sideA
    assert T1.sideB_a == T1.sideA_a
    assert T1.sideB_f == T1.sideA_f
    assert T1.sideB_k == T1.sideB_k
    assert T1.sideB_a_ratio == T1.sideA_a_ratio

    T1 = TANK(D=10.0, horizontal=True, sideA_f=1., sideA_k=0.06, sideA='torispherical', L=3, sideB='same')
    assert T1.sideB == T1.sideA
    assert T1.sideB_a == T1.sideA_a
    assert T1.sideB_f == T1.sideA_f
    assert T1.sideB_k == T1.sideB_k
    assert T1.sideB_a_ratio == T1.sideA_a_ratio

    T1 = TANK(D=10.0, horizontal=True, sideB_f=1., sideB_k=0.06, sideB='torispherical', L=3, sideA='same')
    assert T1.sideB == T1.sideA
    assert T1.sideB_a == T1.sideA_a
    assert T1.sideB_f == T1.sideA_f
    assert T1.sideB_k == T1.sideB_k
    assert T1.sideB_a_ratio == T1.sideA_a_ratio

    # No spec at all
    T1 = TANK(D=10.0, horizontal=True, L=3, sideA='same')
    assert T1.sideB == T1.sideA
    assert T1.sideB_a == T1.sideA_a
    assert T1.sideB_f == T1.sideA_f
    assert T1.sideB_k == T1.sideB_k
    assert T1.sideB_a_ratio == T1.sideA_a_ratio
    assert T1.sideB_a == 0

    with pytest.raises(Exception):
        T1 = TANK(D=10.0, horizontal=True, L=3, sideA='same', sideB='same')

    # Default k, f
    T1 = TANK(D=10.0, horizontal=True, L=3, sideA='torispherical', sideB='torispherical')
    assert T1.sideB == T1.sideA
    assert T1.sideB_a == T1.sideA_a
    assert T1.sideB_f == T1.sideA_f
    assert T1.sideB_k == T1.sideB_k
    assert T1.sideB_a_ratio == T1.sideA_a_ratio
    assert T1.sideB_k == 0.06
    assert T1.sideB_f == 1.0




def test_TANK_issues():
    # GH issue 31
    Tk = TANK(L=3, D=5, horizontal=False,  sideA='torispherical', sideA_f=1, sideA_k=0.1, sideB='torispherical', sideB_f=1, sideB_k=0.1) #DIN28011
    assert_close(Tk.V_total, Tk.V_from_h(Tk.h_max*.9999999999), rtol=1e-12)

    # Issue where checking sideA_a was for truthiness and not  not None
    kwargs = {'L': 2.0, 'horizontal': False, 'L_over_D': None,
              'V': None, 'sideA': 'ellipsoidal', 'sideB': 'ellipsoidal',
              'sideA_a': 0.0, 'sideB_a': 1e-06, 'sideA_a_ratio': None,
              'sideB_a_ratio': None, 'sideA_f': None, 'sideA_k': None,
              'sideB_f': None, 'sideB_k': None}
    assert_close(TANK(D=.5, **kwargs).V_total, 0.39269921259841806, rtol=1e-11)


    # case that failed once
    kwargs = {'D': 0.5, 'L': 2.0, 'horizontal': False, 'L_over_D': None,
                     'V': None, 'sideA': 'ellipsoidal', 'sideB': 'ellipsoidal',
                     'sideA_a': 0.0, 'sideB_a': 0.0, 'sideA_a_ratio': None,
                     'sideB_a_ratio': None, 'sideA_f': None, 'sideA_k': None,
                     'sideB_f': None, 'sideB_k': None}
    TANK(**kwargs)


def assert_TANKs_equal(T1, T2):
    for k, v in T1.__dict__.items():
        if isinstance(v, (float, int)):
            assert_close(v, T2.__dict__[k])
        else:
            assert v == T2.__dict__[k]

def test_add_thickness():
    t = 1e-4
    T1 = TANK(L=3, D=.6, sideA='ellipsoidal', sideA_a = .2, sideB='conical', sideB_a=0.5)
    T1 = T1.add_thickness(t)
    T2 = TANK(L=3+2*t, D=.6+2*t, sideA='ellipsoidal', sideA_a = .2+t, sideB='conical', sideB_a=0.5+t)
    assert_TANKs_equal(T1, T2)

    # Also add a test that there are no default values for `k` and `f` when the tank is not torispherical
    # and the `a` ratios are correctly calculated not default values
    for T in (T1, T2):
        assert T.sideA_f is None
        assert T.sideA_k is None
        assert T.sideB_f is None
        assert T.sideB_k is None
        assert_close(T.sideA_a_ratio, 0.3333888703765412)
        assert_close(T.sideB_a_ratio, 0.8332222592469177)

    t = .1
    T1 = TANK(L=3, D=.6, sideA='spherical', sideA_a = .2, sideB='guppy', sideB_a=0.5)
    T1 = T1.add_thickness(t)
    T2 = TANK(L=3+2*t, D=.6+2*t, sideA='spherical', sideA_a = .2+t, sideB='guppy', sideB_a=0.5+t)
    assert_TANKs_equal(T1, T2)
    for T in (T1, T2):
        assert T.sideA_f is None
        assert T.sideA_k is None
        assert T.sideB_f is None
        assert T.sideB_k is None
        assert_close(T.sideA_a_ratio, 0.375)
        assert_close(T.sideB_a_ratio, .75)

    # Torispherical as well
    t = .15311351231
    T1 = TANK(L=3, D=.6, sideA='torispherical', sideB='torispherical',
             sideA_f=0.9, sideA_k=0.17)
    T1 = T1.add_thickness(t)
    T2 = TANK(L=3+2*t, D=.6+2*t, sideA='torispherical', sideA_f=0.9, sideA_k=0.17, sideB='torispherical')
    assert_TANKs_equal(T1, T2)


@pytest.mark.slow
def test_geometry_tank_chebyshev():
    # Test auto set Chebyshev table
    T = TANK(L=1.2, L_over_D=3.5)
    assert_close(T.h_from_V(T.V_total, 'chebyshev'), T.h_max)

    assert_close(T.h_from_V(.1, 'chebyshev'), 0.2901805880470152, rtol=1e-4)
    assert_close(T.h_from_V(.05, 'chebyshev'), 0.15830377515496144, rtol=1e-4)
    assert_close(T.h_from_V(.02, 'chebyshev'), 0.08101343184833742, rtol=1e-4)

    T = TANK(L=1.2, L_over_D=3.5)
    assert_close(T.V_from_h(T.h_max, 'chebyshev'), T.V_total)


@pytest.mark.slow
def test_geometry_tank_fuzz_h_from_V():
    T = TANK(L=1.2, L_over_D=3.5, sideA='torispherical', sideB='torispherical', sideA_f=1., horizontal=True, sideA_k=0.06, sideB_f=1., sideB_k=0.06)
    T.set_chebyshev_approximators(deg_forward=100, deg_backwards=600)

    # test V_from_h - pretty easy to get right
    for h in linspace(0, T.h_max, 30):
        # It's the top and the bottom of the tank that works poorly
        V1 = T.V_from_h(h, 'full')
        V2 = T.V_from_h(h, 'chebyshev')
        assert_close(V1, V2, rtol=1E-7, atol=1E-7)

    with pytest.raises(Exception):
        T.V_from_h(1E-5, 'NOTAMETHOD')

    # reverse - the spline is also pretty easy, with a limited number of points
    # when the required precision is low
    T.set_table(n=150)
    for V in linspace(0, T.V_total, 30):
        h1 = T.h_from_V(V, 'brenth')
        h2 = T.h_from_V(V, 'spline')
        assert_close(h1, h2, rtol=1E-5, atol=1E-6)

        h3 = T.h_from_V(V, 'chebyshev')
        # Even with a 600-degree polynomial, there will be failures if N
        # is high enough, but the tolerance should just be lowered
        assert_close(h1, h3, rtol=1E-7, atol=1E-7)

    with pytest.raises(Exception):
        T.h_from_V(1E-5, 'NOTAMETHOD')


def test_basic():
    psi = sphericity(10., 2.)
    assert_close(psi, 0.767663317071005)

    a_r = aspect_ratio(.2, 2.)
    assert_close(a_r, 0.1)

    f_circ = circularity(1.5, .1)
    assert_close(f_circ, 1884.9555921538756)

    A = A_cylinder(0.01, .1)
    assert_close(A, 0.0032986722862692833)

    V = V_cylinder(0.01, .1)
    assert_close(V, 7.853981633974484e-06)

    A = A_hollow_cylinder(0.005, 0.01, 0.1)
    assert_close(A, 0.004830198704894308)

    V = V_hollow_cylinder(0.005, 0.01, 0.1)
    assert_close(V, 5.890486225480862e-06)

    A =  A_multiple_hole_cylinder(0.01, 0.1, [(0.005, 1)])
    assert_close(A, 0.004830198704894308)

    V = V_multiple_hole_cylinder(0.01, 0.1, [(0.005, 1)])
    assert_close(V, 5.890486225480862e-06)


def test_HelicalCoil():
    for kwargs in [{'Do': 30, 'H': 20, 'pitch': 5, 'Dt':2},
                   {'Do': 30, 'N': 4, 'pitch': 5, 'Dt':2},
                   {'Do': 30, 'N': 4, 'H': 20, 'Dt':2},
                   {'Do_total': 32, 'N': 4, 'H': 20, 'Dt':2},
                   {'Do_total': 32, 'N': 4, 'H_total': 22, 'Dt':2}]:

        a = HelicalCoil(Di=1.8, **kwargs)
        assert_close(a.N, 4)
        assert_close(a.H, 20)
        assert_close(a.H_total, 22)
        assert_close(a.Do_total, 32)
        assert_close(a.pitch, 5)
        assert_close(a.tube_length, 377.5212621504738)
        assert_close(a.surface_area, 2372.0360474917497)
        # Other parameters
        assert_close(a.curvature, 0.06)
        assert_close(a.helix_angle, 0.053001960689651316)
        assert_close(a.tube_circumference, 94.24777960769379)
        assert_close(a.total_inlet_area, 3.141592653589793)
        assert_close(a.total_volume, 1186.0180237458749)
        # with Di specified
        assert_close(a.Di, 1.8)
        assert_close(a.inner_surface_area,  2134.832442742575)
        assert_close(a.inlet_area, 2.5446900494077327)
        assert_close(a.inner_volume, 960.6745992341587)
        assert_close(a.annulus_area, 0.5969026041820604)
        assert_close(a.annulus_volume, 225.3434245117162)

    # Fusion 360 agrees with the tube length.
    # It says the SA should be 2370.3726964956063057
    # Hopefully its own calculation is flawed

    # Test successfully creating a helix with
    HelicalCoil(Di=1.8, Do=30, H=20, pitch=2, Dt=2)
    with pytest.raises(Exception):
        HelicalCoil(Di=1.8, Do=30, H=20, pitch=1.999, Dt=2)
    with pytest.raises(Exception):
        HelicalCoil(Di=1.8, Do=30, H=20, N=10.0001,  Dt=2)

    # Test Dt < Do
    HelicalCoil(Do=10, H=30, N=2, Dt=10)
    with pytest.raises(Exception):
        HelicalCoil(Do=10, H=30, N=2, Dt=10.00000001)
    with pytest.raises(Exception):
        HelicalCoil(Do_total=20-1E-9, H=30, N=3., Dt=10.000)



def test_PlateExchanger():
    ex = PlateExchanger(amplitude=5E-4, wavelength=3.7E-3, length=1.2, width=.3, d_port=.05, plates=51)

    assert ex.plate_exchanger_identifier == 'L3.7A0.5B45-45'
    assert_close(ex.amplitude, 0.0005)
    assert_close(ex.a, 0.0005)
    assert_close(ex.b, 0.001)
    assert_close(ex.wavelength, 3.7E-3)
    assert_close(ex.pitch, 3.7E-3)

    assert ex.chevron_angle == 45
    assert ex.chevron_angles == (45, 45)

    assert ex.inclination_angle == 45

    assert_close(ex.plate_corrugation_aspect_ratio, 0.5405405405405406)
    assert_close(ex.gamma, 0.5405405405405406)
    assert_close(ex.plate_enlargement_factor, 1.1611862034509677)

    assert_close(ex.D_eq, 0.002)
    assert_close(ex.D_hydraulic, 0.0017223766473078426)

    assert_close(ex.length_port, 1.25)

    assert_close(ex.A_plate_surface, 0.41802703324234836)
    assert_close(ex.A_heat_transfer, 20.483324628875071)
    assert_close(ex.A_channel_flow, 0.0003)
    assert ex.channels == 50
    assert ex.channels_per_fluid == 25

    ex = PlateExchanger(amplitude=5E-4, wavelength=3.7E-3, length=1.2, width=.3, d_port=.05, plates=51, chevron_angles=(30, 60))
    assert ex.chevron_angle == 45
    assert ex.chevron_angles == (30, 60)

    ex = PlateExchanger(amplitude=5E-4, wavelength=3.7E-3)


def plate_enlargement_factor_numerical(amplitude, wavelength):
    from scipy.integrate import quad
    lambda1 = wavelength
    b = amplitude
    gamma = 4*b/lambda1

    def to_int(s):
        return (1 + (gamma*pi/2)**2*cos(2*pi/lambda1*s)**2)**0.5
    main = quad(to_int, 0, lambda1)[0]

    return main/lambda1

def test_plate_enhancement_factor():
    def plate_enlargement_factor_approx(amplitude, wavelength):
        # Approximate formula
        lambda1 = wavelength
        b = amplitude
        A = 2*pi*b/lambda1
        return 1/6.*(1 + (1 + A**2)**0.5 + 4*(1 + 0.5*A**2)**0.5)

    # 1.218 in VDI example
    phi = plate_enlargement_factor_approx(amplitude=0.002, wavelength=0.0126)
    assert_close(phi, 1.217825410973735)
    assert_close(phi, 1.218, rtol=1E-3)

    phi = plate_enlargement_factor_numerical(amplitude=0.002, wavelength=0.0126)
    assert_close(phi, 1.2149896289702244)

@pytest.mark.fuzz
@pytest.mark.slow
def test_plate_enhancement_factor_fuzz():
    # Confirm it's correct to within 1E-7
    for x in linspace(1E-5, 100, 3):
        for y in linspace(1E-5, 100, 3):
            a = plate_enlargement_factor(x, y)
            b = plate_enlargement_factor_numerical(x, y)
            assert_close(a, b, rtol=1E-7)


def test_RectangularFinExchanger():
    PFE = RectangularFinExchanger(0.03, 0.001, 0.012)
    assert_close(PFE.fin_height, 0.03)
    assert_close(PFE.fin_thickness, 0.001)
    assert_close(PFE.fin_spacing, 0.012)

    # calculated values
    assert_close(PFE.channel_height, 0.029)
    assert_close(PFE.blockage_ratio, 0.8861111111111111)
    assert_close(PFE.fin_count, 83.33333333333333)
    assert_close(PFE.Dh, 0.01595)
    assert_close(PFE.channel_width, 0.011)

    # with layers, plate thickness, width, and length (fully defined)
    PFE = RectangularFinExchanger(0.03, 0.001, 0.012, length=1.2, width=2.401, plate_thickness=.005, layers=40)
    assert_close(PFE.A_HX_layer, 19.2)
    assert_close(PFE.layer_fin_count, 200)
    assert_close(PFE.A_HX, 768.0)
    assert_close(PFE.height, 1.4+.005)
    assert_close(PFE.volume, 4.048085999999999)
    assert_close(PFE.A_specific_HX, 189.71928956054794)



def test_RectangularOffsetStripFinExchanger():
    ROSFE = RectangularOffsetStripFinExchanger(fin_length=.05, fin_height=.01, fin_thickness=.003, fin_spacing=.05)
    assert_close(ROSFE.fin_length, 0.05)
    assert_close(ROSFE.fin_height, 0.01)
    assert_close(ROSFE.fin_thickness, 0.003)
    assert_close(ROSFE.fin_spacing, 0.05)
    assert_close(ROSFE.blockage_ratio, 0.348)
    assert_close(ROSFE.blockage_ratio_Kim, 0.34199999999999997)
    assert_close(ROSFE.alpha, 5)
    assert_close(ROSFE.delta, 0.06)
    assert_close(ROSFE.gamma, 0.06)
    assert_close(ROSFE.A_channel, 0.000329)
#    assert_close(ROSFE.SA_fin, 0.005574)
    assert_close(ROSFE.Dh, 0.011804808037316112)
    assert_close(ROSFE.Dh_Kays_London, 0.012185185185185186)
    assert_close(ROSFE.Dh_Joshi_Webb, 0.011319367879456085)

    # With layers, plate thickness, width (fully defined)
#    ROSFE = RectangularOffsetStripFinExchanger(fin_length=.05, fin_height=.01, fin_thickness=.003, fin_spacing=.05, length=1.2, width=2.401, plate_thickness=.005, layers=40)
#    assert_close(ROSFE.A_HX_layer, 0.267552)


def test_HyperbolicCoolingTower():
    pass