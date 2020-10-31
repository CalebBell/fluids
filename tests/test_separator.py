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
import pytest
from fluids.numerics import assert_close, assert_close1d, assert_close2d

@pytest.mark.scipy
def test_K_separator_Watkins_fit():
    import numpy as np
    from fluids.separator import tck_Watkins
    from scipy.interpolate import UnivariateSpline, splev, splrep
    v_factors_Watkins = [0.006, 0.00649546, 0.00700535, 0.00755527, 0.00817788,
        0.00881991, 0.00954676, 0.0103522, 0.0112256, 0.0121947, 0.0132476,
        0.0143655, 0.0156059, 0.0169841, 0.018484, 0.0201165, 0.0219329,
        0.0239133, 0.0260727, 0.0284272, 0.0309944, 0.0339158, 0.037046,
        0.0403924, 0.0441998, 0.0483665, 0.0529259, 0.0579153, 0.0632613,
        0.0692251, 0.0756157, 0.0825962, 0.0902214, 0.0983732, 0.107262,
        0.116954, 0.127753, 0.139047, 0.151614, 0.165317, 0.179934, 0.195845,
        0.21278, 0.231181, 0.251627, 0.272897, 0.295966, 0.32041, 0.347497,
        0.3762, 0.407274, 0.440914, 0.476478, 0.513983, 0.554442, 0.59701,
        0.642851, 0.690966, 0.744018, 0.798268, 0.856477, 0.920588, 0.987716,
        1.05784, 1.13294, 1.21337, 1.29718, 1.38678, 1.4799, 1.58212, 1.68837,
        1.79851, 1.91583, 2.04083, 2.16616, 2.29918, 2.44037, 2.59025, 2.73944,
        2.89725, 3.06414, 3.24064, 3.42731, 3.61822, 3.81976, 4.02527, 4.24949,
        4.47008, 4.71907, 4.96404, 5.22172, 5.43354]

    Kv_Watkins = [0.23722, 0.248265, 0.260298, 0.272915, 0.284578, 0.297283,
        0.309422, 0.322054, 0.33398, 0.345714, 0.355905, 0.366397, 0.377197,
        0.386898, 0.396125, 0.404831, 0.411466, 0.41821, 0.424289, 0.428886,
        0.433533, 0.435832, 0.436547, 0.436468, 0.437982, 0.437898, 0.437815,
        0.436932, 0.435258, 0.434381, 0.431138, 0.427919, 0.42395, 0.420018,
        0.415364, 0.410012, 0.403988, 0.39733, 0.390067, 0.38154, 0.373883,
        0.364378, 0.354467, 0.344197, 0.333613, 0.322767, 0.311704, 0.299923,
        0.289114, 0.277173, 0.265725, 0.254749, 0.243338, 0.232438, 0.221621,
        0.211309, 0.200742, 0.190704, 0.181498, 0.172108, 0.162906, 0.154196,
        0.145952, 0.137897, 0.130049, 0.122647, 0.115667, 0.108885, 0.102502,
        0.0963159, 0.0903385, 0.0847323, 0.0796194, 0.0744062, 0.0695348,
        0.0651011, 0.060839, 0.0567521, 0.0529401, 0.0492041, 0.0458154,
        0.0426601, 0.039722, 0.036919, 0.0343137, 0.0318924, 0.0296419,
        0.0275001, 0.0255595, 0.0237127, 0.0219993, 0.0207107]
    Watkins_interp = UnivariateSpline(v_factors_Watkins, Kv_Watkins, s=.00001)

    tck_Watkins_recalc = splrep(np.log(v_factors_Watkins), np.log(Kv_Watkins), s=0.001, k=3)

    [assert_close1d(i, j, rtol=1e-3) for i, j in zip(tck_Watkins[:-1], tck_Watkins_recalc[:-1])]

#    plt.loglog(v_factors_Watkins, Watkins_interp(v_factors_Watkins))
#    my_vs = np.logspace(np.log10(0.006/10), np.log10(5.43354*10), 1000)
#    plt.loglog(my_vs, np.exp(splev(np.log(my_vs), tck_Watkins)), 'x')
#    plt.show()

def test_K_separator_Watkins():
    calc = [[K_separator_Watkins(0.88, 985.4, 1.3, horizontal, method) for
    method in ['spline', 'branan', 'blackwell']] for horizontal in [False, True]]

    expect = [[0.06361290880381038, 0.06108986837654085, 0.06994527471072351],
    [0.07951613600476297, 0.07636233547067607, 0.0874315933884044]]


    assert_close2d(calc, expect, rtol=1e-4)

    with pytest.raises(Exception):
        K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True, method='BADMETHOD')


def test_K_separator_demister_York():
    from fluids.constants import  psi
    Ks_expect = [0.056387999999999994, 0.056387999999999994, 0.09662736507185091,
                 0.10667999999999998, 0.10520347947487964, 0.1036391539227465, 0.07068690636639535]
    Ks = []
    for P in [.1, 1, 10, 20, 40, 50, 5600]:
        Ks.append(K_separator_demister_York(P*psi))

    assert_close1d(Ks, Ks_expect)

    K  = K_separator_demister_York(25*psi, horizontal=True)
    assert_close(K, 0.13334999999999997)


def test_v_Sounders_Brown():
    v = v_Sounders_Brown(K=0.08, rhol=985.4, rhog=1.3)
    assert_close(v, 2.2010906387516167)


def test_K_Sounders_Brown_theoretical():
    K = K_Sounders_Brown_theoretical(D=150E-6, Cd=0.5)
    assert_close(K, 0.06263114241333939)