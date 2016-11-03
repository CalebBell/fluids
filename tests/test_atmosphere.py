# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from numpy.testing import assert_allclose
from fluids.atmosphere import ATMOSPHERE_1976

# Test values from 'Atmosphere to 86 Km by 2 Km (SI units)', from 
# http://ckw.phys.ncku.edu.tw/public/pub/Notes/Languages/Fortran/FORSYTHE/www.pdas.com/m1.htm
# as provided in atmtabs.html in http://www.pdas.com/atmosdownload.html
H_1 = [-2000, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000, 84000, 86000]
T_1 = [301.15, 288.15, 275.15, 262.17, 249.19, 236.22, 223.25, 216.65, 216.65, 216.65, 216.65, 216.65, 218.57, 220.56, 222.54, 224.53, 226.51, 228.49, 233.74, 239.28, 244.82, 250.35, 255.88, 261.4, 266.92, 270.65, 270.65, 269.03, 263.52, 258.02, 252.52, 247.02, 241.53, 236.04, 230.55, 225.07, 219.58, 214.26, 210.35, 206.45, 202.54, 198.64, 194.74, 190.84, 186.95]
P_1 = [127780, 101320, 79501, 61660, 47218, 35652, 26500, 19399, 14170, 10353, 7565.2, 5529.3, 4047.5, 2971.7, 2188.4, 1616.2, 1197, 889.06, 663.41, 498.52, 377.14, 287.14, 219.97, 169.5, 131.34, 102.3, 79.779, 62.215, 48.338, 37.362, 28.724, 21.959, 16.689, 12.606, 9.4609, 7.0529, 5.2209, 3.8363, 2.8009, 2.0333, 1.4674, 1.0525, 0.75009, 0.53104, 0.37338]
rho_1 = [1.4782, 1.225, 1.0066, 0.81935, 0.66011, 0.52579, 0.41351, 0.31194, 0.22786, 0.16647, 0.12165, 0.08891, 0.06451, 0.046938, 0.034257, 0.025076, 0.01841, 0.013555, 0.0098874, 0.0072579, 0.0053666, 0.0039957, 0.0029948, 0.0022589, 0.0017141, 0.0013167, 0.0010269, 0.00080562, 0.000639, 0.00050445, 0.00039626, 0.00030968, 0.00024071, 0.00018605, 0.00014296, 0.00010917, 0.000082829, 0.000062373, 0.000046385, 0.000034311, 0.000025239, 0.000018458, 0.000013418, 9.6939E-006, 6.9578E-006]
c_1 = [347.89, 340.29, 332.53, 324.59, 316.45, 308.11, 299.53, 295.07, 295.07, 295.07, 295.07, 295.07, 296.38, 297.72, 299.06, 300.39, 301.71, 303.02, 306.49, 310.1, 313.67, 317.19, 320.67, 324.12, 327.52, 329.8, 329.8, 328.81, 325.43, 322.01, 318.56, 315.07, 311.55, 307.99, 304.39, 300.75, 297.06, 293.44, 290.75, 288.04, 285.3, 282.54, 279.75, 276.94, 274.1]
mu_1 = [0.000018515, 0.000017894, 0.00001726, 0.000016612, 0.000015949, 0.000015271, 0.000014577, 0.000014216, 0.000014216, 0.000014216, 0.000014216, 0.000014216, 0.000014322, 0.00001443, 0.000014538, 0.000014646, 0.000014753, 0.000014859, 0.00001514, 0.000015433, 0.000015723, 0.000016009, 0.000016293, 0.000016573, 0.000016851, 0.000017037, 0.000017037, 0.000016956, 0.00001668, 0.000016402, 0.000016121, 0.000015837, 0.000015551, 0.000015262, 0.00001497, 0.000014675, 0.000014377, 0.000014085, 0.000013868, 0.00001365, 0.00001343, 0.000013208, 0.000012985, 0.00001276, 0.000012533]



def test_ATMOSPHERE_1976():
    Ts = [ATMOSPHERE_1976(Z).T for Z in H_1]
    assert_allclose(Ts, T_1, atol=0.005)
    Ps = [ATMOSPHERE_1976(Z).P for Z in H_1]
    assert_allclose(Ps, P_1, rtol=5E-5)
    rhos = [ATMOSPHERE_1976(Z).rho for Z in H_1]
    assert_allclose(rhos, rho_1, rtol=5E-5)
    cs = [ATMOSPHERE_1976(Z).v_sonic for Z in H_1]
    assert_allclose(cs, c_1, rtol=5E-5)
    mus = [ATMOSPHERE_1976(Z).mu for Z in H_1]
    assert_allclose(mus, mu_1, rtol=5E-5)
    
    assert_allclose(ATMOSPHERE_1976(1000, dT=1).T, 282.6510223716947)
    
    # Check thermal conductivity with: http://www.aerospaceweb.org/design/scripts/atmosphere/
    assert_allclose(ATMOSPHERE_1976(1000).k, 0.0248133634493)
    # Other possible additions: 
    # mean air particle speed; mean collision frequency; mean free path; mole volume; total number density
