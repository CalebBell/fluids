# -*- coding: utf-8 -*-
# type: ignore
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2019, 2020, 2021, 2022, 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicensse, and/or sell
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
"""
from __future__ import division
from math import (sin, exp, pi, fabs, copysign, log, isinf, isnan, acos, cos, sin,
                  atan2, asinh, sqrt, gamma)
from cmath import sqrt as csqrt, log as clog

__all__ = ['roots_quadratic', 'roots_quartic', 'roots_cubic_a1']
third = 1.0/3.0
sixth = 1.0/6.0
ninth = 1.0/9.0
twelfth = 1.0/12.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0


root_three = 1.7320508075688772 # sqrt(3.0)
one_27 = 1.0/27.0
complex_factor = 0.8660254037844386j # (sqrt(3)*0.5j)

def roots_quadratic(a, b, c):
    if a == 0.0:
        root = -c/b
        return (root, root)
    D = b*b - 4.0*a*c
    a_inv_2 = 0.5/a
    if D < 0.0:
        D = sqrt(-D)
        x1 = (-b + D*1.0j)*a_inv_2
        x2 = (-b - D*1.0j)*a_inv_2
    else:
        D = sqrt(D)
        x1 = (D - b)*a_inv_2
        x2 = -(b + D)*a_inv_2
    return (x1, x2)

def roots_quartic(a, b, c, d, e):
    # There is no divide by zero check. A should be 1 for best numerical results
    # Multiple order of magnitude differences still can cause problems
    # Like  [1, 0.0016525874561771799, 106.8665062954208, 0.0032802613917246727, 0.16036091315844248]
    x0 = 1.0/a
    x1 = b*x0
    x2 = -x1*0.25
    x3 = c*x0
    x4 = b*b*x0*x0
    x5 = -two_thirds*x3 + 0.25*x4
    x6 = x3 - 0.375*x4
    x6_2 = x6*x6
    x7 = x6_2*x6
    x8 = d*x0
    x9 = x1*(-0.5*x3 + 0.125*x4)
    x10 = (x8 + x9)*(x8 + x9)
    x11 = e*x0
    x12 = x1*(x1*(-0.0625*x3 + 0.01171875*x4) + 0.25*x8) # 0.01171875 = 3/256
    x13 = x6*(x11 - x12)
    x14 = -.125*x10 + x13*third - x7/108.0
    x15 = 2.0*(x14 + 0.0j)**(third)
    x16 = csqrt(-x15 + x5)
    x17 = 0.5*x16
    x18 = -x17 + x2
    x19 = -four_thirds*x3
    x20 = 0.5*x4
    x21 = x15 + x19 + x20
    x22 = 2.0*x8 + 2.0*x9
    x23 = x22/x16
    x24 = csqrt(x21 + x23)*0.5
    x25 = -x11 + x12 - twelfth*x6_2
    x27 = (0.0625*x10 - x13*sixth + x7/216.0 + csqrt(0.25*x14*x14 + one_27*x25*x25*x25))**(third)
    x28 = 2.0*x27
    x29 = two_thirds*x25/(x27)
    x30 = csqrt(x28 - x29 + x5)
    x31 = 0.5*x30
    x32 = x2 - x31
    x33 = x19 + x20 - x28 + x29
    x34 = x22/x30
    x35 = csqrt(x33 + x34)*0.5
    x36 = x17 + x2
    x37 = csqrt(x21 - x23)*0.5
    x38 = x2 + x31
    x39 = csqrt(x33 - x34)*0.5
    return ((x32 - x35), (x32 + x35), (x38 - x39), (x38 + x39))

def roots_cubic_a1(b, c, d):
    # Output from mathematica
    t1 = b*b
    t2 = t1*b
    t4 = c*b
    t9 = c*c
    t16 = d*d
    t19 = csqrt(12.0*t9*c + 12.0*t2*d - 54.0*t4*d - 3.0*t1*t9 + 81.0*t16)
    t22 = (-8.0*t2 + 36.0*t4 - 108.0*d + 12.0*t19)**third
    root1 = t22*sixth - 6.0*(c*third - t1*ninth)/t22 - b*third
    t28 = (c*third - t1*ninth)/t22
    t101 = -t22*twelfth + 3.0*t28 - b*third
    t102 =  root_three*(t22*sixth + 6.0*t28)

    root2 = t101 + 0.5j*t102
    root3 = t101 - 0.5j*t102

    return (root1, root2, root3)
