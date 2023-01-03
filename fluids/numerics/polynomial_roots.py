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
