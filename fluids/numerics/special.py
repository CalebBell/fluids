# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import (sin, exp, pi, fabs, copysign, log, isinf, acos, cos, sin,
                  atan2, asinh, sqrt, gamma)
from cmath import sqrt as csqrt, log as clog

try:
    from math import log1p
except:
    log1p = log


__all__ = ['py_hypot', 'py_cacos', 'py_catan', 'py_catanh', 'trunc_exp',
           'trunc_log']

DBL_MAX = 1.7976931348623157e+308
CM_LARGE_DOUBLE = DBL_MAX/4.
CM_SQRT_LARGE_DOUBLE = sqrt(CM_LARGE_DOUBLE)
DBL_MIN = 2.2250738585072013830902327173324040642192159804623318306e-308
CM_SQRT_DBL_MIN = sqrt(DBL_MIN)

def py_hypot(x, y):
    x = fabs(x)
    y = fabs(y)
    if x < y:
        x, y = y, x
    if x == 0.0:
        return 0.0
    yx = y/x
    return x*sqrt(1.0 + yx*yx)

def py_cacos(z):
    # After CPython https://github.com/python/cpython/blob/e9e7d284c434768333fdfb53a3663eae74cb995a/Modules/cmathmodule.c#L237
    # Without the special cases
    # Implemented only because micropython is missing this function
    s1 = csqrt(1. - z.real - z.imag*1.0j)
    s2 = csqrt(1. + z.real + z.imag*1.0j)
    r =  2.*atan2(s1.real, s2.real) + asinh(s2.real*s1.imag - s2.imag*s1.real)*1.0j
    return r

def py_catan(x):
    # Implemented only because micropython is missing this function
    return 0.5j*(clog(1.0 - 1.0j*x) - clog(1.0 + 1.0j*x))

def py_catanh(z):
    # Does not contain special values
    if z.real < 0.0:
        # works
        res = py_catanh(-z.real + z.imag*1j)
        return -res.real +res.imag*1j
    ay = fabs(z.imag)
    if (z.real > CM_SQRT_LARGE_DOUBLE or ay > CM_SQRT_LARGE_DOUBLE):
        h = py_hypot(z.real/2., z.imag/2.)
        real = z.real/4./h/h
        imag = -copysign(pi/2., -z.imag)
    elif (z.real == 1. and ay < CM_SQRT_DBL_MIN):
        if (ay == 0.):
            real = inf
            imag = z.imag
        else:
            real = -log(sqrt(ay)/sqrt(py_hypot(ay, 2.)))
            imag = copysign(atan2(2., -ay)/2, z.imag)
    else:
        real = log1p(4.*z.real/((1-z.real)*(1-z.real) + ay*ay))/4.
        imag = -atan2(-2.*z.imag, (1-z.real)*(1+z.real) - ay*ay)/2.
    return real + imag*1.0j

def trunc_exp(x, trunc=1.7976931348622732e+308):
    # maximum value occurs at 709.782712893384 exactly
    try:
        return exp(x)
    except:
        # Really exp(709.7) 1.6549840276802644e+308
        return trunc

def trunc_log(x, trunc=-744.4400719213812):
    # 5e-324 is the smallest floating point number above zero and its log is -744.4400719213812
    if x == 0.0:
        return trunc
    return log(x)
#    try:
#        return log(x)
#    except ValueError as e:
#        if x == 0:
#            return trunc
#        else:
#            raise e
