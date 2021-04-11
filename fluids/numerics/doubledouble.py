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
from math import sqrt as msqrt

__all__ = ['add_dd', 'mul_noerrors_dd', 'mul_dd', 'div_dd', 'sqrt_dd',
           'square_dd', 'mul_imag_dd', 'mul_imag_noerrors_dd', 'sqrt_imag_dd',
           'add_imag_dd', 'imag_inv_dd', 'div_imag_dd', 'cbrt_imag_dd',
           'cbrt_dd', 'cube_dd', 'cbrt_explicit_dd']

third = 1/3.0

def add_dd(x0, y0, x1, y1):
    '''Add two floating point doule doubles.
    args: first number main, first number small...
    '''
    r = x0 + x1
    t = r - x0
    e = (x0 - (r - t)) + (x1 - t)
    e += y0 + y1
    r2 = r + e
    e = e - (r2 - r)
    return r2, e

def mul_noerrors_dd(x0, x1):
    '''Multiply two floating point numbers which were previously only
    doubles, and return ther
    '''
    u = x0*134217729.0
    v = x1*134217729.0
    s = u - (u - x0)
    t = v - (v - x1)
    f = x0 - s
    g = x1 - t
    r = x0*x1
    e = ((s*t - r) + s*g + f*t) + f*g
    return r, e

def mul_dd(x0, y0, x1, y1):
    u = x0*134217729.0
    v = x1*134217729.0
    s = u - (u - x0)
    t = v - (v - x1)
    f = x0 - s
    g = x1 - t
    r = x0*x1
    e = ((s*t - r) + s*g + f*t) + f*g
    e += x0*y1 + y0*x1
    r0 = r + e
    e = e - (r0 - r)
    return r0, e

def div_dd(x0, y0, x1, y1):
    # Creating a 1/x operation would save 1 add, one multiply only!
    r = x0/x1
    u = r*134217729.0
    v = x1*134217729.0
    s2 = u - (u - r)
    t = v - (v - x1)
    f = r - s2
    g = x1 - t
    s = r*x1
    f = ((s2*t - s) + s2*g + f*t) + f*g
    e = (x0 - s - f + y0 - r*y1)/x1
    r0 = r + e
    e = e - (r0 - r)
    return r0, e

def sqrt_dd(x, y):
    if x == 0.0:
        return (0.0, 0.0)
    r = msqrt(x)
    u = r*134217729.0
    s2 = u - (u - r)
    f2 = r - s2
    s = r*r
    f = ((s2*s2 - s) + 2.0*s2*f2) + f2*f2
    e = (x - s - f + y)*0.5/r
    r0 = r + e
    e = e - (r0 - r)
    return r0, e

def square_dd(x0, y0):
    # main part, second part - as fast as possible
    u = x0*134217729.0
    s = u - (u - x0)
    f = x0 - s
    r = x0*x0
    e = ((s*s - r) + 2.0*s*f) + f*f + 2.0*x0*y0
    r0 = r + e
    e = e - (r0 - r)
    return r0, e

def mul_imag_dd(xrr, xre, xcr, xce, yrr, yre, ycr, yce):
    # TODO Make one for one number having zero complex number
    zrr, zre = mul_dd(xrr, xre, yrr, yre)
    wrr, wre = mul_dd(xcr, xce, ycr, yce)
    zrr, zre = add_dd(zrr, zre, -wrr, -wre)

    zcr, zce = mul_dd(xrr, xre, ycr, yce)
    wrr, wre = mul_dd(xcr, xce, yrr, yre)
    zcr, zce = add_dd(zcr, zce, wrr, wre)
    return zrr, zre, zcr, zce

def mul_imag_noerrors_dd(xrr, xcr, yrr, ycr):
    zrr, zre = mul_noerrors_dd(xrr, yrr)
    wrr, wre = mul_noerrors_dd(xcr, -ycr)
    zrr, zre = add_dd(zrr, zre, wrr, wre)

    zcr, zce = mul_noerrors_dd(xrr, ycr)
    wrr, wre = mul_noerrors_dd(xcr, yrr)
    zcr, zce = add_dd(zcr, zce, wrr, wre)
    return zrr, zre, zcr, zce

def sqrt_imag_dd(xrr, xre, xcr, xce):
    if xcr == 0.0:
        if xrr > 0.0:
            zrr, zre = sqrt_dd(xrr, xre)
            return zrr, zre, 0.0, 0.0
        zrr, zre = sqrt_dd(-xrr, -xre)
        return 0.0, 0.0, zrr, zre
    # 2 square, 3 add, 3 sqrt
    xrr2, xre2 = square_dd(xrr, xre)
    xcr2, xce2 = square_dd(xcr, xce)
    wr, we = add_dd(xrr2, xre2, xcr2, xce2)
    wr, we = sqrt_dd(wr, we)

    zrr, zre = add_dd(wr, we, xrr, xre)
    zrr *= 0.5
    zre *= 0.5 # checks out
    zrr, zre = sqrt_dd(zrr, zre) # real part of answer

    zcr, zce = add_dd(wr, we, -xrr, -xre)
    zcr *= 0.5
    zce *= 0.5 # checks out
    zcr, zce = sqrt_dd(zcr, zce) # real part of answer
    return (zrr, zre, zcr, zce)

def add_imag_dd(xrr, xre, xcr, xce, yrr, yre, ycr, yce):
    zrr, zre = add_dd(xrr, xre, yrr, yre)
    zcr, zce = add_dd(xcr, xce, ycr, yce)
    return zrr, zre, zcr, zce

def imag_inv_dd(xrr, xre, xcr, xce):
    cr2, ce2 = square_dd(xrr, xre)
    wr, we = square_dd(xcr, xce)
    wr, we = add_dd(cr2, ce2, wr, we)
    xrr, xre = div_dd(xrr, xre, wr, we)
    xcr, xce = div_dd(xcr, xce, wr, we)
    return xrr, xre, -xcr, -xce

def div_imag_dd(xrr, xre, xcr, xce, yrr, yre, ycr, yce):
    # TODO try to make one for the case the numerator has no complex number
    # as that is used.
    cr2, ce2 = square_dd(yrr, yre)
    wr, we = square_dd(ycr, yce)
    wr, we = add_dd(cr2, ce2, wr, we)

    nlr, nle = mul_dd(xrr, xre, yrr, yre)
    cr2, ce2 = mul_dd(xcr, xce, ycr, yce)
    nlr, nle = add_dd(nlr, nle, cr2, ce2)

    nrr, nre = mul_dd(xcr, xce, yrr, yre)
    cr2, ce2 = mul_dd(xrr, xre, ycr, yce)
    nrr, nre = add_dd(nrr, nre, -cr2, -ce2)

    xrr, xre = div_dd(nlr, nle, wr, we)
    xcr, xce = div_dd(nrr, nre, wr, we)
    return xrr, xre, xcr, xce

def cbrt_imag_dd(xrr, xre, xcr, xce):
    # start off at the double precision solution
    y_guess = (xrr + xcr*1.0j)**(-1.0/3.)
    yr, yc = y_guess.real, y_guess.imag
    # one newton iteration
    t0rr, t0re, t0cr, t0ce = mul_imag_noerrors_dd(yr, yc, yr, yc)  # have y*y
    t0rr, t0re, t0cr, t0ce = mul_imag_dd(t0rr, t0re, t0cr, t0ce, yr, 0.0, yc, 0.0)  # have y*y*y
    t0rr, t0re, t0cr, t0ce = mul_imag_dd(t0rr, t0re, t0cr, t0ce, xrr, xre, xcr, xce)   # have x*y*y*y

    # here, we flip the signs on the complex bits
    # add do an add to the real bits
    t0rr, t0re = add_dd(1.0, 0.0, -t0rr, -t0re)
#     t0rr, t0re, t0cr, t0ce = add_imag_dd(1.0, 0.0, 0.0, 0.0, -t0rr, -t0re, -t0cr, -t0ce) # have 1-x*y*y*y ; should be able to optimize this
    t0rr, t0re, t0cr, t0ce = mul_imag_dd(yr, 0.0, yc, 0.0, t0rr, t0re, -t0cr, -t0ce) # have y*(1-x*y*y*y)

    t0rr, t0re, t0cr, t0ce = mul_imag_dd(t0rr, t0re, t0cr, t0ce, 0.3333333333333333, 1.850371707708594e-17, 0.0, 0.0) # have third_dd*y*(1-x*y*y*y); should be able to optimize this
    t0rr, t0re, t0cr, t0ce = add_imag_dd(yr, 0.0, yc, 0.0, t0rr, t0re, t0cr, t0ce)  # have y
    return imag_inv_dd(t0rr, t0re, t0cr, t0ce)

def cbrt_dd(xr, xe):
    # http://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
    yr = (xr**(-1.0/3.))
    ye = 0.0

    w0r, w0e = cube_dd(yr, ye)
#     w0r, w0e = mul_dd(w0r, w0e, yr, ye)
    w0r, w0e = mul_dd(w0r, w0e, xr, xe)
    w0r, w0e = add_dd(1.0, 0.0, -w0r, -w0e)
    w0r, w0e = mul_dd(w0r, w0e, yr, ye)
    yr, ye = add_dd(yr, ye, w0r, w0e)

    # Do it again, most of the time probably don't need this? Turn it off on EOS?
    w0r, w0e = cube_dd(yr, ye)
    w0r, w0e = mul_dd(w0r, w0e, xr, xe)
    w0r, w0e = add_dd(1.0, 0.0, -w0r, -w0e)
    w0r, w0e = mul_dd(w0r, w0e, yr, ye)
    yr, ye = add_dd(yr, ye, third*w0r, third*w0e)
    return div_dd(1.0, 0.0, yr, ye)

def cube_dd(x0, y0):
    # main part, second part - as fast as possible
    u = x0*134217729.0
    s = u - (u - x0)
    f = x0 - s
    r = x0*x0
    e = ((s*s - r) + 2.0*s*f) + f*f + 2.0*x0*y0
    r0 = r + e
    e0 = e - (r0 - r)

    v = r0*134217729.0
    t = v - (v - r0)
    g = r0 - t
    r = x0*r0
    e = ((s*t - r) + s*g + f*t) + f*g + x0*e0 + y0*r0
    r0 = r + e
    e = e - (r0 - r)
    return r0, e

def cbrt_explicit_dd(xr, xe):
    # http://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
    yr = (xr**(-1.0/3.))
    ye = 0.0

#     w0r, w0e = cube_dd(yr, ye)
    # Couple of things commented out at the start since ye is zero
    # Cannot seem to make ot work good.
    u = yr*134217729.0
    s = u - (u - yr)
    f = yr - s
    r = yr*yr
    w0e = ((s*s - r) + 2.0*s*f) + f*f + 2.0*yr*ye
    w0r = r + w0e
    e0 = w0e - (w0r - r)

    v = w0r*134217729.0
    t = v - (v - w0r)
    g = w0r - t
    r = yr*w0r
    w0e = ((s*t - r) + s*g + f*t) + f*g + yr*e0 + ye*w0r
    w0r = r + w0e
    w0e = w0e - (w0r - r)


    w0r, w0e = mul_dd(w0r, w0e, xr, xe)
    w0r, w0e = add_dd(1.0, 0.0, -w0r, -w0e)
    w0r, w0e = mul_dd(w0r, w0e, yr, ye)
    yr, ye = add_dd(yr, ye, w0r, w0e)

    # Do it again, most of the time probably don't need this? Turn it off on EOS?
    u = yr*134217729.0
    s = u - (u - yr)
    f = yr - s
    r = yr*yr
    w0e = ((s*s - r) + 2.0*s*f) + f*f + 2.0*yr*ye
    w0r = r + w0e
    e0 = w0e - (w0r - r)

    v = w0r*134217729.0
    t = v - (v - w0r)
    g = w0r - t
    r = yr*w0r
    w0e = ((s*t - r) + s*g + f*t) + f*g + yr*e0 + ye*w0r
    w0r = r + w0e
    w0e = w0e - (w0r - r)
    w0r, w0e = mul_dd(w0r, w0e, xr, xe)
    w0r, w0e = add_dd(1.0, 0.0, -w0r, -w0e)
    w0r, w0e = mul_dd(w0r, w0e, yr, ye)
    yr, ye = add_dd(yr, ye, third*w0r, third*w0e)
    return div_dd(1.0, 0.0, yr, ye)
