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
from math import sqrt, log
from fluids.numerics.polynomial_evaluation import horner
__all__ = ['polyint', 'polyint_over_x', 'polyder', 'quadratic_from_points',
'deflate_cubic_real_roots',  'exp_poly_ln_tau_coeffs3', 'exp_poly_ln_tau_coeffs2',
'polynomial_offset_scale', 'stable_poly_to_unstable',
]

def stable_poly_to_unstable(coeffs, low, high):
    if high != low:
        from numpy.polynomial import Polynomial
        # Handle the case of no transformation, no limits
        my_poly = Polynomial([-0.5*(high + low)*2.0/(high - low), 2.0/(high - low)])
        coeffs = horner(coeffs, my_poly).coef[::-1].tolist()
    return coeffs





def polynomial_offset_scale(xmin, xmax):
    range_inv = 1.0/(xmax - xmin)
    offset = (-xmax - xmin)*range_inv
    scale = 2.0*range_inv
    return offset, scale


def polyint(coeffs):
    """not quite a copy of numpy's version because this was faster to
    implement.
    Tried out a bunch of optimizations, and this hits a good balance
    between CPython and pypy speed."""
#    return ([0.0] + [c/(i+1) for i, c in enumerate(coeffs[::-1])])[::-1]
    N = len(coeffs)
    out = [0.0]*(N+1)
    for i in range(N):
        out[i] = coeffs[i]/(N-i)
    return out

def polyint_over_x(coeffs):
    N = len(coeffs)
    Nm1 = N - 1
    poly_terms = [0.0]*N
    for i in range(Nm1):
        poly_terms[i] = coeffs[i]/(Nm1-i)
    if N:
        log_coef = coeffs[-1]
        return poly_terms, log_coef
    else:
        return poly_terms, 0.0
#    N = len(coeffs)
#    log_coef = coeffs[-1]
#    Nm1 = N - 1
#    poly_terms = [coeffs[Nm1-i]/i for i in range(N-1, 0, -1)]
#    poly_terms.append(0.0)
#    return poly_terms, log_coef
#    coeffs = coeffs[::-1]
#    log_coef = coeffs[0]
#    poly_terms = [0.0]
#    for i in range(1, len(coeffs)):
#        poly_terms.append(coeffs[i]/i)
#    return list(reversed(poly_terms)), log_coef

def polyder(c, m=1):
    """not quite a copy of numpy's version because this was faster to
    implement."""
    cnt = m

    if cnt == 0:
        return c

    n = len(c)
    if cnt >= n:
        c = c[:1]*0
    else:
        for i in range(cnt): # normally only happens once
            n = n - 1

            der = [0.0]*n
            for j in range(n, 0, -1):
                der[j - 1] = j*c[j]
            c = der
    return c

def quadratic_from_points(x0, x1, x2, f0, f1, f2):
    '''
    from sympy import *
    f, a, b, c, x, x0, x1, x2, f0, f1, f2 = symbols('f, a, b, c, x, x0, x1, x2, f0, f1, f2')

    func = a*x**2 + b*x + c
    Eq0 = Eq(func.subs(x, x0), f0)
    Eq1 = Eq(func.subs(x, x1), f1)
    Eq2 = Eq(func.subs(x, x2), f2)
    sln = solve([Eq0, Eq1, Eq2], [a, b, c])
    cse([sln[a], sln[b], sln[c]], optimizations='basic', symbols=utilities.iterables.numbered_symbols(prefix='v'))
    '''

    v0 = -x2
    v1 = f0*(v0 + x1)
    v2 = f2*(x0 - x1)
    v3 = f1*(v0 + x0)
    v4 = x2*x2
    v5 = x0*x0
    v6 = x1*x1
    v7 = 1.0/(v4*x0 + v5*x1 + v6*x2 - (v4*x1  + v5*x2 + v6*x0))
    v8 = -v4
    a = v7*(v1 + v2 - v3)
    b = -v7*(f0*(v6 + v8) - f1*(v5 + v8) + f2*(v5 - v6))
    c = v7*(v1*x1*x2 + v2*x0*x1 - v3*x0*x2)
    return (a, b, c)

def quadratic_from_f_ders(x, v, d1, d2):
    '''from sympy import *
    f, a, b, c, x, v, d1, d2 = symbols('f, a, b, c, x, v, d1, d2')

    f0 = a*x**2 + b*x + c
    f1 = diff(f0, x)
    f2 = diff(f0, x, 2)

    solve([Eq(f0, v), Eq(f1, d1), Eq(f2, d2)], [a, b, c])
    '''
    a = d2*0.5
    b = d1 - d2*x
    c = -d1*x + d2*x*x*0.5 + v
    return (a, b, c)


def exp_poly_ln_tau_coeffs2(T, Tc, val, der):
    '''
    from sympy import *
    T, Tc, T0, T1, T2, sigma0, sigma1, sigma2 = symbols('T, Tc, T0, T1, T2, sigma0, sigma1, sigma2')
    val, der = symbols('val, der')
    from sympy.abc import a, b, c
    from fluids.numerics import horner
    coeffs = [a, b]
    lntau = log(1 - T/Tc)
    sigma = exp(horner(coeffs, lntau))
    d0 = diff(sigma, T)
    Eq0 = Eq(sigma,val)
    Eq1 = Eq(d0, der)
    s = solve([Eq0, Eq1], [a, b])
    '''
    c0 = der*(T - Tc)/val
    c1 = (-T*der*log((-T + Tc)/Tc) + Tc*der*log((-T + Tc)/Tc) + val*log(val))/val
    return (c0, c1)

def exp_poly_ln_tau_coeffs3(T, Tc, val, der, der2):
    '''
    from sympy import *
    T, Tc, T0, T1, T2, sigma0, sigma1, sigma2 = symbols('T, Tc, T0, T1, T2, sigma0, sigma1, sigma2')
    val, der, der2 = symbols('val, der, der2')
    from sympy.abc import a, b, c
    from fluids.numerics import horner
    coeffs = [a, b, c]
    lntau = log(1 - T/Tc)
    sigma = exp(horner(coeffs, lntau))
    d0 = diff(sigma, T)
    
    Eq0 = Eq(sigma,val)
    Eq1 = Eq(d0, der)
    Eq2 = Eq(diff(d0, T), der2)
    
    # s = solve([Eq0, Eq1], [a, b])
    s = solve([Eq0, Eq1, Eq2], [a, b, c])
    '''
    a = (-T**2*der**2 + T**2*der2*val + 2*T*Tc*der**2 - 2*T*Tc*der2*val + T*der*val - Tc**2*der**2 + Tc**2*der2*val - Tc*der*val)/(2*val**2)
    b = (T**2*der**2*log((-T + Tc)/Tc) - T**2*der2*val*log((-T + Tc)/Tc) - 2*T*Tc*der**2*log((-T + Tc)/Tc) + 2*T*Tc*der2*val*log((-T + Tc)/Tc) - T*der*val*log((-T + Tc)/Tc) + T*der*val + Tc**2*der**2*log((-T + Tc)/Tc) - Tc**2*der2*val*log((-T + Tc)/Tc) + Tc*der*val*log((-T + Tc)/Tc) - Tc*der*val)/val**2
    c = (-T**2*der**2*log((-T + Tc)/Tc)**2 + T**2*der2*val*log((-T + Tc)/Tc)**2 + 2*T*Tc*der**2*log((-T + Tc)/Tc)**2 - 2*T*Tc*der2*val*log((-T + Tc)/Tc)**2 + T*der*val*log((-T + Tc)/Tc)**2 - 2*T*der*val*log((-T + Tc)/Tc) - Tc**2*der**2*log((-T + Tc)/Tc)**2 + Tc**2*der2*val*log((-T + Tc)/Tc)**2 - Tc*der*val*log((-T + Tc)/Tc)**2 + 2*Tc*der*val*log((-T + Tc)/Tc) + 2*val**2*log(val))/(2*val**2)
    return (a, b, c)

def deflate_cubic_real_roots(b, c, d, x0):
    F = b + x0
    G = -d/x0

    D = F*F - 4.0*G
#     if D < 0.0:
#         D = (-D)**0.5
#         x1 = (-F + D*1.0j)*0.5
#         x2 = (-F - D*1.0j)*0.5
#     else:
    if D < 0.0:
        return (0.0, 0.0)
    D = sqrt(D)
    x1 = 0.5*(D - F)#(D - c)*0.5
    x2 = 0.5*(-F - D) #-(c + D)*0.5
    return (x1, x2)


