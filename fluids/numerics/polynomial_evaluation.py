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
from math import exp

__all__ = ['horner', 'horner_and_der', 'horner_and_der2', 'horner_and_der3', 'horner_and_der4', 'horner_backwards', 'exp_horner_backwards']


def horner(coeffs, x):
    r'''Evaluates a polynomial defined by coefficienfs `coeffs` at a specified
    scalar `x` value, using the horner method. This is the most efficient
    formula to evaluate a polynomial (assuming non-zero coefficients for all
    terms). This has been added to the `fluids` library because of the need to
    frequently evaluate polynomials; and `NumPy`'s polyval is actually quite
    slow for scalar values.

    Note that the coefficients are reversed compared to the common form; the
    first value is the coefficient of the highest-powered x term, and the last
    value in `coeffs` is the constant offset value.

    Parameters
    ----------
    coeffs : iterable[float]
        Coefficients of polynomial, [-]
    x : float
        Point at which to evaluate the polynomial, [-]

    Returns
    -------
    val : float
        The evaluated value of the polynomial, [-]

    Notes
    -----
    For maximum speed, provide a list of Python floats and `x` should also be
    of type `float` to avoid either `NumPy` types or slow python ints.

    Compare the speed with numpy via:

    >>> coeffs = np.random.uniform(0, 1, size=15)
    >>> coeffs_list = coeffs.tolist()

    %timeit np.polyval(coeffs, 10.0)

    `np.polyval` takes on the order of 15 us; `horner`, 1 us.

    Examples
    --------
    >>> horner([1.0, 3.0], 2.0)
    5.0

    >>> horner([21.24288737657324, -31.326919865992743, 23.490607246508382, -14.318875366457021, 6.993092901276407, -2.6446094897570775, 0.7629439408284319, -0.16825320656035953, 0.02866101768198035, -0.0038190069303978003, 0.0004027586707189051, -3.394447111198843e-05, 2.302586717011523e-06, -1.2627393196517083e-07, 5.607585274731649e-09, -2.013760843818914e-10, 5.819957519561292e-12, -1.3414794055766234e-13, 2.430101267966631e-15, -3.381444175898971e-17, 3.4861255675373234e-19, -2.5070616549039004e-21, 1.122234904781319e-23, -2.3532795334141448e-26], 300.0)
    1.9900667478569642e+58

    References
    ----------
    .. [1] "Horner’s Method." Wikipedia, October 6, 2018.
    https://en.wikipedia.org/w/index.php?title=Horner%27s_method&oldid=862709437.
    '''
    tot = 0.0
    for c in coeffs:
        tot = tot*x + c
    return tot


def horner_and_der(coeffs, x):
    # Coefficients in same order as for horner
    f = 0.0
    der = 0.0
    for a in coeffs:
        der = x*der + f
        f = x*f + a
    return (f, der)

def horner_and_der2(coeffs, x):
    # Coefficients in same order as for horner
    f, der, der2 = 0.0, 0.0, 0.0
    for a in coeffs:
        der2 = x*der2 + der
        der = x*der + f
        f = x*f + a
    return (f, der, der2 + der2)

def horner_and_der3(coeffs, x):
    # Coefficients in same order as for horner
    # Tested
    f, der, der2, der3 = 0.0, 0.0, 0.0, 0.0
    for a in coeffs:
        der3 = x*der3 + der2
        der2 = x*der2 + der
        der = x*der + f
        f = x*f + a
    return (f, der, der2 + der2, der3*6.0)

def horner_and_der4(coeffs, x):
    # Coefficients in same order as for horner
    # Tested
    f, der, der2, der3, der4 = 0.0, 0.0, 0.0, 0.0, 0.0
    for a in coeffs:
        der4 = x*der4 + der3
        der3 = x*der3 + der2
        der2 = x*der2 + der
        der = x*der + f
        f = x*f + a
    return (f, der, der2 + der2, der3*6.0, der4*24.0)

def horner_backwards(x, coeffs):
    return horner(coeffs, x)

def exp_horner_backwards(x, coeffs):
    return exp(horner(coeffs, x))
