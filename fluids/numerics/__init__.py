# -*- coding: utf-8 -*-
# type: ignore
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import sys
from fluids.numerics.arrays import (solve as py_solve, inv, dot, norm2, inner_product, eye,
                     array_as_tridiagonals, tridiagonals_as_array,
                     solve_tridiagonal, subset_matrix)

from fluids.numerics.special import (py_hypot, py_cacos, py_catan, py_catanh, 
                                     trunc_exp, trunc_log)


__all__ = ['isclose', 'horner', 'horner_and_der', 'horner_and_der2',
           'horner_and_der3', 'quadratic_from_f_ders', 'chebval', 'interp',
           'linspace', 'logspace', 'cumsum', 'diff', 'basic_damping',
           'is_poly_negative', 'is_poly_positive',
           'implementation_optimize_tck', 'tck_interp2d_linear',
           'bisect', 'ridder', 'brenth', 'newton', 'secant', 'halley',
           'splev', 'bisplev', 'derivative', 'jacobian', 'hessian',
           'normalize', 'oscillation_checker',
           'IS_PYPY', 'roots_cubic', 'roots_quartic', 'newton_system',
           'broyden2', 'basic_damping', 'solve_2_direct', 'solve_3_direct',
           'solve_4_direct', 'sincos', 'horner_and_der4',
           'lambertw', 'ellipe', 'gamma', 'gammaincc', 'erf',
           'i1', 'i0', 'k1', 'k0', 'iv', 'mean', 'polylog2',
           'numpy', 'nquad', 'catanh',
           'polyint_over_x', 'horner_log', 'polyint', 'chebder',
           'polyder', 'make_damp_initial', 'quadratic_from_points',
           'OscillationError', 'UnconvergedError', 'caching_decorator',
           'NoSolutionError', 'SamePointError', 'NotBoundedError',
           'damping_maintain_sign', 'oscillation_checking_wrapper',
           'trunc_exp', 'trunc_log', 'fit_integral_linear_extrapolation',
           'fit_integral_over_T_linear_extrapolation',
           'poly_fit_integral_value', 'poly_fit_integral_over_T_value',
           'evaluate_linear_fits', 'evaluate_linear_fits_d',
           'evaluate_linear_fits_d2',
           'best_bounding_bounds', 'newton_minimize', 'array_as_tridiagonals',
           'tridiagonals_as_array', 'solve_tridiagonal', 'subset_matrix',
           'assert_close', 'assert_close1d', 'assert_close2d', 'assert_close3d',
           'assert_close4d', 'translate_bound_func', 'translate_bound_jac',
           'translate_bound_f_jac',
           'quad', 'quad_adaptive',

           # Complex number math missing in micropython
           'cacos', 'catan',
           'deflate_cubic_real_roots',

           'root', 'minimize', 'fsolve',
           ]

from fluids.numerics import doubledouble
from fluids.numerics.doubledouble import *
__all__.extend(doubledouble.__all__)


__numba_additional_funcs__ = ['py_bisplev', 'py_splev', 'binary_search',
                              'py_lambertw', '_lambertw_err', 'newton_err',
                              'norm2', 'py_solve', 'func_35_splev', 'func_40_splev',
                              'quad_adaptive', 'fixed_quad_Gauss_Kronrod',
                              'halley_compat_numba',
                              ]
nan = float("nan")
inf = float("inf")




SKIP_DEPENDENCIES = False # for testing

class FakePackage(object):
    pkg = None
    def __getattr__(self, name):
        raise ImportError('%s in not installed and required by this feature' %(self.pkg))

    def __init__(self, pkg):
        self.pkg = pkg

version_components = sys.version.split('.')
PY_MAJOR, PY_MINOR = int(version_components[0]), int(version_components[1])
PY37 = (PY_MAJOR, PY_MINOR) >= (3, 7)

try:
    # The right way imports the platform module which costs to ms to load!
    # implementation = platform.python_implementation()
    IS_PYPY = 'PyPy' in sys.version
except AttributeError:
    IS_PYPY = False

# Hacks for numba - allow the module to run in different ways
try:
    IS_PYPY = FORCE_PYPY
except:
    pass
try:
    array_if_needed
except:
    array_if_needed = lambda x: x

try:
    is_micropython = sys.implementation.name == 'micropython'
    if is_micropython:
        IS_PYPY = True
except:
    is_micropython = False

try:
    is_ironpython =  sys.implementation.name == 'ironpython'
    if is_ironpython:
        IS_PYPY = True
except:
    is_ironpython = False

if is_micropython:
    hypot = py_hypot


def sincos(x):
    return sin(x), cos(x)

try:
    if IS_PYPY:

        def sincos(x):
            # fast implementation based of cephes and go
            PI4A = 7.85398125648498535156e-1
            PI4B = 3.77489470793079817668e-8
            PI4C = 2.69515142907905952645e-15
            M4PI = 1.273239544735162542821171882678754627704620361328125 #// 4/pi
            sinSign, cosSign = False, False
            if x < 0:
                x = -x
                sinSign = True

            j = int(x * M4PI)
            y = float(j)

            if j&1 == 1:
                j += 1
                y += 1
            j &= 7
            if j > 3:
                j -= 4
                sinSign, cosSign = not sinSign, not cosSign
            if j > 1:
                cosSign = not cosSign
            z = ((x - y*PI4A) - y*PI4B) - y*PI4C
            zz = z * z
            cos = 1.0 - 0.5*zz + zz*zz*((((((-1.13585365213876817300E-11*zz)+2.08757008419747316778E-9)
                                           *zz+-2.75573141792967388112E-7)*zz+2.48015872888517045348E-5)
                                         *zz+-1.38888888888730564116E-3)*zz+4.16666666666665929218E-2)
            sin = z + z*zz*((((((1.58962301576546568060E-10*zz)+-2.50507477628578072866E-8)*zz+2.75573136213857245213E-6)
                              *zz+-1.98412698295895385996E-4)*zz+8.33333333332211858878E-3)*zz+-1.66666666666666307295E-1)
            if j == 1 or j == 2:
                sin, cos = cos, sin
            if cosSign :
                cos = -cos
            if sinSign:
                sin = -sin
            return sin, cos
except:
    pass

try:
    from cmath import acos as cacos, atan as catan, atanh as catanh, isclose as cisclose
except:
    cacos = py_cacos
    catan = py_catan
    catanh = py_catanh

_wraps = None
def my_wraps():
    global _wraps
    if _wraps is not None:
        return _wraps
    from functools import wraps
    _wraps = wraps
    return _wraps


#IS_PYPY = True # for testing

if not SKIP_DEPENDENCIES:
    try:
        # Regardless of actual interpreter, fall back to pure python implementations
        # if scipy and numpy are not available.
        import numpy
        np = numpy
    except ImportError:
        # Allow a fake numpy to be imported, but will raise an excption on any use
        numpy = FakePackage('numpy')
        IS_PYPY = True
else:
    numpy = FakePackage('numpy')
    IS_PYPY = True

IS_PYPY_OR_SKIP_DEPENDENCIES = IS_PYPY or SKIP_DEPENDENCIES
np = numpy

#IS_PYPY = True

try:
    from sys import float_info
    epsilon = float_info.epsilon
except:
    # Probably micropython
    epsilon = 2.220446049250313e-16

one_epsilon_larger = 1.0 + epsilon
one_epsilon_smaller = 1.0 - epsilon
zero_epsilon_smaller = 1.0 - epsilon

_iter = 100
_xtol = 1e-12
_rtol = epsilon*2.0

third = 1.0/3.0
sixth = 1.0/6.0
ninth = 1.0/9.0
twelfth = 1.0/12.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0

root_three = 1.7320508075688772 # sqrt(3.0)
one_27 = 1.0/27.0
complex_factor = 0.8660254037844386j # (sqrt(3)*0.5j)


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

    return [root1, root2, root3]


def roots_cubic_a2(a, b, c, d):
    # Output from maple
    t2 = a*a
    t3 = d*d
    t10 = c*c
    t14 = b*b
    t15 = t14*b
    t20 = csqrt(-18.0*a*b*c*d + 4.0*a*t10*c + 4.0*t15*d - t14*t10 + 27.0*t2*t3)
    t31 = (36.0*c*b*a + 12.0*root_three*t20*a - 108.0*d*t2 - 8.0*t15)**third
    t32 = 1.0/a
    root1 = t31*t32*sixth - two_thirds*(3.0*a*c - t14)*t32/t31 - b*t32*third
    t33 = t31*t32
    t40 = (3.0*a*c - t14)*t32/t31

    t50 = -t33*twelfth + t40*third - b*t32*third
    t51 = 0.5j*root_three *(t33*sixth + two_thirds*t40)
    root2 = t50 + t51
    root3 = t50 - t51
    return [root1, root2, root3]


def roots_cubic(a, b, c, d):
    r'''Cubic equation solver based on a variety of sources, algorithms, and
    numerical tools. It seems evident after some work that no analytical
    solution using floating points will ever have high-precision results
    for all cases of inputs. Some other solvers, such as NumPy's roots
    which uses eigenvalues derived using some BLAS, seem to provide bang-on
    answers for all values coefficients. However, they can be quite slow - and
    where possible there is still a need for analytical solutions to obtain
    15-35x speed, such as when using PyPy.

    A particular focus of this routine is where a=1, b is a small number in the
    range -10 to 10 - a common occurrence in cubic equations of state.

    Parameters
    ----------
    a : float
        Coefficient of x^3, [-]
    b : float
        Coefficient of x^2, [-]
    c : float
        Coefficient of x, [-]
    d : float
        Added coefficient, [-]

    Returns
    -------
    roots : tuple(float)
        The evaluated roots of the polynomial, 1 value when a and b are zero,
        two values when a is zero, and three otherwise, [-]

    Notes
    -----
    For maximum speed, provide Python floats. Compare the speed with numpy via:

    %timeit roots_cubic(1.0, 100.0, 1000.0, 10.0)
    %timeit np.roots([1.0, 100.0, 1000.0, 10.0])

    %timeit roots_cubic(1.0, 2.0, 3.0, 4.0)
    %timeit np.roots([1.0, 2.0, 3.0, 4.0])

    The speed is ~15-35 times faster; or using PyPy, 240-370 times faster.

    Examples
    --------
    >>> roots_cubic(1.0, 100.0, 1000.0, 10.0)
    (-0.0100100190, -88.731288, -11.25870159)

    References
    ----------
    .. [1] "Solving Cubic Equations." Accessed January 5, 2019.
       http://www.1728.org/cubic2.htm.

    '''
    '''
    Notes
    -----
    Known issue is inputs that look like
    1, -0.999999999978168, 1.698247818501352e-11, -8.47396642608142e-17
    Errors grown unbound, starting when b is -.99999 and close to 1.
    '''
    if a == 0.0:
        if b == 0.0:
            return (-d/c, )
        D = c*c - 4.0*b*d
        b_inv_2 = 0.5/b
        if D < 0.0:
            D = sqrt(-D)
            x1 = (-c + D*1.0j)*b_inv_2
            x2 = (-c - D*1.0j)*b_inv_2
        else:
            D = sqrt(D)
            x1 = (D - c)*b_inv_2
            x2 = -(c + D)*b_inv_2
        return (x1, x2)
    a_inv = 1.0/a
    a_inv2 = a_inv*a_inv
    bb = b*b
    '''Herbie modifications for f:
    c*a_inv - b_a*b_a*third
    '''

    b_a = b*a_inv
    b_a2 = b_a*b_a
    f = c*a_inv - b_a2*third
#    f = (3.0*c*a_inv - bb*a_inv2)*third
    g = ((2.0*(bb*b) * a_inv2*a_inv) - (9.0*b*c)*(a_inv2) + (27.0*d*a_inv))*one_27
#    g = (((2.0/(a/b))/((a/b) * (a/b)) + d*27.0/a) - (9.0/a*b)*c/a)/27.0

    h = (0.25*(g*g) + (f*f*f)*one_27)
#    print(f, g, h)
    '''h has no savings on precision - 0.4 error to 0.2.
    '''
#    print(f, g, h, 'f, g, h')
    if h == 0.0 and g == 0.0 and f == 0.0:
        if d/a >= 0.0:
            x = -((d*a_inv)**(third))
        else:
            x = (-d*a_inv)**(third)
        return (x, x, x)
    elif h > 0.0:
        # Happy with these formulas - double doubles should be fast.
        # No complex numbers are needed here.
#        print('basic')
        # 1 real root, 2 imag
        root_h = sqrt(h)
        R = -0.5*g + root_h

        # It is possible to save one of the power of thirds!
        if R >= 0.0:
            S = R**third
        else:
            S = -((-R)**third)
        T = -(0.5*g) - root_h
        if T >= 0.0:
            U = (T**(third))
        else:
            U = -(((-T)**(third)))

        SU = S + U
        b_3a = b*(third*a_inv)
        t1 = -0.5*SU - b_3a
        t2 = (S - U)*complex_factor
        x1 = SU - b_3a
        # x1 is OK actually in some tests? the issue is x2, x3?
        x2 = t1 + t2
        x3 = t1 - t2

    else:
#    elif h <= 0.0:
        t2 = a*a
        t3 = d*d
        t10 = c*c
        t14 = b*b
        t15 = t14*b

        '''This method is inaccurate when choice_term is too small; but still
        more accurate than the other method.
        '''
        choice_term = -18.0*a*b*c*d + 4.0*a*t10*c + 4.0*t15*d - t14*t10 + 27.0*t2*t3
        if (abs(choice_term) > 1e-12 or abs(b + 1.0) < 1e-7):
#            print('mine')
            t32 = 1.0/a
            t20 = csqrt(choice_term)
            t31 = (36.0*c*b*a + 12.0*root_three*t20*a - 108.0*d*t2 - 8.0*t15)**third
            t33 = t31*t32
            t32_t31 = t32/t31

            x1 = (t33*sixth - two_thirds*(3.0*a*c - t14)*t32_t31 - b*t32*third).real
            t40 = (3.0*a*c - t14)*t32_t31

            t50 = -t33*twelfth + t40*third - b*t32*third
            t51 = 0.5j*root_three*(t33*sixth + two_thirds*t40)
            x2 = (t50 + t51).real
            x3 = (t50 - t51).real
        else:
#            print('other')
            # 3 real roots
            # example is going in here
            i = sqrt(((g*g)*0.25) - h)
            j = i**third # There was a saving for j but it was very weird with if statements!
            '''Clamied nothing saved for k.
            '''
            k = acos(-0.5*g/i)
#            L = -j

#            N, M = sincos(k*third)
#            N *= root_three
            k_third = k*third
            M = cos(k_third)
            N = root_three*sin(k_third)
            P = -b_a*third

            # Direct formula for x1
            x1 = 2.0*j*M + P
            x2 = P - j*(M + N)
            x3 = P - j*(M - N)
    return (x1, x2, x3)


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

def mean(data):
    # Much faster than the statistics.mean module
    return sum(data)/len(data)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    """Port of numpy's linspace to pure python.

    Does not support dtype, and returns lists of floats.
    """
    num = int(num)
    start = start * 1.
    stop = stop * 1.

    if num <= 0:
        return []
    if endpoint:
        if num == 1:
            return [start]
        step = (stop-start)/float((num-1))
        if num == 1:
            step = nan

        y = [start]
        for _ in range(num-2):
            y.append(y[-1] + step)
        y.append(stop)
    else:
        step = (stop-start)/float(num)
        if num == 1:
            step = nan
        y = [start]
        for _ in range(num-1):
            y.append(y[-1] + step)

    if retstep:
        return y, step
    else:
        return y


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    y = linspace(start, stop, num=num, endpoint=endpoint)
    for i in range(len(y)):
        y[i] = base**y[i]
#    return [base**yi for yi in y]
    return y


def product(l):
    # Helper in some functions
    tot = 1.0
    for i in l:
        tot *= i
    return tot


def cumsum(a):
    # Does not support multiple dimensions
    sums = [a[0]]
    for i in a[1:]:
        sums.append(sums[-1] + i)
    return sums


def diff(a, n=1, axis=-1):
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got %s" %(n))
#    nd = 1 # hardcode
    diffs = []
    for i in range(1, len(a)):
        delta = a[i] - a[i-1]
        diffs.append(delta)

    if n > 1:
        return diff(diffs, n-1)
    return diffs



central_diff_weights_precomputed = {
 (1, 3): [-0.5, 0.0, 0.5],
 (1, 5): [0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333],
 (1, 7): [-0.016666666666666666, 0.15, -0.75, 0.0, 0.75, -0.15, 0.016666666666666666],
 (1, 9): [0.0035714285714285713, -0.0380952380952381, 0.2, -0.8, 0.0, 0.8, -0.2, 0.0380952380952381,
          -0.0035714285714285713],
 (1, 11): [-0.0007936507936507937, 0.00992063492063492, -0.05952380952380952, 0.23809523809523808, -0.8333333333333334,
           0.0, 0.8333333333333334, -0.23809523809523808, 0.05952380952380952, -0.00992063492063492,
           0.0007936507936507937],
 (1, 13): [0.00018037518037518038, -0.0025974025974025974, 0.017857142857142856, -0.07936507936507936,
           0.26785714285714285, -0.8571428571428571, 0.0, 0.8571428571428571, -0.26785714285714285, 0.07936507936507936,
           -0.017857142857142856, 0.0025974025974025974, -0.00018037518037518038],
 (1, 15): [-4.1625041625041625e-05, 0.0006798756798756799, -0.005303030303030303, 0.026515151515151516,
           -0.09722222222222222, 0.2916666666666667, -0.875, 0.0, 0.875, -0.2916666666666667, 0.09722222222222222,
           -0.026515151515151516, 0.005303030303030303, -0.0006798756798756799, 4.1625041625041625e-05],
 (1, 17): [9.712509712509713e-06, -0.0001776001776001776, 0.001554001554001554, -0.008702408702408702,
           0.03535353535353535, -0.11313131313131314, 0.3111111111111111, -0.8888888888888888, 0.0, 0.8888888888888888,
           -0.3111111111111111, 0.11313131313131314, -0.03535353535353535, 0.008702408702408702, -0.001554001554001554,
           0.0001776001776001776, -9.712509712509713e-06],
 (1, 19): [-2.285296402943462e-06, 4.6277252159605104e-05, -0.00044955044955044955, 0.002797202797202797,
           -0.012587412587412588, 0.044055944055944055, -0.12727272727272726, 0.32727272727272727, -0.9, 0.0, 0.9,
           -0.32727272727272727, 0.12727272727272726, -0.044055944055944055, 0.012587412587412588,
           -0.002797202797202797, 0.00044955044955044955, -4.6277252159605104e-05, 2.285296402943462e-06],
 (2, 3): [1.0, -2.0, 1.0],
 (2, 5): [-0.08333333333333333, 1.3333333333333333, -2.5, 1.3333333333333333, -0.08333333333333333],
 (2, 7): [0.011111111111111112, -0.15, 1.5, -2.7222222222222223, 1.5, -0.15, 0.011111111111111112],
 (2, 9): [-0.0017857142857142857, 0.025396825396825397, -0.2, 1.6, -2.8472222222222223, 1.6, -0.2, 0.025396825396825397,
          -0.0017857142857142857],
 (2, 11): [0.00031746031746031746, -0.00496031746031746, 0.03968253968253968, -0.23809523809523808, 1.6666666666666667,
           -2.9272222222222224, 1.6666666666666667, -0.23809523809523808, 0.03968253968253968, -0.00496031746031746,
           0.00031746031746031746],
 (2, 13): [-6.012506012506013e-05, 0.001038961038961039, -0.008928571428571428, 0.05291005291005291,
           -0.26785714285714285, 1.7142857142857142, -2.9827777777777778, 1.7142857142857142, -0.26785714285714285,
           0.05291005291005291, -0.008928571428571428, 0.001038961038961039, -6.012506012506013e-05],
 (2, 15): [1.1892869035726179e-05, -0.00022662522662522663, 0.0021212121212121214, -0.013257575757575758,
           0.06481481481481481, -0.2916666666666667, 1.75, -3.02359410430839, 1.75, -0.2916666666666667,
           0.06481481481481481, -0.013257575757575758, 0.0021212121212121214, -0.00022662522662522663,
           1.1892869035726179e-05],
 (2, 17): [-2.428127428127428e-06, 5.074290788576503e-05, -0.000518000518000518, 0.003480963480963481,
           -0.017676767676767676, 0.07542087542087542, -0.3111111111111111, 1.7777777777777777, -3.05484410430839,
           1.7777777777777777, -0.3111111111111111, 0.07542087542087542, -0.017676767676767676, 0.003480963480963481,
           -0.000518000518000518, 5.074290788576503e-05, -2.428127428127428e-06],
 (2, 19): [5.078436450985471e-07, -1.1569313039901276e-05, 0.00012844298558584272, -0.0009324009324009324,
           0.005034965034965035, -0.022027972027972027, 0.08484848484848485, -0.32727272727272727, 1.8,
           -3.0795354623330815, 1.8, -0.32727272727272727, 0.08484848484848485, -0.022027972027972027,
           0.005034965034965035, -0.0009324009324009324, 0.00012844298558584272, -1.1569313039901276e-05,
           5.078436450985471e-07],
 (3, 5): [-0.5, 1.0, 0.0, -1.0, 0.5],
 (3, 7): [0.125, -1.0, 1.625, 0.0, -1.625, 1.0, -0.125],
 (3, 9): [-0.029166666666666667, 0.30000000000000004, -1.4083333333333332, 2.033333333333333, 0.0, -2.033333333333333,
          1.4083333333333332, -0.30000000000000004, 0.029166666666666667],
# (3, 11): [0.006779100529100529, -0.08339947089947089, 0.48303571428571435, -1.7337301587301588, 2.3180555555555555,
#           0.0, -2.3180555555555555, 1.7337301587301588, -0.48303571428571435, 0.08339947089947089,
#           -0.006779100529100529],
# (3, 13): [-0.0015839947089947089, 0.02261904761904762, -0.1530952380952381, 0.6572751322751322, -1.9950892857142857,
#           2.527142857142857, 0.0, -2.527142857142857, 1.9950892857142857, -0.6572751322751322, 0.1530952380952381,
#           -0.02261904761904762, 0.0015839947089947089],
# (3, 15): [0.0003724747474747475, -0.006053691678691679, 0.04682990620490621, -0.2305699855699856, 0.8170667989417989,
#           -2.2081448412698412, 2.6869345238095237, 0.0, -2.6869345238095237, 2.2081448412698412, -0.8170667989417989,
#           0.2305699855699856, -0.04682990620490621, 0.006053691678691679, -0.0003724747474747475],
# (3, 17): [-8.810006131434702e-05, 0.0016058756058756059, -0.013982697196982911, 0.07766492766492766,
#           -0.31074104136604136, 0.9613746993746994, -2.384521164021164, 2.81291761148904, 0.0, -2.81291761148904,
#           2.384521164021164, -0.9613746993746994, 0.31074104136604136, -0.07766492766492766, 0.013982697196982911,
#           -0.0016058756058756059, 8.810006131434702e-05],
# (3, 19): [2.0943672729387014e-05, -0.00042319882498453924, 0.00409817266067266, -0.025376055161769447,
#           0.11326917130488559, -0.3904945471195471, 1.0909741462241462, -2.532634817563389, 2.9147457482993198, 0.0,
#           -2.9147457482993198, 2.532634817563389, -1.0909741462241462, 0.3904945471195471, -0.11326917130488559,
#           0.025376055161769447, -0.00409817266067266, 0.00042319882498453924, -2.0943672729387014e-05],
# (4, 5): [1.0, -4.0, 6.0, -4.0, 1.0],
# (4, 7): [-0.16666666666666666, 2.0, -6.5, 9.333333333333334, -6.5, 2.0, -0.16666666666666666],
# (4, 9): [0.029166666666666667, -0.4, 2.8166666666666664, -8.133333333333333, 11.375, -8.133333333333333,
#          2.8166666666666664, -0.4, 0.029166666666666667],
# (4, 11): [-0.005423280423280424, 0.08339947089947089, -0.644047619047619, 3.4674603174603176, -9.272222222222222,
#           12.741666666666665, -9.272222222222222, 3.4674603174603176, -0.644047619047619, 0.08339947089947089,
#           -0.005423280423280424],
# (4, 13): [0.0010559964726631393, -0.018095238095238095, 0.1530952380952381, -0.8763668430335096, 3.9901785714285714,
#           -10.108571428571429, 13.717407407407407, -10.108571428571429, 3.9901785714285714, -0.8763668430335096,
#           0.1530952380952381, -0.018095238095238095, 0.0010559964726631393],
# (5, 7): [-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5],
# (5, 9): [0.16666666666666669, -1.5, 4.333333333333333, -4.833333333333334, 0.0, 4.833333333333334, -4.333333333333333,
#          1.5, -0.16666666666666669],
# (5, 11): [-0.04513888888888889, 0.5277777777777778, -2.71875, 6.5, -6.729166666666667, 0.0, 6.729166666666667, -6.5,
#           2.71875, -0.5277777777777778, 0.04513888888888889],
# (5, 13): [0.011491402116402117, -0.16005291005291006, 1.033399470899471, -3.9828042328042326, 8.39608134920635,
#           -8.246031746031747, 0.0, 8.246031746031747, -8.39608134920635, 3.9828042328042326, -1.033399470899471,
#           0.16005291005291006, -0.011491402116402117]
 }

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
    return x1, x2


def central_diff_weights(points, divisions=1):
    # Check the cache
    if (divisions, points) in central_diff_weights_precomputed:
        return central_diff_weights_precomputed[(divisions, points)]

    if points < divisions + 1:
        raise ValueError("Points < divisions + 1, cannot compute")
    if points % 2 == 0:
        raise ValueError("Odd number of points required")
    ho = points >> 1

    x = [[xi] for xi in range(-ho, ho+1)]
    X = []
    for xi in x:
        line = [1.0] + [xi[0]**k for k in range(1, points)]
        X.append(line)
    factor = product(range(1, divisions + 1))
#    from scipy.linalg import inv
    from sympy import Matrix
    # The above coefficients were generated from a routine which used Fractions
    # Additional coefficients cannot reliable be computed with numpy or floating
    # point numbers - the error is too great
    # sympy must be used for reliability
    inverted = [[float(j) for j in i] for i in Matrix(X).inv().tolist()]
    w = [i*factor for i in inverted[divisions]]
#    w = [i*factor for i in (inv(X)[divisions]).tolist()]
    central_diff_weights_precomputed[(divisions, points)] = w
    return w


def derivative(func, x0, dx=1.0, n=1, args=(), order=3, scalar=True,
               lower_limit=None, upper_limit=None):
    """Reimplementation of SciPy's derivative function, with more cached
    coefficients and without using numpy. If new coefficients not cached are
    needed, they are only calculated once and are remembered.

    Support for vector value functions has also been added.
    """
    if order < n + 1:
        raise ValueError
    if order % 2 == 0:
        raise ValueError
    weights = central_diff_weights(order, n)
    ho = order >> 1
    denominator = 1.0/product([dx]*n)
    if scalar:
        max_x = x0 + (order - 1 - ho)*dx
        if upper_limit is not None and max_x > upper_limit:
            x0 -= (max_x - x0)
        min_x = x0 + -ho*dx
        if lower_limit is not None and min_x < lower_limit:
            x0 += (x0 - min_x)

        tot = 0.0
        for k in range(order):
            if weights[k] != 0.0:
                tot += weights[k]*func(x0 + (k - ho)*dx, *args)
        return tot*denominator
    else:
        numerators = None
        for k in range(order):
            f = func(x0 + (k - ho)*dx, *args)
            if numerators is None:
                N = len(f)
                numerators = [0.0]*N
            for i in range(N):
                numerators[i] += weights[k]*f[i]
        return [num*denominator for num in numerators]




kronrod_weights = {
#    2: [0.1979797979797974, 0.4909090909090922, 0.6222222222222221, 0.4909090909090904, 0.1979797979797974],
#    3: [0.10465622602646653, 0.2684880898683364, 0.4013974147759613, 0.45091653865847436, 0.40139741477596186, 0.26848808986833306, 0.10465622602646701],
#    4: [0.06297737366547282, 0.17005360533572325, 0.26679834045228396, 0.3269491896014517, 0.3464429818901376, 0.32694918960145064, 0.2667983404522837,
#        0.17005360533572333, 0.06297737366547272
#    ],
#    5: [0.042582036751078814, 0.11523331662247298, 0.1868007965564952, 0.2410403392286488, 0.27284980191255903, 0.28298741785749215, 0.2728498019125575,
#        0.24104033922864848, 0.1868007965564929, 0.11523331662247287, 0.04258203675108158
#    ],
#    6: [0.03039615411981852, 0.08369444044690531, 0.1373206046344473, 0.1810719943231391, 0.21320965227196226, 0.23377086411699666, 0.2410725801734652,
#        0.23377086411699363, 0.21320965227196081, 0.1810719943231382, 0.13732060463444687, 0.08369444044690623, 0.030396154119819826
#    ],
#    7: [0.02293532201053038, 0.0630920926299766, 0.10479001032225017, 0.14065325971552478, 0.16900472663927035, 0.19035057806478545, 0.2044329400753001,
#        0.20948214108472712, 0.2044329400752992, 0.1903505780647853, 0.16900472663926686, 0.14065325971552592, 0.10479001032225005, 0.06309209262997834,
#        0.022935322010529363
#    ],
#    8: [0.017822383320711625, 0.04943939500213785, 0.08248229893135677, 0.11164637082684088, 0.13626310925517557, 0.15665260616818655, 0.17207060855521186,
#        0.18140002506803385, 0.18444640574468968, 0.1814000250680349, 0.1720706085552107, 0.1566526061681889, 0.13626310925517285, 0.11164637082683933,
#        0.08248229893135825, 0.04943939500213947, 0.017822383320710476
#    ],
#    9: [0.014304775643840032, 0.03963189516026222, 0.0665181559402729, 0.09079068168872734, 0.11178913468441762, 0.13000140685534053, 0.14523958838436735,
#        0.15641352778848416, 0.16286282744011463, 0.16489601282834765, 0.16286282744011538, 0.15641352778848497, 0.14523958838436518,
#        0.13000140685534053, 0.11178913468441844, 0.09079068168872709, 0.0665181559402743, 0.03963189516026152, 0.01430477564383906
#    ],
    10: [0.011694638867371172, 0.03255816230796458, 0.05475589657435185, 0.07503967481091954, 0.09312545458369915, 0.10938715880229993, 0.12349197626206619,
        0.1347092173114735, 0.1427759385770589, 0.14773910490133732, 0.1494455540029151, 0.14773910490133846, 0.14277593857705978, 0.13470921731147467,
        0.1234919762620664, 0.10938715880229763, 0.09312545458369788, 0.0750396748109198, 0.05475589657435227, 0.03255816230796462, 0.01169463886737192
    ],
#    11: [0.009765441045959541, 0.02715655468210335, 0.045829378564428196, 0.06309742475037379, 0.07866457193222803, 0.092953098596902, 0.1058720744813897,
#        0.11673950246104702, 0.12515879910031777, 0.13128068422980624, 0.13519357279988428, 0.1365777947111174, 0.13519357279988467, 0.1312806842298052,
#        0.12515879910031882, 0.11673950246104742, 0.10587207448139022, 0.09295309859690085, 0.07866457193222773, 0.06309742475037532,
#        0.04582937856442653, 0.027156554682104143, 0.009765441045960747
#    ],
#    12: [0.008257711433166826, 0.023036084038981483, 0.03891523046929862, 0.053697017607756393, 0.06725090705084145, 0.07992027533360162,
#        0.09154946829504902, 0.1016497322790617, 0.11002260497764366, 0.11671205350175794, 0.12162630352394828, 0.12458416453615645,
#        0.12555689390547403, 0.12458416453615662, 0.12162630352394885, 0.11671205350175637, 0.11002260497764366, 0.10164973227906021,
#        0.09154946829504902, 0.07992027533360205, 0.06725090705084015, 0.05369701760775601, 0.038915230469299136, 0.023036084038982187,
#        0.008257711433168356
#    ],
#    13: [0.007087846351247695, 0.019753746382707126, 0.03344358998955114, 0.046279017973829016, 0.05811521042311424, 0.06930363324778273,
#        0.0798059621694777, 0.08916844187754189, 0.09714173487607722, 0.10383060116904075, 0.10926635109528536, 0.11321025917152877, 0.1154887990912863,
#        0.11620961236306047, 0.11548879909128668, 0.11321025917152944, 0.10926635109528575, 0.10383060116903994, 0.09714173487607768,
#        0.08916844187754075, 0.0798059621694761, 0.06930363324778134, 0.05811521042311445, 0.0462790179738303, 0.033443589989552346,
#        0.019753746382705984, 0.007087846351248617
#    ],
#    14: [0.006139558686377161, 0.01714845890993743, 0.02904870126150814, 0.04025059487268863, 0.050691543260464045, 0.06066712586674275,
#        0.07010297900274666, 0.0786557972496211, 0.08618376286946026, 0.09273683001785471, 0.09826492647210357, 0.1026166273213988, 0.10573163984176341,
#        0.10762642111411824, 0.10827006650642954, 0.10762642111411819, 0.10573163984176337, 0.10261662732139956, 0.09826492647210401,
#        0.09273683001785424, 0.08618376286945752, 0.07865579724962145, 0.07010297900274708, 0.060667125866742444, 0.050691543260465384,
#        0.040250594872688804, 0.029048701261508613, 0.01714845890993556, 0.00613955868637814
#    ],
#    15: [0.005377479872922721, 0.015007947329316597, 0.02546084732671451, 0.03534636079137684, 0.044589751324763977, 0.053481524690928775,
#        0.06200956780067099, 0.06985412131872938, 0.07684968075772279, 0.08308050282313152, 0.08856444305621097, 0.09312659817082411,
#        0.09664272698362378, 0.09917359872179198, 0.10076984552387584, 0.10133000701479176, 0.10076984552387498, 0.09917359872179111,
#        0.09664272698362311, 0.09312659817082504, 0.08856444305621132, 0.08308050282313374, 0.07684968075772115, 0.06985412131872824,
#        0.06200956780067089, 0.05348152469092814, 0.04458975132476507, 0.03534636079137597, 0.0254608473267153, 0.01500794732931606,
#        0.005377479872923303
#    ],
#    16: [0.0047427770492463476, 0.01325793068809046, 0.022498859440048875, 0.031260543647380616, 0.03951295120242179, 0.04750621597640688,
#        0.055205633095423194, 0.062358806011834494, 0.06886299519153054, 0.07476982388560004, 0.08005394126371923, 0.08459580379259089,
#        0.08833750257911316, 0.0912920328281922, 0.09343867406092181, 0.09472840124723077, 0.09515421608049866, 0.09472840124722949,
#        0.09343867406092161, 0.09129203282819215, 0.08833750257911213, 0.0845958037925905, 0.08005394126371795, 0.07476982388559944,
#        0.06886299519153126, 0.062358806011835174, 0.05520563309542224, 0.047506215976407244, 0.03951295120242178, 0.031260543647380366,
#        0.02249885944004935, 0.013257930688091162, 0.0047427770492472505
#    ],
#    17: [0.00421897579377578, 0.011785837562288765, 0.020022233953294232, 0.027856722457863144, 0.035249746751878815, 0.04244263020500057,
#        0.04943614182396047, 0.055994044530093004, 0.06200091526823072, 0.06752801671813287, 0.07258989011419156, 0.07705622304620212,
#        0.08083667844081736, 0.08397257199123528, 0.08649053220565191, 0.08830878229760371, 0.08936184586778975, 0.08969642194398147,
#        0.08936184586778813, 0.08830878229760494, 0.08649053220565157, 0.0839725719912349, 0.08083667844081745, 0.07705622304620285,
#        0.07258989011419015, 0.06752801671813129, 0.06200091526822982, 0.05599404453009317, 0.04943614182395916, 0.042442630205000643,
#        0.03524974675187985, 0.027856722457863275, 0.020022233953294787, 0.011785837562289191, 0.004218975793776932
#    ],
#    18: [0.00377351012843988, 0.010554430567545547, 0.01793337620570976, 0.024964402753740674, 0.03163401984884463, 0.0381537583251753,
#        0.044509195165090054, 0.0505075715021042, 0.056072390382993734, 0.061253565791358926, 0.06603995819281894, 0.07033745002422243,
#        0.07409702738131525, 0.0773320701646388, 0.08003001949509578, 0.08214399978333053, 0.08365485143716135, 0.08456933354407732,
#        0.08487813861267707, 0.08456933354407585, 0.08365485143716228, 0.08214399978332948, 0.08003001949509637, 0.07733207016463836,
#        0.07409702738131455, 0.07033745002422402, 0.06603995819281755, 0.06125356579135898, 0.05607239038299394, 0.05050757150210246,
#        0.044509195165090026, 0.03815375832517691, 0.031634019848844154, 0.024964402753740036, 0.017933376205711227, 0.01055443056754483,
#        0.0037735101284401234
#    ],
#    19: [0.0033981531196335965, 0.009499205461711157, 0.016153414369089493, 0.02250869575220442, 0.028544238117114772, 0.03446235286288065,
#        0.04027002324951411, 0.04578597308792263, 0.05092622550624418, 0.05575220103348164, 0.060277411503209075, 0.0644023522619292,
#        0.06805978049813398, 0.07128142476493002, 0.07408407453373855, 0.07640286193981659, 0.07818691068574178, 0.07946568358535473,
#        0.08026621602085142, 0.08054560329299826, 0.08026621602085034, 0.07946568358535418, 0.07818691068574216, 0.07640286193981571,
#        0.07408407453373904, 0.07128142476492985, 0.06805978049813312, 0.06440235226192832, 0.06027741150320817, 0.055752201033481275,
#        0.05092622550624404, 0.04578597308792251, 0.04027002324951481, 0.03446235286288103, 0.028544238117114873, 0.02250869575220419,
#        0.016153414369091356, 0.009499205461710144, 0.0033981531196347167
#    ],
#         87: [0.00016881480210456692, 0.00047314868570756235, 0.0008076859860306628, 0.0011322615841296547, 0.00144806597820222, 0.0017668191481724435, 0.0020903160499537304, 0.002411576726915393, 0.0027285063230227296, 0.0030454007869937876, 0.0033639128646212944, 0.0036808253607571195, 0.003994559817314871, 0.004307345076427869, 0.004620335406184352, 0.004931600306915534, 0.00523996592505234, 0.005546814695915935, 0.00585298027614558, 0.006157129826902876, 0.006458347449208747, 0.0067575841722384, 0.007055469839437327, 0.007351002065619594, 0.007643435225722091, 0.007933469441304792, 0.008221599308725454, 0.008507026852701362, 0.008789124673244655, 0.009068433903060025, 0.00934535574073612, 0.00961922899522448, 0.009889512165959614, 0.010156639610742505, 0.01042094604150296, 0.010681867318519509, 0.010938926800718108, 0.011192484212786687, 0.011442825936038695, 0.011689459907180238, 0.01193196007256122, 0.012170632572570447, 0.01240572822845446, 0.012636810511622156, 0.012863493923529042, 0.013086045502712524, 0.013304689821902624, 0.013519034459386238, 0.013728727196381123, 0.013934006347263819, 0.014135077241799326, 0.0143315833864912, 0.01452320041557479, 0.01471014561481331, 0.014892610489607272, 0.015070268437801156, 0.01524281878982064, 0.015410463680651272, 0.01557338508816692, 0.015731281717590702, 0.01588387332676284, 0.016031351508057137, 0.01617389219804666, 0.016311215832305204, 0.016443059955062258, 0.016569609315104587, 0.016691036698090543, 0.016807081400707274, 0.01691749655723537, 0.017022463108090843, 0.017122153135412682, 0.017216322428248873, 0.017304737826468557, 0.017387579005841453, 0.0174650194618558, 0.017536829459282623, 0.017602787871331006, 0.01766307528646667, 0.017717868485658792, 0.017766950435146724, 0.017810110503721263, 0.01784753209761472, 0.017879396979083733, 0.017905499199739466, 0.017925637157192337, 0.017939998782076135, 0.017948772394930045, 0.017951761607250333, 0.01794877239493056, 0.017939998782076676, 0.01792563715719179, 0.017905499199739282, 0.017879396979084167, 0.017847532097614346, 0.01781011050372197, 0.017766950435146787, 0.017717868485658865, 0.01766307528646623, 0.017602787871330267, 0.017536829459281828, 0.01746501946185603, 0.01738757900584135, 0.017304737826468315, 0.017216322428249966, 0.01712215313541374, 0.017022463108090982, 0.016917496557235175, 0.016807081400708086, 0.016691036698090224, 0.016569609315104577, 0.016443059955061293, 0.016311215832305943, 0.016173892198046793, 0.01603135150805684, 0.015883873326762922, 0.0157312817175899, 0.015573385088166592, 0.015410463680651668, 0.015242818789820901, 0.015070268437801513, 0.01489261048960673, 0.014710145614813493, 0.014523200415575395, 0.014331583386491533, 0.01413507724179865, 0.013934006347263444, 0.013728727196380418, 0.013519034459385928, 0.01330468982190219, 0.013086045502712456, 0.01286349392352816, 0.012636810511621802, 0.012405728228453924, 0.012170632572569789, 0.011931960072561447, 0.011689459907180488, 0.011442825936038951, 0.011192484212786695, 0.010938926800718342, 0.010681867318518458, 0.010420946041502708, 0.01015663961074195, 0.009889512165958998, 0.009619228995224303, 0.009345355740736385, 0.009068433903060022, 0.00878912467324535, 0.008507026852701478, 0.008221599308725935, 0.007933469441304955, 0.00764343522572274, 0.007351002065619289, 0.007055469839437192, 0.006757584172238039, 0.006458347449209268, 0.0061571298269029575, 0.005852980276145424, 0.00554681469591638, 0.005239965925052114, 0.004931600306916382, 0.00462033540618409, 0.004307345076429006, 0.003994559817315303, 0.0036808253607556987, 0.00336391286462138, 0.0030454007869938063, 0.0027285063230224034, 0.0024115767269154683, 0.0020903160499540895, 0.0017668191481723982, 0.0014480659782023144, 0.0011322615841305992, 0.000807685986030332, 0.0004731486857077167, 0.00016881480210494433],
#    300: [1.431338551855931e-05, 4.012227180120215e-05, 6.850298129947514e-05, 9.605899747347381e-05, 0.00012290072040166636, 0.0001500289025800392, 0.00017760205103018156, 0.0002050376837370814, 0.00023216802273282316, 0.00025936376117528543, 0.000286770923337297, 0.0003141263938546069, 0.0003413064333028032, 0.0003685059563224142, 0.0003958282766766556, 0.00042311960183710317, 0.00045029066797970557, 0.00047746412489335334, 0.0005047156371897132, 0.0005319422306222964, 0.0005590766158027022, 0.0005862043554606536, 0.0006133833418229676, 0.0006405386949363323, 0.0006676176085674061, 0.0006946838416537453, 0.0007217832962185991, 0.0007488584417973781, 0.0007758664924472269, 0.0008028571692832445, 0.0008298678975835217, 0.0008568527124189767, 0.0008837760635199613, 0.0009106780294555469, 0.0009375898032692981, 0.0009644735729342993, 0.000991299201423038, 0.0010180998154806211, 0.001044901876521459, 0.0010716735696885908, 0.0010983889275928184, 0.0011250758736232511, 0.0011517571804946123, 0.00117840559504653, 0.0012049984403109213, 0.0012315596269869425, 0.0012581089880280057, 0.0012846228351818792, 0.0013110811414252545, 0.00133750464674385, 0.001363910797264291, 0.0013902787534588901, 0.0014165906596462637, 0.0014428646992844724, 0.0014691163496578206, 0.0014953270906152946, 0.001521480872306304, 0.0015475937814626148, 0.00157367964899441, 0.0015997218734369169, 0.0016257059263124127, 0.001651646149529894, 0.001677554980842586, 0.001703417427928488, 0.0017292202586412138, 0.0017549763445420545, 0.0017806969321644153, 0.0018063683949693567, 0.0018319786165106394, 0.001857539215647843, 0.0018830604109505785, 0.0019085297473710398, 0.0019339360773297323, 0.001959289942067989, 0.0019846006657902662, 0.002009856807725714, 0.002035048068429775, 0.0020601840541859453, 0.0020852733053521944, 0.00211030526669342, 0.002135270386619155, 0.002160177454021052, 0.0021850343177347107, 0.002209831201507028, 0.0022345592175448837, 0.0022592264352258922, 0.002283840089667751, 0.002308391094568184, 0.0023328711548821417, 0.0023572877027127443, 0.002381647425556331, 0.00240594185202409, 0.002430163219323544, 0.002454318391951272, 0.002478413566344175, 0.0025024408222929406, 0.0025263928773852835, 0.00255027608798504, 0.0025740962081959693, 0.0025978458144661396, 0.0026215180600127255, 0.0026451188441838666, 0.0026686535209823115, 0.0026921151165751785, 0.0027154971809750543, 0.0027388052007325915, 0.0027620441665595044, 0.0027852075136901313, 0.002808289155056237, 0.002831294202880455, 0.0028542273168385085, 0.002877082305828928, 0.002899853416015445, 0.0029225454189358694, 0.002945162671626956, 0.002967699325674426, 0.0029901499343276, 0.0030125189580153013, 0.003034810476250164, 0.0030570189560660717, 0.0030791392346825115, 0.00310117548753865, 0.0031231315389228563, 0.0031450021472680554, 0.0031667824132465515, 0.003188476250471062, 0.003210087247886691, 0.0032316104340022714, 0.0032530411546735326, 0.0032743830823033305, 0.003295639588282603, 0.003316805952230262, 0.003337877748850066, 0.0033588584277646333, 0.003379751158776646, 0.003400551455680898, 0.0034212551073882925, 0.0034418653572642383, 0.0034623851879040167, 0.0034828103321097002, 0.0035031367798385575, 0.0035233675830591412, 0.0035435055501479742, 0.0035635466192892535, 0.0035834869696225067, 0.0036033294751299954, 0.003623076781728268, 0.0036427250207152315, 0.003662270549687044, 0.0036817160767749556, 0.0037010640961099225, 0.0037203109210253195, 0.003739453077859395, 0.0037584931198988143, 0.003777433399208016, 0.003796270401119918, 0.003815000811903878, 0.003833627040006793, 0.003852151304287865, 0.0038705702529859497, 0.003888880724274518, 0.003907084990886804, 0.003925185146571497, 0.003943177994202738, 0.003961060516554395, 0.00397883485897057, 0.00399650299750876, 0.004014061882142001, 0.004031508633574408, 0.0040488452773837025, 0.004066073678737227, 0.004083190927833456, 0.0041001942772090185, 0.004117085639660952, 0.004133866775714363, 0.004150534909511804, 0.004167087419841768, 0.00418352611313258, 0.004199852651012914, 0.004216064385826003, 0.004232158817490263, 0.0042481376519647165, 0.004264002457277361, 0.004279750708703243, 0.004295380022593513, 0.004310892009868351, 0.004326288149837712, 0.004341566035873309, 0.004356723396447817, 0.004371761752443555, 0.004386682498963286, 0.004401483343036293, 0.0044161621212866105, 0.004430720269180384, 0.004445159101775619, 0.004459476435677051, 0.004473670212010881, 0.004487741785091177, 0.0045016923937820415, 0.004515519960510983, 0.0045292225275432845, 0.0045428013719821725, 0.004556257660061908, 0.004569589416572178, 0.004582794781821561, 0.004595874959351039, 0.004608831046059929, 0.004621661165914976, 0.004634363554417776, 0.00464693934491508, 0.00465938956802471, 0.004671712443956493, 0.004683906300772354, 0.0046959722047512755, 0.004707911123050588, 0.0047197213694184096, 0.004731401362055238, 0.004742952103059206, 0.004754374498742801, 0.004765666953897418, 0.004776827974631882, 0.004787858501534513, 0.004798759382490675, 0.004809529111038303, 0.004820166279139213, 0.004830671768345388, 0.004841046370347547, 0.004851288665314363, 0.004861397329170941, 0.004871373186717853, 0.004881216975513267, 0.0048909273604094364, 0.004900503099558432, 0.004909944963118174, 0.0049192536364175265, 0.004928427867200543, 0.004937466494252367, 0.004946370235033431, 0.0049551397243954, 0.004963773791329307, 0.004972271353793974, 0.004980633078343443, 0.004988859550960298, 0.004996949680372135, 0.005004902462380824, 0.005012718514281037, 0.0050203983746564115, 0.005027941030594458, 0.00503534555452279, 0.005042612515983653, 0.005049742407509773, 0.005056734293288795, 0.0050635873212702565, 0.0050703020146247325, 0.005076878821053563, 0.0050833168806995, 0.005089615416040483, 0.005095774905126578, 0.005101795751935285, 0.0051076771715272175, 0.0051134184600114575, 0.005119020051454203, 0.0051244823071031915, 0.005129804516003444, 0.005134986047095221, 0.005140027291483333, 0.005144928568571508, 0.005149689240552303, 0.0051543087484908, 0.005158787441443115, 0.005163125597752833, 0.005167322652015288, 0.0051713781168029205, 0.005175292299931676, 0.005179065439366613, 0.0051826970414525384, 0.005186186689737414, 0.005189534651502356, 0.005192741124918436, 0.005195805687514126, 0.005198727993361184, 0.005201508269816953, 0.0052041466757448375, 0.005206642859375407, 0.005208996544935003, 0.00521120792036838, 0.00521327710562835, 0.005215203819244274, 0.005216987855309183, 0.005218629362770544, 0.005220128422976571, 0.005221484824432338, 0.005222698430884936, 0.005223769352611614, 0.005224697632566832, 0.005225483128989441, 0.005226125775143653, 0.005226625642871415, 0.005226982736856913, 0.005227196984904545, 0.005227268389736952, 0.005227196984904609, 0.005226982736857298, 0.005226625642871094, 0.005226125775143343, 0.005225483128989314, 0.005224697632566667, 0.005223769352611394, 0.005222698430885047, 0.005221484824432276, 0.005220128422976045, 0.005218629362770703, 0.005216987855309369, 0.005215203819244113, 0.005213277105628448, 0.005211207920368179, 0.005208996544935323, 0.005206642859375223, 0.005204146675744738, 0.005201508269816841, 0.005198727993361313, 0.005195805687514172, 0.0051927411249182535, 0.00518953465150263, 0.005186186689737238, 0.0051826970414525254, 0.0051790654393667, 0.005175292299931799, 0.005171378116803026, 0.005167322652015363, 0.005163125597752643, 0.005158787441443411, 0.00515430874849076, 0.005149689240552316, 0.005144928568571719, 0.005140027291483478, 0.00513498604709535, 0.005129804516003328, 0.0051244823071030276, 0.005119020051454297, 0.005113418460011166, 0.005107677171526961, 0.005101795751935227, 0.005095774905126261, 0.005089615416040522, 0.005083316880699735, 0.005076878821053743, 0.005070302014624845, 0.0050635873212700805, 0.0050567342932884085, 0.005049742407509847, 0.00504261251598381, 0.005035345554522823, 0.005027941030594158, 0.005020398374656165, 0.005012718514281176, 0.005004902462380897, 0.00499694968037226, 0.0049888595509603095, 0.004980633078343566, 0.004972271353793644, 0.004963773791329042, 0.004955139724395721, 0.004946370235034016, 0.004937466494252536, 0.004928427867200534, 0.004919253636417707, 0.004909944963118104, 0.004900503099558809, 0.0048909273604097105, 0.0048812169755131024, 0.004871373186717716, 0.004861397329170951, 0.004851288665314311, 0.004841046370347232, 0.004830671768345188, 0.004820166279138959, 0.004809529111038251, 0.004798759382490338, 0.0047878585015345716, 0.004776827974631735, 0.004765666953897307, 0.0047543744987427914, 0.004742952103059202, 0.0047314013620551616, 0.004719721369418516, 0.004707911123050433, 0.004695972204750941, 0.0046839063007723, 0.004671712443956289, 0.004659389568024603, 0.004646939344914928, 0.004634363554417283, 0.00462166116591521, 0.004608831046059741, 0.004595874959351329, 0.004582794781821686, 0.004569589416571941, 0.004556257660061593, 0.004542801371981853, 0.0045292225275430625, 0.0045155199605110725, 0.004501692393782684, 0.004487741785091521, 0.0044736702120110345, 0.0044594764356769125, 0.004445159101775285, 0.0044307202691803455, 0.004416162121286692, 0.004401483343036429, 0.004386682498963625, 0.004371761752443638, 0.004356723396447825, 0.004341566035873333, 0.004326288149837637, 0.004310892009868086, 0.00429538002259381, 0.0042797507087033665, 0.004264002457277777, 0.004248137651964907, 0.004232158817490341, 0.004216064385826206, 0.004199852651012641, 0.004183526113132338, 0.004167087419841528, 0.004150534909512119, 0.004133866775714176, 0.004117085639661458, 0.0041001942772092145, 0.00408319092783341, 0.004066073678737072, 0.004048845277383841, 0.004031508633574328, 0.004014061882142042, 0.003996502997508904, 0.003978834858970341, 0.00396106051655424, 0.003943177994202945, 0.003925185146572065, 0.003907084990886625, 0.0038888807242744516, 0.003870570252985675, 0.0038521513042880833, 0.003833627040007188, 0.0038150008119035845, 0.0037962704011200703, 0.003777433399207321, 0.003758493119898556, 0.003739453077858988, 0.0037203109210250346, 0.0037010640961101984, 0.0036817160767748554, 0.0036622705496868126, 0.00364272502071535, 0.0036230767817284986, 0.003603329475130338, 0.0035834869696225024, 0.0035635466192891576, 0.00354350555014782, 0.0035233675830593134, 0.0035031367798384993, 0.003482810332109164, 0.003462385187904296, 0.003441865357264365, 0.0034212551073889708, 0.003400551455680581, 0.0033797511587767763, 0.003358858427764282, 0.003337877748850245, 0.0033168059522300543, 0.00329563958828269, 0.0032743830823032723, 0.0032530411546732876, 0.003231610434001921, 0.0032100872478863954, 0.003188476250471022, 0.0031667824132466152, 0.0031450021472676755, 0.003123131538923296, 0.003101175487538483, 0.003079139234681824, 0.0030570189560658587, 0.0030348104762501467, 0.003012518958015103, 0.0029901499343269566, 0.0029676993256746486, 0.0029451626716272246, 0.0029225454189359683, 0.002899853416015462, 0.0028770823058289967, 0.0028542273168381985, 0.0028312942028804098, 0.002808289155056215, 0.00278520751368993, 0.0027620441665596666, 0.002738805200732458, 0.002715497180975006, 0.0026921151165756846, 0.0026686535209825175, 0.0026451188441835257, 0.0026215180600122068, 0.002597845814466621, 0.0025740962081962533, 0.0025502760879849754, 0.00252639287738533, 0.002502440822293173, 0.0024784135663443014, 0.002454318391950843, 0.0024301632193235606, 0.0024059418520241804, 0.0023816474255563412, 0.0023572877027127643, 0.002332871154882404, 0.0023083910945681076, 0.002283840089667744, 0.0022592264352261303, 0.0022345592175454193, 0.002209831201507626, 0.0021850343177344657, 0.0021601774540208397, 0.0021352703866189507, 0.0021103052666933977, 0.0020852733053521146, 0.002060184054186014, 0.002035048068429977, 0.002009856807725957, 0.0019846006657902758, 0.00195928994206817, 0.0019339360773296312, 0.0019085297473711007, 0.0018830604109507608, 0.0018575392156482725, 0.0018319786165107877, 0.0018063683949691585, 0.00178069693216468, 0.0017549763445423654, 0.0017292202586409347, 0.0017034174279280823, 0.0016775549808421297, 0.001651646149530077, 0.00162570592631277, 0.0015997218734367048, 0.001573679648994387, 0.0015475937814620998, 0.0015214808723062155, 0.0014953270906152233, 0.0014691163496581476, 0.0014428646992844813, 0.0014165906596466105, 0.0013902787534589855, 0.0013639107972643685, 0.0013375046467437846, 0.0013110811414257816, 0.0012846228351825408, 0.0012581089880279953, 0.0012315596269869729, 0.001204998440311574, 0.001178405595046257, 0.0011517571804948782, 0.0011250758736229627, 0.001098388927592197, 0.001071673569688066, 0.0010449018765211608, 0.0010180998154806183, 0.0009912992014234517, 0.0009644735729341865, 0.0009375898032692254, 0.000910678029455875, 0.0008837760635201092, 0.0008568527124191105, 0.0008298678975834106, 0.0008028571692830543, 0.0007758664924465403, 0.0007488584417971681, 0.000721783296218569, 0.0006946838416533995, 0.0006676176085676541, 0.000640538694936538, 0.0006133833418229547, 0.0005862043554614876, 0.000559076615802472, 0.0005319422306224783, 0.0005047156371895406, 0.0004774641248931814, 0.0004502906679796702, 0.00042311960183722124, 0.00039582827667671103, 0.00036850595632271756, 0.00034130643330288764, 0.0003141263938549482, 0.0002867709233374423, 0.0002593637611755845, 0.0002321680227330003, 0.00020503768373677231, 0.00017760205102980336, 0.00015002890258022111, 0.00012290072040175952, 9.605899747384604e-05, 6.850298129895686e-05, 4.012227180096977e-05, 1.4313385518582305e-05],

}

kronrod_points = {
#    2: [-0.9258200997725493, -0.5773502691896243, 4.440892098500626e-16, 0.5773502691896262, 0.9258200997725514],
#    3: [-0.9604912687080189, -0.7745966692414805, -0.4342437493468012, 5.551115123125783e-16, 0.43424374934680354, 0.7745966692414835, 0.9604912687080203],
#    4: [-0.9765602507375716, -0.8611363115940519, -0.6402862174963088, -0.3399810435848555, -3.3306690738754696e-16, 0.3399810435848566, 0.6402862174963104,
#        0.8611363115940526, 0.9765602507375731
#    ],
#    5: [-0.9840853600948419, -0.9061798459386637, -0.7541667265708493, -0.5384693101056838, -0.2796304131617827, -5.551115123125783e-16, 0.2796304131617834,
#        0.5384693101056831, 0.7541667265708492, 0.9061798459386639, 0.9840853600948425
#    ],
#    6: [-0.9887032026126787, -0.9324695142031516, -0.821373340865027, -0.6612093864662647, -0.4631182124753035, -0.23861918608319632, -
#        2.220446049250313e-16, 0.23861918608319654, 0.4631182124753045, 0.6612093864662645, 0.821373340865028, 0.9324695142031519, 0.9887032026126789
#    ],
#    7: [-0.9914553711208121, -0.949107912342757, -0.8648644233597671, -0.7415311855993938, -0.5860872354676903, -0.40584515137739663, -0.20778495500789784,
#        3.3306690738754696e-16, 0.20778495500789873, 0.40584515137739796, 0.5860872354676913, 0.7415311855993948, 0.8648644233597691,
#        0.9491079123427586, 0.9914553711208126
#    ],
#    8: [-0.9933798758817153, -0.9602898564975361, -0.8941209068474557, -0.7966664774136257, -0.6723540709451575, -0.5255324099163289, -0.3607010979281312, -
#        0.18343464249564911, 4.440892098500626e-16, 0.1834346424956501, 0.3607010979281319, 0.5255324099163294, 0.6723540709451585, 0.7966664774136267,
#        0.8941209068474564, 0.9602898564975363, 0.9933798758817162
#    ],
#    9: [-0.9946781606773386, -0.9681602395076251, -0.9149635072496775, -0.8360311073266354, -0.7344867651839325, -0.6133714327005891, -0.475462479112459, -
#        0.3242534234038089, -0.16422356361498636, 3.3306690738754696e-16, 0.16422356361498702, 0.3242534234038096, 0.4754624791124603,
#        0.613371432700591, 0.7344867651839341, 0.8360311073266359, 0.914963507249678, 0.9681602395076261, 0.9946781606773403
#    ],
    10: [-0.9956571630258076, -0.9739065285171714, -0.9301574913557086, -0.8650633666889832, -0.7808177265864168, -0.679409568299023, -0.562757134668605, -
        0.4333953941292482, -0.29439286270146015, -0.14887433898163127, 2.220446049250313e-16, 0.1488743389816316, 0.2943928627014607,
        0.43339539412924755, 0.5627571346686051, 0.6794095682990244, 0.7808177265864169, 0.8650633666889844, 0.9301574913557082, 0.9739065285171717,
        0.9956571630258081
    ],
#    11: [-0.9963696138895425, -0.9782286581460574, -0.9416771085780676, -0.8870625997680958, -0.8160574566562212, -0.7301520055740492, -0.6305995201619655, -
#        0.5190961292068118, -0.39794414095237773, -0.26954315595234524, -0.13611300079936217, -4.440892098500626e-16, 0.13611300079936206,
#        0.2695431559523449, 0.39794414095237773, 0.5190961292068117, 0.6305995201619654, 0.7301520055740492, 0.816057456656221, 0.8870625997680953,
#        0.9416771085780679, 0.978228658146057, 0.9963696138895426
#    ],
#    12: [-0.9969339225295953, -0.9815606342467201, -0.950537795943121, -0.9041172563704741, -0.8435581241611524, -0.7699026741943046, -0.6840598954700551, -
#        0.5873179542866177, -0.4813394504781564, -0.3678314989981798, -0.24850574832046946, -0.12523340851146914, -3.3306690738754696e-16,
#        0.1252334085114689, 0.24850574832046957, 0.36783149899818013, 0.4813394504781572, 0.5873179542866174, 0.6840598954700562, 0.7699026741943048,
#        0.8435581241611532, 0.9041172563704747, 0.9505377959431213, 0.9815606342467192, 0.9969339225295955
#    ],
#    13: [-0.9973661769948244, -0.984183054718588, -0.95755246838608, -0.9175983992229771, -0.8653331602663435, -0.8015780907333098, -0.7269488493206314, -
#        0.64234933944034, -0.5490799579565359, -0.4484927510364468, -0.34183246302180625, -0.23045831595513466, -0.11597108974493386, -
#        3.3306690738754696e-16, 0.11597108974493409, 0.2304583159551351, 0.3418324630218067, 0.44849275103644715, 0.5490799579565372,
#        0.6423493394403403, 0.726948849320632, 0.80157809073331, 0.8653331602663444, 0.917598399222978, 0.9575524683860812, 0.9841830547185881,
#        0.9973661769948249
#    ],
#    14: [-0.9977205937565432, -0.9862838086968115, -0.9631583382788527, -0.9284348836635736, -0.8829146632520561, -0.8272013150697641, -0.7617567525622055, -
#        0.6872929048116849, -0.6047893659409216, -0.5152486363581534, -0.41965589764297806, -0.31911236892788875, -0.21483591853348383, -
#        0.108054948707343, 5.551115123125783e-16, 0.10805494870734478, 0.21483591853348527, 0.3191123689278901, 0.41965589764297917, 0.5152486363581541,
#        0.6047893659409218, 0.6872929048116857, 0.7617567525622057, 0.8272013150697651, 0.8829146632520571, 0.9284348836635736, 0.9631583382788532,
#        0.9862838086968123, 0.9977205937565431
#    ],
#    15: [-0.9980022986933965, -0.9879925180204847, -0.9677390756791384, -0.937273392400706, -0.8972645323440815, -0.8482065834104267, -0.7904185014424661, -
#        0.7244177313601698, -0.6509967412974171, -0.5709721726085399, -0.48508186364023964, -0.3941513470775635, -0.2991800071531687, -
#        0.20119409399743393, -0.1011420669187173, 1.1102230246251565e-16, 0.10114206691871763, 0.20119409399743426, 0.2991800071531686,
#        0.3941513470775634, 0.48508186364023986, 0.570972172608539, 0.6509967412974169, 0.7244177313601701, 0.7904185014424661, 0.8482065834104272,
#        0.897264532344082, 0.937273392400706, 0.9677390756791392, 0.9879925180204855, 0.9980022986933971
#    ],
#    16: [-0.9982392741454447, -0.9894009349916497, -0.9715059509693929, -0.9445750230732326, -0.9091576670123429, -0.8656312023878314, -0.8142402870624446, -
#        0.7554044083550029, -0.6897411066817623, -0.6178762444026438, -0.5404076763521388, -0.45801677765722726, -0.37148378087841594, -
#        0.28160355077925914, -0.18916857901808393, -0.09501250983763732, -2.220446049250313e-16, 0.0950125098376372, 0.18916857901808315,
#        0.2816035507792588, 0.3714837808784164, 0.45801677765722737, 0.5404076763521395, 0.6178762444026438, 0.6897411066817624, 0.7554044083550031,
#        0.8142402870624446, 0.8656312023878316, 0.9091576670123429, 0.9445750230732325, 0.9715059509693925, 0.9894009349916499, 0.9982392741454444
#    ],
#    17: [-0.998432970606058, -0.9905754753144171, -0.9746592569674308, -0.9506755217687672, -0.9190961368038919, -0.8802391537269854, -0.8342740928501335, -
#        0.7815140038968009, -0.722472287372409, -0.6576711592166902, -0.587569212334035, -0.5126905370864765, -0.43368729520979854, -0.3512317634538762, -
#        0.2659465074516818, -0.17848418149584755, -0.08958563942522657, 0.0, 0.0895856394252269, 0.17848418149584822, 0.26594650745168247,
#        0.3512317634538764, 0.43368729520979965, 0.5126905370864772, 0.5875692123340355, 0.6576711592166908, 0.7224722873724099, 0.7815140038968015,
#        0.8342740928501343, 0.8802391537269859, 0.9190961368038917, 0.9506755217687678, 0.974659256967431, 0.9905754753144174, 0.9984329706060581
#    ],
#    18: [-0.998599165412682, -0.9915651684209306, -0.9773103807748602, -0.9558239495713969, -0.927504501573343, -0.8926024664975549, -0.8512493734337165, -
#        0.8037049589725229, -0.7503804817033572, -0.6916870430603526, -0.6280037548638335, -0.5597708310739471, -0.48750904258507766, -
#        0.4117511614628424, -0.3330234227914082, -0.2518862256915051, -0.16893682806654864, -0.08477501304173507, 0.0, 0.0847750130417354,
#        0.1689368280665493, 0.2518862256915053, 0.33302342279140795, 0.41175116146284263, 0.48750904258507877, 0.5597708310739478, 0.6280037548638338,
#        0.6916870430603532, 0.7503804817033577, 0.803704958972523, 0.8512493734337176, 0.8926024664975558, 0.9275045015733433, 0.9558239495713977,
#        0.977310380774861, 0.9915651684209309, 0.9985991654126828
#    ],
#    19: [-0.9987380120803147, -0.9924068438435842, -0.9795723007892643, -0.9602081521348296, -0.9346640001655193, -0.903155903614818, -0.865773542685335, -
#        0.822714656537143, -0.7743288590095345, -0.7209661773352292, -0.6629230657449079, -0.6005453046616815, -0.5342756325305535, -
#        0.46457074137596144, -0.3918512308715205, -0.3165640999636299, -0.23922492831824171, -0.16035864564022484, -0.0804519227440117, -
#        3.3306690738754696e-16, 0.08045192274401136, 0.16035864564022517, 0.23922492831824105, 0.31656409996363, 0.3918512308715202, 0.464570741375961,
#        0.5342756325305533, 0.6005453046616811, 0.6629230657449078, 0.7209661773352294, 0.7743288590095347, 0.8227146565371429, 0.8657735426853352,
#        0.9031559036148178, 0.9346640001655193, 0.9602081521348299, 0.9795723007892634, 0.9924068438435844, 0.9987380120803145
#    ],
#         87: [-0.9999373393176647, -0.999622350432484, -0.998981834567236, -0.9980107215629824, -0.9967203521912877, -0.9951134581379919, -0.9931850143689511, -0.9909336501109647, -0.9883633935160456, -0.985476606068421, -0.9822719826624027, -0.978749341003817, -0.9749114465404709, -0.9707605197472842, -0.9662966427216388, -0.961520436936073, -0.9564344462027763, -0.9510410013893145, -0.945341012434413, -0.9393357199742063, -0.9330277577952324, -0.9264196798912376, -0.9195130155009865, -0.9123095290941232, -0.9048120659287133, -0.8970234547675697, -0.8889457425315686, -0.8805811598520423, -0.8719328161808033, -0.8630038376297071, -0.8537967282834917, -0.8443141443970357, -0.8345594815785893, -0.8245361702557805, -0.8142471292361737, -0.8036954080581786, -0.7928846943811525, -0.7818187205454386, -0.7705007861856665, -0.7589343057213496, -0.7471232552715923, -0.735071660505314, -0.7227831721000328, -0.7102615427114733, -0.6975110271436125, -0.6845359314028241, -0.6713402293006723, -0.6579279856713498, -0.6443037198655162, -0.6304720018972505, -0.6164371014898609, -0.6022033695423357, -0.5877755725413023, -0.5731585255345304, -0.5583567669734248, -0.5433749073022942, -0.5282179401678413, -0.5128909045194527, -0.49739858004223114, -0.4817458096226672, -0.4659377920214407, -0.449979767165285, -0.43387672799250576, -0.417633722072869, -0.4012561295371335, -0.3847493668670974, -0.3681186117187589, -0.3513690879260418, -0.33450633184746614, -0.3175359108501159, -0.30046315821599157, -0.28329344500409226, -0.266032437514764, -0.24868582730667765, -0.2312590736832678, -0.21375766534010188, -0.19618737131686093, -0.17855397985304466, -0.16086304622927927, -0.14312014673100815, -0.12533112522788015, -0.10750183850894302, -0.08963790744155298, -0.07174496550149989, -0.05382890297085785, -0.03589561662680141, -0.017950762293751588, 0.0, 0.017950762293751477, 0.03589561662680074, 0.05382890297085752, 0.071744965501499, 0.08963790744155298, 0.10750183850894268, 0.1253311252278798, 0.14312014673100815, 0.16086304622927938, 0.17855397985304444, 0.19618737131686126, 0.21375766534010154, 0.23125907368326792, 0.2486858273066771, 0.2660324375147639, 0.28329344500409204, 0.30046315821599134, 0.3175359108501157, 0.3345063318474658, 0.3513690879260414, 0.3681186117187585, 0.38474936686709693, 0.4012561295371332, 0.4176337220728691, 0.4338767279925052, 0.4499797671652843, 0.46593779202144003, 0.4817458096226668, 0.49739858004223125, 0.5128909045194523, 0.5282179401678414, 0.5433749073022933, 0.5583567669734248, 0.5731585255345304, 0.5877755725413025, 0.6022033695423353, 0.6164371014898609, 0.6304720018972505, 0.6443037198655157, 0.6579279856713494, 0.671340229300672, 0.6845359314028239, 0.6975110271436127, 0.7102615427114729, 0.7227831721000324, 0.7350716605053138, 0.7471232552715918, 0.7589343057213493, 0.7705007861856658, 0.7818187205454378, 0.7928846943811524, 0.8036954080581777, 0.8142471292361735, 0.8245361702557805, 0.8345594815785895, 0.8443141443970352, 0.8537967282834915, 0.8630038376297067, 0.8719328161808031, 0.8805811598520419, 0.8889457425315681, 0.8970234547675693, 0.9048120659287131, 0.9123095290941229, 0.9195130155009866, 0.9264196798912372, 0.933027757795232, 0.9393357199742057, 0.9453410124344126, 0.9510410013893136, 0.9564344462027765, 0.9615204369360731, 0.9662966427216388, 0.9707605197472843, 0.9749114465404707, 0.9787493410038163, 0.9822719826624021, 0.9854766060684204, 0.988363393516045, 0.9909336501109646, 0.9931850143689511, 0.995113458137992, 0.9967203521912877, 0.9980107215629823, 0.998981834567236, 0.9996223504324843, 0.9999373393176647],
#         300: [-0.9999946872996457, -0.9999679782184366, -0.9999136584901585, -0.9998312829844194, -0.9997217884834974, -0.9995853734488863, -0.9994215723741114, -0.9992302218632736, -0.9990116062590783, -0.9987658603530949, -0.9984928023779717, -0.9981923380734435, -0.9978646126899144, -0.9975097171758793, -0.9971275560703853, -0.996718072051, -0.9962813600745217, -0.9958174891208241, -0.9953264031603122, -0.9948080667630033, -0.9942625517027517, -0.9936899152742682, -0.9930901239339065, -0.9924631568471005, -0.9918090737598455, -0.9911279255509896, -0.990419693459469, -0.9896843673150096, -0.9889220003217568, -0.9881326399103605, -0.9873162780530513, -0.9864729129322606, -0.9856025942467834, -0.9847053677808956, -0.9837812338860811, -0.9828301976412723, -0.9818523070673794, -0.9808476074619001, -0.9798161060814252, -0.9787578139322586, -0.9776727785275657, -0.9765610454590224, -0.9754226279104833, -0.9742575421410374, -0.9730658359415845, -0.9718475557430342, -0.9706027199771077, -0.9693313496680812, -0.9680334934315086, -0.9667091989287729, -0.9653584893488589, -0.9639813901171553, -0.9625779510657028, -0.9611482213733913, -0.9596922286202931, -0.9582100023531643, -0.9567015939075111, -0.9551670541938286, -0.9536064149019075, -0.9520197094793031, -0.9504069909786793, -0.9487683122037178, -0.9471037087320028, -0.9454132177338136, -0.9436968941399522, -0.9419547927700981, -0.9401869529103559, -0.9383934153067652, -0.9365742368903639, -0.9347294745903799, -0.9328591712533739, -0.9309633710773335, -0.929042133086303, -0.9270955163900743, -0.9251235672708156, -0.9231263332721165, -0.9211038755812251, -0.9190562555418369, -0.9169835227643917, -0.9148857280450582, -0.9127629347867887, -0.9106152066064295, -0.9084425963487075, -0.9062451579796027, -0.9040229571561673, -0.901776059796223, -0.8995045218950815, -0.8972084005137191, -0.894887763590253, -0.8925426793619207, -0.8901732068988839, -0.8877794062884896, -0.8853613477675033, -0.8829191019031915, -0.8804527307710444, -0.8779622974209855, -0.8754478743981893, -0.8729095346039717, -0.8703473430544713, -0.8677613657021737, -0.8651516774038297, -0.8625183533931811, -0.8598614615661291, -0.8571810707206511, -0.8544772580226088, -0.8517501010316768, -0.848999670465588, -0.846226037913026, -0.8434292828416308, -0.8406094851262689, -0.8377667182508701, -0.8349010565417854, -0.832012581756866, -0.8291013760716741, -0.8261675156824684, -0.8232110776015575, -0.8202321458616895, -0.8172308049213011, -0.8142071336364427, -0.8111612116546523, -0.8080931252649417, -0.8050029611877987, -0.8018908008875274, -0.7987567265968607, -0.7956008268394583, -0.7924231905743322, -0.7892239018232119, -0.786003045354458, -0.7827607119020659, -0.7794969926375797, -0.7762119740898052, -0.7729057435134399, -0.7695783938260492, -0.7662300183834612, -0.7628607061714925, -0.7594705468820133, -0.7560596355871452, -0.752628067796661, -0.7491759349034655, -0.7457033289874226, -0.7422103472441275, -0.7386970873050263, -0.7351636429201902, -0.7316101085081914, -0.7280365833550897, -0.7244431671799338, -0.7208299560399418, -0.7171970466429054, -0.7135445403305545, -0.709872538873781, -0.706181140586744, -0.7024704444166954, -0.6987405537245666, -0.69499157229575, -0.6912236006508845, -0.6874367399265875, -0.6836310954649563, -0.679806773027049, -0.675963875289201, -0.6721025055269298, -0.6682227710239814, -0.6643247794768312, -0.6604086356663699, -0.6564744449561277, -0.6525223165305882, -0.6485523599800562, -0.6445646821384352, -0.640559390405937, -0.6365365958255456, -0.632496409838541, -0.6284389412798634, -0.6243642995346044, -0.6202725974607572, -0.6161639483065122, -0.612038462855417, -0.6078962524251541, -0.6037374316440418, -0.5995621155219637, -0.5953704167381737, -0.591162448490157, -0.5869383271307362, -0.5826981693851709, -0.5784420897750377, -0.5741702033243279, -0.569882628063465, -0.5655794823857194, -0.5612608826011143, -0.5569269455063359, -0.552577790761474, -0.5482135383794409, -0.5438343064043345, -0.539440213351214, -0.5350313804609139, -0.5306079293166595, -0.5261699796417558, -0.5217176516148028, -0.5172510680075144, -0.512770351923185, -0.5082756247089657, -0.5037670081516556, -0.4992446265030859, -0.4947086043354987, -0.49015906456404057, -0.48559613052787265, -0.4810199279073166, -0.47643058269160643, -0.4718282193075505, -0.4672129625903425, -0.4625849395963533, -0.4579442776790468, -0.45329110272009054, -0.4486255409938984, -0.4439477208796667, -0.43925777104156594, -0.4345558187588534, -0.4298419916878913, -0.42511641947672607, -0.42037923204598227, -0.415630558014791, -0.4108705263637342, -0.4060992679550248, -0.40131691391079194, -0.3965235941318921, -0.39171943886495786, -0.38690458013101003, -0.3820791501980705, -0.3772432801901535, -0.37239710156135253, -0.36754074743549325, -0.36267435117025193, -0.3577980450538263, -0.35291196168878125, -0.3480162352451315, -0.34311100011337237, -0.338196389686529, -0.3332725376562593, -0.3283395791815773, -0.323397649628395, -0.31844688343484195, -0.31348741532192625, -0.30851938137992496, -0.3035429178922302, -0.29855816028200666, -0.29356524423952535, -0.28856430672807254, -0.28355548489008364, -0.27853891507336437, -0.27351473387703384, -0.26848307907865343, -0.2634440886207914, -0.258397899715193, -0.25334464980911076, -0.24828447743519133, -0.2432175212767862, -0.23814391934859436, -0.23306380988500508, -0.22797733211414206, -0.22288462540037246, -0.21778582850010708, -0.21268108037361766, -0.20757052088450112, -0.2024542900179942, -0.1973325272107358, -0.19220537208739352, -0.18707296508667115, -0.1819354467541754, -0.17679295714507726, -0.17164563648674025, -0.16649362573227822, -0.1613370659268425, -0.15617609768225016, -0.15101086176667544, -0.14584149958664194, -0.1406681526257254, -0.13549096199033794, -0.13031006892741814, -0.12512561523562193, -0.1199377427755659, -0.11474659308606205, -0.10955230783063419, -0.10435502913854622, -0.09915489918583553, -0.09395205988140298, -0.08874665324306696, -0.0835388216689601, -0.07832870758870736, -0.07311645321890214, -0.06790220086927956, -0.06268609314491269, -0.05746827266700372, -0.0522488818973752, -0.04702806337524312, -0.04180595985053315, -0.03658271407386193, -0.03135846868977743, -0.026133366404510694, -0.020907550050621282, -0.01568116244584883, -0.010454346354956456, -0.0052272445887175945, 0.0, 0.0052272445887177055, 0.0104543463549569, 0.01568116244584905, 0.020907550050621393, 0.026133366404511027, 0.03135846868977776, 0.036582714073862044, 0.04180595985053359, 0.04702806337524368, 0.052248881897375865, 0.057468272667004605, 0.06268609314491314, 0.0679022008692799, 0.07311645321890248, 0.07832870758870769, 0.08353882166896032, 0.08874665324306752, 0.09395205988140376, 0.09915489918583587, 0.10435502913854666, 0.10955230783063508, 0.1147465930860625, 0.11993774277556657, 0.12512561523562216, 0.13031006892741892, 0.13549096199033805, 0.14066815262572618, 0.14584149958664216, 0.1510108617666761, 0.15617609768225016, 0.16133706592684272, 0.16649362573227855, 0.17164563648674003, 0.17679295714507792, 0.18193544675417572, 0.18707296508667137, 0.1922053720873934, 0.19733252721073624, 0.2024542900179942, 0.20757052088450112, 0.21268108037361744, 0.21778582850010697, 0.22288462540037235, 0.22797733211414206, 0.23306380988500508, 0.23814391934859414, 0.2432175212767861, 0.24828447743519155, 0.25334464980911076, 0.2583978997151932, 0.2634440886207915, 0.2684830790786531, 0.27351473387703407, 0.27853891507336426, 0.2835554848900833, 0.2885643067280719, 0.29356524423952524, 0.2985581602820062, 0.30354291789222965, 0.30851938137992485, 0.3134874153219265, 0.31844688343484184, 0.323397649628395, 0.3283395791815772, 0.33327253765625897, 0.33819638968652843, 0.3431110001133719, 0.34801623524513126, 0.3529119616887808, 0.3577980450538262, 0.3626743511702518, 0.36754074743549336, 0.37239710156135264, 0.3772432801901534, 0.38207915019807037, 0.3869045801310097, 0.39171943886495786, 0.3965235941318921, 0.4013169139107917, 0.4060992679550246, 0.410870526363734, 0.4156305580147909, 0.4203792320459817, 0.42511641947672574, 0.42984199168789106, 0.43455581875885274, 0.4392577710415656, 0.44394772087966616, 0.44862554099389795, 0.45329110272008966, 0.45794427767904666, 0.46258493959635283, 0.4672129625903423, 0.4718282193075505, 0.4764305826916061, 0.4810199279073162, 0.4855961305278721, 0.49015906456404, 0.4947086043354981, 0.49924462650308543, 0.5037670081516556, 0.508275624708965, 0.5127703519231843, 0.517251068007514, 0.5217176516148028, 0.5261699796417558, 0.5306079293166588, 0.5350313804609135, 0.5394402133512137, 0.5438343064043343, 0.5482135383794404, 0.5525777907614735, 0.5569269455063353, 0.5612608826011143, 0.5655794823857192, 0.5698826280634648, 0.5741702033243274, 0.578442089775038, 0.5826981693851705, 0.5869383271307357, 0.5911624484901564, 0.5953704167381735, 0.5995621155219635, 0.6037374316440416, 0.6078962524251535, 0.6120384628554165, 0.6161639483065119, 0.6202725974607567, 0.6243642995346039, 0.6284389412798628, 0.6324964098385406, 0.6365365958255453, 0.6405593904059368, 0.6445646821384348, 0.6485523599800556, 0.6525223165305876, 0.6564744449561274, 0.6604086356663694, 0.6643247794768309, 0.6682227710239813, 0.6721025055269294, 0.6759638752892005, 0.6798067730270486, 0.6836310954649557, 0.687436739926587, 0.6912236006508838, 0.69499157229575, 0.6987405537245661, 0.7024704444166952, 0.706181140586744, 0.7098725388737805, 0.713544540330554, 0.7171970466429054, 0.7208299560399416, 0.7244431671799337, 0.7280365833550895, 0.7316101085081913, 0.7351636429201899, 0.7386970873050263, 0.742210347244127, 0.7457033289874225, 0.7491759349034652, 0.752628067796661, 0.7560596355871447, 0.7594705468820128, 0.7628607061714925, 0.7662300183834609, 0.7695783938260488, 0.7729057435134394, 0.7762119740898051, 0.77949699263758, 0.7827607119020656, 0.7860030453544584, 0.7892239018232124, 0.7924231905743324, 0.7956008268394587, 0.7987567265968609, 0.8018908008875274, 0.8050029611877982, 0.8080931252649417, 0.8111612116546523, 0.8142071336364427, 0.8172308049213008, 0.8202321458616895, 0.8232110776015571, 0.826167515682468, 0.8291013760716742, 0.8320125817568655, 0.8349010565417858, 0.8377667182508699, 0.8406094851262688, 0.8434292828416307, 0.846226037913026, 0.848999670465588, 0.8517501010316768, 0.8544772580226087, 0.8571810707206517, 0.859861461566129, 0.8625183533931813, 0.8651516774038295, 0.8677613657021737, 0.870347343054471, 0.8729095346039718, 0.8754478743981893, 0.8779622974209857, 0.8804527307710446, 0.8829191019031915, 0.8853613477675031, 0.88777940628849, 0.890173206898884, 0.8925426793619202, 0.8948877635902533, 0.8972084005137192, 0.8995045218950817, 0.9017760597962232, 0.9040229571561679, 0.9062451579796031, 0.9084425963487072, 0.9106152066064294, 0.9127629347867888, 0.9148857280450583, 0.9169835227643919, 0.9190562555418367, 0.9211038755812248, 0.9231263332721162, 0.9251235672708154, 0.927095516390074, 0.9290421330863031, 0.9309633710773336, 0.932859171253374, 0.93472947459038, 0.9365742368903638, 0.9383934153067653, 0.940186952910356, 0.9419547927700982, 0.9436968941399521, 0.9454132177338137, 0.9471037087320029, 0.948768312203718, 0.9504069909786793, 0.9520197094793031, 0.9536064149019075, 0.9551670541938287, 0.9567015939075112, 0.9582100023531646, 0.9596922286202932, 0.9611482213733913, 0.9625779510657029, 0.9639813901171553, 0.965358489348859, 0.9667091989287728, 0.9680334934315088, 0.9693313496680812, 0.9706027199771078, 0.9718475557430342, 0.9730658359415845, 0.9742575421410374, 0.9754226279104833, 0.9765610454590226, 0.9776727785275656, 0.9787578139322587, 0.9798161060814253, 0.9808476074619004, 0.9818523070673794, 0.9828301976412724, 0.9837812338860812, 0.9847053677808958, 0.9856025942467834, 0.9864729129322606, 0.9873162780530514, 0.9881326399103604, 0.9889220003217569, 0.9896843673150097, 0.9904196934594691, 0.9911279255509898, 0.9918090737598456, 0.9924631568471006, 0.9930901239339065, 0.9936899152742682, 0.9942625517027519, 0.9948080667630034, 0.9953264031603122, 0.9958174891208242, 0.9962813600745218, 0.9967180720510002, 0.9971275560703854, 0.9975097171758793, 0.9978646126899146, 0.9981923380734435, 0.9984928023779718, 0.998765860353095, 0.9990116062590781, 0.9992302218632736, 0.9994215723741117, 0.9995853734488863, 0.9997217884834975, 0.9998312829844194, 0.9999136584901583, 0.9999679782184367, 0.9999946872996458],

}

legendre_weights = {
#    2: [1.0, 1.0],
#    3: [0.5555555555555557, 0.8888888888888888, 0.5555555555555557],
#    4: [0.3478548451374537, 0.6521451548625462, 0.6521451548625462, 0.3478548451374537],
#    5: [0.23692688505618942, 0.4786286704993662, 0.568888888888889, 0.4786286704993662, 0.23692688505618942],
#    6: [0.17132449237916975, 0.36076157304813894, 0.46791393457269137, 0.46791393457269137, 0.36076157304813894, 0.17132449237916975],
#    7: [0.12948496616887065, 0.2797053914892766, 0.3818300505051183, 0.41795918367346896, 0.3818300505051183, 0.2797053914892766, 0.12948496616887065],
#    8: [0.10122853629037669, 0.22238103445337434, 0.31370664587788705, 0.36268378337836177, 0.36268378337836177, 0.31370664587788705, 0.22238103445337434,
#        0.10122853629037669
#    ],
#    9: [0.08127438836157472, 0.18064816069485712, 0.26061069640293566, 0.3123470770400028, 0.33023935500125967, 0.3123470770400028, 0.26061069640293566,
#        0.18064816069485712, 0.08127438836157472
#    ],
    10: [0.06667134430868807, 0.14945134915058036, 0.219086362515982, 0.2692667193099965, 0.295524224714753, 0.295524224714753, 0.2692667193099965,
        0.219086362515982, 0.14945134915058036, 0.06667134430868807
    ],
#    11: [0.055668567116173164, 0.1255803694649047, 0.18629021092773443, 0.23319376459199068, 0.26280454451024676, 0.2729250867779009, 0.26280454451024676,
#        0.23319376459199068, 0.18629021092773443, 0.1255803694649047, 0.055668567116173164
#    ],
#    12: [0.04717533638651202, 0.10693932599531888, 0.1600783285433461, 0.20316742672306565, 0.23349253653835464, 0.2491470458134027, 0.2491470458134027,
#        0.23349253653835464, 0.20316742672306565, 0.1600783285433461, 0.10693932599531888, 0.04717533638651202
#    ],
#    13: [0.04048400476531588, 0.0921214998377286, 0.13887351021978736, 0.17814598076194552, 0.20781604753688857, 0.22628318026289715, 0.2325515532308739,
#        0.22628318026289715, 0.20781604753688857, 0.17814598076194552, 0.13887351021978736, 0.0921214998377286, 0.04048400476531588
#    ],
#    14: [0.035119460331752374, 0.0801580871597603, 0.12151857068790296, 0.1572031671581934, 0.18553839747793763, 0.20519846372129555, 0.21526385346315766,
#        0.21526385346315766, 0.20519846372129555, 0.18553839747793763, 0.1572031671581934, 0.12151857068790296, 0.0801580871597603,
#        0.035119460331752374
#    ],
#    15: [0.030753241996118647, 0.07036604748810807, 0.10715922046717177, 0.1395706779261539, 0.16626920581699378, 0.18616100001556188, 0.19843148532711125,
#        0.2025782419255609, 0.19843148532711125, 0.18616100001556188, 0.16626920581699378, 0.1395706779261539, 0.10715922046717177, 0.07036604748810807,
#        0.030753241996118647
#    ],
#    16: [0.027152459411754037, 0.062253523938647706, 0.09515851168249259, 0.12462897125553403, 0.14959598881657676, 0.16915651939500262, 0.1826034150449236,
#        0.18945061045506859, 0.18945061045506859, 0.1826034150449236, 0.16915651939500262, 0.14959598881657676, 0.12462897125553403,
#        0.09515851168249259, 0.062253523938647706, 0.027152459411754037
#    ],
#    17: [0.02414830286854952, 0.0554595293739866, 0.08503614831717908, 0.11188384719340365, 0.13513636846852523, 0.15404576107681012, 0.16800410215644995,
#        0.17656270536699253, 0.17944647035620653, 0.17656270536699253, 0.16800410215644995, 0.15404576107681012, 0.13513636846852523,
#        0.11188384719340365, 0.08503614831717908, 0.0554595293739866, 0.02414830286854952
#    ],
#    18: [0.02161601352648413, 0.04971454889496922, 0.07642573025488925, 0.10094204410628699, 0.12255520671147836, 0.14064291467065063, 0.15468467512626521,
#        0.16427648374583273, 0.16914238296314363, 0.16914238296314363, 0.16427648374583273, 0.15468467512626521, 0.14064291467065063,
#        0.12255520671147836, 0.10094204410628699, 0.07642573025488925, 0.04971454889496922, 0.02161601352648413
#    ],
#    19: [0.01946178822972761, 0.04481422676569981, 0.06904454273764107, 0.09149002162244985, 0.11156664554733375, 0.1287539625393362, 0.14260670217360638,
#        0.15276604206585945, 0.15896884339395415, 0.16105444984878345, 0.15896884339395415, 0.15276604206585945, 0.14260670217360638,
#        0.1287539625393362, 0.11156664554733375, 0.09149002162244985, 0.06904454273764107, 0.04481422676569981, 0.01946178822972761
#    ],
#         87: [0.0009691097381757348, 0.0022546907537549384, 0.003539271655388117, 0.004819456238501684, 0.00609346204763528, 0.007359623648817653, 0.008616302838488783, 0.009861877713702014, 0.011094741940560695, 0.012313306030048123, 0.013515999118245947, 0.0147012708872398, 0.015867593518826113, 0.017013463643001287, 0.0181374042653544, 0.01923796666535654, 0.020313732260655384, 0.021363314433802724, 0.022385360318485367, 0.02337855254266003, 0.024341610926167583, 0.025273294130557043, 0.026172401258933563, 0.027037773403735973, 0.02786829514042919, 0.028662895965176235, 0.02942055167462312, 0.03014028568601895, 0.030821170295962166, 0.031462327876150824, 0.032062932004589706, 0.03262220853080143, 0.03313943657366203, 0.03361394945057692, 0.034045135536799345, 0.03443243905378225, 0.03477536078554787, 0.03507345872215154, 0.03532634862941022, 0.03553370454416061, 0.035695259194409426, 0.03581080434383377, 0.03588019106018702, 0.03590332990726553, 0.03588019106018702, 0.03581080434383377, 0.035695259194409426, 0.03553370454416061, 0.03532634862941022, 0.03507345872215154, 0.03477536078554787, 0.03443243905378225, 0.034045135536799345, 0.03361394945057692, 0.03313943657366203, 0.03262220853080143, 0.032062932004589706, 0.031462327876150824, 0.030821170295962166, 0.03014028568601895, 0.02942055167462312, 0.028662895965176235, 0.02786829514042919, 0.027037773403735973, 0.026172401258933563, 0.025273294130557043, 0.024341610926167583, 0.02337855254266003, 0.022385360318485367, 0.021363314433802724, 0.020313732260655384, 0.01923796666535654, 0.0181374042653544, 0.017013463643001287, 0.015867593518826113, 0.0147012708872398, 0.013515999118245947, 0.012313306030048123, 0.011094741940560695, 0.009861877713702014, 0.008616302838488783, 0.007359623648817653, 0.00609346204763528, 0.004819456238501684, 0.003539271655388117, 0.0022546907537549384, 0.0009691097381757348],
#         300: [8.217779368318766e-05, 0.000191285544654933, 0.00030053396541231163, 0.0004097635734033533, 0.0005189512134602094, 0.0006280829779724268, 0.0007371464159049574, 0.0008461294290789873, 0.0009550200344351969, 0.0010638062980465963, 0.001172476313703897, 0.001281018195428937, 0.0013894200749916825, 0.0014976701014677357, 0.0016057564416445678, 0.0017136672808561744, 0.0018213908240185923, 0.0019289152967680268, 0.0020362289466670158, 0.0021433200444213973, 0.002250176885138956, 0.002356787789583519, 0.002463141105430975, 0.0025692252085409055, 0.002675028504212449, 0.002780539428452415, 0.0028857464492311412, 0.0029906380677445906, 0.0030952028196669814, 0.0031994292764019384, 0.003303306046330757, 0.003406821776058694, 0.0035099651516518333, 0.0036127248998770794, 0.0037150897894316927, 0.0038170486321696183, 0.003918590284326712, 0.004019703647735881, 0.004120377671042578, 0.004220601350909488, 0.004320363733222272, 0.0044196539142855205, 0.004518461042012614, 0.004616774317114607, 0.004714582994277626, 0.004811876383340857, 0.004908643850461695, 0.005004874819279474, 0.0051005587720714925, 0.0051956852509013034, 0.005290243858763116, 0.0053842242607182395, 0.0054776161850220466, 0.005570409424251355, 0.005662593836414576, 0.005754159346065033, 0.005845095945399296, 0.005935393695350927, 0.006025042726679271, 0.006114033241045046, 0.006202355512083302, 0.0062899998864659286, 0.00637695678495604, 0.006463216703456925, 0.006548770214047811, 0.006633607966017162, 0.006717720686882713, 0.0068010991834066194, 0.006883734342598478, 0.0069656171327117265, 0.00704673860423293, 0.007127089890856632, 0.007206662210456636, 0.0072854468660452166, 0.007363435246723478, 0.00744061882862333, 0.007516989175837821, 0.00759253794134379, 0.007667256867914979, 0.007741137789022482, 0.007814172629730136, 0.007886353407574561, 0.007957672233438363, 0.008028121312413226, 0.008097692944651201, 0.008166379526205846, 0.008234173549864183, 0.008301067605967079, 0.008367054383218206, 0.008432126669483946, 0.00849627735258214, 0.00855949942105765, 0.0086217859649506, 0.00868313017655067, 0.008743525351140993, 0.008802964887731923, 0.008861442289781127, 0.008918951165904783, 0.008975485230575371, 0.009031038304809223, 0.009085604316841996, 0.009139177302791153, 0.009191751407309061, 0.009243320884222412, 0.009293880097160322, 0.009343423520170318, 0.00939194573832249, 0.009439441448300992, 0.009485905458984029, 0.009531332692011101, 0.009575718182337876, 0.009619057078779164, 0.009661344644538864, 0.009702576257727958, 0.009742747411869234, 0.00978185371639035, 0.00981989089710335, 0.009856854796671895, 0.009892741375065664, 0.009927546710002122, 0.009961266997374775, 0.009993898551669387, 0.010025437806366347, 0.01005588131433098, 0.01008522574818994, 0.010113467900694973, 0.010140604685073598, 0.010166633135366204, 0.010191550406750467, 0.010215353775852252, 0.010238040641043068, 0.010259608522724598, 0.010280055063599757, 0.010299378028930195, 0.010317575306780569, 0.010334644908249447, 0.010350584967686661, 0.010365393742897078, 0.010379069615331337, 0.010391611090262378, 0.010403016796949039, 0.01041328548878585, 0.010422416043439223, 0.010430407462970154, 0.010437258873943282, 0.01044296952752239, 0.010447538799552137, 0.010450966190626462, 0.010453251326142981, 0.010454393956344087, 0.010454393956344087, 0.010453251326142981, 0.010450966190626462, 0.010447538799552137, 0.01044296952752239, 0.010437258873943282, 0.010430407462970154, 0.010422416043439223, 0.01041328548878585, 0.010403016796949039, 0.010391611090262378, 0.010379069615331337, 0.010365393742897078, 0.010350584967686661, 0.010334644908249447, 0.010317575306780569, 0.010299378028930195, 0.010280055063599757, 0.010259608522724598, 0.010238040641043068, 0.010215353775852252, 0.010191550406750467, 0.010166633135366204, 0.010140604685073598, 0.010113467900694973, 0.01008522574818994, 0.01005588131433098, 0.010025437806366347, 0.009993898551669387, 0.009961266997374775, 0.009927546710002122, 0.009892741375065664, 0.009856854796671895, 0.00981989089710335, 0.00978185371639035, 0.009742747411869234, 0.009702576257727958, 0.009661344644538864, 0.009619057078779164, 0.009575718182337876, 0.009531332692011101, 0.009485905458984029, 0.009439441448300992, 0.00939194573832249, 0.009343423520170318, 0.009293880097160322, 0.009243320884222412, 0.009191751407309061, 0.009139177302791153, 0.009085604316841996, 0.009031038304809223, 0.008975485230575371, 0.008918951165904783, 0.008861442289781127, 0.008802964887731923, 0.008743525351140993, 0.00868313017655067, 0.0086217859649506, 0.00855949942105765, 0.00849627735258214, 0.008432126669483946, 0.008367054383218206, 0.008301067605967079, 0.008234173549864183, 0.008166379526205846, 0.008097692944651201, 0.008028121312413226, 0.007957672233438363, 0.007886353407574561, 0.007814172629730136, 0.007741137789022482, 0.007667256867914979, 0.00759253794134379, 0.007516989175837821, 0.00744061882862333, 0.007363435246723478, 0.0072854468660452166, 0.007206662210456636, 0.007127089890856632, 0.00704673860423293, 0.0069656171327117265, 0.006883734342598478, 0.0068010991834066194, 0.006717720686882713, 0.006633607966017162, 0.006548770214047811, 0.006463216703456925, 0.00637695678495604, 0.0062899998864659286, 0.006202355512083302, 0.006114033241045046, 0.006025042726679271, 0.005935393695350927, 0.005845095945399296, 0.005754159346065033, 0.005662593836414576, 0.005570409424251355, 0.0054776161850220466, 0.0053842242607182395, 0.005290243858763116, 0.0051956852509013034, 0.0051005587720714925, 0.005004874819279474, 0.004908643850461695, 0.004811876383340857, 0.004714582994277626, 0.004616774317114607, 0.004518461042012614, 0.0044196539142855205, 0.004320363733222272, 0.004220601350909488, 0.004120377671042578, 0.004019703647735881, 0.003918590284326712, 0.0038170486321696183, 0.0037150897894316927, 0.0036127248998770794, 0.0035099651516518333, 0.003406821776058694, 0.003303306046330757, 0.0031994292764019384, 0.0030952028196669814, 0.0029906380677445906, 0.0028857464492311412, 0.002780539428452415, 0.002675028504212449, 0.0025692252085409055, 0.002463141105430975, 0.002356787789583519, 0.002250176885138956, 0.0021433200444213973, 0.0020362289466670158, 0.0019289152967680268, 0.0018213908240185923, 0.0017136672808561744, 0.0016057564416445678, 0.0014976701014677357, 0.0013894200749916825, 0.001281018195428937, 0.001172476313703897, 0.0010638062980465963, 0.0009550200344351969, 0.0008461294290789873, 0.0007371464159049574, 0.0006280829779724268, 0.0005189512134602094, 0.0004097635734033533, 0.00030053396541231163, 0.000191285544654933, 8.217779368318766e-05],
}
#legendre_points = {
#    2: [-0.5773502691896257, 0.5773502691896257],
#    3: [-0.7745966692414834, 0.0, 0.7745966692414834],
#    4: [-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526],
#    5: [-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664],
#    6: [-0.932469514203152, -0.6612093864662645, -0.23861918608319693, 0.23861918608319693, 0.6612093864662645, 0.932469514203152],
#    7: [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585],
#    8: [-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978, 0.18343464249564978, 0.525532409916329, 0.7966664774136267,
#        0.9602898564975362
#    ],
#    9: [-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0, 0.3242534234038089, 0.6133714327005904, 0.8360311073266358,
#        0.9681602395076261
#    ],
#    10: [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 0.4333953941292472,
#        0.6794095682990244, 0.8650633666889845, 0.9739065285171717
#    ],
#    11: [-0.978228658146057, -0.8870625997680953, -0.7301520055740494, -0.5190961292068118, -0.26954315595234496, 0.0, 0.26954315595234496,
#        0.5190961292068118, 0.7301520055740494, 0.8870625997680953, 0.978228658146057
#    ],
#    12: [-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689, 0.1252334085114689,
#        0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192
#    ],
#    13: [-0.9841830547185881, -0.9175983992229779, -0.8015780907333099, -0.6423493394403402, -0.4484927510364468, -0.23045831595513477, 0.0,
#        0.23045831595513477, 0.4484927510364468, 0.6423493394403402, 0.8015780907333099, 0.9175983992229779, 0.9841830547185881
#    ],
#    14: [-0.9862838086968123, -0.9284348836635735, -0.827201315069765, -0.6872929048116855, -0.5152486363581541, -0.31911236892788974, -0.10805494870734367,
#        0.10805494870734367, 0.31911236892788974, 0.5152486363581541, 0.6872929048116855, 0.827201315069765, 0.9284348836635735, 0.9862838086968123
#    ],
#    15: [-0.9879925180204854, -0.937273392400706, -0.8482065834104272, -0.7244177313601701, -0.5709721726085388, -0.3941513470775634, -0.20119409399743451,
#        0.0, 0.20119409399743451, 0.3941513470775634, 0.5709721726085388, 0.7244177313601701, 0.8482065834104272, 0.937273392400706, 0.9879925180204854
#    ],
#    16: [-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -
#        0.09501250983763745, 0.09501250983763745, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318,
#        0.9445750230732326, 0.9894009349916499
#    ],
#    17: [-0.9905754753144174, -0.9506755217687678, -0.8802391537269859, -0.7815140038968014, -0.6576711592166908, -0.5126905370864769, -0.3512317634538763, -
#        0.17848418149584785, 0.0, 0.17848418149584785, 0.3512317634538763, 0.5126905370864769, 0.6576711592166908, 0.7815140038968014,
#        0.8802391537269859, 0.9506755217687678, 0.9905754753144174
#    ],
#    18: [-0.9915651684209309, -0.9558239495713978, -0.8926024664975557, -0.8037049589725231, -0.6916870430603532, -0.5597708310739475, -0.41175116146284263, -
#        0.2518862256915055, -0.08477501304173529, 0.08477501304173529, 0.2518862256915055, 0.41175116146284263, 0.5597708310739475, 0.6916870430603532,
#        0.8037049589725231, 0.8926024664975557, 0.9558239495713978, 0.9915651684209309
#    ],
#    19: [-0.9924068438435844, -0.96020815213483, -0.9031559036148179, -0.8227146565371428, -0.7209661773352294, -0.600545304661681, -0.46457074137596094, -
#        0.31656409996362983, -0.1603586456402254, 0.0, 0.1603586456402254, 0.31656409996362983, 0.46457074137596094, 0.600545304661681,
#        0.7209661773352294, 0.8227146565371428, 0.9031559036148179, 0.96020815213483, 0.9924068438435844
#    ],
##         87: [-0.9996223504324843, -0.9980107215629823, -0.995113458137992, -0.9909336501109646, -0.9854766060684204, -0.9787493410038163, -0.9707605197472843, -0.9615204369360731, -0.9510410013893136, -0.9393357199742057, -0.9264196798912372, -0.9123095290941229, -0.8970234547675694, -0.8805811598520419, -0.8630038376297066, -0.8443141443970352, -0.8245361702557805, -0.8036954080581777, -0.7818187205454378, -0.7589343057213493, -0.7350716605053138, -0.7102615427114729, -0.6845359314028236, -0.6579279856713494, -0.6304720018972505, -0.6022033695423352, -0.5731585255345304, -0.5433749073022933, -0.5128909045194523, -0.4817458096226668, -0.44997976716528437, -0.4176337220728691, -0.3847493668670969, -0.3513690879260413, -0.31753591085011573, -0.28329344500409187, -0.24868582730667713, -0.2137576653401016, -0.17855397985304422, -0.14312014673100815, -0.10750183850894277, -0.07174496550149943, -0.035895616626800894, 0.0, 0.035895616626800894, 0.07174496550149943, 0.10750183850894277, 0.14312014673100815, 0.17855397985304422, 0.2137576653401016, 0.24868582730667713, 0.28329344500409187, 0.31753591085011573, 0.3513690879260413, 0.3847493668670969, 0.4176337220728691, 0.44997976716528437, 0.4817458096226668, 0.5128909045194523, 0.5433749073022933, 0.5731585255345304, 0.6022033695423352, 0.6304720018972505, 0.6579279856713494, 0.6845359314028236, 0.7102615427114729, 0.7350716605053138, 0.7589343057213493, 0.7818187205454378, 0.8036954080581777, 0.8245361702557805, 0.8443141443970352, 0.8630038376297066, 0.8805811598520419, 0.8970234547675694, 0.9123095290941229, 0.9264196798912372, 0.9393357199742057, 0.9510410013893136, 0.9615204369360731, 0.9707605197472843, 0.9787493410038163, 0.9854766060684204, 0.9909336501109646, 0.995113458137992, 0.9980107215629823, 0.9996223504324843],
##         300: [-0.9999679782184367, -0.9998312829844194, -0.9995853734488863, -0.9992302218632736, -0.998765860353095, -0.9981923380734435, -0.9975097171758793, -0.9967180720510002, -0.9958174891208242, -0.9948080667630034, -0.9936899152742682, -0.9924631568471006, -0.9911279255509899, -0.9896843673150097, -0.9881326399103604, -0.9864729129322606, -0.9847053677808957, -0.9828301976412724, -0.9808476074619004, -0.9787578139322587, -0.9765610454590226, -0.9742575421410374, -0.9718475557430343, -0.9693313496680812, -0.9667091989287728, -0.9639813901171553, -0.9611482213733913, -0.9582100023531646, -0.9551670541938287, -0.9520197094793031, -0.948768312203718, -0.9454132177338137, -0.9419547927700982, -0.9383934153067653, -0.93472947459038, -0.9309633710773336, -0.927095516390074, -0.9231263332721162, -0.9190562555418367, -0.9148857280450583, -0.9106152066064294, -0.9062451579796031, -0.9017760597962232, -0.8972084005137192, -0.8925426793619202, -0.88777940628849, -0.8829191019031915, -0.8779622974209857, -0.8729095346039718, -0.8677613657021737, -0.8625183533931814, -0.8571810707206516, -0.8517501010316768, -0.846226037913026, -0.8406094851262689, -0.8349010565417858, -0.8291013760716742, -0.8232110776015571, -0.817230804921301, -0.8111612116546523, -0.8050029611877982, -0.7987567265968609, -0.7924231905743324, -0.7860030453544584, -0.77949699263758, -0.7729057435134394, -0.7662300183834609, -0.7594705468820128, -0.752628067796661, -0.7457033289874226, -0.7386970873050261, -0.7316101085081913, -0.7244431671799336, -0.7171970466429053, -0.7098725388737805, -0.7024704444166952, -0.6949915722957499, -0.6874367399265869, -0.6798067730270485, -0.6721025055269295, -0.664324779476831, -0.6564744449561274, -0.6485523599800557, -0.6405593904059368, -0.6324964098385406, -0.6243642995346039, -0.6161639483065119, -0.6078962524251537, -0.5995621155219635, -0.5911624484901564, -0.5826981693851706, -0.5741702033243276, -0.5655794823857192, -0.5569269455063353, -0.5482135383794405, -0.5394402133512136, -0.530607929316659, -0.5217176516148028, -0.5127703519231844, -0.5037670081516558, -0.4947086043354983, -0.4855961305278722, -0.4764305826916062, -0.4672129625903424, -0.45794427767904683, -0.44862554099389806, -0.43925777104156566, -0.42984199168789095, -0.4203792320459818, -0.410870526363734, -0.40131691391079155, -0.3917194388649578, -0.38207915019807054, -0.37239710156135264, -0.36267435117025176, -0.35291196168878075, -0.34311100011337214, -0.333272537656259, -0.32339764962839507, -0.3134874153219266, -0.3035429178922298, -0.29356524423952524, -0.2835554848900835, -0.27351473387703407, -0.26344408862079155, -0.25334464980911087, -0.24321752127678611, -0.233063809885005, -0.22288462540037263, -0.21268108037361766, -0.2024542900179944, -0.19220537208739363, -0.18193544675417583, -0.17164563648674022, -0.16133706592684258, -0.15101086176667575, -0.140668152625726, -0.1303100689274186, -0.1199377427755664, -0.10955230783063467, -0.0991548991858358, -0.08874665324306741, -0.07832870758870747, -0.0679022008692799, -0.057468272667004265, -0.0470280633752434, -0.03658271407386212, -0.02613336640451104, -0.015681162445849026, -0.005227244588717748, 0.005227244588717748, 0.015681162445849026, 0.02613336640451104, 0.03658271407386212, 0.0470280633752434, 0.057468272667004265, 0.0679022008692799, 0.07832870758870747, 0.08874665324306741, 0.0991548991858358, 0.10955230783063467, 0.1199377427755664, 0.1303100689274186, 0.140668152625726, 0.15101086176667575, 0.16133706592684258, 0.17164563648674022, 0.18193544675417583, 0.19220537208739363, 0.2024542900179944, 0.21268108037361766, 0.22288462540037263, 0.233063809885005, 0.24321752127678611, 0.25334464980911087, 0.26344408862079155, 0.27351473387703407, 0.2835554848900835, 0.29356524423952524, 0.3035429178922298, 0.3134874153219266, 0.32339764962839507, 0.333272537656259, 0.34311100011337214, 0.35291196168878075, 0.36267435117025176, 0.37239710156135264, 0.38207915019807054, 0.3917194388649578, 0.40131691391079155, 0.410870526363734, 0.4203792320459818, 0.42984199168789095, 0.43925777104156566, 0.44862554099389806, 0.45794427767904683, 0.4672129625903424, 0.4764305826916062, 0.4855961305278722, 0.4947086043354983, 0.5037670081516558, 0.5127703519231844, 0.5217176516148028, 0.530607929316659, 0.5394402133512136, 0.5482135383794405, 0.5569269455063353, 0.5655794823857192, 0.5741702033243276, 0.5826981693851706, 0.5911624484901564, 0.5995621155219635, 0.6078962524251537, 0.6161639483065119, 0.6243642995346039, 0.6324964098385406, 0.6405593904059368, 0.6485523599800557, 0.6564744449561274, 0.664324779476831, 0.6721025055269295, 0.6798067730270485, 0.6874367399265869, 0.6949915722957499, 0.7024704444166952, 0.7098725388737805, 0.7171970466429053, 0.7244431671799336, 0.7316101085081913, 0.7386970873050261, 0.7457033289874226, 0.752628067796661, 0.7594705468820128, 0.7662300183834609, 0.7729057435134394, 0.77949699263758, 0.7860030453544584, 0.7924231905743324, 0.7987567265968609, 0.8050029611877982, 0.8111612116546523, 0.817230804921301, 0.8232110776015571, 0.8291013760716742, 0.8349010565417858, 0.8406094851262689, 0.846226037913026, 0.8517501010316768, 0.8571810707206516, 0.8625183533931814, 0.8677613657021737, 0.8729095346039718, 0.8779622974209857, 0.8829191019031915, 0.88777940628849, 0.8925426793619202, 0.8972084005137192, 0.9017760597962232, 0.9062451579796031, 0.9106152066064294, 0.9148857280450583, 0.9190562555418367, 0.9231263332721162, 0.927095516390074, 0.9309633710773336, 0.93472947459038, 0.9383934153067653, 0.9419547927700982, 0.9454132177338137, 0.948768312203718, 0.9520197094793031, 0.9551670541938287, 0.9582100023531646, 0.9611482213733913, 0.9639813901171553, 0.9667091989287728, 0.9693313496680812, 0.9718475557430343, 0.9742575421410374, 0.9765610454590226, 0.9787578139322587, 0.9808476074619004, 0.9828301976412724, 0.9847053677808957, 0.9864729129322606, 0.9881326399103604, 0.9896843673150097, 0.9911279255509899, 0.9924631568471006, 0.9936899152742682, 0.9948080667630034, 0.9958174891208242, 0.9967180720510002, 0.9975097171758793, 0.9981923380734435, 0.998765860353095, 0.9992302218632736, 0.9995853734488863, 0.9998312829844194, 0.9999679782184367],
#}


def jacobian(f, x0, scalar=True, perturbation=1e-9, zero_offset=1e-7, args=(),
             **kwargs):
    """def test_fun(x):

    # test case - 2 inputs, 3 outputs - should work fine
    x2 = x[0]*x[0]
    return np.array([x2*exp(x[1]), x2*sin(x[1]), x2*cos(x[1])])

    def easy_fun(x):
        x = x[0]
        return 5*x*x - 3*x - 100
    """
    # For scalar - returns list, size of input variables
    # For vector - returns list of list - size of input variables * output variables
    # Could add backwards/complex, multiple evaluations, detection of poor condition
    # types and limits
    base = f(x0, *args, **kwargs)
    x = list(x0)
    nx = len(x0)

    gradient = []
    for i in range(nx):
        delta = x0[i]*(perturbation)
        if delta == 0:
            delta = zero_offset

        x[i] += delta

        point = f(x, *args, **kwargs)
        if scalar:
            dy = (point - base)/delta
            gradient.append(dy)
        else:
            delta_inv = 1.0/delta
            dys = [delta_inv*(p - b) for p, b in zip(point, base)]
            gradient.append(dys)

        x[i] -= delta
    if not scalar:
        # Transpose to be in standard form
        return list(map(list, zip(*gradient)))
    return gradient


def hessian(f, x0, scalar=True, perturbation=1e-9, zero_offset=1e-7, full=True, args=(), **kwargs):
    # Takes n**2/2 + 3*n/2 + 1 function evaluations! Can still be quite fast.
    # For scalar - returns list[list], size of input variables
    # For vector - returns list of list of list - size of input variables * input variables * output variables
    # Could add backwards/complex, multiple evaluations, detection of poor condition
    # types and limits, jacobian as output, fevals


    base = f(x0, *args, **kwargs)
    nx = len(x0)

    if not isinstance(base, (float, int, complex)):
        try:
            ny = len(base)
        except:
            ny = 1
    else:
        ny = 1

    deltas = []
    for i in range(nx):
        delta = x0[i]*(perturbation)
        if delta == 0.0:
            delta = zero_offset
        deltas.append(delta)
    deltas_inv = [1.0/di for di in deltas]


    x_perturb = list(x0)

    fs_perturb_i = []
    for i in range(nx):
        x_perturb[i] += deltas[i]
        f_perturb_i = f(x_perturb, *args, **kwargs)
        fs_perturb_i.append(f_perturb_i)
        x_perturb[i] -= deltas[i]

    if full:
        hessian = [[None]*nx for _ in range(nx)]
    else:
        hessian = []

    for i in range(nx):
        if not full:
            row = []
        f_perturb_i = fs_perturb_i[i]
        x_perturb[i] += deltas[i]

        for j in range(i+1):
            f_perturb_j = fs_perturb_i[j]
            x_perturb[j] += deltas[j]
            f_perturb_ij = f(x_perturb, *args, **kwargs)

            if scalar:
                dii0 = (f_perturb_i - base)*deltas_inv[i]
                dii1 = (f_perturb_ij - f_perturb_j)*deltas_inv[i]
                dij = (dii1 - dii0)*deltas_inv[j]
            else:
#                 dii0s = [(fi - bi)*deltas_inv[i] for fi, bi in zip(f_perturb_i, base)]
#                 dii1s = [(fij - fj)*deltas_inv[i] for fij, fj in zip(f_perturb_ij, f_perturb_j)]
#                 dij = [(di1 - di0)*deltas_inv[j] for di1, di0 in zip(dii1s, dii0s)]
                # Saves a good amount of time
                dij = [((f_perturb_ij[m] - f_perturb_j[m]) - (f_perturb_i[m] - base[m]))*deltas_inv[j]*deltas_inv[i]
                       for m in range(ny)]

            if not full:
                row.append(dij)
            else:
                hessian[i][j] = hessian[j][i] = dij

            x_perturb[j] -= deltas[j]
        if not full:
            hessian.append(row)
        x_perturb[i] -= deltas[i]
    return hessian


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
    v7 = 1.0/(v4*x0 - v4*x1 + v5*x1 - v5*x2 - v6*x0 + v6*x2)
    v8 = -v4
    a = v7*(v1 + v2 - v3)
    b = -v7*(f0*(v6 + v8) - f1*(v5 + v8) + f2*(v5 - v6))
    c = v7*(v1*x1*x2 + v2*x0*x1 - v3*x0*x2)
    return [a, b, c]



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
    return [a, b, c]

def is_poly_positive(poly, domain=None, rand_pts=10, j_tol=1e-12, root_perturb=1e-12):
    # Returns True if positive everywhere in the specified domain (or globally)
    if domain is None:
        # 1e-100 to 1e100
        pts = logspace(-100, 100, rand_pts//2)
        pts += [-i for i in pts]
    else:
        pts = linspace(domain[0], domain[1], rand_pts)

    for p in pts:
        if horner(poly, p) < 0.0:
            return False

    roots = np.roots(poly)
    for root in roots:
        r = root.real
        if abs(root.imag/r) < j_tol:
            if (domain is not None) and (r < domain[0] or r > domain[1]):
                continue
            eps_high, eps_low = r*(1.0 + root_perturb), r*(1.0 - root_perturb)
            if horner(poly, eps_high) < 0:
                return False
            if horner(poly, eps_low) < 0:
                return False
    return True

def is_poly_negative(poly, domain=None, rand_pts=10, j_tol=1e-12, root_perturb=1e-12):
    # Returns True if negative everywhere in the specified domain (or globally)
    poly = [-i for i in poly]# Changes the sign of all polynomial calculated values
    return is_poly_positive(poly, domain=domain, rand_pts=rand_pts, j_tol=j_tol, root_perturb=root_perturb)


def polyder(c, m=1, scl=1, axis=0):
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
    log_coef = coeffs[-1]
    Nm1 = N - 1
    poly_terms = [0.0]*N
    for i in range(Nm1):
        poly_terms[i] = coeffs[i]/(Nm1-i)
    return poly_terms, log_coef
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
#
def chebder(c, m=1):
    """not quite a copy of numpy's version because this was faster to
    implement."""
    c = list(c)
    cnt = int(m)
    if cnt == 0:
        return c

    n = len(c)
    if cnt >= n:
        c = []
    else:
        for i in range(cnt):
            n = n - 1
#            c *= scl
            der = [0.0 for _ in range(n)]
            for j in range(n, 2, -1):
                der[j - 1] = (j + j)*c[j]
                c[j - 2] += (j*c[j])/(j - 2.0)
            if n > 1:
                der[1] = 4.0*c[2]
            der[0] = c[1]
            c = der
    return c

def horner_log(coeffs, log_coeff, x):
    """Technically possible to save one addition of the last term of coeffs is
    removed but benchmarks said nothing was saved."""
    tot = 0.0
    for c in coeffs:
        tot = tot*x + c
    return tot + log_coeff*log(x)


def fit_integral_linear_extrapolation(T1, T2, int_coeffs, Tmin, Tmax,
                                      Tmin_value, Tmax_value,
                                      Tmin_slope, Tmax_slope):
    # Order T1, T2 so T2 is always larger for simplicity
    flip = T1 > T2
    if flip:
        T1, T2 = T2, T1

    tot = 0.0
    if T1 < Tmin:
        T2_low = T2 if T2 < Tmin else Tmin
        x1 = Tmin_value - Tmin_slope*Tmin
        tot += T2_low*(0.5*Tmin_slope*T2_low + x1) - T1*(0.5*Tmin_slope*T1 + x1)
    if (Tmin <= T1 <= Tmax) or (Tmin <= T2 <= Tmax) or (T1 <= Tmin and T2 >= Tmax):
        T1_mid = T1 if T1 > Tmin else Tmin
        T2_mid = T2 if T2 < Tmax else Tmax
        tot += (horner(int_coeffs, T2_mid) - horner(int_coeffs, T1_mid))

    if T2 > Tmax:
        T1_high = T1 if T1 > Tmax else Tmax
        x1 = Tmax_value - Tmax_slope*Tmax
        tot += T2*(0.5*Tmax_slope*T2 + x1) - T1_high*(0.5*Tmax_slope*T1_high + x1)
    if flip:
        return -tot
    return tot

def poly_fit_integral_value(T, int_coeffs, Tmin, Tmax, Tmin_value, Tmax_value,
                            Tmin_slope, Tmax_slope):
    # Can still save 1 horner evaluation (all of them for height T), but will be VERY messy.
    if T < Tmin:
        x1 = Tmin_value - Tmin_slope*Tmin
        tot = T*(0.5*Tmin_slope*T + x1)
        return tot
    if (Tmin <= T <= Tmax):
        tot1 = horner(int_coeffs, T) - horner(int_coeffs, Tmin)
        x1 = Tmin_value - Tmin_slope*Tmin
        tot = Tmin*(0.5*Tmin_slope*Tmin + x1)
        return tot + tot1
    else:
        x1 = Tmin_value - Tmin_slope*Tmin
        tot = Tmin*(0.5*Tmin_slope*Tmin + x1)

        tot1 = horner(int_coeffs, Tmax) - horner(int_coeffs, Tmin)

        x1 = Tmax_value - Tmax_slope*Tmax
        tot2 = T*(0.5*Tmax_slope*T + x1) - Tmax*(0.5*Tmax_slope*Tmax + x1)
        return tot1 + tot + tot2

def fit_integral_over_T_linear_extrapolation(T1, T2, T_int_T_coeffs,
                                            poly_fit_log_coeff, Tmin, Tmax,
                                      Tmin_value, Tmax_value,
                                      Tmin_slope, Tmax_slope):
    # Order T1, T2 so T2 is always larger for simplicity
    flip = T1 > T2
    if flip:
        T1, T2 = T2, T1

    tot = 0.0
    if T1 < Tmin:
        T2_low = T2 if T2 < Tmin else Tmin
        x1 = Tmin_value - Tmin_slope*Tmin
        tot += (Tmin_slope*T2_low + x1*log(T2_low)) - (Tmin_slope*T1 + x1*log(T1))
    if (Tmin <= T1 <= Tmax) or (Tmin <= T2 <= Tmax) or (T1 <= Tmin and T2 >= Tmax):
        T1_mid = T1 if T1 > Tmin else Tmin
        T2_mid = T2 if T2 < Tmax else Tmax
        tot += (horner_log(T_int_T_coeffs, poly_fit_log_coeff, T2_mid)
                    - horner_log(T_int_T_coeffs, poly_fit_log_coeff, T1_mid))
    if T2 > Tmax:
        T1_high = T1 if T1 > Tmax else Tmax
        x1 = Tmax_value - Tmax_slope*Tmax
        tot += (Tmax_slope*T2 + x1*log(T2)) - (Tmax_slope*T1_high + x1*log(T1_high))
    if flip:
        return -tot
    return tot


def poly_fit_integral_over_T_value(T, T_int_T_coeffs, poly_fit_log_coeff,
                                   Tmin, Tmax, Tmin_value, Tmax_value,
                                   Tmin_slope, Tmax_slope):
    if T < Tmin:
        x1 = Tmin_value - Tmin_slope*Tmin
        tot = (Tmin_slope*T + x1*log(T))
        return tot
    if (Tmin <= T <= Tmax):
        tot1 = (horner_log(T_int_T_coeffs, poly_fit_log_coeff, T)
                    - horner_log(T_int_T_coeffs, poly_fit_log_coeff, Tmin))

        x1 = Tmin_value - Tmin_slope*Tmin
        tot = (Tmin_slope*Tmin + x1*log(Tmin))
        return tot + tot1
    else:
        x1 = Tmin_value - Tmin_slope*Tmin
        tot = (Tmin_slope*Tmin + x1*log(Tmin))

        tot1 = (horner_log(T_int_T_coeffs, poly_fit_log_coeff, Tmax)
                    - horner_log(T_int_T_coeffs, poly_fit_log_coeff, Tmin))
        x2 = Tmax_value -Tmax*Tmax_slope
        tot2 = (-Tmax_slope*(Tmax - T) + x2*log(T) - x2*log(Tmax))
        return tot1 + tot + tot2



def evaluate_linear_fits(data, x):
    calc = []
    low_limits, high_limits, coeffs = data[0], data[3], data[6]
    for i in range(len(data[0])):
        if x < low_limits[i]:
            v = (x - low_limits[i])*data[1][i] + data[2][i]
        elif x > high_limits[i]:
            v = (x - high_limits[i])*data[4][i] + data[5][i]
        else:
            v = 0.0
            for c in coeffs[i]:
                v = v*x + c
#               v = horner(coeffs[i], x)
        calc.append(v)
    return calc


def evaluate_linear_fits_d(data, x):
    calc = []
    low_limits, high_limits, dcoeffs = data[0], data[3], data[7]
    for i in range(len(data[0])):
        if x < low_limits[i]:
            dv = data[1][i]
        elif x > high_limits[i]:
            dv = data[4][i]
        else:
            dv = 0.0
            for c in dcoeffs[i]:
                dv = dv*x + c
        calc.append(dv)
    return calc


def evaluate_linear_fits_d2(data, x):
    calc = []
    low_limits, high_limits, d2coeffs = data[0], data[3], data[8]
    for i in range(len(data[0])):
        d2v = 0.0
        if low_limits[i] < x < high_limits[i]:
            for c in d2coeffs[i]:
                d2v = d2v*x + c
        calc.append(d2v)
    return calc


def chebval(x, c):
    # Pure Python implementation of numpy.polynomial.chebyshev.chebval
    # This routine is faster in CPython as well as PyPy
    len_c = len(c)
    if len_c == 1:
        c0, c1 = c[0], 0.0
    elif len_c == 2:
        c0, c1 = c[0], c[1]
    else:
        x2 = 2.0*x
        c0, c1 = c[-2], c[-1]
        for i in range(3, len_c + 1):
            c0_prev = c0
            c0 = c[-i] - c1
            c1 = c0_prev + c1*x2
    return c0 + c1*x


def binary_search(key, arr, size=None):
    if size is None:
        size = len(arr)
    imin = 0
    imax = size
    if key > arr[size - 1]:
        return size
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if key >= arr[imid]: imin = imid + 1
        else: imax = imid
    return imin - 1


def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    """Pure python and therefore slow version of the standard library isclose.
    Works on older versions of python though! Hasn't been unit tested, but has
    been tested.

    manual unit testing:

    from math import isclose as isclose2
    from random import uniform
    for i in range(10000000):
        a = uniform(-1, 1)
        b = uniform(-1, 1)
        rel_tol = uniform(0, 1)
        abs_tol = uniform(0, .001)
        ans1 = isclose(a, b, rel_tol, abs_tol)
        ans2 = isclose2(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        try:
            assert ans1 == ans2
        except:
            print(a, b, rel_tol, abs_tol)
    """
    if (rel_tol < 0.0 or abs_tol < 0.0 ):
        raise ValueError('Negative tolerances')

    if ((a.real == b.real) and (a.imag == b.imag)):
        return True

    if (isinf(a.real) or isinf(a.imag) or
        isinf(b.real) or isinf(b.imag)):
        return False

    diff = abs(a - b)
    return (((diff <= rel_tol*abs(b)) or
             (diff <= rel_tol*abs(a))) or (diff <= abs_tol))
try:
    from math import isclose
except ImportError:
    pass

def assert_close(a, b, rtol=1e-7, atol=0.0):
    if a is b:
        # Nice to handle None
        return True

    if __debug__:
        # Do not run these branches in -O, -OO mode
        try:
            try:
                assert isclose(a, b, rel_tol=rtol, abs_tol=atol)
                return
            except:
                assert cisclose(a, b, rel_tol=rtol, abs_tol=atol)
                return
        except:
            pass
    from numpy.testing import assert_allclose
    return assert_allclose(a, b, rtol=rtol, atol=atol)

def assert_close1d(a, b, rtol=1e-7, atol=0.0):
    N = len(a)
    if N != len(b):
        raise ValueError("Variables are not the same length: %d, %d" %(N, len(b)))
    for i in range(N):
        assert_close(a[i], b[i], rtol=rtol, atol=atol)

def assert_close2d(a, b, rtol=1e-7, atol=0.0):
#    N = len(a)
#    if N != len(b):
#        raise ValueError("Variables are not the same length: %d, %d" %(N, len(b)))
#    for i in range(N):
#        assert_close1d(a[i], b[i], rtol=rtol, atol=atol)
    N = len(a)
    if N != len(b):
        raise ValueError("Variables are not the same length: %d, %d" %(N, len(b)))
    if not __debug__:
        # Do not run these branches in -O, -OO mode
        from numpy.testing import assert_allclose
        return assert_allclose(a, b, rtol=rtol, atol=atol)
    for i in range(N):
        a0, b0 = a[i], b[i]
        N0 = len(a0)
        if N0 != len(b0):
            raise ValueError("Variables are not the same length: %d, %d" %(N0, len(b0)))
        for j in range(N0):
#            assert_close(a0[j], b0[j], rtol=rtol, atol=atol)
            good = True
            a1, b1 = a0[j], b0[j]
            if a1 is b1:
                # Nice to handle None
                pass
            else:
                try:
                    try:
                        good = isclose(a1, b1, rel_tol=rtol, abs_tol=atol)
                    except:
                        good = cisclose(a1, b1, rel_tol=rtol, abs_tol=atol)
                except:
                    pass
            if not good:
                from numpy.testing import assert_allclose
                return assert_allclose(a1, b1, rtol=rtol, atol=atol)






def assert_close3d(a, b, rtol=1e-7, atol=0.0):
    N = len(a)
    if N != len(b):
        raise ValueError("Variables are not the same length: %d, %d" %(N, len(b)))
    for i in range(N):
        assert_close2d(a[i], b[i], rtol=rtol, atol=atol)

def assert_close4d(a, b, rtol=1e-7, atol=0.0):
    N = len(a)
    if N != len(b):
        raise ValueError("Variables are not the same length: %d, %d" %(N, len(b)))
#    for i in range(N):
#        assert_close3d(a[i], b[i], rtol=rtol, atol=atol)
    for i in range(N):
        a0, b0 = a[i], b[i]
        N0 = len(a0)
        if N0 != len(b0):
            raise ValueError("Variables are not the same length: %d, %d" %(N0, len(b0)))
        for j in range(N0):
            assert_close2d(a0[j], b0[j], rtol=rtol, atol=atol)

def interp(x, dx, dy, left=None, right=None, extrapolate=False):
    """One-dimensional linear interpolation routine inspired/ reimplemented from
    NumPy for extra speed for scalar values (and also numpy).

    Returns the one-dimensional piecewise linear interpolant to a function
    with a given value at discrete data-points.

    Parameters
    ----------
    x : float
        X-coordinate of the interpolated values, [-]
    dx : list[float]
        X-coordinates of the data points, must be increasing, [-]
    dy : list[float]
        Y-coordinates of the data points; same length as `dx`, [-]
    left : float, optional
        Value to return for `x < dx[0]`, default is `dy[0]`, [-]
    right : float, optional
        Value to return for `x > dx[-1]`, default is `dy[-1]`, [-]
    extrapolate : bool, optional
        If True, for the cases of `left` and/or `right` not given, a linear
        extrapolation will be performed outside of bounds, [-]

    Returns
    -------
    y : float
        The interpolated value, [-]

    Notes
    -----
    This function is "unsafe" in that it assumes the x-coordinates
    are increasing. It also does not check for nan's, that `dx` and `dy`
    are the same length, and that `x` is scalar.

    Performance is 40-50% of that of NumPy under CPython.

    Examples
    --------
    >>> interp(2.5, [1, 2, 3], [3, 2, 0])
    1.0
    """
    lendx = len(dx)
    j = binary_search(x, dx, lendx)
    if (j == -1):
        if left is not None:
            return left
        elif extrapolate:
            j = 0
            return (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j])*(x - dx[j]) + dy[j]
        else:
            return dy[0]
    elif (j == lendx - 1):
        return dy[j]
    elif (j == lendx):
        if right is not None:
            return right
        elif extrapolate:
            j = -2
            return (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j])*(x - dx[j]) + dy[j]
        else:
            return dy[-1]
    else:
        return (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j])*(x - dx[j]) + dy[j]


def interp2d_linear(x, y, xs, ys, vals):
    # Same as RectBivariateSpline, s=0, kx=1, ky=1 (and better performance)
    if y < ys[0]:
        i0, i1 = 0, 1
        y_dat = ys[i0], ys[i1]
    elif y > ys[-1]:
        i0, i1 = -2, -1
    else:
        for i in range(len(ys)):
            if ys[i] >= y:
                i0, i1 = i-1, i
                break
    y_low, y_high = ys[i0], ys[i1]

    v_low = interp(x, xs, vals[i0], extrapolate=True)
    v_high = interp(x, xs, vals[i1], extrapolate=True)

    return v_low + (y-y_low)/(y_high-y_low)*(v_high-v_low)


try:
    _array = np.array
except:
    pass
def implementation_optimize_tck(tck, force_numpy=False):
    """Converts 1-d or 2-d splines calculated with SciPy's `splrep` or.

    `bisplrep` to a format for fastest computation - lists in PyPy, and numpy
    arrays otherwise.

    Only implemented for 3 and 5 length `tck`s.
    """
    if IS_PYPY_OR_SKIP_DEPENDENCIES and not force_numpy:
        return tuple(tck)
    else:
        size = len(tck)
        if size == 3:
            return (_array(tck[0]), _array(tck[1]), tck[2])
        elif size == 5:
            return (_array(tck[0]), _array(tck[1]), _array(tck[2]), tck[3], tck[4])
        else:
            raise NotImplementedError


def tck_interp2d_linear(x, y, z, kx=1, ky=1):
    if kx != 1 or ky != 1:
        raise ValueError("Only linear formulations are currently implemented")
    # copy is not a method of lists in python 2
    x = list(x)
    x.insert(0, x[0])
    x.append(x[-1])

    y = list(y)
    y.insert(0, y[0])
    y.append(y[-1])

    # c needs to be transposed, and made 1d
    c = [z[j][i] for i in range(len(z[0])) for j in range(len(z))]

    tck = [x, y, c, 1, 1]
    return implementation_optimize_tck(tck)



def caching_decorator(f, full=False):
    from functools import wraps
    cache = {}
    info_cache = {}
    wraps = my_wraps()
    @wraps(f)
    def wrapper(x, *args, **kwargs):
        has_info = 'info' in kwargs
        if x in cache:
            if 'info' in kwargs:
                kwargs['info'][:] = info_cache[x]
            return cache[x]

        err = f(x, *args, **kwargs)
        cache[x] = err
        if has_info:
            info_cache[x] = list(kwargs['info'])
        return err
    if full:
        return wrapper, cache, info_cache
    return wrapper


def translate_bound_func(func, bounds=None, low=None, high=None):
    if bounds is not None:
        low = [i[0] for i in bounds]
        high = [i[1] for i in bounds]

    def new_f(x, *args, **kwargs):
        """Function for a solver to call when using the bounded variables."""
        x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = (low[i] + (high[i] - low[i])/(1.0 + exp(-x[i])))
        # Return the actual results
        return func(x, *args, **kwargs)

    def translate_into(x):
        x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = -log((high[i] - x[i])/(x[i] - low[i]))
        return x

    def translate_outof(x):
        x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = (low[i] + (high[i] - low[i])/(1.0 + exp(-x[i])))
        return x
    return new_f, translate_into, translate_outof


def translate_bound_jac(jac, bounds=None, low=None, high=None):
    if bounds is not None:
        low = [i[0] for i in bounds]
        high = [i[1] for i in bounds]

    def new_j(x):
        x_base = [float(i) for i in x]
        N = len(x)
        for i in range(N):
            x_base[i] = (low[i] + (high[i] - low[i])/(1.0 + exp(-x[i])))
        jac_base = jac(x_base)
        try:
            jac_base = [i for i in jac_base]
            for i in range(N):
                v = (high[i] - low[i])*exp(-x[i])*jac_base[i]
                v *= (1.0 + exp(-x[i]))**-2
                jac_base[i] = v
            return jac_base
        except:
            raise NotImplementedError("Fail")

    def translate_into(x):
        x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = -log((high[i] - x[i])/(x[i] - low[i]))
        return x

    def translate_outof(x):
        x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = (low[i] + (high[i] - low[i])/(1.0 + exp(-x[i])))
        return x
    return new_j, translate_into, translate_outof


def translate_bound_f_jac(f, jac, bounds=None, low=None, high=None,
                          inplace_jac=False, as_np=False):
    if bounds is not None:
        low = [i[0] for i in bounds]
        high = [i[1] for i in bounds]

    exp_terms = [0.0]*len(low)

    def new_f_j(x, *args):
        x_base = [i for i in x]
        N = len(x)
        for i in range(N):
            exp_terms[i] = ei = trunc_exp(-x[i])
            x_base[i] = (low[i] + (high[i] - low[i])/(1.0 + ei))

        if jac is True:
            f_base, jac_base = f(x_base, *args)
        else:
            f_base = f(x_base, *args)
            jac_base = jac(x_base, *args)
        try:
            if type(jac_base[0]) is list or (isinstance(jac_base, np.ndarray) and len(jac_base.shape) == 2):
                if not inplace_jac:
                    jac_base = [[j for j in i] for i in jac_base]

                for i in range(len(jac_base)):
                    for j in range(len(jac_base[i])):
                        # Checked numerically
                        t = (1.0 + exp_terms[j])
                        jac_base[i][j] = (high[j] - low[j])*exp_terms[j]*jac_base[i][j]/(t*t)
            else:
                if not inplace_jac:
                    jac_base = [i for i in jac_base]
                for i in range(N):
                    t = (1.0 + exp_terms[i])
                    jac_base[i] = (high[i] - low[i])*exp_terms[i]*jac_base[i]/(t*t)
            if as_np:
                jac_base = np.array(jac_base)
            return f_base, jac_base
        except:
            raise NotImplementedError("Fail")

    def translate_into(x):
        #x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = -trunc_log((high[i] - x[i])/(x[i] - low[i]))
        return x

    def translate_outof(x):
        #x = [float(i) for i in x]
        for i in range(len(x)):
            x[i] = (low[i] + (high[i] - low[i])/(1.0 + trunc_exp(-x[i])))
        return x
    return new_f_j, translate_into, translate_outof


class OscillationError(Exception):
    """Error raised when a derivative-based method is not converging."""

class UnconvergedError(Exception):
    """Error raised when maxiter has been reached in an optimization problem."""

    def __repr__(self):
        return ('UnconvergedError("Failed to converge; maxiter (%d) reached, value=%g, error %g)"' %(self.maxiter, self.point, self.err))

    def __init__(self, message, iterations=None, err=None, point=None):
        super(UnconvergedError, self).__init__(message)
        self.point = point
        self.iterations = iterations
        self.err = err

class SamePointError(UnconvergedError):
    """Error raised when two trial points in a root finding problem have the
    same error."""
    def __repr__(self):
        return 'TODO'

    def __init__(self, message, iterations=None, err=None, q1=None, p1=None, q0=None, p0=None):
        super(UnconvergedError, self).__init__(message)
        self.q1 = q1
        self.p1 = p1
        self.q0 = q0
        self.p0 = p0
        self.iterations = iterations
        self.err = err


class NoSolutionError(Exception):
    """Error raised when detected that there is no actual solution to a
    problem."""

class NotBoundedError(Exception):
    """Error raised when a bisection type algorithm fails because its initial
    bounds do not bound the solution."""
class DiscontinuityError(Exception):
    """Error raised when a bisection type algorithm fails because there is a
    discontinuity."""

def damping_maintain_sign(x, step, damping=1.0, factor=0.5):
    """Damping function which will maintain the sign of the variable being
    manipulated. If the step puts it at the other sign, the distance between `x`
    and `step` will be shortened by the multiple of `factor`; i.e. if factor is
    `x`, the new value of `x` will be 0 exactly.

    The provided `damping` is applied as well.

    Parameters
    ----------
    x : float
        Previous value in iteration, [-]
    step : float
        Change in `x`, [-]
    damping : float, optional
        The damping factor to be applied always, [-]
    factor : float, optional
        If the calculated step changes sign, this factor will be used instead
        of the step, [-]

    Returns
    -------
    x_new : float
        The new value in the iteration, [-]

    Notes
    -----

    Examples
    --------
    >>> damping_maintain_sign(100, -200, factor=.5)
    50.0
    """
    if isinstance(x, list):
        return [damping_maintain_sign(x[i], step[i], damping, factor) for i in range(len(x))]
    positive = x > 0.0
    step_x = x + step

    if (positive and step_x < 0) or (not positive and step_x > 0.0):
#        print('damping')
        step = -factor*x
    return x + step*damping


def make_damp_initial(steps=5, damping=1.0, *args):
    steps_holder = [steps]
    def damping_func(x, step, damping=damping, *args):
        if steps_holder[0] <= 0:
            # Do not dampen at all
            if isinstance(x, list):
                return [xi + dxi for xi, dxi in zip(x, step)]
            return x + step
        else:
            steps_holder[0] -= 1
            if isinstance(x, list):
                return [xi + dxi*damping for xi, dxi in zip(x, step)]
            return x + step*damping
    return damping_func



def oscillation_checking_wrapper(f, minimum_progress=0.3,
                                 both_sides=False, full=True,
                                 good_err=None):
    checker = oscillation_checker(minimum_progress=minimum_progress,
                                 both_sides=both_sides, good_err=good_err)
    wraps = my_wraps()
    @wraps(f)
    def wrapper(x, *args, **kwargs):
        err_test = err = f(x, *args, **kwargs)
        if not isinstance(err, (float, int, complex)):
            err_test = err[0]
        try:
            oscillating = checker(x, err_test)
        except:
            oscillating = False # Zero division error probably

        if oscillating:
            raise OscillationError("Oscillating")

        return err
    if full:
        return wrapper, checker
    return wrapper


class oscillation_checker(object):
    def __init__(self, minimum_progress=0.3, both_sides=False, good_err=None):
        self.minimum_progress = minimum_progress
        self.both_sides = both_sides
        self.xs_neg = []
        self.xs_pos = []

        self.ys_neg = []
        self.ys_pos = []

        # Provide a number that if the error is under this, no longer be able to return False
        # For example, near phase boundaries newton could be bisecting as it overshoots
        # each step, but is still converging fine
        self.good_err = good_err

    def clear(self):
        self.xs_neg = []
        self.xs_pos = []

        self.ys_neg = []
        self.ys_pos = []

    def is_solve_oscilating(self, x, y):
        if y == 0.0:
            return False
        xs_neg, xs_pos, ys_neg, ys_pos = self.xs_neg, self.xs_pos, self.ys_neg, self.ys_pos
        minimum_progress = self.minimum_progress

        if y < 0.0:
            xs_neg.append(x)
            ys_neg.append(y)
        else:
            xs_pos.append(x)
            ys_pos.append(y)
        if len(xs_pos) > 1 and len(xs_neg) > 1:
            if y < 0:
                dy_cur = y - ys_neg[-2]
                dy_other = ys_pos[-1] - ys_pos[-2]
                gain_neg = abs(dy_cur/y)
                gain_pos = abs(dy_other/ys_pos[-1])
            else:
                dy_cur = y - ys_pos[-2]
                dy_other = ys_neg[-1] - ys_neg[-2]
                gain_pos = abs(dy_cur/y)
                gain_neg = abs(dy_other/ys_neg[-1])

#            print(gain_pos, gain_neg, y)
            if self.both_sides:
                if gain_pos < minimum_progress and gain_neg < minimum_progress:
                    if self.good_err is not None and min(abs(ys_neg[-1]), abs(ys_pos[-1])) < self.good_err:
                        return False
                    return True
            else:
                if gain_pos < minimum_progress or gain_neg < minimum_progress:
                    if self.good_err is not None and min(abs(ys_neg[-1]), abs(ys_pos[-1])) < self.good_err:
                        return False
                    return True
        return False

    __call__ = is_solve_oscilating


def best_bounding_bounds(low, high, f=None, xs_pos=None, ys_pos=None,
                         xs_neg=None, ys_neg=None, fa=None, fb=None):
    r'''Given:
        1) A presumed bracketing interval such as very far out limits on
           physical bounds
        2) A history of a non-bounded search algorithm which did not converge

    Find the best bracketing points which get the algorithm as close to the
    solution as possible.

    Parameters
    ----------
    low : float
        Low bracketing interval (`f` has opposite sign at `low` than `high`),
        [-]
    high : float
        High bracketing interval (`f` has opposite sign at `high` than `low`),
        [-]
    f : callable, optional
        1D function to be solved, [-]
    xs_pos : list[float]
        Unsorted list of `x` values of points with positive `y` values
        previously evaluated, [-]
    ys_pos : list[float]
        Unsorted list of `y` values of points with positive `y` values
        previously evaluated, [-]
    xs_neg : list[float]
        Unsorted list of `x` values of points with negative `y` values
        previously evaluated, [-]
    ys_neg : list[float]
        Unsorted list of `y` values of points with negative `y` values
        previously evaluated, [-]
    fa : float, optional
        Value of function at `low`, used instead of recalculating if provided,
        [-]
    fb : float, optional
        Value of function at `high`, used instead of recalculating if provided,
        [-]

    Returns
    -------
    low : float
        Low bracketing interval (`f` has opposite sign at `low` than `high`),
        [-]
    high : float
        High bracketing interval (`f` has opposite sign at `high` than `low`),
        [-]
    fa : float
        Value of function at `low`, [-]
    fb : float, optional
        Value of function at `high`, [-]

    Notes
    -----
    Negative and/or positive history values can be omitted, but both the `x`
    and `y` lists should be skipped if so.

    More work could be done to handle better the case if the bounds not
    bracketing the root but the function history doing so.
    '''
    if fa is None:
        fa = f(low)
    if fb is None:
        fb = f(high)

    if ys_pos:
        y_min_pos = min(ys_pos)
        x_min_pos = xs_pos[ys_pos.index(y_min_pos)]

        if fa > 0:
            if y_min_pos < fa:
                fa, low = y_min_pos, x_min_pos
        else:
            if y_min_pos < fb:
                fb, high = y_min_pos, x_min_pos

    if ys_neg:
        y_min_neg = max(ys_neg)
        x_min_neg = xs_neg[ys_neg.index(y_min_neg)]

        if fa < 0:
            if y_min_neg > fa:
                fa, low = y_min_neg, x_min_neg
        else:
            if y_min_pos > fb:
                fb, high = y_min_neg, x_min_neg

    if fa*fb > 0:
        raise ValueError("Bounds and previous history do not contain bracketing points")
    return low, high, fa, fb

def bisect(f, a, b, args=(), xtol=1e-12, rtol=2.220446049250313e-16, maxiter=100,
              ytol=None):
    """Port of SciPy's C bisect routine."""
    fa = f(a, *args)
    fb = f(b, *args)
    if fa*fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")
    elif fa == 0.0:
        return a
    elif fb == 0.0:
        return b

    dm = b - a
#    iterations = 0.0

    for i in range(maxiter):
        dm *= 0.5
        xm = a + dm
        fm = f(xm, *args)
        if fm*fa >= 0.0:
            a = xm
        abs_dm = fabs(dm)

        if fm == 0.0:
            return xm
        elif ytol is not None:
            if (abs_dm < (xtol + rtol*abs_dm)) and abs(fm) < ytol:
                return xm
        elif (abs_dm < (xtol + rtol*abs_dm)):
            return xm
#        elif gap_detection:
#            dy_dx = abs((fm - fa)/(a-b))
#            if dy_dx > dy_dx_limit:
#                raise DiscontinuityError("Discontinuity detected")

    raise UnconvergedError("Failed to converge after %d iterations" %maxiter)


def ridder(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter,
              full_output=False, disp=True):
    a_abs, b_abs = fabs(a), fabs(b)
    tol = xtol + rtol*(a_abs if a_abs < b_abs else b_abs)

    fa = f(a, *args)
    fb = f(b, *args)
    if fa*fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")
    elif fa == 0.0:
        return a
    elif fb == 0.0:
        return b

    for i in range(maxiter):
        dm = 0.5*(b - a)
        xm = a + dm
        fm = f(xm, *args)
        dn = copysign(1.0/sqrt(fm*fm - fa*fb), fb - fa)*fm*dm

        dn_abs, dm_abs_tol = fabs(dn), fabs(dm) - 0.5*tol
        xn = xm - copysign((dn_abs if dn_abs < dm_abs_tol else dm_abs_tol), dn)
        fn = f(xn, *args)

        if (fn*fm < 0.0):
            a = xn
            fa = fn
            b = xm
            fb = fm
        elif (fn*fa < 0.0):
            b = xn
            fb = fn
        else:
            a = xn
            fa = fn
        tol = xtol + rtol*xn
        if (fn == 0.0 or fabs(b - a) < tol):
            return xn
    raise UnconvergedError("Failed to converge after %d iterations" %maxiter) # numba: delete
#    raise UnconvergedError("Failed to converge") # numba: uncomment

def brenth(f, xa, xb, args=(),
            xtol=1e-12, rtol=4.440892098500626e-16, maxiter=100, ytol=None,
            full_output=False, disp=True, q=False,
            fa=None, fb=None, kwargs={}):
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    if fa is None:
        fpre = f(xpre, *args, **kwargs)
    else:
        fpre = fa

    if fb is None:
        fcur = f(xcur, *args, **kwargs)
    else:
        fcur = fb

    if fpre*fcur > 0.0:
        raise NotBoundedError("f(a) and f(b) must have different signs")
    elif fpre == 0.0:
        return xa
    elif fcur == 0.0:
        return xb

    for i in range(maxiter):
        if fpre*fcur < 0.0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        # Breaks a bunch of tests
#        if fpre == fcur:
#            raise UnconvergedError("Failed to converge - reached equal points after %d iterations" %i)
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = 0.5*(xtol + rtol*abs(xcur))
        sbis = 0.5*(xblk - xcur)

        if ytol is not None:
            if fcur == 0.0 or (abs(sbis) < delta) and abs(fcur) < ytol:
                return xcur
        else:
            if fcur == 0.0 or (abs(sbis) < delta):
                return xcur

        if (abs(spre) > delta and abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                if q:
                    stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))
                else:
                    stry = -fcur*(fblk - fpre)/(fblk*dpre - fpre*dblk)
            if (abs(stry + stry) < min(abs(spre), 3.0*abs(sbis) - delta)):
                # accept step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            if sbis > 0.0:
                xcur += delta
            else:
                xcur -= delta

        fcur = f(xcur, *args, **kwargs)
    raise UnconvergedError("Failed to converge after %d iterations" %maxiter)


def secant(func, x0, args=(), maxiter=100, low=None, high=None, damping=1.0,
           xtol=1.48e-8, ytol=None, x1=None, require_eval=False,
           f0=None, f1=None, bisection=False, same_tol=1.0, kwargs={},
           require_xtol=True):
    p0 = 1.0*x0
    # Logic to take a small step to calculate the approximate derivative
    if x1 is not None:
        p1 = x1
    else:
        if x0 >= 0.0:
            p1 = x0*1.0001 + 1e-4
        else:
            p1 = x0*1.0001 - 1e-4
        # May need to truncate p1
        if low is not None and p1 < low:
            p1 = low
        if high is not None and p1 > high:
            p1 = high

    # Are we already converged on either point? Do not consider checking xtol
    # if so.
    if f0 is None:
        q0 = func(p0, *args, **kwargs)
    else:
        q0 = f0
    if (ytol is not None and abs(q0) < ytol and not require_xtol) or q0 == 0.0:
        return p0

    if f1 is None:
        q1 = func(p1, *args, **kwargs)
    else:
        q1 = f1
    if (ytol is not None and abs(q1) < ytol and not require_xtol) or q1 == 0.0:
        return p1

    if bisection:
        a, b = None, None
        if q1 < 0.0:
            a = p1
        else:
            b = p1
        if q0 < 0.0:
            a = p0
        else:
            b = p0

    for i in range(maxiter):
        # Calculate new point, and truncate if necessary

        if q1 != q0:
            p = p1 - q1*(p1 - p0)/(q1 - q0)*damping
        else:
            p = p1

        if low is not None and p < low:
            p = low
        if high is not None and p > high:
            p = high

        # After computing new point
        if bisection and a is not None and b is not None:
            if not (a < p < b) or (b < p < a):
                p = 0.5*(a + b)

        # Check the exit conditions
        if ytol is not None and xtol is not None:
            # Meet both tolerance - new value is under ytol, and old value
            if abs(q1) < ytol and (not require_xtol or abs(p0 - p1) <= abs(xtol*p0)):
#            if abs(p0 - p1) <= abs(xtol*p0) and abs(q1) < ytol:
                if require_eval:
                    return p1
                return p
        elif xtol is not None:
            if abs(p0 - p1) <= abs(xtol*p0) and not (p0 == p1 and (p0 == low or p0 == high)):
                if require_eval:
                    return p1
                return p
        elif ytol is not None:
            if abs(q1) < ytol:
                if require_eval:
                    return p1
                return p

        # Check to quit after convergence check - may meet criteria
        if q1 == q0:
            # Are we close enough? Run the checks again
            if xtol is not None:
                xtol *= same_tol
            if ytol is not None:
                ytol *= same_tol

            if ytol is not None and xtol is not None:
                # Meet both tolerance - new value is under ytol, and old value
                if abs(p0 - p1) <= abs(xtol * p0) and abs(q1) < ytol:
                    return p
            elif xtol is not None:
                if abs(p0 - p1) <= abs(xtol * p0) and not (p0 == p1 and (p0 == low or p0 == high)):
                    return p
            elif ytol is not None:
                if abs(q1) < ytol:
                    return p

            # Cannot proceed, raise an error
            raise SamePointError("Convergence failed - previous points are the same", q1=q1, p1=p1, q0=q0, p0=p0)


        # Swap the points around
        p0 = p1
        q0 = q1
        p1 = p
        q1 = func(p1, *args, **kwargs)
        if q1 == 0.0:
            return p1
        if bisection:
            if q1 < 0.0:
                a = p1
            else:
                b = p1

    raise UnconvergedError("Failed to converge", iterations=i, point=p, err=q1)


def halley_compat_numba(func, x, *args):
    a, b = func(x, *args)
    return a, b, 0.0

def newton(func, x0, fprime=None, args=(), tol=None, maxiter=100,
           fprime2=None, low=None, high=None, damping=1.0, ytol=None,
           xtol=1.48e-8, require_eval=False, damping_func=None,
           bisection=False, gap_detection=False, dy_dx_limit=1e100,
           max_bound_hits=4, kwargs={}):
    """Newton's method designed to be mostly compatible with SciPy's
    implementation, with a few features added and others now implemented.

    1) No tracking of how many iterations have progressed.
    2) No ability to return a RootResults object
    3) No warnings on some cases of bad input (low tolerance, no iterations)
    4) Ability to accept True for either fprime or fprime2, which means that
       they are included in the return value of func
    5) No handling for inf or nan!
    6) Special handling for functions which need to ensure an evaluation at
       the final point
    7) Damping as a constant or a fraction
    8) Ability to perform bisection, optionally specifying a maximum range
    9) Ability to specify minimum and maximum iteration values
    10) Ability to specify a tolerance in the `y` direction
    11) Ability to pass in keyword arguments as well as positional arguments

    From scipy, with some modifications!
    https://github.com/scipy/scipy/blob/v1.1.0/scipy/optimize/zeros.py#L66-L206
    """
    if tol is not None:
        xtol = tol
    p0 = 1.0*x0
#    p1 = p0 = 1.0*x0
#    fval0 = None
    if bisection:
        a, b = None, None
        fa, fb = None, None

    fprime2_included = fprime2 == True
    fprime_included = fprime == True
    if fprime2_included:
        func2 = func
    hit_low, hit_high = 0, 0

    for it in range(maxiter):
#        if fprime2_included: # numba: uncomment
#            fval, fder, fder2 = func(p0, *args) # numba: uncomment
#        else: # numba: uncomment
#            fval, fder = func(p0, *args) # numba: uncomment
#            fder2 = 0.0 # numba: uncomment

        if fprime2_included: # numba: DELETE
            fval, fder, fder2 = func2(p0, *args, **kwargs) # numba: DELETE
        elif fprime_included: # numba: DELETE
            fval, fder = func(p0, *args, **kwargs)
        elif fprime2 is not None: #numba: DELETE
            fval = func(p0, *args, **kwargs) #numba: DELETE
            fder = fprime(p0, *args, **kwargs) #numba: DELETE
            fder2 = fprime2(p0, *args, **kwargs) #numba: DELETE
        else: #numba: DELETE
            fval = func(p0, *args, **kwargs) #numba: DELETE
            fder = fprime(p0, *args, **kwargs) #numba: DELETE

        if fval == 0.0:
            return p0 # Cannot continue or already finished
        elif fder == 0.0:
            if ytol is None or abs(fval) < ytol:
                return p0
            else:
                raise UnconvergedError("Derivative became zero; maxiter (%d) reached, value=%f " %(maxiter, p0))

        if bisection:
            if fval < 0.0:
                a = p0
                fa = fval
            else:
                b = p0 # b always has positive value of error
                fb = fval

        fder_inv = 1.0/fder
        # Compute the next point
        step = fval*fder_inv
        if damping_func is not None:
            if fprime2 is not None:
                step = step/(1.0 - 0.5*step*fder2*fder_inv)
            p = damping_func(p0, -step, damping)
#                variable, derivative, damping_factor

        elif fprime2 is None:
            p = p0 - step*damping
        else:
            p = p0 - step/(1.0 - 0.5*step*fder2*fder_inv)*damping

        if bisection and a is not None and b is not None:
            if (not (a < p < b) and not (b < p < a)):
#                if p < 0.0:
#                    if p < a:
#                    print('bisecting')
                p = 0.5*(a + b)
#                else:
#                    if p > b:
#                        p = 0.5*(a + b)
                if gap_detection:
                    # Change in function value required to get goal in worst case
                    dy_dx = abs((fa- fb)/(a-b))
                    if dy_dx > dy_dx_limit: #or dy_dx > abs(fder)*10:
                        raise DiscontinuityError("Discontinuity detected")

        if low is not None and p < low:
            hit_low += 1
            if p0 == low and hit_low > max_bound_hits:
                if abs(fval) < ytol:
                    return low
                else:
                    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " % (maxiter, p))
                # Stuck - not going to converge, hammering the boundary. Check ytol
            p = low
        if high is not None and p > high:
            hit_high += 1
            if p0 == high and hit_high > max_bound_hits:
                if abs(fval) < ytol:
                    return high
                else:
                    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " % (maxiter, p))
            p = high




        # p0 is last point (fval at that point), p is new

        if ytol is not None and xtol is not None:
            # Meet both tolerance - new value is under ytol, and old value
            if abs(p - p0) < abs(xtol*p) and abs(fval) < ytol:
                if require_eval:
                    return p0
                return p
        elif xtol is not None:
            if abs(p - p0) < abs(xtol*p):
                if require_eval:
                    return p0
                return p
        elif ytol is not None:
            if abs(fval) < ytol:
                if require_eval:
                    return p0
                return p

#            fval0, fval1 = fval, fval0
#            p0, p1 = p, p0
# need a history of fval also
        p0 = p
#    else:
#        return secant(func, x0, args=args, maxiter=maxiter, low=low, high=high,
#                      damping=damping,
#                      xtol=xtol, ytol=ytol, kwargs=kwargs)
#
    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " %(maxiter, p))

def halley(func, x0, args=(), maxiter=100,
           low=None, high=None, damping=1.0, ytol=None,
           xtol=1.48e-8, require_eval=False, damping_func=None,
           bisection=False,
           max_bound_hits=4, kwargs={}):
    p0 = 1.0*x0
    if bisection:
        a, b = None, None
        fa, fb = None, None

    hit_low, hit_high = 0, 0

    for it in range(maxiter):
        fval, fder, fder2 = func(p0, *args)
        if fval == 0.0:
            return p0 # Cannot continue or already finished
        elif fder == 0.0:
            if ytol is None or abs(fval) < ytol:
                return p0
            else:
                raise UnconvergedError("Derivative became zero; maxiter (%d) reached, value=%f " %(maxiter, p0))

        if bisection:
            if fval < 0.0:
                a = p0
                fa = fval
            else:
                b = p0 # b always has positive value of error
                fb = fval

        fder_inv = 1.0/fder
        # Compute the next point
        step = fval*fder_inv
        if damping_func is not None:
            step = step/(1.0 - 0.5*step*fder2*fder_inv)
            p = damping_func(p0, -step, damping)
        else:
            p = p0 - step/(1.0 - 0.5*step*fder2*fder_inv)*damping

        if bisection and a is not None and b is not None:
            if (not (a < p < b) and not (b < p < a)):
                p = 0.5*(a + b)

        if low is not None and p < low:
            hit_low += 1
            if p0 == low and hit_low > max_bound_hits:
                if abs(fval) < ytol:
                    return low
                else:
                    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " % (maxiter, p))
                # Stuck - not going to converge, hammering the boundary. Check ytol
            p = low
        if high is not None and p > high:
            hit_high += 1
            if p0 == high and hit_high > max_bound_hits:
                if abs(fval) < ytol:
                    return high
                else:
                    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " % (maxiter, p))
            p = high




        # p0 is last point (fval at that point), p is new

        if ytol is not None and xtol is not None:
            # Meet both tolerance - new value is under ytol, and old value
            if abs(p - p0) < abs(xtol*p) and abs(fval) < ytol:
                if require_eval:
                    return p0
                return p
        elif xtol is not None:
            if abs(p - p0) < abs(xtol*p):
                if require_eval:
                    return p0
                return p
        elif ytol is not None:
            if abs(fval) < ytol:
                if require_eval:
                    return p0
                return p

        p0 = p
    raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%f " %(maxiter, p))

def newton_err(F):
    err = sum([abs(i) for i in F])
    return err

def basic_damping(x, dx, damping, *args):
    N = len(x)
    x2 = [0.0]*N
    for i in range(N):
        x2[i] = x[i] + dx[i]*damping
    return x2

def solve_2_direct(mat, vec):
    ab = mat[0]
    cd = mat[1]
    a, b = ab[0], ab[1]
    c, d = cd[0], cd[1]
    e, f = vec[0], vec[1]

    x0 = 1.0/(a*d - b*c)

    sln = [0.0]*2
    sln[0] = x0*(-b*f + d*e)
    sln[1] = x0*(a*f - c*e)
    return sln

def solve_3_direct(mat, vec):
    a, b, c = mat[0]
    d, e, f = mat[1]
    g, h, i = mat[2]
    j, k, l = vec

    x0 = a*e
    x1 = -b*d + x0
    x2 = a*i - c*g
    x3 = a*f - c*d
    x4 = a*h - b*g
    x5 = x1*x2 - x3*x4
    x6 = 1.0/x5
    x7 = b*x3 - c*x1
    x8 = 1.0/x1
    x9 = d*x4 - g*x1
    x10 = j*x8
    x11 = a*k
    x12 = a*l

    ans = [0.0]*3
    ans[0] = x6*(-k*x8*(b*x5 + x4*x7) + l*x7 + x10*(x0*x5 + x7*x9)/a)
    ans[1] = x6*(-x10*(d*x5 + x3*x9) + x11*x2 - x12*x3)
    ans[2] = x6*(j*x9 + x1*x12 - x11*x4)
    return ans

def solve_4_direct(mat, vec):
    a, b, c, d = mat[0]
    e, f, g, h = mat[1]
    i, j, k, l = mat[2]
    m, n, o, p = mat[3]
    q, r, s, t = vec
    x0 = a*f
    x1 = -b*e + x0
    x2 = x1*(a*k - c*i)
    x3 = a*g - c*e
    x4 = a*j - b*i
    x5 = x2 - x3*x4
    x6 = a*h - d*e
    x7 = a*n - b*m
    x8 = x1*(a*p - d*m) - x6*x7
    x9 = x1*(a*l - d*i) - x4*x6
    x10 = x1*(a*o - c*m) - x3*x7
    x11 = -x10*x9 + x5*x8
    x12 = 1.0/x11
    x13 = -b*x3 + c*x1
    x14 = x13*x9
    x15 = x5*(-b*x6 + d*x1)
    x16 = -x14 + x15
    x17 = 1.0/x5
    x18 = s*x17
    x19 = x10*x4 - x5*x7
    x20 = 1.0/x1
    x21 = x17*x20
    x22 = e*x4 - i*x1
    x23 = x5*(e*x7 - m*x1)
    x24 = x10*x22
    x25 = x23 - x24
    x26 = q*x17
    x27 = x20*x26
    x28 = x3*x9
    x29 = x5*x6
    x30 = a*t
    x31 = -x28 + x29
    x32 = a*r
    x33 = a*s*x1
    x34 = x1*x30
    x35 = -x23 + x24
    ans = [0.0]*4
    ans[0] = x12*(-r*x21*(-x11*(-b*x5 + x13*x4) + x16*x19) + t*(x14 - x15)
             - x18*(-x10*x16 + x11*x13) - x27*(-x11*(x0*x5 - x13*x22) + x16*x25)/a)
    ans[1] = x12*(-a*x18*(-x10*x31 + x11*x3) - x21*x32*(-x11*x2 + x19*x31)
             - x27*(x11*(e*x5 + x22*x3) + x25*x31) + x30*(x28 - x29))
    ans[2] = x12*(-x17*x32*(x11*x4 + x19*x9) + x26*(x11*x22 + x35*x9) + x33*x8 - x34*x9)
    ans[3] = x12*(-q*x35 - x10*x33 + x19*x32 + x34*x5)
    return ans


def newton_system(f, x0, jac, xtol=None, ytol=None, maxiter=100, damping=1.0,
                  args=(), damping_func=None, solve_func=py_solve): # numba: delete
#                  args=(), damping_func=None, solve_func=np.linalg.solve): # numba: uncomment
    jac_also = True if jac == True else False


    if jac_also:
        fcur, j = f(x0, *args)
    else: # numba: delete
        fcur = f(x0, *args)  # numba: delete


    err0 = 0.0
    for v in fcur:
        err0 += abs(v)

    if xtol is None and (ytol is not None and err0 < ytol):
        return x0, 0
    else:
        x = x0
        if not jac_also:  # numba: delete
            j = jac(x, *args)  # numba: delete

    iteration = 1
    while iteration < maxiter:
#        dx = solve_func(j, -fcur) # numba: uncomment
        dx = solve_func(j, [-v for v in fcur]) # numba: delete
        if damping_func is None:
#            x = x + dx*damping # numba: uncomment
            x = [xi + dxi*damping for xi, dxi in zip(x, dx)] # numba: delete
        else:
            x = damping_func(x, dx, damping, *args)
        if jac_also:
            fcur, j = f(x, *args)
        else:  # numba: delete
            fcur = f(x, *args)  # numba: delete

        iteration += 1

        err0 = 0.0
        for v in fcur:
            err0 += abs(v)
        if xtol is not None:
            if (norm2(fcur) < xtol) and (ytol is None or err0 < ytol):
                break
        elif ytol is not None:
            if err0 < ytol:
                break

        if not jac_also:  # numba: delete
            j = jac(x, *args)  # numba: delete

    if xtol is not None and norm2(fcur) > xtol:
        raise UnconvergedError("Failed to converge")
#        raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%s" %(maxiter, x))
    if ytol is not None:
        err0 = 0.0
        for v in fcur:
            err0 += abs(v)
        if err0 > ytol:
#            raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%s" %(maxiter, x))
            raise UnconvergedError("Failed to converge")

    return x, iteration

def newton_minimize(f, x0, jac, hess, xtol=None, ytol=None, maxiter=100, damping=1.0,
                  args=(), damping_func=None):
    jac_also = True if jac == True else False
    hess_also = True if hess == True else False
    def err(F):
        err = sum([abs(i) for i in F])
        return err
    if hess_also:
        fcur, j, h = f(x0, *args)
    elif jac_also:
        fcur, j = f(x0, *args)
        h = hess(x0, *args)
    else:
        fcur = f(x0, *args)
        j = jac(x0, *args)
        h = hess(x0, *args)
    iter = 0
    x = x0
    while iter < maxiter:
        dx = py_solve(h, [-v for v in j])
        if damping_func is None:
            x = [xi + dxi*damping for xi, dxi in zip(x, dx)]
        else:
            x = damping_func(x, dx, damping)
        if hess_also:
            fcur, j, h = f(x, *args)
        elif jac_also:
            fcur, j = f(x, *args)
            h = hess(x, *args)
        else:
            fcur = f(x, *args)
            j = jac(x, *args)
            h = hess(x, *args)

        iter += 1
        if xtol is not None and norm2(j) < xtol:
            break
        if ytol is not None and err(j) < ytol:
            break

    if xtol is not None and norm2(j) > xtol:
        raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%s " %(maxiter, x))
    if ytol is not None and err(j) > ytol:
        raise UnconvergedError("Failed to converge; maxiter (%d) reached, value=%s " %(maxiter, x))

    return x, iter


def broyden2(xs, fun, jac, xtol=1e-7, maxiter=100, jac_has_fun=False,
             skip_J=False, args=()):
    iter = 0
    if skip_J:
        fcur = fun(xs, *args)
        N = len(fcur)
        J = eye(N)
    elif jac_has_fun:
        fcur, J = jac(xs, *args)
        J = inv(J)
    else:
        fcur = fun(xs, *args)
        J = inv(jac(xs, *args))

    N = len(fcur)
    eqns = range(N)

    err = 0.0
    for fi in fcur:
        err += abs(fi)

    while err > xtol and iter < maxiter:
        s = dot(J, fcur)

        xs = [xs[i] - s[i] for i in eqns]

        fnew = fun(xs, *args)
        z = [fnew[i] - fcur[i] for i in eqns]

        u = dot(J, z)

        d = [-i for i in s]


        dmu = [d[i]-u[i] for i in eqns]
        dmu_d = inner_product(dmu, d)
        den_inv = 1.0/inner_product(d, u)
        factor = den_inv*dmu_d
        J_delta = [[factor*j for j in row] for row in J]
        for i in eqns:
            for j in eqns:
                J[i][j] += J_delta[i][j]

        fcur = fnew
        iter += 1
        err = 0.0
        for fi in fcur:
            err += abs(fi)

    return xs, iter

def normalize(values):
    r'''Simple function which normalizes a series of values to be from 0 to 1,
    and for their sum to add to 1.

    .. math::
        x = \frac{x}{sum_i x_i}

    Parameters
    ----------
    values : array-like
        array of values

    Returns
    -------
    fractions : array-like
        Array of values from 0 to 1

    Notes
    -----
    Does not work on negative values, or handle the case where the sum is zero.

    Examples
    --------
    >>> normalize([3, 2, 1])
    [0.5, 0.3333333333333333, 0.16666666666666666]
    '''
    tot_inv = 1.0/sum(values)
    return [i*tot_inv for i in values]

# NOTE: the first value of each array is used; it is only the indexes that
# are adjusted for fortran
def func_35_splev(arg, t, l, l1, k2, nk1):
    # minus 1 index
    if arg >= t[l-1] or l1 == k2:
        arg, t, l, l1, nk1, leave = func_40_splev(arg, t, l, l1, nk1)
        # Always leaves here
        return arg, t, l, l1, k2, nk1

    l1 = l
    l = l - 1
    arg, t, l, l1, k2, nk1 = func_35_splev(arg, t, l, l1, k2, nk1)
    return arg, t, l, l1, k2, nk1

def func_40_splev(arg, t, l, l1, nk1):
    if arg < t[l1-1] or l == nk1: # minus 1 index
        return arg, t, l, l1, nk1, 1
    l = l1
    l1 = l + 1
    arg, t, l, l1, nk1, leave = func_40_splev(arg, t, l, l1, nk1)
    return arg, t, l, l1, nk1, leave

def py_splev(x, tck, ext=0, t=None, c=None, k=None):
    """Evaluate a B-spline using a pure-python port of FITPACK's splev. This is
    not fully featured in that it does not support calculating derivatives.
    Takes the knots and coefficients of a B-spline tuple, and returns the value
    of the smoothing polynomial.

    Parameters
    ----------
    x : float or list[float]
        An point or array of points at which to calculate and return the value
        of the  spline, [-]
    tck : 3-tuple
        Ssequence of length 3 returned by
        `splrep` containing the knots, coefficients, and degree
        of the spline, [-]
    ext : int, optional, default 0
        If `x` is not within the range of the spline, this handles the
        calculation of the results.

        * For ext=0, extrapolate the value
        * For ext=1, return 0 as the value
        * For ext=2, raise a ValueError on that point and fail to return values
        * For ext=3, return the boundary value as the value

    Returns
    -------
    y : list
        The array of calculated values; the spline function evaluated at
        the points in `x`, [-]

    Notes
    -----
    The speed of this for a scalar value in CPython is approximately 15%
    slower than SciPy's FITPACK interface. In PyPy, this is 10-20 times faster
    than using it (benchmarked on PyPy 6).

    There could be more bugs in this port.
    """
    e = ext
    if tck is not None:
        t, c, k = tck
    x = [x]
#    if isinstance(x, (float, int, complex)):
#        x = [x]

    m = 1#len(x)
    n = len(t)
    y = [] # output array

    k1 = k + 1
    k2 = k1 + 1
    nk1 = n - k1
    tb = t[k1-1] # -1 to get right index
    te = t[nk1 ]  # -1 to get right index; + 1 - 1
    l = k1
    l1 = l + 1

    for i in range(0, m): # m is only 1
        # i only used in loop for 1
        arg = x[i]
        if arg < tb or arg > te:
            if e == 0:
                arg, t, l, l1, k2, nk1 = func_35_splev(arg, t, l, l1, k2, nk1)
            elif e == 1:
                y.append(0.0)
                continue
            elif e == 2:
                raise ValueError("X value not in domain; set `ext` to 0 to "
                                 "extrapolate")
            elif e == 3:
                if arg < tb:
                    arg = tb
                else:
                    arg = te
                arg, t, l, l1, k2, nk1 = func_35_splev(arg, t, l, l1, k2, nk1)
        else:
            arg, t, l, l1, k2, nk1 = func_35_splev(arg, t, l, l1, k2, nk1)

        # Local arrays used in fpbspl and to carry its result
        h = [0.0]*20
        hh = [0.0]*19

        fpbspl(t, n, k, arg, l, h, hh)
        sp = 0.0E0
        ll = l - k1
        for j in range(0, k1):
            ll = ll + 1
            sp = sp + c[ll-1]*h[j] # -1 to get right index
        y.append(sp)
    return y[0]
#    if len(y) == 1:
#        return y[0]
#    return y


def py_bisplev(x, y, tck, dx=0, dy=0):
    """Evaluate a bivariate B-spline or its derivatives. For scalars, returns a
    float; for other inputs, mimics the formats of SciPy's `bisplev`.

    Parameters
    ----------
    x : float or list[float]
        x value (rank 1), [-]
    y : float or list[float]
        y value (rank 1), [-]
    tck : tuple(list, list, list, int, int)
        Tuple of knot locations, coefficients, and the degree of the spline,
        [tx, ty, c, kx, ky], [-]
    dx : int, optional
        Order of partial derivative with respect to `x`, [-]
    dy : int, optional
        Order of partial derivative with respect to `y`, [-]

    Returns
    -------
    values : float or list[list[float]]
        Calculated values from spline or their derivatives; according to the
        same format as SciPy's `bisplev`, [-]

    Notes
    -----
    Use `bisplrep` to generate the `tck` representation; there is no Python
    port of it.
    """
    tx, ty, c, kx, ky = tck
    if isinstance(x, (float, int)):
        x = [x]
    if isinstance(y, (float, int)):
        y = [y]

    z = [[cy_bispev(tx, ty, c, kx, ky, [xi], [yi])[0] for yi in y] for xi in x]
    if len(x) == len(y) == 1:
        return z[0][0]
    return z


def fpbspl(t, n, k, x, l, h, hh):
    """subroutine fpbspl evaluates the (k+1) non-zero b-splines of degree k at
    t(l) <= x < t(l+1) using the stable recurrence relation of de boor and cox.

    All arrays are 1d! Optimized the assignment and order and so on.
    """
    h[0] = 1.0
    for j in range(1, k + 1):
        hh[0:j] = h[0:j]
        h[0] = 0.0
        for i in range(j):
            li = l+i
            f = hh[i]/(t[li] - t[li - j])
            h[i] = h[i] + f*(t[li] - x)
            h[i + 1] = f*(x - t[li - j])


def init_w(t, k, x, lx, w):
    tb = t[k]
    n = len(t)
    m = len(x)
    h = [0]*6
    hh = [0]*5
    te = t[n - k - 1]
    l1 = k + 1
    l2 = l1 + 1
    for i in range(m):
        arg = x[i]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
        while not (arg < t[l1] or l1 == (n - k - 1)):
            l1 = l2
            l2 = l1 + 1
        fpbspl(t, n, k, arg, l1, h, hh)

        lx[i] = l1 - k - 1
        for j in range(k + 1):
            w[i][j] = h[j]


def cy_bispev(tx, ty, c, kx, ky, x, y):
    """Possible optimization: Do not evaluate derivatives, ever."""
    nx = len(tx)
    ny = len(ty)
    mx = len(x)
    my = len(y)

    kx1 = kx + 1
    ky1 = ky + 1

    nkx1 = nx - kx1
    nky1 = ny - ky1

    wx = [[0.0]*kx1]*mx
    wy = [[0.0]*ky1]*my
    lx = [0]*mx
    ly = [0]*my

    size_z = mx*my

    z = [0.0]*size_z
    init_w(tx, kx, x, lx, wx)
    init_w(ty, ky, y, ly, wy)

    for j in range(my):
        for i in range(mx):
            sp = 0.0
            err = 0.0
            for i1 in range(kx1):
                for j1 in range(ky1):
                    l2 = lx[i]*nky1 + ly[j] + i1*nky1 + j1
                    a = c[l2]*wx[i][i1]*wy[j][j1] - err
                    tmp = sp + a
                    err = (tmp - sp) - a
                    sp = tmp
            z[j*mx + i] += sp
    return z


p_0_70 = array_if_needed([0.06184590404457956, -0.7460693871557973, 2.2435704485433376, -2.1944070385048526, 0.3382265629285811, 0.2791966558569478])
q_0_07 = array_if_needed([-0.005308735283483908, 0.1823421262956287, -1.2364596896290079, 2.9897802200092296, -2.9365321202088004, 1.0])
p_07_099 = array_if_needed([7543860.817140365, -10254250.429758755, -4186383.973408412, 7724476.972409749, -3130743.609030545, 600806.068543299, -62981.15051292659, 3696.7937385473397, -114.06795167646395, 1.4406337969700391])
q_07_099 = array_if_needed([-1262997.3422452002, 10684514.56076485, -16931658.916668657, 10275996.02842749, -3079141.9506451315, 511164.4690136096, -49254.56172495263, 2738.0399260270983, -81.36790509581284, 1.0])
p_099_1 = array_if_needed([8.548256176424551e+34, 1.8485781239087334e+35, -2.1706889553798647e+34, 8.318563643438321e+32, -1.559802348661511e+31, 1.698939241177209e+29, -1.180285031647229e+27, 5.531049937687143e+24, -1.8085903366375877e+22, 4.203276811951035e+19, -6.98211620300421e+16, 82281997048841.92, -67157299796.61345, 36084814.54808544, -11478.108105137717, 1.6370226052761176])
q_099_1 = array_if_needed([-1.9763570499484274e+35, 1.4813997374958851e+35, -1.4773854824041134e+34, 5.38853721252814e+32, -9.882387315028929e+30, 1.0635231532999732e+29, -7.334629044071992e+26, 3.420655574477631e+24, -1.1147787784365177e+22, 2.584530363912858e+19, -4.285376337404043e+16, 50430830490687.56, -41115254924.43107, 22072284.971253656, -7015.799744041691, 1.0])

def polylog2(x):
    r'''Simple function to calculate PolyLog(2, x) from ranges 0 <= x <= 1,
    with relative error guaranteed to be < 1E-7 from 0 to 0.99999. This
    is a Pade approximation, with three coefficient sets with splits at 0.7
    and 0.99. An exception is raised if x is under 0 or above 1.


    Parameters
    ----------
    x : float
        Value to evaluate PolyLog(2, x) T

    Returns
    -------
    y : float
        Evaluated result

    Notes
    -----
    Efficient (2-4 microseconds). No implementation of this function exists in
    SciPy. Derived with mpmath's pade approximation.
    Required for the entropy integral of
    :obj:`thermo.heat_capacity.Zabransky_quasi_polynomial`.

    Examples
    --------
    >>> polylog2(0.5)
    0.5822405264516294
    '''
    if 0 <= x <= 0.7:
        p = p_0_70
        q = q_0_07
        offset = 0.26
    elif 0.7 < x <= 0.99:
        p = p_07_099
        q = q_07_099
        offset = 0.95
    elif 0.99 < x <= 1:
        p = p_099_1
        q = q_099_1
        offset = 0.999
    else:
        raise ValueError('Approximation is valid between 0 and 1 only.')
    x = x - offset
    return horner(p, x)/horner(q, x)


global sp_root
sp_root = None
def root(*args, **kwargs):
    global sp_root
    if sp_root is None:
        from scipy.optimize import root as sp_root
    return sp_root(*args, **kwargs)

global sp_minimize
sp_minimize = None
def minimize(*args, **kwargs):
    global sp_minimize
    if sp_minimize is None:
        from scipy.optimize import minimize as sp_minimize
    return sp_minimize(*args, **kwargs)


global sp_fsolve
sp_fsolve = None
def fsolve(*args, **kwargs):
    global sp_fsolve
    if sp_fsolve is None:
        from scipy.optimize import fsolve as sp_fsolve
    return sp_fsolve(*args, **kwargs)

def fixed_quad_Gauss_Kronrod(f, a, b, k_points, k_weights, l_weights, args):
    '''
    Note: This type of function cannot be fast in numba right now, the function
    call is too slow!
    https://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    '''
    val_gauss_kronrod = val_gauss_legendre = 0.0
    diff = b - a
    fact = 0.5*diff
    N = len(k_points)
#     center = N//2
#     sp = [0.0]*N
#     fx_gk = [0.0]*N
#     fx_gl = [0.0]*center
    k = 0
    for i in range(N):
        x0 = 0.5*(1.0 - k_points[i])
        x1 = 0.5*(1.0 + k_points[i])
#         sp[i] =
        x = a*x0 + b*x1
#         fx_gk[i] =
        y = f(x, *args)
        val_gauss_kronrod += fact*y*k_weights[i]
        if i%2:
#             fx_gl[k] = y
            val_gauss_legendre += fact*y*l_weights[k]
            k += 1
    return val_gauss_kronrod, val_gauss_kronrod-val_gauss_legendre

def quad_adaptive(f, a, b, args=(), kronrod_points=array_if_needed(kronrod_points[10]),
                  kronrod_weights=array_if_needed(kronrod_weights[10]), legendre_weights=array_if_needed(legendre_weights[10]),
                  epsrel=1.49e-8, epsabs=1.49e-8, depth=0, points=None):
    # Disregard `points` for now
    area, err_abs = fixed_quad_Gauss_Kronrod(f, a, b, kronrod_points, kronrod_weights,
                                             legendre_weights, args)
    # Match behavior, documented at https://www.johndcook.com/blog/2012/03/20/scipy-integration/
    #and with a good test case at https://www.johndcook.com/blog/2012/03/20/scipy-integration/
    if (abs(err_abs) < epsabs or (area == 0.0 or abs(err_abs/area) < epsrel)) or depth > 6:
#        print((a, b), area, abs(err_abs),  epsabs, abs(err_abs/area), epsrel, depth)
        return area, err_abs

    mid = a + (b-a)*0.5
    area_A, err_abs_A = quad_adaptive(f, a, mid, args, kronrod_points, kronrod_weights, legendre_weights,
                    epsrel=epsrel, epsabs=epsabs*0.5, depth=depth+1)
    area_B, err_abs_B = quad_adaptive(f, mid, b, args, kronrod_points, kronrod_weights, legendre_weights,
                    epsrel=epsrel, epsabs=epsabs*0.5, depth=depth+1)
    return area_A + area_B, abs(err_abs_A) + abs(err_abs_B)

global sp_quad
sp_quad = None
def lazy_quad(f, a, b, args=(), epsrel=1.49e-08, epsabs=1.49e-8, **kwargs):
    global sp_quad
    if not IS_PYPY:
        if sp_quad is None:
            from scipy.integrate import quad as sp_quad
        return sp_quad(f, a, b, args, epsrel=epsrel, epsabs=epsabs, **kwargs)
    else:
        return quad_adaptive(f, a, b, ags=args, epsrel=epsrel, epsabs=epsabs)
#        n = 300
#        return fixed_quad_Gauss_Kronrod(f, a, b, kronrod_points[n], kronrod_weights[n], legendre_weights[n], args)


def _call_nquad(x, func, range_funcs, epsrel, epsabs, *args):
    return nquad(func, range_funcs, args=(x,) +args , epsrel=epsrel, epsabs=epsabs)

def nquad(func, ranges, args=(), epsrel=1.48e-8, epsabs=1.48e-8):
    my_low, my_high = ranges[-1](*args)
    if len(ranges) == 1:
        return quad_adaptive(func, my_low, my_high, args=args, epsrel=epsrel, epsabs=epsabs)
    #
    return quad_adaptive(_call_nquad, my_low, my_high,
                         args=(func, ranges[:-1], epsrel, epsabs)  +args,
                         epsrel=epsrel, epsabs=epsabs)

def dblquad(func, a, b, hfun, gfun, args=(), epsrel=1.48e-12, epsabs=1.48e-15):
    """Nominally working, but trying to use it has exposed the expected bugs in
    `quad_adaptive`."""
    def inner_func(y, *args):
        full_args = (y,)+args
        quad_fluids = quad_adaptive(func, hfun(y, *args), gfun(y, *args), args=full_args, epsrel=epsrel, epsabs=epsabs)[0]
        # from scipy.integrate import quad as quad2
        # quad_sp = quad2(func, hfun(y), gfun(y), args=full_args, epsrel=epsrel, epsabs=epsabs)[0]
        # print(quad_fluids, quad_sp, hfun(y), gfun(y), full_args, )
        return quad_fluids
#         return quad(func, hfun(y), gfun(y), args=(y,)+args, epsrel=epsrel, epsabs=epsabs)[0]
    return quad_adaptive(inner_func, a, b, args=args, epsrel=epsrel, epsabs=epsabs)

if IS_PYPY:
    quad = quad_adaptive
else:
    quad = lazy_quad


# interp, horner, derivative methods (and maybe newton?) should always be used.
if not IS_PYPY:
    def splev(*args, **kwargs):
        from scipy.interpolate import splev
        return splev(*args, **kwargs)
    def bisplev(*args, **kwargs):
        from scipy.interpolate import bisplev
        return bisplev(*args, **kwargs)

else:
    splev, bisplev = py_splev, py_bisplev

# Try out mpmath for special functions anyway
has_scipy = False
if not SKIP_DEPENDENCIES and not is_micropython and not is_ironpython:
    has_scipy = True
    # try:
    #     import scipy
    #     has_scipy = True
    # except ImportError:
    #     has_scipy = False
else:
    has_scipy = False

erf = None
try:
    from math import erf
except ImportError:
    # python 2.6 or other implementations?
    pass




def _lambertw_err(x, y):
    return x*exp(x) - y
def py_lambertw(y, k=0):
    """For x > 0, the is always only one real solution For -1/e < x < 0, two
    real solutions.

    Besides compatibility with scipy, the result should have a complex part
    because micropython doesn't support .real on floats
    """
    # Works for real inputs only, two main branches
    if k == 0:
        # Branches dead at -1
        # -1 is hard limit for real in this branch
        # 700 is safe upper limit for exp
        # Input should be between -1 and +BIGNUMBER
        return brenth(_lambertw_err, -1.0, 700.0, (y,)) + 0.0j
    elif k == -1:
        # Input should be between 0 and -1/e
        # not a big input range!
        return brenth(_lambertw_err, -700.0, -1.0, (y,)) + 0.0j
    else:
        raise ValueError("Other branches not supported")
#has_scipy = False
if IS_PYPY:
    lambertw = py_lambertw

if has_scipy:
    if not IS_PYPY:
        def lambertw(*args, **kwargs):
            from scipy.special import lambertw
            return lambertw(*args, **kwargs)
    def ellipe(m):
        from scipy.special import ellipe
        return ellipe(m)
    def gammaincc(*args, **kwargs):
        from scipy.special import gammaincc
        return gammaincc(*args, **kwargs)
    def i1(*args, **kwargs):
        from scipy.special import i1
        return i1(*args, **kwargs)
    def i0(*args, **kwargs):
        from scipy.special import i0
        return i0(*args, **kwargs)
    def k1(*args, **kwargs):
        from scipy.special import k1
        return k1(*args, **kwargs)
    def k0(*args, **kwargs):
        from scipy.special import k0
        return k0(*args, **kwargs)
    def iv(*args, **kwargs):
        from scipy.special import iv
        return iv(*args, **kwargs)
    def hyp2f1(*args, **kwargs):
        from scipy.special import hyp2f1
        return hyp2f1(*args, **kwargs)
    def ellipkinc(phi, m):
        from scipy.special import ellipkinc
        return ellipkinc(phi, m)
    def ellipeinc(phi, m):
        from scipy.special import ellipeinc
        return ellipeinc(phi, m)
    if erf is None:
        def erf(*args, **kwargs):
            from scipy.special import erf
            return erf(*args, **kwargs)


#    from scipy.special import lambertw, ellipe, gammaincc # fluids
#    from scipy.special import i1, i0, k1, k0, iv # ht
#    from scipy.special import hyp2f1
#    if erf is None:
#        from scipy.special import erf
else:
    lambertw = py_lambertw
        # scipy is not available... fall back to mpmath as a Pure-Python implementation
        # However, lazy load all functions
    #    from mpmath import lambertw # Same branches as scipy, supports .real
            # Figured out this definition from test_precompute_gammainc.py in scipy
    if erf is None:
        def erf(*args, **kwargs):
            import mpmath
            return mpmath.erf(*args, **kwargs)

    def gammaincc(a, x):
        import mpmath
        return mpmath.gammainc(a, a=x, regularized=True)

    def ellipe(*args, **kwargs):
        import mpmath
        return mpmath.ellipe(*args, **kwargs)

    def gammaincc(a, x):
        import mpmath
        return mpmath.gammainc(a, a=x, regularized=True)

    def iv(*args, **kwargs):
        import mpmath
        return mpmath.besseli(*args, **kwargs)

    def hyp2f1(*args, **kwargs):
        import mpmath
        return mpmath.hyp2f1(*args, **kwargs)

    def iv(*args, **kwargs):
        import mpmath
        return mpmath.besseli(*args, **kwargs)
    def i1(x):
        import mpmath
        return mpmath.mpmath.besseli(1, x)
    def i0(x):
        import mpmath
        return mpmath.mpmath.besseli(0, x)
    def k1(x):
        import mpmath
        return mpmath.mpmath.besselk(1, x)
    def k0(x):
        import mpmath
        return mpmath.mpmath.besselk(0, x)
    def ellipkinc(phi, m):
        import mpmath
        return mpmath.mpmath.ellipf(phi, m)
    def ellipeinc(phi, m):
        import mpmath
        return mpmath.ellipe.ellipeinc(phi, m)

try:
    if FORCE_PYPY:
        lambertw = py_lambertw
        from scipy.special import ellipe, gammaincc, gamma, ellipkinc, ellipeinc # fluids
        from scipy.special import i1, i0, k1, k0, iv # ht
        from scipy.special import hyp2f1
        if erf is None:
            from scipy.special import erf

except:
    pass
