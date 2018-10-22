# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import sin, exp, pi, fabs, copysign
import sys
from sys import float_info

__all__ = ['horner', 'interp',
           'implementation_optimize_tck', 'tck_interp2d_linear',
           'bisect', 'ridder', 'brenth', 'newton', 
           'splev', 'bisplev', 'derivative',
           'IS_PYPY',
           ]

try:
    # The right way imports the platform module which costs to ms to load!
    #     implementation = platform.python_implementation()
    IS_PYPY = 'PyPy' in sys.version
except AttributeError:
    IS_PYPY = False
  
try:
    import numpy as np
except ImportError:
    IS_PYPY = True

#IS_PYPY = True

epsilon = float_info.epsilon
_iter = 100
_xtol = 1e-12
_rtol = float_info.epsilon*2.0
one_epsilon_larger = 1.0 + float_info.epsilon
one_epsilon_smaller = 1.0 - float_info.epsilon


def product(l):
    tot = 1.0
    for i in l:
        tot *= i
    return tot

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
 (3, 11): [0.006779100529100529, -0.08339947089947089, 0.48303571428571435, -1.7337301587301588, 2.3180555555555555,
           0.0, -2.3180555555555555, 1.7337301587301588, -0.48303571428571435, 0.08339947089947089,
           -0.006779100529100529],
 (3, 13): [-0.0015839947089947089, 0.02261904761904762, -0.1530952380952381, 0.6572751322751322, -1.9950892857142857,
           2.527142857142857, 0.0, -2.527142857142857, 1.9950892857142857, -0.6572751322751322, 0.1530952380952381,
           -0.02261904761904762, 0.0015839947089947089],
 (3, 15): [0.0003724747474747475, -0.006053691678691679, 0.04682990620490621, -0.2305699855699856, 0.8170667989417989,
           -2.2081448412698412, 2.6869345238095237, 0.0, -2.6869345238095237, 2.2081448412698412, -0.8170667989417989,
           0.2305699855699856, -0.04682990620490621, 0.006053691678691679, -0.0003724747474747475],
 (3, 17): [-8.810006131434702e-05, 0.0016058756058756059, -0.013982697196982911, 0.07766492766492766,
           -0.31074104136604136, 0.9613746993746994, -2.384521164021164, 2.81291761148904, 0.0, -2.81291761148904,
           2.384521164021164, -0.9613746993746994, 0.31074104136604136, -0.07766492766492766, 0.013982697196982911,
           -0.0016058756058756059, 8.810006131434702e-05],
 (3, 19): [2.0943672729387014e-05, -0.00042319882498453924, 0.00409817266067266, -0.025376055161769447,
           0.11326917130488559, -0.3904945471195471, 1.0909741462241462, -2.532634817563389, 2.9147457482993198, 0.0,
           -2.9147457482993198, 2.532634817563389, -1.0909741462241462, 0.3904945471195471, -0.11326917130488559,
           0.025376055161769447, -0.00409817266067266, 0.00042319882498453924, -2.0943672729387014e-05],
 (4, 5): [1.0, -4.0, 6.0, -4.0, 1.0],
 (4, 7): [-0.16666666666666666, 2.0, -6.5, 9.333333333333334, -6.5, 2.0, -0.16666666666666666],
 (4, 9): [0.029166666666666667, -0.4, 2.8166666666666664, -8.133333333333333, 11.375, -8.133333333333333,
          2.8166666666666664, -0.4, 0.029166666666666667],
 (4, 11): [-0.005423280423280424, 0.08339947089947089, -0.644047619047619, 3.4674603174603176, -9.272222222222222,
           12.741666666666665, -9.272222222222222, 3.4674603174603176, -0.644047619047619, 0.08339947089947089,
           -0.005423280423280424],
 (4, 13): [0.0010559964726631393, -0.018095238095238095, 0.1530952380952381, -0.8763668430335096, 3.9901785714285714,
           -10.108571428571429, 13.717407407407407, -10.108571428571429, 3.9901785714285714, -0.8763668430335096,
           0.1530952380952381, -0.018095238095238095, 0.0010559964726631393],
 (5, 7): [-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5],
 (5, 9): [0.16666666666666669, -1.5, 4.333333333333333, -4.833333333333334, 0.0, 4.833333333333334, -4.333333333333333,
          1.5, -0.16666666666666669],
 (5, 11): [-0.04513888888888889, 0.5277777777777778, -2.71875, 6.5, -6.729166666666667, 0.0, 6.729166666666667, -6.5,
           2.71875, -0.5277777777777778, 0.04513888888888889],
 (5, 13): [0.011491402116402117, -0.16005291005291006, 1.033399470899471, -3.9828042328042326, 8.39608134920635,
           -8.246031746031747, 0.0, 8.246031746031747, -8.39608134920635, 3.9828042328042326, -1.033399470899471,
           0.16005291005291006, -0.011491402116402117]
 }
 
 
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
    from scipy.linalg import inv
    # The above coefficients were generated from a routine which used Fractions
    w = [i*factor for i in (inv(X)[divisions]).tolist()]
    central_diff_weights_precomputed[(divisions, points)] = w
    return w


def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    '''Reimplementation of SciPy's derivative function, with more cached
    coefficients and without using numpy. If new coefficients not cached are
    needed, they are only calculated once and are remembered.
    '''
    if order < n + 1:
        raise ValueError
    if order % 2 == 0:
        raise ValueError
    weights = central_diff_weights(order, n)
    tot = 0.0
    ho = order >> 1
    for k in range(order):
        tot += weights[k]*func(x0 + (k - ho)*dx, *args)
    return tot/product([dx]*n)


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

    References
    ----------
    .. [1] "Horner’s Method." Wikipedia, October 6, 2018. 
    https://en.wikipedia.org/w/index.php?title=Horner%27s_method&oldid=862709437.
    '''
    tot = 0.0
    for c in coeffs:
        tot = tot*x + c
    return tot


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


def interp(x, dx, dy, left=None, right=None):
    '''One-dimensional linear interpolation routine inspired/
    reimplemented from NumPy for extra speed for scalar values
    (and also numpy).
    
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
    '''
    lendx = len(dx)
    j = binary_search(x, dx, lendx)
    if (j == -1):
        if left is not None:
            return left
        else:
            return dy[0]
    elif (j == lendx - 1):
        return dy[j]
    elif (j == lendx):
        if right is not None:
            return right
        else:
            return dy[-1]
    else:
        return (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j])*(x - dx[j]) + dy[j]


def implementation_optimize_tck(tck):
    '''Converts 1-d or 2-d splines calculated with SciPy's `splrep` or
    `bisplrep` to a format for fastest computation - lists in PyPy, and numpy
    arrays otherwise.
    
    Only implemented for 3 and 5 length `tck`s.
    '''
    if IS_PYPY:
        return tck
    else:
        if len(tck) == 3:
            tck[0] = np.array(tck[0])
            tck[1] = np.array(tck[1])
        elif len(tck) == 5:
            tck[0] = np.array(tck[0])
            tck[1] = np.array(tck[1])
            tck[2] = np.array(tck[2])
        else:
            raise NotImplementedError
    return tck


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


def py_bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    '''Port of SciPy's C bisect routine.
    '''
    fa = f(a, *args)
    fb = f(b, *args)
    if fa*fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs") 
    elif fa == 0.0:
        return a
    elif fb == 0.0:
        return b

    dm = b - a
    iterations = 0.0
    
    for i in range(maxiter):
        dm *= 0.5
        xm = a + dm
        fm = f(xm, *args)
        if fm*fa >= 0.0:
            a = xm
        abs_dm = fabs(dm)
        if (fm == 0.0 or (abs_dm < xtol + rtol*abs_dm)):
            return xm
    raise ValueError("Failed to converge after %d iterations" %maxiter)


def py_ridder(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=_iter,
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
        dn = copysign((fm*fm - fa*fb)**-0.5, fb - fa)*fm*dm
    
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
    raise ValueError("Failed to converge after %d iterations" %maxiter)


def py_brenth(f, xa, xb, args=(),
            xtol=_xtol, rtol=_rtol, maxiter=_iter,
            full_output=False, disp=True, q=False):
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    if fpre*fcur > 0.0:
        raise ValueError("f(a) and f(b) must have different signs") 
    elif fpre == 0.0:
        return xa
    elif fcur == 0.0:
        return xb

    for i in range(maxiter):
        if fpre*fcur < 0.0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        if fabs(fblk) < fabs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre
        
        delta = 0.5*(xtol + rtol*fabs(xcur))
        sbis = 0.5*(xblk - xcur)
        
        if fcur == 0.0 or (fabs(sbis) < delta):
            return xcur
        
        if (fabs(spre) > delta and fabs(fcur) < fabs(fpre)):
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
            if (2.0*fabs(stry) < min(fabs(spre), 3.0*fabs(sbis) - delta)):
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
        if fabs(scur) > delta:
            xcur += scur
        else:
            if sbis > 0.0:
                xcur += delta
            else:
                xcur -= delta
        
        fcur = f(xcur, *args)
    raise ValueError("Failed to converge after %d iterations" %maxiter)


def py_newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=_iter,
              fprime2=None, low=None, high=None, damping=1.0):
    '''Newton's method designed to be mostly compatible with SciPy's 
    implementation, with a few features added and others now implemented.
    
    1) No tracking ofo how many iterations have progressed.
    2) No ability to return a RootResults object
    3) No warnings on some cases of bad input ( low tolerance, no iterations)
    4) 
    
    From scipy, with very minor modifications!
    https://github.com/scipy/scipy/blob/v1.1.0/scipy/optimize/zeros.py#L66-L206
    '''
    p0 = 1.0*x0
    if fprime is not None:
        for iter in range(maxiter):
            fder = fprime(p0, *args)
            if fder == 0.0:
                # Cannot coninue
                return p0
            fval = func(p0, *args)
            step = fval/fder*damping
            if fprime2 is None:
                p = p0 - step
            else:
                fder2 = fprime2(p0, *args)
                p = p0 - step/(1.0 - 0.5*step*fder2/fder)
                
                if low is not None and p < low:
                    p = low
                if high is not None and p > high:
                    p = high
                        
            if abs(p - p0) < tol:
                # complete
                return p
            p0 = p
    else:
        # Logic to take a small step to calculate the approximate derivative
        if x0 >= 0.0:
            p1 = x0*1.0001 + 1e-4
        else:
            p1 = x0*1.0001 - 1e-4
        if low is not None and p1 < low:
            p1 = low
        if high is not None and p1 > high:
            p1 = high
        q0 = func(p0, *args)
        q1 = func(p1, *args)
        for _ in range(maxiter):
            if q1 == q0:
                return 0.5*(p1 + p0)
            else:
                p = p1 - q1*(p1 - p0)/(q1 - q0)*damping
            if low is not None and p < low:
                p = low
            if high is not None and p > high:
                p = high

            if abs(p - p1) < tol:
                return p
            p0 = p1
            q0 = q1
            p1 = p
            q1 = func(p1, *args)
    raise ValueError("Failed to converge; maxiter (%d) reached, value=%f " %(maxiter, p))


def py_splev(x, tck, ext=0):
    '''Evaluate a B-spline using a pure-python port of FITPACK's splev. This is
    not fully featured in that it does not support calculating derivatives.
    Takes the knots and coefficients of a B-spline tuple, and returns 
    the value of the smoothing polynomial.  
    
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
    '''
    e = ext
    t, c, k = tck
    if isinstance(x, (float, int, complex)):
        x = [x]

    # NOTE: the first value of each array is used; it is only the indexes that 
    # are adjusted for fortran
    def func_35(arg, t, l, l1, k2, nk1):    
        # minus 1 index
        if arg >= t[l-1] or l1 == k2:
            arg, t, l, l1, nk1, leave = func_40(arg, t, l, l1, nk1)
            # Always leaves here
            return arg, t, l, l1, k2, nk1
        
        l1 = l
        l = l - 1
        arg, t, l, l1, k2, nk1 = func_35(arg, t, l, l1, k2, nk1)
        return arg, t, l, l1, k2, nk1
    
    def func_40(arg, t, l, l1, nk1):
        if arg < t[l1-1] or l == nk1: # minus 1 index
            return arg, t, l, l1, nk1, 1
        l = l1
        l1 = l + 1
        arg, t, l, l1, nk1, leave = func_40(arg, t, l, l1, nk1)
        return arg, t, l, l1, nk1, leave
    
    m = len(x)
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
                arg, t, l, l1, k2, nk1 = func_35(arg, t, l, l1, k2, nk1)
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
                arg, t, l, l1, k2, nk1 = func_35(arg, t, l, l1, k2, nk1)
        else:
            arg, t, l, l1, k2, nk1 = func_35(arg, t, l, l1, k2, nk1)

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
    if len(y) == 1:
        return y[0]
    return y


def py_bisplev(x, y, tck, dx=0, dy=0):
    '''Evaluate a bivariate B-spline or its derivatives.
    For scalars, returns a float; for other inputs, mimics the formats of 
    SciPy's `bisplev`.
    
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
    '''
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
    """
    subroutine fpbspl evaluates the (k+1) non-zero b-splines of
    degree k at t(l) <= x < t(l+1) using the stable recurrence
    relation of de boor and cox.
    
    All arrays are 1d!
    Optimized the assignment and order and so on.
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
    '''Possible optimization: Do not evaluate derivatives, ever.
    '''
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


# interp, horner, derivative methods (and maybe newton?) should always be used.
if not IS_PYPY:
    from scipy.interpolate import splev, bisplev
    from scipy.optimize import newton, bisect, ridder, brenth
else:
    splev, bisplev = py_splev, py_bisplev
    newton, bisect, ridder, brenth = py_newton, py_bisect, py_ridder, py_brenth