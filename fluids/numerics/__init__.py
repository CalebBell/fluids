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
from math import sin, exp, pi, fabs, copysign, log, isinf, acos, cos, sin
import sys
from sys import float_info
from .arrays import solve as py_solve

__all__ = ['isclose', 'horner', 'chebval', 'interp',
           'linspace', 'logspace', 'cumsum', 'diff',
           'implementation_optimize_tck', 'tck_interp2d_linear',
           'bisect', 'ridder', 'brenth', 'newton', 
           'splev', 'bisplev', 'derivative', 'normalize',
           'IS_PYPY', 'roots_cubic', 'roots_quartic', 'newton_system',
           'lambertw', 'ellipe', 'gamma', 'gammaincc', 'erf',
           'i1', 'i0', 'k1', 'k0', 'iv',
           'numpy',
           'polyint_over_x', 'horner_log', 'polyint',
           ]

nan = float("nan")

class FakePackage(object):
    pkg = None
    def __getattr__(self, name):
        raise ImportError('%s in not installed and required by this feature' %(self.pkg))
        
    def __init__(self, pkg):
        self.pkg = pkg


try:
    # The right way imports the platform module which costs to ms to load!
    # implementation = platform.python_implementation()
    IS_PYPY = 'PyPy' in sys.version
except AttributeError:
    IS_PYPY = False
  
try:
    # Regardless of actual interpreter, fall back to pure python implementations
    # if scipy and numpy are not available.
    import numpy
    import scipy
except ImportError:
    # Allow a fake numpy to be imported, but will raise an excption on any use
    numpy = FakePackage('numpy')
    IS_PYPY = True

np = numpy

#IS_PYPY = True

epsilon = float_info.epsilon
one_epsilon_larger = 1.0 + float_info.epsilon
one_epsilon_smaller = 1.0 - float_info.epsilon
zero_epsilon_smaller = 1.0 - float_info.epsilon

_iter = 100
_xtol = 1e-12
_rtol = float_info.epsilon*2.0

third = 1.0/3.0
sixth = 1.0/6.0
ninth = 1.0/9.0
twelfth = 1.0/12.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0

root_three = (3.0)**0.5
one_27 = 1.0/27.0
complex_factor = 0.8660254037844386j # (sqrt(3)*0.5j)

from cmath import sqrt as csqrt
def roots_cubic_a1(b, c, d):
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
    t2 = a*a
    t3 = d*d
    t10 = c*c
    t14 = b*b
    t15 = t14*b
    t20 = csqrt(-18.0*a*b*c*d + 4.0*a*t10*c + 4.0*t15*d - t14*t10 + 27.0*t2*t3)
    t31 = (36.0*c*b*a + 12.0*root_three*t20*a - 108.0*d*t2 - 8.0*t15)**third
    t32 = 1.0/a
    root1 = t31*t32*sixth - two_thirds*( 3.0*a*c - t14)*t32/t31 - b*t32*third
    t33 = t31*t32
    t40 = (3.0*a*c - t14)*t32/t31
    
    t50 = -t33*twelfth + t40*third - b*t32*third
    t51 = 0.5j*root_three *(t33*sixth + two_thirds*t40)
    root2 = t50 + t51
    root3 = t50 - t51
    return [root1, root2, root3]
    
from math import sqrt




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
    range -10 to 10 - a common occurence in cubic equations of state.

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
    (-0.010010019045111562, -88.73128838313305, -11.258701597821826)
    
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
    if b == 0.0 and a == 0.0: 
        return (-d/c, )
    elif a == 0.0:
        D = c*c - 4.0*b*d
        b_inv_2 = 0.5/b
        if D < 0.0:
            D = (-D)**0.5
            x1 = (-c + D*1.0j)*b_inv_2
            x2 = (-c - D*1.0j)*b_inv_2
        else:
            D = D**0.5
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
    '''h has no savings on precicion - 0.4 error to 0.2.
    '''
#    print(f, g, h, 'f, g, h')
    if h == 0.0 and g == 0.0 and f == 0.0:
        if d/a >= 0.0:
            x = -((d*a_inv)**(third))
        else:
            x = (-d*a_inv)**(third)
        return (x, x, x)
    elif h > 0.0:
#        print('basic')
        # 1 real root, 2 imag
        root_h = h**0.5
        R = -(0.5*g) + root_h
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
    elif h <= 0.0:
        t2 = a*a
        t3 = d*d
        t10 = c*c
        t14 = b*b
        t15 = t14*b
        
        '''This method is inaccurate when choice_term is too small; but still
        more accurate than the other method.
        '''
        choice_term = -18.0*a*b*c*d + 4.0*a*t10*c + 4.0*t15*d - t14*t10 + 27.0*t2*t3
        if abs(choice_term) > 1e-12 or abs(b + 1.0) < 1e-7:
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
            i = (((g*g)*0.25) - h)**0.5
            j = i**third # There was a saving for j but it was very weird with if statements!
            '''Clamied nothing saved for k.
            '''
            k = acos(-(g/(2.0*i)))
            L = -j
    
            # Would be nice to be able to compute the sin and cos at the same time
            M = cos(k*third)
            N = root_three*sin(k*third)
            P = -(b*(third*a_inv))
    
            # Direct formula for x1
            x1 = 2.0*j*M - b*(third*a_inv)
            x2 = L*(M + N) + P
            x3 = L*(M - N) + P
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
    x15 = 2.0*x14**(third)
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
    x27 = (0.0625*x10 - x13*sixth + x7/216 + csqrt(0.25*x14*x14 + one_27*x25*x25*x25))**(third)
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

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    '''Port of numpy's linspace to pure python. Does not support dtype, and 
    returns lists of floats.
    '''
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
    return [base**yi for yi in y]


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

def polyint(coeffs):
    '''not quite a copy of numpy's version because this was faster to implement'''
    return ([0.0] + [c/(i+1) for i, c in enumerate(coeffs[::-1])])[::-1]


def polyint_over_x(coeffs):
    coeffs = coeffs[::-1]
    log_coef = coeffs[0]
    poly_terms = [0.0]
    for i in range(1, len(coeffs)):
        poly_terms.append(coeffs[i]/i)
    return list(reversed(poly_terms)), log_coef


def horner_log(coeffs, log_coeff, x):
    '''Technically possible to save one addition of the last term of 
    coeffs is removed but benchmarks said nothing was saved'''
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

def fit_integral_over_T_linear_extrapolation(T1, T2, T_int_T_coeffs,
                                            best_fit_log_coeff, Tmin, Tmax, 
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
        tot += (horner_log(T_int_T_coeffs, best_fit_log_coeff, T2_mid) 
                    - horner_log(T_int_T_coeffs, best_fit_log_coeff, T1_mid))
    if T2 > Tmax:
        T1_high = T1 if T1 > Tmax else Tmax
        x1 = Tmax_value - Tmax_slope*Tmax
        tot += (Tmax_slope*T2 + x1*log(T2)) - (Tmax_slope*T1_high + x1*log(T1_high))
    if flip:
        return -tot
    return tot


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
    '''Pure python and therefore slow version of the standard library isclose.
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
    
    '''
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


def py_newton(func, x0, fprime=None, args=(), tol=None, maxiter=_iter,
              fprime2=None, low=None, high=None, damping=1.0, ytol=0.0,
              xtol=1.48e-8):
    '''Newton's method designed to be mostly compatible with SciPy's 
    implementation, with a few features added and others now implemented.
    
    1) No tracking ofo how many iterations have progressed.
    2) No ability to return a RootResults object
    3) No warnings on some cases of bad input ( low tolerance, no iterations)
    4) Ability to accept True for either fprime or fprime2, which means that
       they are included in the return value of func
    5) No handling for inf or nan!
    
    From scipy, with very minor modifications!
    https://github.com/scipy/scipy/blob/v1.1.0/scipy/optimize/zeros.py#L66-L206
    '''
    if tol is not None:
        xtol = tol
    p0 = 1.0*x0
    if fprime is not None:
        fprime2_included = fprime2 == True
        fprime_included = fprime == True
        
        for iter in range(maxiter):
            if fprime2_included:
                fval, fder, fder2 = func(p0, *args)
            elif fprime_included:
                fval, fder = func(p0, *args)
            elif fprime2 is not None:
                fval = func(p0, *args)
                fder = fprime(p0, *args)
                fder2 = fprime2(p0, *args)
            else:
                fval = func(p0, *args)
                fder = fprime(p0, *args)
            
            if fder == 0.0:
                # Cannot coninue
                return p0
            step = fval/fder*damping
            if fprime2 is None:
                p = p0 - step
            else:
                p = p0 - step/(1.0 - 0.5*step*fder2/fder)
                
            if low is not None and p < low:
                p = low
            if high is not None and p > high:
                p = high
                        
            if abs(p - p0) < xtol:
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
        if abs(q0) < ytol:
            return p0
        
        q1 = func(p1, *args)
        if abs(q1) < ytol:
            return p1
        
        for _ in range(maxiter):
            if q1 == q0:
                return 0.5*(p1 + p0)
            else:
                p = p1 - q1*(p1 - p0)/(q1 - q0)*damping
            if low is not None and p < low:
                p = low
            if high is not None and p > high:
                p = high

            p0 = p1
            q0 = q1
            p1 = p
            q1 = func(p1, *args)
            
#            if abs(p - p1) < tol:
#            if abs(q1) < ytol:
#            if abs(p - p1) < tol or abs(q1) < ytol:
            if abs(p - p1) < xtol and abs(q1) < ytol:
                return p
            
    raise ValueError("Failed to converge; maxiter (%d) reached, value=%f " %(maxiter, p))


def newton_system(f, x0, jac, xtol=None, ytol=None, maxiter=100, damping=1.0,
                  args=(), damping_func=None):
    jac_also = True if jac == True else False
    
    def err(F):
        err = sum([abs(i) for i in F])
        return err

    if jac_also:
        fcur, j = f(x0, *args)
    else:
        fcur = f(x0, *args)
        
    if ytol is not None and err(fcur) < ytol:
        return x0, 0
    else:
        x = x0
        if not jac_also:
            j = jac(x, *args)
            
    iter = 1
    while iter < maxiter:
        dx = py_solve(j, [-v for v in fcur])
        if damping_func is None:
            x = [xi + dxi*damping for xi, dxi in zip(x, dx)]
        else:
            x = damping_func(x, dx, damping)
        if jac_also:
            fcur, j = f(x, *args)
        else:
            fcur = f(x, *args)
        
        iter += 1
        if xtol is not None and np.linalg.norm(fcur, ord=2) < xtol:
            break
        if ytol is not None and err(fcur) < ytol:
            break
            
        if not jac_also:
            j = jac(x, *args)
    return x, iter

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
    
    # Try out mpmath for special functions anyway
has_scipy = False
try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False
    
erf = None
try:
    from math import erf
except ImportError:
    # python 2.6 or other implementations?
    pass


from math import gamma # Been there a while



def my_lambertw(y):
    def err(x):
        return x*exp(x) - y
    return py_brenth(err, 1e-300, 700.0)

#has_scipy = False

if has_scipy:
    from scipy.special import lambertw, ellipe, gammaincc, gamma # fluids
    from scipy.special import i1, i0, k1, k0, iv # ht
    from scipy.special import hyp2f1    
    if erf is None:
        from scipy.special import erf
else:
    import mpmath
    # scipy is not available... fall back to mpmath as a Pure-Python implementation
    from mpmath import lambertw # Same branches as scipy, supports .real
    lambertw = my_lambertw
    from mpmath import ellipe # seems the same so far        
    

    # Figured out this definition from test_precompute_gammainc.py in scipy
    gammaincc = lambda a, x: mpmath.gammainc(a, a=x, regularized=True)
    iv = mpmath.besseli
    i1 = lambda x: mpmath.besseli(1, x)
    i0 = lambda x: mpmath.besseli(0, x)
    k1 = lambda x: mpmath.besselk(1, x)
    k0 = lambda x: mpmath.besselk(0, x)
    
    if erf is None:
        from mpmath import erf
