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
from cmath import sqrt as csqrt
from math import acos, cos, sin, sqrt

__all__ = ['roots_quadratic', 'roots_quartic', 'roots_cubic_a1', 'roots_cubic_a2',
'roots_cubic']
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
    return (root1, root2, root3)

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
    """
    Notes
    -----
    Known issue is inputs that look like
    1, -0.999999999978168, 1.698247818501352e-11, -8.47396642608142e-17
    Errors grown unbound, starting when b is -.99999 and close to 1.
    """
    if a == 0.0:
        if b == 0.0:
            root = -d/c
            return (root, root, root)
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
        return (x1, x1, x2)
    a_inv = 1.0/a
    a_inv2 = a_inv*a_inv
    bb = b*b
    """Herbie modifications for f:
    c*a_inv - b_a*b_a*third
    """

    b_a = b*a_inv
    b_a2 = b_a*b_a
    f = c*a_inv - b_a2*third
#    f = (3.0*c*a_inv - bb*a_inv2)*third
    g = ((2.0*(bb*b) * a_inv2*a_inv) - (9.0*b*c)*(a_inv2) + (27.0*d*a_inv))*one_27
#    g = (((2.0/(a/b))/((a/b) * (a/b)) + d*27.0/a) - (9.0/a*b)*c/a)/27.0

    h = (0.25*(g*g) + (f*f*f)*one_27)
#    print(f, g, h)
    """h has no savings on precision - 0.4 error to 0.2.
    """
#    print(f, g, h, 'f, g, h')
    if h == 0.0 and g == 0.0 and f == 0.0:
        if d/a >= 0.0:
            # print('A'*50, locals())
            x = -((d*a_inv)**(third))
        else:
            # print('B'*50, locals())
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
            U = -((-T)**(third))

        SU = S + U
        b_3a = b*(third*a_inv)
        t1 = -0.5*SU - b_3a
        x1 = SU - b_3a
        t2 = (S - U)*complex_factor
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

        """This method is inaccurate when choice_term is too small; but still
        more accurate than the other method.
        """
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
            # 3 real roots
            # example is going in here
            i = sqrt(((g*g)*0.25) - h)
            j = i**third # There was a saving for j but it was very weird with if statements!
            """Clamied nothing saved for k.
            """
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

