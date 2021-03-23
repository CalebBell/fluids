# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell.

<Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.

This module contains functionality for calculating parameters about different
geometrical forms that come up in engineering practice.

Implemented geometry objects are tanks, helical coils, cooling towers,
air coolers, compact heat exchangers, and plate and frame heat exchangers.

Additional functionality for typical catalyst/adsorbent pellet shapes is also
included.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Main Interfaces
---------------
.. autoclass :: TANK
    :members:
    :undoc-members:
    :show-inheritance:
.. autofunction:: V_tank
.. autofunction:: V_from_h
.. autofunction:: SA_tank
.. autofunction:: SA_from_h
.. autoclass :: HelicalCoil
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass :: PlateExchanger
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass :: AirCooledExchanger
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass :: HyperbolicCoolingTower
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass :: RectangularFinExchanger
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass :: RectangularOffsetStripFinExchanger
    :members:
    :undoc-members:
    :show-inheritance:

Tank Volume Functions
---------------------
.. autofunction:: V_partial_sphere
.. autofunction:: V_horiz_conical
.. autofunction:: V_horiz_ellipsoidal
.. autofunction:: V_horiz_guppy
.. autofunction:: V_horiz_spherical
.. autofunction:: V_horiz_torispherical
.. autofunction:: V_vertical_conical
.. autofunction:: V_vertical_ellipsoidal
.. autofunction:: V_vertical_spherical
.. autofunction:: V_vertical_torispherical
.. autofunction:: V_vertical_conical_concave
.. autofunction:: V_vertical_ellipsoidal_concave
.. autofunction:: V_vertical_spherical_concave
.. autofunction:: V_vertical_torispherical_concave

Tank Surface Area Functions
---------------------------
.. autofunction:: SA_partial_sphere
.. autofunction:: SA_ellipsoidal_head
.. autofunction:: SA_conical_head
.. autofunction:: SA_guppy_head
.. autofunction:: SA_torispheroidal
.. autofunction:: SA_partial_cylindrical_body
.. autofunction:: SA_partial_horiz_conical_head
.. autofunction:: SA_partial_horiz_spherical_head
.. autofunction:: SA_partial_horiz_ellipsoidal_head
.. autofunction:: SA_partial_horiz_guppy_head
.. autofunction:: SA_partial_horiz_torispherical_head
.. autofunction:: SA_partial_vertical_conical_head
.. autofunction:: SA_partial_vertical_ellipsoidal_head
.. autofunction:: SA_partial_vertical_spherical_head
.. autofunction:: SA_partial_vertical_torispherical_head

Miscellaneous Geometry Functions
--------------------------------
.. autofunction:: pitch_angle_solver
.. autofunction:: plate_enlargement_factor
.. autofunction:: a_torispherical
.. autofunction:: A_partial_circle

Pellet Properties
-----------------
.. autofunction:: sphericity
.. autofunction:: aspect_ratio
.. autofunction:: circularity
.. autofunction:: A_cylinder
.. autofunction:: V_cylinder
.. autofunction:: A_hollow_cylinder
.. autofunction:: V_hollow_cylinder
.. autofunction:: A_multiple_hole_cylinder
.. autofunction:: V_multiple_hole_cylinder

"""

from __future__ import division
from math import (pi, sin, cos, tan, asin, acos, atan, acosh, log, radians,
                  degrees, sqrt)
from cmath import sqrt as csqrt
from fluids.constants import inch
from fluids.core import PY3
from fluids.numerics import (cacos, catan, secant, brenth, ellipe, ellipkinc,
                             ellipeinc, horner, chebval, linspace, derivative,
                             quad, translate_bound_func)

__all__ = ['TANK', 'HelicalCoil', 'PlateExchanger', 'RectangularFinExchanger',
           'RectangularOffsetStripFinExchanger', 'HyperbolicCoolingTower',
           'AirCooledExchanger',
           'SA_partial_sphere',
           'V_partial_sphere', 'V_horiz_conical',
           'V_horiz_ellipsoidal', 'V_horiz_guppy', 'V_horiz_spherical',
           'V_horiz_torispherical', 'V_vertical_conical',
           'V_vertical_ellipsoidal', 'V_vertical_spherical',
           'V_vertical_torispherical', 'V_vertical_conical_concave',
           'V_vertical_ellipsoidal_concave', 'V_vertical_spherical_concave',
           'V_vertical_torispherical_concave', 'a_torispherical',
           'SA_ellipsoidal_head', 'SA_conical_head', 'SA_guppy_head',
           'SA_torispheroidal', 'SA_partial_cylindrical_body',
           'A_partial_circle', 'SA_partial_horiz_conical_head',
           'SA_partial_horiz_spherical_head', 'SA_partial_horiz_guppy_head',
           'SA_partial_horiz_ellipsoidal_head', 'SA_partial_horiz_torispherical_head',
           'SA_partial_vertical_conical_head', 'SA_partial_vertical_spherical_head',
           'SA_partial_vertical_torispherical_head', 'SA_partial_vertical_ellipsoidal_head',

           'V_from_h', 'V_tank', 'SA_from_h', 'SA_tank', 'sphericity',
           'aspect_ratio', 'circularity', 'A_cylinder', 'V_cylinder',
           'A_hollow_cylinder', 'V_hollow_cylinder',
           'A_multiple_hole_cylinder', 'V_multiple_hole_cylinder',
           'pitch_angle_solver', 'plate_enlargement_factor']


### Spherical Vessels, partially filled


def SA_partial_sphere(D, h):
    r'''Calculates surface area of a partial sphere according to [1]_.
    If h is half of D, the shape is half a sphere. No bottom is considered in
    this function. Valid inputs are positive values of D and h, with h always
    smaller or equal to D.

    .. math::
        a = \sqrt{h(2r - h)}

    .. math::
        A = \pi(a^2 + h^2)

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    h : float
        Height, as measured from the cap to where the sphere is cut off [m]

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    >>> SA_partial_sphere(1., 0.7)
    2.199114857512855

    References
    ----------
    .. [1] Weisstein, Eric W. "Spherical Cap." Text. Accessed December 22, 2015.
       http://mathworld.wolfram.com/SphericalCap.html.'''
    r = D*0.5
    a = sqrt(h*(2.*r - h))
    return pi*(a*a + h*h)


def V_partial_sphere(D, h):
    r'''Calculates volume of a partial sphere according to [1]_.
    If h is half of D, the shape is half a sphere. No bottom is considered in
    this function. Valid inputs are positive values of D and h, with h always
    smaller or equal to D.

    .. math::
        a = \sqrt{h(2r - h)}

    .. math::
        V = 1/6 \pi h(3a^2 + h^2)

    Parameters
    ----------
    D : float
        Diameter of the sphere, [m]
    h : float
        Height, as measured up to where the sphere is cut off, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    >>> V_partial_sphere(1., 0.7)
    0.4105014400690663

    References
    ----------
    .. [1] Weisstein, Eric W. "Spherical Cap." Text. Accessed December 22, 2015.
       http://mathworld.wolfram.com/SphericalCap.html.'''
    if h <= 0.0:
        return 0.0
    r = 0.5*D
    a = sqrt(h*(2.*r - h))
    return 1/6.*pi*h*(3.*a*a + h*h)



#def V_horizontal_bullet(D, L, H, b=None):
#    # As in GPSA
#    if not b:
#        b = 0.25*D # elliptical 2:1 heads
#    Ze = H/D
#    Zc = H/D
#    K1 = 2*b/D
#    alpha = 2*atan(H/sqrt(2*H*D/2 - H**2))
#    fZc = (alpha - sin(alpha)*cos(alpha))/pi
#    fZe = -H**2/D**2*(-3 + 2*H/D)
#    V = 1/6.*pi*K1*D**3*fZe + 1/4.*pi*D**2*L*fZc
#    return V

#print(V_horizontal_bullet(1., 5., .4999999999999, 0.000000000000000001))

#def V_vertical_bullet(D, L, H, b=None):
#    K1 = 2*b/D
#    Ze = (H1 + H2)/K1*D # is divided by D?
#    fZe = -((H1 + H2))
#
#    V = 1/6.*pi*K1*D**3*fZe + 1/4.*pi*D**2*L*fZc
#    return V



### Functions as developed by Dan Jones

def V_horiz_conical(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with conical ends, according to [1]_.

    .. math::
        V_f = A_fL + \frac{2aR^2}{3}K, \;\;0 \le h < R\\

    .. math::
        V_f = A_fL + \frac{2aR^2}{3}\pi/2,\;\; h = R\\

    .. math::
        V_f = A_fL + \frac{2aR^2}{3}(\pi-K), \;\; R< h \le 2R

    .. math::
        K = \cos^{-1} M + M^3\cosh^{-1} \frac{1}{M} - 2M\sqrt{1 - M^2}

    .. math::
        M = \left|\frac{R-h}{R}\right|

    .. math::
        Af = R^2\cos^{-1}\frac{R-h}{R} - (R-h)\sqrt{2Rh - h^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    a : float
        Distance the cone head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]
    headonly : bool, optional
        Function returns only the volume of a single head side if True

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_horiz_conical(D=108., L=156., a=42., h=36)/231
    2041.1923581273443

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    R = 0.5*D
    R_third = R/3.0
    t0 = (R-h)/R
    if t0 < -1.0 or t0 > 1.0:
        raise ValueError("Unphysical height")
    Af = R*R*acos(t0) - (R-h)*sqrt(h*(R + R - h))
    M = abs(t0)
    if h == R:
        Vf = a*R*R_third*pi
    else:
        K = acos(M) + M*M*M*acosh(1./M) - 2.*M*sqrt(1.-M*M)
        if 0. <= h < R:
            Vf = 2.*a*R*R_third*K
        else:
        # elif R < h <= 2.0*R:
            Vf = 2.*a*R*R_third*(pi - K)
    if headonly:
        Vf = 0.5*Vf
    else:
        Vf += Af*L
    return Vf


def V_horiz_ellipsoidal(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with ellipsoidal ends, according to [1]_.

    .. math::
        V_f = A_fL + \pi a h^2\left(1 - \frac{h}{3R}\right)

    .. math::
        Af = R^2\cos^{-1}\frac{R-h}{R} - (R-h)\sqrt{2Rh - h^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    a : float
        Distance the ellipsoidal head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]
    headonly : bool, optional
        Function returns only the volume of a single head side if True

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_horiz_ellipsoidal(D=108, L=156, a=42, h=36)/231.
    2380.9565415578145

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    R = 0.5*D
    Af = R*R*acos((R-h)/R) - (R-h)*sqrt(2*R*h - h*h)
    Vf = pi*a*h*h*(1 - h/(3.*R))
    if headonly:
        Vf = 0.5*Vf
    else:
        Vf += Af*L
    return Vf


def V_horiz_guppy(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with guppy heads, according to [1]_.

    .. math::
        V_f = A_fL + \frac{2aR^2}{3}\cos^{-1}\left(1 - \frac{h}{R}\right)
        +\frac{2a}{9R}\sqrt{2Rh - h^2}(2h-3R)(h+R)

    .. math::
        Af = R^2\cos^{-1}\frac{R-h}{R} - (R-h)\sqrt{2Rh - h^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    a : float
        Distance the guppy head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]
    headonly : bool, optional
        Function returns only the volume of a single head side if True

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_horiz_guppy(D=108., L=156., a=42., h=36)/231.
    1931.7208029476762

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    R = 0.5*D
    x0 = sqrt(2.*R*h - h*h)
    Af = R*R*acos((R-h)/R) - (R-h)*x0
    Vf = 2.*a*R*R/3.*acos(1. - h/R) + 2.*a/9./R*x0*(2.0*h - 3.0*R)*(h + R)
    if headonly:
        Vf = Vf*0.5
    else:
        Vf += Af*L
    return Vf

def _V_horiz_spherical_toint(x, r2, R2, den_inv):
    x2 = x*x
    return (r2 - x2)*atan(sqrt((R2 - x2)*den_inv))


def V_horiz_spherical(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with spherical heads, according to [1]_.

    .. math::
        V_f = A_fL + \frac{\pi a}{6}(3R^2 + a^2),\;\; h = R, |a|\le R

    .. math::
        V_f = A_fL + \frac{\pi a}{3}(3R^2 + a^2),\;\; h = D, |a|\le R

    .. math::
        V_f = A_fL + \pi a h^2\left(1 - \frac{h}{3R}\right),\;\; h = 0,
        \text{ or } |a| = 0, R, -R

    .. math::
        V_f = A_fL + \frac{a}{|a|}\left\{\frac{2r^3}{3}\left[\cos^{-1}
        \frac{R^2 - rw}{R(w-r)} + \cos^{-1}\frac{R^2 + rw}{R(w+r)}
        - \frac{z}{r}\left(2 + \left(\frac{R}{r}\right)^2\right)
        \cos^{-1}\frac{w}{R}\right] - 2\left(wr^2 - \frac{w^3}{3}\right)
        \tan^{-1}\frac{y}{z} + \frac{4wyz}{3}\right\}
        ,\;\; h \ne R, D; a \ne 0, R, -R, |a| \ge 0.01D

    .. math::
        V_f = A_fL + \frac{a}{|a|}\left[2\int_w^R(r^2 - x^2)\tan^{-1}
        \sqrt{\frac{R^2-x^2}{r^2-R^2}}dx - A_f z\right]
        ,\;\; h \ne R, D; a \ne 0, R, -R, |a| < 0.01D

    .. math::
        Af = R^2\cos^{-1}\frac{R-h}{R} - (R-h)\sqrt{2Rh - h^2}

    .. math::
        r = \frac{a^2 + R^2}{2|a|}

    .. math::
        w = R - h

    .. math::
        y = \sqrt{2Rh-h^2}

    .. math::
        z = \sqrt{r^2 - R^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    a : float
        Distance the spherical head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]
    headonly : bool, optional
        Function returns only the volume of a single head side if True

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_horiz_spherical(D=108., L=156., a=42., h=36)/231.
    2303.9615116986183

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    R = D/2.
    r = (a*a + R*R)/2./abs(a)
    w = R - h
    y = sqrt(2*R*h - h**2)
    z = sqrt(r**2 - R**2)
    Af = R**2*acos((R-h)/R) - (R-h)*sqrt(2*R*h - h**2)

    if h == R and abs(a) <= R:
        Vf = pi*a/6*(3*R**2 + a**2)
    elif h == D and abs(a) <= R:
        Vf = pi*a/3*(3*R**2 + a**2)
    elif h == 0 or a == 0 or a == R or a == -R or z == 0.0:
        Vf = pi*a*h**2*(1 - h/3./R)
    elif abs(a) >= 0.01*D:
        Vf = a/abs(a)*(
        2*r**3/3.*(acos((R**2 - r*w)/(R*(w-r))) + acos((R**2+r*w)/(R*(w+r)))
        - z/r*(2+(R/r)**2)*acos(w/R))
        - 2*(w*r**2 - w**3/3)*atan(y/z) + 4*w*y*z/3)
    else:
        r2 = r*r
        R2 = R*R
        den_inv = 1.0/(r2 - R2)
        integrated = quad(_V_horiz_spherical_toint, w, R, args=(r2, R2, den_inv))[0] # , epsrel=1.49e-13,
        Vf = a/abs(a)*(2*integrated - Af*z)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


def V_horiz_torispherical_toint_1(x, w, c10, c11):
    # No analytical integral available in MP
    n = c11 + sqrt(c10 - x*x)
    n2 = n*n
    t = sqrt(n2 - w*w)
    return n2*asin(t/n) - w*t

def V_horiz_torispherical_toint_2(x, w, c10, c11, g, g2):
    # No analytical integral available in MP
    n = c11 + sqrt(c10 - x*x)
    n2 = n*n
    n_inv = 1.0/n
    ans = n2*(acos(w*n_inv) - acos(g*n_inv)) - w*sqrt(n2 - w*w) + g*sqrt(n2 - g2)
    return ans

def V_horiz_torispherical_toint_3(x, r2, g2, z_inv):
    # There is an analytical integral in MP, but for all cases we seem to
    # get ZeroDivisionError: 0.0 cannot be raised to a negative power
    x2 = x*x
    return (r2 - x2)*atan(sqrt(g2 - x2)*z_inv)

def V_horiz_torispherical(D, L, f, k, h, headonly=False):
    r'''Calculates volume of a tank with torispherical heads, according to [1]_.

    .. math::
        V_f  = A_fL + 2V_1, \;\; 0 \le h \le h_1\\
        V_f  = A_fL + 2(V_{1,max} + V_2 + V_3), \;\; h_1 < h < h_2\\
        V_f  = A_fL + 2[2V_{1,max} - V_1(h=D-h) + V_{2,max} + V_{3,max}]
        , \;\; h_2 \le h \le D

    .. math::
        V_1 = \int_0^{\sqrt{2kDh - h^2}} \left[n^2\sin^{-1}\frac{\sqrt
        {n^2-w^2}}{n} - w\sqrt{n^2-w^2}\right]dx

    .. math::
        V_2 = \int_0^{kD\cos\alpha}\left[n^2\left(\cos^{-1}\frac{w}{n}
        - \cos^{-1}\frac{g}{n}\right) - w\sqrt{n^2 - w^2} + g\sqrt{n^2
        - g^2}\right]dx

    .. math::
        V_3 = \int_w^g(r^2 - x^2)\tan^{-1}\frac{\sqrt{g^2 - x^2}}{z}dx
        - \frac{z}{2}\left(g^2\cos^{-1}\frac{w}{g} - w\sqrt{2g(h-h_1)
        - (h-h_1)^2}\right)

    .. math::
        V_{1,max} = v_1(h=h_1)

    .. math::
        v_{2,max} = v_2(h=h_2)

    .. math::
        v_{3,max} = \frac{\pi a_1}{6}(3g^2 + a_1^2)

    .. math::
        a_1 = fD(1-\cos\alpha)

    .. math::
        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

    .. math::
        n = R - kD + \sqrt{k^2D^2-x^2}

    .. math::
        g = r\sin\alpha

    .. math::
        r = fD

    .. math::
        h_2 = D - h_1

    .. math::
        w = R - h

    .. math::
        z = \sqrt{r^2- g^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called both dish radius and
        also crown radius and has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]
    h : float
        Height, as measured up to where the fluid ends, [m]
    headonly : bool, optional
        Function returns only the volume of a single head side if True

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=36)/231.
    2028.62667

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    if f is None or k is None:
        raise ValueError("Missing f or k")
    R = 0.5*D
    R2 = R*R
    hh = h*h
    Af = R2*acos((R-h)/R) - (R-h)*sqrt(2.0*R*h - hh)
    r = f*D
    alpha = asin((1.0 - 2.0*k)/(2.*(f - k)))
    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)
    a1 = r*(1.0 - cos_alpha)
    g = r*sin_alpha
    z = r*cos_alpha
    h1 = k*D*(1.0 - sin_alpha)
    h2 = D - h1

    # Chebfun in Python failed on these functions
    c10 = k*k*D*D
    c11 = R - k*D
    g2 = g*g
    r2 = r*r

    if 0.0 <= h <= h1:
        w = R - h
        Vf = 2.0*quad(V_horiz_torispherical_toint_1, 0.0, sqrt(2.0*k*D*h - hh), (w, c10, c11))[0]
    elif h1 < h < h2:
        w = R - h
        wmax1 = R - h1
        V1max = quad(V_horiz_torispherical_toint_1, 0.0, sqrt(2.0*k*D*h1 - h1*h1), (wmax1,c10, c11))[0]
        V2 = quad(V_horiz_torispherical_toint_2, 0.0, k*D*cos_alpha, (w, c10, c11, g, g2))[0]
        V3 = quad(V_horiz_torispherical_toint_3, w, g , (r2, g2, 1.0/z))[0] - 0.5*z*(g*g*acos(w/g) -w*sqrt(2.0*g*(h-h1) - (h-h1)*(h-h1)))
        Vf = 2.0*(V1max + V2 + V3)
    else:
        w = R - h
        wmax1 = R - h1
        wmax2 = R - h2
        wwerird = R - (D - h)

        V1max = quad(V_horiz_torispherical_toint_1, 0.0, sqrt(2.0*k*D*h1-h1*h1), (wmax1,c10, c11))[0]
        V1weird = quad(V_horiz_torispherical_toint_1, 0.0, sqrt(2.0*k*D*(D-h)-(D-h)*(D-h)), (wwerird,c10, c11))[0]
        V2max = quad(V_horiz_torispherical_toint_2, 0.0, k*D*cos_alpha, (wmax2, c10, c11, g, g2))[0]
        V3max = pi*a1/6.*(3.0*g*g + a1*a1)
        Vf = 2.0*(2.0*V1max - V1weird + V2max + V3max)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


### Begin vertical tanks

def V_vertical_conical(D, a, h):
    r'''Calculates volume of a vertical tank with a convex conical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V_f = \frac{\pi}{4}\left(\frac{Dh}{a}\right)^2\left(\frac{h}{3}\right),\; h < a

    .. math::
        V_f = \frac{\pi D^2}{4}\left(h - \frac{2a}{3}\right),\; h\ge a

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the cone head extends under the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_conical(132., 33., 24)/231.
    250.67461381371024

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    if h < a:
        Vf = pi/4*(D*h/a)**2*(h/3.)
    else:
        Vf = pi*D**2/4*(h - 2*a/3.)
    return Vf


def V_vertical_ellipsoidal(D, a, h):
    r'''Calculates volume of a vertical tank with a convex ellipsoidal bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V_f = \frac{\pi}{4}\left(\frac{Dh}{a}\right)^2 \left(a - \frac{h}{3}\right),\; h < a

    .. math::
        V_f = \frac{\pi D^2}{4}\left(h - \frac{a}{3}\right),\; h \ge a

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the ellipsoid head extends under the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_ellipsoidal(132., 33., 24)/231.
    783.3581681678445

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    if h < a:
        Vf = pi/4*(D*h/a)**2*(a - h/3.)
    else:
        Vf = pi*D**2/4*(h - a/3.)
    return Vf


def V_vertical_spherical(D, a, h):
    r'''Calculates volume of a vertical tank with a convex spherical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V_f = \frac{\pi h^2}{4}\left(2a + \frac{D^2}{2a} - \frac{4h}{3}\right),\; h < a

    .. math::
        V_f = \frac{\pi}{4}\left(\frac{2a^3}{3} - \frac{aD^2}{2} + hD^2\right),\; h\ge a

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the spherical head extends under the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_spherical(132., 33., 24)/231.
    583.6018352850442

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    if h < a:
        Vf = pi*h**2/4*(2*a + D**2/2/a - 4*h/3)
    else:
        Vf = pi/4*(2*a**3/3 - a*D**2/2 + h*D**2)
    return Vf


def V_vertical_torispherical(D, f, k, h):
    r'''Calculates volume of a vertical tank with a convex torispherical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V_f = \frac{\pi h^2}{4}\left(2a_1 + \frac{D_1^2}{2a_1}
        - \frac{4h}{3}\right),\; 0 \le h \le a_1

    .. math::
        V_f = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\pi u\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        + \frac{\pi tu^2}{2} - \frac{\pi u^3}{3} + \pi D(1-2k)\left[
        \frac{2u-t}{4}\sqrt{s+tu-u^2} + \frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\left(\cos^{-1}\frac{t-2u}{2kD}-\alpha\right)\right]
        ,\; a_1 < h \le a_1 + a_2

    .. math::
        V_f = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\frac{\pi t}{2}\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        +\frac{\pi  t^3}{12} + \pi D(1-2k)\left[\frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\sin^{-1}(\cos\alpha)\right]
        + \frac{\pi D^2}{4}[h-(a_1+a_2)] ,\;  a_1 + a_2 < h

    .. math::
        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

    .. math::
        a_1 = fD(1-\cos\alpha)

    .. math::
        a_2 = kD\cos\alpha

    .. math::
        D_1 = 2fD\sin\alpha

    .. math::
        s = (kD\sin\alpha)^2

    .. math::
        t = 2a_2

    .. math::
        u = h - fD(1-\cos\alpha)

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_torispherical(D=132., f=1.0, k=0.06, h=24)/231.
    904.0688283793

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    if h <= 0.0:
        return 0.0
    if f is None or k is None:
        raise ValueError("f and k are required")
    alpha = asin((1.0 - 2.0*k)/(2.0*(f-k)))
    sin_alpha = sin(alpha)
    cos_alpha = cos(alpha)
    a1 = f*D*(1.0 - cos_alpha)
    a2 = k*D*cos_alpha
    D1 = 2.0*f*D*sin_alpha

    x1 = k*D*sin_alpha
    s = x1*x1
    t = a2 + a2
    u = h - f*D*(1.0 - cos_alpha)
    h2 = h*h

    if 0.0 <= h <= a1:
        Vf = 0.25*pi*h2*(a1 + a1 + 0.5*D1*D1/a1 - (4.0/3.0)*h)
    elif a1 < h <= a1 + a2:
        x2 = (0.5*D - k*D)
        u2 = u*u
        Vf = (0.25*pi*a1*((2.0/3.0)*a1*a1 + 0.5*D1*D1) + pi*u*(x2*x2 + s)
        + pi*u2*(0.5*t - u/3.) + pi*D*(1.0 - 2.0*k)*(0.25*(2.0*u - t)*sqrt(s + t*u
                - u2) + 0.25*t*sqrt(s) + 0.5*k*k*D*D*(acos((t - 2.0*u)/(2.0*k*D)) - alpha)))
    else:
        Vf = pi/4*(2*a1**3/3. + a1*D1**2/2.) + pi*t/2.*((D/2 - k*D)**2
        + s) + pi*t**3/12. + pi*D*(1 - 2*k)*(t*sqrt(s)/4
        + k**2*D**2/2*asin(cos(alpha))) + pi*D**2/4*(h - (a1 + a2))
    return Vf


### Begin vertical tanks with concave heads

def V_vertical_conical_concave(D, a, h):
    r'''Calculates volume of a vertical tank with a concave conical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V = \frac{\pi D^2}{12} \left(3h + a - \frac{(a+h)^3}{a^2}\right)
        ,\;\; 0 \le h < |a|

    .. math::
        V = \frac{\pi D^2}{12} (3h + a ),\;\; h \ge |a|

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Negative distance the cone head extends inside the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_conical_concave(D=113., a=-33, h=15)/231
    251.15825565795188

    References
    ----------
    .. [1] Jones, D. "Compute Fluid Volumes in Vertical Tanks." Chemical
       Processing. December 18, 2003.
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if h <= 0.0:
        return 0.0
    if h < abs(a):
        Vf = pi*D**2/12.*(3*h + a - (a+h)**3/a**2)
    else:
        Vf = pi*D**2/12.*(3*h + a)
    return Vf


def V_vertical_ellipsoidal_concave(D, a, h):
    r'''Calculates volume of a vertical tank with a concave ellipsoidal bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V = \frac{\pi D^2}{12} \left(3h + 2a - \frac{(a+h)^2(2a-h)}{a^2}\right)
        ,\;\; 0 \le h < |a|

    .. math::
        V = \frac{\pi D^2}{12} (3h + 2a ),\;\; h \ge |a|

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Negative distance the eppilsoid head extends inside the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_ellipsoidal_concave(D=113., a=-33, h=15)/231
    44.84968851034856

    References
    ----------
    .. [1] Jones, D. "Compute Fluid Volumes in Vertical Tanks." Chemical
       Processing. December 18, 2003.
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if h <= 0.0:
        return 0.0
    if h < abs(a):
        Vf = pi*D**2/12.*(3*h + 2*a - (a+h)**2*(2*a-h)/a**2)
    else:
        Vf = pi*D**2/12.*(3*h + 2*a)
    return Vf


def V_vertical_spherical_concave(D, a, h):
    r'''Calculates volume of a vertical tank with a concave spherical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V = \frac{\pi}{12}\left[3D^2h + \frac{a}{2}(3D^2 + 4a^2) + (a+h)^3
        \left(4 - \frac{3D^2 + 12a^2}{2a(a+h)}\right)\right],\;\; 0 \le h < |a|

    .. math::
        V = \frac{\pi}{12}\left[3D^2h + \frac{a}{2}(3D^2 + 4a^2) \right]
        ,\;\;  h \ge |a|

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Negative distance the spherical head extends inside the main cylinder, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_spherical_concave(D=113., a=-33, h=15)/231
    112.81405437348528

    References
    ----------
    .. [1] Jones, D. "Compute Fluid Volumes in Vertical Tanks." Chemical
       Processing. December 18, 2003.
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if h <= 0.0:
        return 0.0
    if h < abs(a):
        Vf = pi/12*(3*D**2*h + a/2.*(3*D**2 + 4*a**2) + (a+h)**3*(4 - (3*D**2+12*a**2)/(2.*a*(a+h))))
    else:
        Vf = pi/12*(3*D**2*h + a/2.*(3*D**2 + 4*a**2))
    return Vf


def V_vertical_torispherical_concave(D, f, k, h):
    r'''Calculates volume of a vertical tank with a concave torispherical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V = \frac{\pi D^2 h}{4} - v_1(h=a_1+a_2) + v_1(h=a_1 + a_2 -h),\; 0 \le h < a_2

    .. math::
        V = \frac{\pi D^2 h}{4} - v_1(h=a_1+a_2) + v_2(h=a_1 + a_2 -h),\; a_2 \le h < a_1 + a_2

    .. math::
        V = \frac{\pi D^2 h}{4} - v_1(h=a_1+a_2) + 0,\; h \ge a_1 + a_2

    .. math::
        v_1 = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\pi u\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        + \frac{\pi tu^2}{2} - \frac{\pi u^3}{3} + \pi D(1-2k)\left[
        \frac{2u-t}{4}\sqrt{s+tu-u^2} + \frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\left(\cos^{-1}\frac{t-2u}{2kD}-\alpha\right)\right]

    .. math::
        v_2 = \frac{\pi h^2}{4}\left(2a_1 + \frac{D_1^2}{2a_1} - \frac{4h}{3}\right)

    .. math::
        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

    .. math::
        a_1 = fD(1-\cos\alpha)

    .. math::
        a_2 = kD\cos\alpha

    .. math::
        D_1 = 2fD\sin\alpha

    .. math::
        s = (kD\sin\alpha)^2

    .. math::
        t = 2a_2

    .. math::
        u = h - fD(1-\cos\alpha)

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    Matching example from [1]_, with inputs in inches and volume in gallons.

    >>> V_vertical_torispherical_concave(D=113., f=0.71, k=0.081, h=15)/231
    103.88569287163769

    References
    ----------
    .. [1] Jones, D. "Compute Fluid Volumes in Vertical Tanks." Chemical
       Processing. December 18, 2003.
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if h <= 0.0:
        return 0.0
    alpha = asin((1-2*k)/(2.*(f-k)))
    a1 = f*D*(1-cos(alpha))
    a2 = k*D*cos(alpha)
    D1 = 2*f*D*sin(alpha)
    s = (k*D*sin(alpha))**2
    t = 2*a2
    def V1(h):
        u = h-f*D*(1-cos(alpha))
        v1 = pi/4*(2*a1**3/3. + a1*D1**2/2.) + pi*u*((D/2.-k*D)**2 +s)
        v1 += pi*t*u**2/2. - pi*u**3/3.
        v1 += pi*D*(1-2*k)*((2*u-t)/4.*sqrt(s+t*u-u**2) + t*sqrt(s)/4.
        + k**2*D**2/2.*(acos((t-2*u)/(2*k*D)) -alpha))
        return v1
    def V2(h):
        v2 = pi*h**2/4.*(2*a1 + D1**2/(2.*a1) - 4*h/3.)
        return v2
    if 0 <= h < a2:
        Vf = pi*D**2*h/4 - V1(a1+a2) + V1(a1+a2-h)
    elif a2 <= h < a1 + a2:
        Vf = pi*D**2*h/4 - V1(a1+a2) + V2(a1+a2-h)
    else:
        Vf = pi*D**2*h/4 - V1(a1+a2)
    return Vf


### Total surface area of heads, orientation-independent

def SA_ellipsoidal_head(D, a):
    r'''Calculates the surface area of an ellipsoidal head according to [1]_ and [2]_.
    The formula below is for the full shape, the result of which is halved. The
    formula is for :math:`a < R`. In the equations, `a` is the same and `c` is `D`.

    .. math::
         \text{SA} = 2\pi a^2 + \frac{\pi c^2}{e_1}\ln\left(\frac{1+e_1}{1-e_1}
         \right)

    .. math::
        e_1 = \sqrt{1 - \frac{c^2}{a^2}}

    For the case of :math:`a \ge R` from [2]_, which is needed to make the tank
    head volume grow linearly with length:

    .. math::
        \text{SA} = 2\pi R^2 + \frac{2\pi a^2 R}{\sqrt{a^2 - R^2}}\cos^{-1}\frac{R}{|a|}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the ellipsoidal head extends, [m]

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    Spherical case

    >>> SA_ellipsoidal_head(2, 1)
    6.283185307179586
    >>> SA_ellipsoidal_head(2, 1.5)
    8.459109081729984

    References
    ----------
    .. [1] Weisstein, Eric W. "Spheroid." Text. Accessed March 14, 2016.
       http://mathworld.wolfram.com/Spheroid.html.
    .. [2] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if D == a*2.0:
        return 0.5*pi*D*D # necessary to avoid a division by zero when D == a
    R = 0.5*D
    if a < R:
        R, a = min((R, a)), max((R, a))
        e1 = sqrt(1.0 - R*R/(a*a))

        if e1 != 1.0:
#        try:
            log_term = log((1.0 + e1)/(1.0 - e1))
#        except ZeroDivisionError:
        else:
            # Limit as a goes to zero relative to D; may only be ~6 orders of
            # magnitude smaller than D and will still occur
            log_term = 0.0

        return (2.0*pi*a*a + pi*R*R/e1*log_term)*0.5
    else:
        return pi*R*R + pi*a*a*R*1.0/sqrt(a*a - R*R)*acos(R/abs(a))


def SA_conical_head(D, a):
    r'''Calculates the surface area of a conical head according to [1]_.

    .. math::
        SA = \frac{\pi D}{2} \sqrt{a^2 + \left(\frac{D}{2}\right)^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the conical head extends, [m]

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    >>> SA_conical_head(2, 1)
    4.442882938158366

    References
    ----------
    .. [1] Weisstein, Eric W. "Cone." Text. Accessed March 14, 2016.
       http://mathworld.wolfram.com/Cone.html.'''
    return 0.5*pi*D*sqrt(a*a + 0.25*D*D)


def SA_guppy_head(D, a):
    r'''Calculates the surface area of a guppy head according to [1]_.
    Some work was involved in combining formulas for the ellipse of the head,
    and the conic section on the sides.

    .. math::
        SA = \frac{\pi  D}{4}\sqrt{D^2 + a^2} + \frac{\pi D}{2}a

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the conical head extends, [m]

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    >>> SA_guppy_head(2, 1)
    6.654000019110157

    References
    ----------
    .. [1] Weisstein, Eric W. "Cone." Text. Accessed March 14, 2016.
       http://mathworld.wolfram.com/Cone.html.'''
    return 0.25*pi*D*sqrt(a*a + D*D) + 0.5*pi*D*a


def SA_torispheroidal(D, f, k):
    r'''Calculates surface area of a torispherical head according to [1]_.
    Somewhat involved. Equations are adapted to be used for a full head.

    .. math::
        SA = S_1 + S_2

    .. math::
        S_1 = 2\pi D^2 f_d \alpha

    .. math::
        S_2 = 2\pi D^2 f_k\left(\alpha - \alpha_1 + (0.5 - f_k)\left(\sin^{-1}
        \left(\frac{\alpha-\alpha_2}{f_k}\right) - \sin^{-1}\left(\frac{
        \alpha_1-\alpha_2}{f_k}\right)\right)\right)

    .. math::
        \alpha_1 = f_d\left(1 - \sqrt{1 - \left(\frac{0.5 - f_k}{f_d-f_k}
        \right)^2}\right)

    .. math::
        \alpha_2 = f_d - \sqrt{f_d^2 - 2f_d f_k + f_k - 0.25}

    .. math::
        \alpha = \frac{a}{D_i}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    Example from [1]_.

    >>> SA_torispheroidal(D=2.54, f=1.039370079, k=0.062362205)
    6.00394283477063

    References
    ----------
    .. [1] Honeywell. "Calculate Surface Areas and Cross-sectional Areas in
       Vessels with Dished Heads". https://www.honeywellprocess.com/library/marketing/whitepapers/WP-VesselsWithDishedHeads-UniSimDesign.pdf
       Whitepaper. 2014.
    '''
    D2 = D*D
    x1 = 2.0*pi*D2
    k_inv = 1.0/k
    x2 = ((0.5 - k)/(f-k))
    alpha_1 = f*(1.0 - sqrt(1.0 - x2*x2))
    alpha_2 = f - sqrt(f*f - 2.0*f*k + k - 0.25)
    alpha = alpha_1 # Up to top of dome
    S1 = x1*f*alpha_1
    alpha = alpha_2 # up to top of torus
    S2_sub = asin((alpha-alpha_2)*k_inv) - asin((alpha_1-alpha_2)*k_inv)
    S2 = x1*k*(alpha - alpha_1 + (0.5 - k) *S2_sub)
    return S1 + S2


def SA_tank(D, L, sideA=None, sideB=None, sideA_a=0,
            sideB_a=0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None):
    r'''Calculates the surface are of a cylindrical tank with optional heads.
    In the degenerate case of being provided with only `D` and `L`, provides
    the surface area of a cylinder.

    Parameters
    ----------
    D : float
        Diameter of the cylindrical section of the tank, [m]
    L : float
        Length of the main cylindrical section of the tank, [m]
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section, [m]
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section, [m]
    sideA_f : float, optional
        Dish-radius parameter for side A; fD  = dish radius [1/m]
    sideA_k : float, optional
        knuckle-radius parameter for side A; kD = knuckle radius [1/m]
    sideB_f : float, optional
        Dish-radius parameter for side B; fD  = dish radius [1/m]
    sideB_k : float, optional
        knuckle-radius parameter for side B; kD = knuckle radius [1/m]

    Returns
    -------
    SA : float
        Surface area of the tank [m^2]
    sideA_SA : float
        Surface area of only `sideA` [m^2]
    sideB_SA : float
        Surface area of only `sideB` [m^2]
    lateral_SA : float
        Surface area of cylindrical section of tank [m^2]

    Examples
    --------
    Cylinder, Spheroid, Long Cones, and spheres. All checked.

    >>> SA_tank(D=2, L=2)[0]
    18.84955592153876
    >>> SA_tank(D=1., L=0, sideA='ellipsoidal', sideA_a=2, sideB='ellipsoidal',
    ... sideB_a=2)[0]
    10.124375616183062
    >>> SA_tank(D=1., L=5, sideA='conical', sideA_a=2, sideB='conical',
    ... sideB_a=2)[0]
    22.18452243965656
    >>> SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',
    ... sideB_a=0.5)[0]
    18.84955592153876
    '''
    # Side A
    if sideA == 'conical':
        sideA_SA = SA_conical_head(D=D, a=sideA_a)
    elif sideA == 'ellipsoidal':
        sideA_SA = SA_ellipsoidal_head(D=D, a=sideA_a)
    elif sideA == 'guppy':
        sideA_SA = SA_guppy_head(D=D, a=sideA_a)
    elif sideA == 'spherical':
        sideA_SA = pi*(sideA_a*sideA_a + 0.25*D*D) # (SA_partial_sphere(D=D, h=sideA_a)
    elif sideA == 'torispherical':
        if sideA_f is None:
            raise ValueError("Missing torispherical `f` parameter for sideA")
        if sideA_k is None:
            raise ValueError("Missing torispherical `k` parameter for sideA")
        sideA_SA = SA_torispheroidal(D=D, f=sideA_f, k=sideA_k)
    else:
        sideA_SA = pi/4*D**2 # Circle
    # Side B
    if sideB == 'conical':
        sideB_SA = SA_conical_head(D=D, a=sideB_a)
    elif sideB == 'ellipsoidal':
        sideB_SA = SA_ellipsoidal_head(D=D, a=sideB_a)
    elif sideB == 'guppy':
        sideB_SA = SA_guppy_head(D=D, a=sideB_a)
    elif sideB == 'spherical':
        sideB_SA = pi*(sideB_a*sideB_a + 0.25*D*D)#SA_partial_sphere(D=D, h=sideB_a)
    elif sideB == 'torispherical':
        if sideB_f is None:
            raise ValueError("Missing torispherical `f` parameter for sideB")
        if sideB_k is None:
            raise ValueError("Missing torispherical `k` parameter for sideB")
        sideB_SA = SA_torispheroidal(D=D, f=sideB_f, k=sideB_k)
    else:
        sideB_SA = pi/4*D**2 # Circle

    lateral_SA = pi*D*L

    SA = sideA_SA + sideB_SA + lateral_SA
    return SA, sideA_SA, sideB_SA, lateral_SA


def V_tank(D, L, horizontal=True, sideA=None, sideB=None, sideA_a=0.0,
           sideB_a=0.0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None):
    r'''Calculates the total volume of a vertical or horizontal tank with
    different head types.

    Parameters
    ----------
    D : float
        Diameter of the cylindrical section of the tank, [m]
    L : float
        Length of the main cylindrical section of the tank, [m]
    horizontal : bool, optional
        Whether or not the tank is a horizontal or vertical tank
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section, [m]
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section, [m]
    sideA_f : float, optional
        Dimensionless dish-radius parameter for side A; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideA_k : float, optional
        Dimensionless knuckle-radius parameter for side A; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]
    sideB_f : float, optional
        Dimensionless dish-radius parameter for side B; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideB_k : float, optional
        Dimensionless knuckle-radius parameter for side B; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]

    Returns
    -------
    V : float
        Total volume [m^3]
    sideA_V : float
        Volume of only `sideA` [m^3]
    sideB_V : float
        Volume of only `sideB` [m^3]
    lateral_V : float
        Volume of cylindrical section of tank [m^3]

    Examples
    --------
    >>> V_tank(D=1.5, L=5., horizontal=False, sideA='conical',
    ... sideB='conical', sideA_a=2., sideB_a=1.)
    (10.602875205865551, 1.1780972450961726, 0.5890486225480863, 8.835729338221293)
    '''
    if sideA is not None and sideA not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side A')
    if sideB is not None and sideB not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side B')
    R = 0.5*D
    V = sideA_V = sideB_V = lateral_V = 0.0
    if horizontal:
        # Conical case
        if sideA == 'conical' and sideB == 'conical' and sideA_a == sideB_a:
            sideB_V = sideA_V = V_horiz_conical(D, L, sideA_a, D, headonly=True)
        else:
            if sideA == 'conical':
                sideA_V = V_horiz_conical(D, L, sideA_a, D, headonly=True)
            if sideB == 'conical':
                sideB_V = V_horiz_conical(D, L, sideB_a, D, headonly=True)
        # Elliosoidal case
        if sideA == 'ellipsoidal' and sideB == 'ellipsoidal' and sideA_a == sideB_a:
            sideB_V = sideA_V = V_horiz_ellipsoidal(D, L, sideA_a, D, headonly=True)
        else:
            if sideA == 'ellipsoidal':
                sideA_V = V_horiz_ellipsoidal(D, L, sideA_a, D, headonly=True)
            if sideB == 'ellipsoidal':
                sideB_V = V_horiz_ellipsoidal(D, L, sideB_a, D, headonly=True)
        # Guppy case
        if sideA == 'guppy' and sideB == 'guppy' and sideA_a == sideB_a:
            sideB_V = sideA_V = V_horiz_guppy(D, L, sideA_a, D, headonly=True)
        else:
            if sideA == 'guppy':
                sideA_V = V_horiz_guppy(D, L, sideA_a, D, headonly=True)
            if sideB == 'guppy':
                sideB_V = V_horiz_guppy(D, L, sideB_a, D, headonly=True)
        # Spherical case
        if sideA == 'spherical' and sideB == 'spherical' and sideA_a == sideB_a:
            sideB_V = sideA_V = V_horiz_spherical(D, L, sideA_a, D, headonly=True)
        else:
            if sideA == 'spherical':
                sideA_V = V_horiz_spherical(D, L, sideA_a, D, headonly=True)
            if sideB == 'spherical':
                sideB_V = V_horiz_spherical(D, L, sideB_a, D, headonly=True)
        # Torispherical case
        if (sideA == 'torispherical' and sideB == 'torispherical'
            and (sideA_f == sideB_f) and (sideA_k == sideB_k)):
            sideB_V = sideA_V = V_horiz_torispherical(D, L, sideA_f, sideA_k, D, headonly=True)
        else:
            if sideA == 'torispherical':
                sideA_V = V_horiz_torispherical(D, L, sideA_f, sideA_k, D, headonly=True)
            if sideB == 'torispherical':
                sideB_V = V_horiz_torispherical(D, L, sideB_f, sideB_k, D, headonly=True)
        Af = R*R*acos((R-D)/R) - (R-D)*sqrt(2.0*R*D - D*D)
        lateral_V = L*Af
    else:
        # Bottom head
        if sideA == 'conical':
            sideA_V = V_vertical_conical(D, sideA_a, h=sideA_a)
        if sideA == 'ellipsoidal':
            sideA_V = V_vertical_ellipsoidal(D, sideA_a, h=sideA_a)
        if sideA == 'spherical':
            sideA_V = V_vertical_spherical(D, sideA_a, h=sideA_a)
        if sideA == 'torispherical':
            sideA_V = V_vertical_torispherical(D, sideA_f, sideA_k, h=sideA_a)

        # Cylindrical section
        lateral_V = pi/4*D**2*L # All middle

        if sideB == 'conical':
            sideB_V = V_vertical_conical(D, sideB_a, h=sideB_a)
        if sideB == 'ellipsoidal':
            sideB_V = V_vertical_ellipsoidal(D, sideB_a, h=sideB_a)
        if sideB == 'spherical':
            sideB_V = V_vertical_spherical(D, sideB_a, h=sideB_a)
        if sideB == 'torispherical':
            sideB_V = V_vertical_torispherical(D, sideB_f, sideB_k, h=sideB_a)
    return lateral_V + sideA_V + sideB_V, sideA_V, sideB_V, lateral_V


def SA_partial_cylindrical_body(L, D, h):
    r'''Calculates the partial area of a cylinder's body in the context of
    a horizontal cylindrical vessel and liquid partially
    filling it. This computes the wetted surface area of the bottom of the
    cylinder.

    .. math::
        \text{SA} = L D \cos^{-1}\left(\frac{D - 2h}{D}\right)

    Parameters
    ----------
    L : float
        Length of the cylinder, [m]
    D : float
        Diameter of the cylinder, [m]
    h : float
        Height measured from bottom of cylinder to liquid level, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D`. and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    Examples
    --------
    >>> SA_partial_cylindrical_body(L=200.0, D=96., h=22.0)
    19168.852890279868

    References
    ----------
    .. [1] Weisstein, Eric W. "Circular Segment." Text. Wolfram Research, Inc.
       Accessed May 10, 2020. https://mathworld.wolfram.com/CircularSegment.html.
    '''
    if h < 0.0:
        return 0.0
    elif h > D:
        h = D
    C = D*acos((D - h - h)/D)
    return C*L


def A_partial_circle(D, h):
    r'''Calculates the partial area of a circle, in the context of the circle
    being an end cap to a horizontal cylindrical vessel and liquid partially
    filling it. This computes the wetted surface area of one of the end caps.

    Multiply this by two to obtain the wetted area of two end caps.

    .. math::
        \text{SA} = R^2\cos^{-1}\frac{(R - h)}{R} - (R - h)\sqrt{(2Rh - h^2)}

    Parameters
    ----------
    D : float
        Diameter of the circle, [m]
    h : float
        Height measured from bottom of circle to liquid level, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    Examples
    --------
    >>> A_partial_circle(D=96., h=22.0)
    1251.2018147383194

    References
    ----------
    .. [1] Weisstein, Eric W. "Circular Segment." Text. Wolfram Research, Inc.
       Accessed May 10, 2020. https://mathworld.wolfram.com/CircularSegment.html.
    '''
    if h > D:
        h = D # Catch the case of a computed `h` being trivially larger than `D` due to floating point
    elif h < 0.0:
        return 0.0
    R = 0.5*D
    SA = R*R*acos((R - h)/R) - (R - h)*sqrt(2.0*R*h - h*h)
    if SA < 0.0:
        SA = 0.0 # Catch trig errors
    return SA


def SA_partial_horiz_conical_head(D, a, h):
    r'''Calculates the partial area of a conical tank head in the context of
    a horizontal vessel and liquid partially
    filling it. This computes the wetted surface area of one of the conical
    heads only.

    .. math::
        \text{SA} = \frac{\sqrt{(a^2 + R^2)}}{R}\left[R^2\cos^{-1}\left(\frac{
        (R-h)}{R}\right) - (R-h)\sqrt{(2Rh - h^2)}\right]

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the cone head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one conical tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    Examples
    --------
    >>> SA_partial_horiz_conical_head(D=72., a=48.0, h=24.0)
    1980.0498315169873

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if h > D:
        h = D
    elif h < 0:
        return 0.0
    R = 0.5*D
    R_inv = 1.0/R
    return sqrt(a*a + R*R)*R_inv*(R*R*acos((R-h)*R_inv) - (R-h)*sqrt(2.0*R*h - h*h))

def _SA_partial_horiz_spherical_head_to_int(x, R2, a4, c1, c2):
    x2 = x*x
    to_pow = (R2 - x2)/(c2 - a4*x2)
    if to_pow < 0.0: to_pow = 0.0
    num = c1*sqrt(to_pow)
    try:
        return asin(num)
    except:
        # Tried to asin a number just slightly higher than 1
        return 0.5*pi

def SA_partial_horiz_spherical_head(D, a, h):
    r'''Calculates the partial area of a spherical tank head in the context of
    a horizontal vessel and liquid partially
    filling it. This computes the wetted surface area of one of the spherical
    heads only.

    .. math::
        \text{SA} = \frac{a^2 + R^2}{|a|}\int_{R-h}^R
        \sin^{-1} \frac{2|a|\sqrt{R^2-x^2}} {\sqrt{(a^2+R^2)^2 - (2ax)^2}} dx

    For the special case of :math:`|a| = R` :

    .. math::
        \text{SA} = \pi R h

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the spherical head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one spherical tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    A symbolic attempt did not suggest any analytical integrals were available.

    Examples
    --------
    >>> SA_partial_horiz_spherical_head(D=72., a=48.0, h=24.0)
    2027.2672

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    R = 0.5*D
    if a == R:
        return pi*R*h
    elif h < 0.0:
        return 0.0
    elif h > D:
        h = D

    fact = (a*a + R*R)/abs(a)
    R2 = R*R
    a2 = a + a
    a4 = a2*a2
    c1 = 2.0*abs(a)
    c2 = (a*a + R2)*(a*a + R2)

    SA = fact*quad(_SA_partial_horiz_spherical_head_to_int, R-h, R, args=(R2, a4, c1, c2))[0]
    return SA


def _SA_partial_horiz_ellipsoidal_head_to_int_dbl(x, y, c1, R2, R4, h):
    y2 = y*y
    x2 = x*x
    num = c1*(x2 + y2) - R4
    den = x2 + (y2 - R2) # Brackets help numerical truncation; zero div without it
    try:
        return sqrt(num/den)
    except:
         # Equation is undefined for y == R when x is zero; avoid it
        return _SA_partial_horiz_ellipsoidal_head_to_int_dbl(x, y*(1.0 - 1e-12), c1, R2, R4, h)

def _SA_partial_horiz_ellipsoidal_head_limits(x, c1, R2, R4, h):
    return [0.0, sqrt(R2 - x*x)]

def _SA_partial_horiz_ellipsoidal_head_limits2(c1, R2, R4, h):
    R = sqrt(R2)
    return [R-h, R]

def _SA_partial_horiz_ellipsoidal_head_to_int(y, c1, R2, R4):
    y2 = y*y
    t0 = c1*y2
    x6 = c1*(y2 - R2)/(t0 - R4)
    ans = sqrt(R4 - t0)*float(ellipe(x6))
    return ans

def SA_partial_horiz_ellipsoidal_head(D, a, h):
    r'''Calculates the partial area of a ellipsoidal tank head in the context of
    a horizontal vessel and liquid partially
    filling it. This computes the wetted surface area of one of the ellipsoidal
    heads only.

    .. math::
        \text{SA} = \frac{2}{R} \int_{R-h}^R \int_0^{\sqrt{R^2 - x^2}}
        \sqrt{ \frac{(R^2 - a^2)x^2
        + (R^2 - a^2)y^2 - R^4} {x^2 + y^2 - R^2}} dy dx

    After extensive manipulation, the first integral was solved analytically,
    extending the result of [1]_ with greater performance.

    .. math::
        \text{SA} = \frac{2}{R} \int_{R-h}^R \frac{\left(\frac{R^{4} - R^{2}
        \left(R^{2} - a^{2}\right)}{R^{2} - y^{2}}\right)^{0.5} \left(R^{2}
        - y^{2}\right)^{0.5} E{\left(\frac{\left(- R^{2} + y^{2}\right) \left(R^{2}
        - a^{2}\right)}{- R^{4} + y^{2} \left(R^{2} - a^{2}\right)} \right)}}
        {\left(\frac{R^{4} - R^{2} \left(R^{2} - a^{2}\right)}{R^{4} - y^{2}
        \left(R^{2} - a^{2}\right)}\right)^{0.5}}

    Where :math:`E(x)` is the complete elliptic integral of the second kind,
    calculated with SciPy's link to the cephes library.

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the ellipsoidal head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one ellipsoidal tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    The original numerical double integral is extremely nasty - there are places
    where f(x) -> infinity but that have a bounded area. quadpack's numerical
    integration handles this well, but adaptive inetgration which is not
    aware of singularities does not.

    Examples
    --------
    >>> SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=24.0)
    3401.233622547

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    R = 0.5*D
    if h < 0.0:
        return 0.0
    elif h > D:
        h = D
    R2 = R*R
    R4 = R2*R2
    a2 = a*a
    c1 = R2 - a2
#    from fluids.numerics import dblquad
#    from scipy.integrate import dblquad, nquad
##    quad_val = nquad(_SA_partial_horiz_ellipsoidal_head_to_int, ranges=[_SA_partial_horiz_ellipsoidal_head_limits, _SA_partial_horiz_ellipsoidal_head_limits2],
##                     args=(c1, R2, R4, h))[0]
#    quad_val = dblquad(_SA_partial_horiz_ellipsoidal_head_to_int, R-h, R, lambda x: 0.0, lambda x: (R2 - x*x)**0.5,
#                       args=(c1, R2, R4, h))[0]
    quad_val = quad(_SA_partial_horiz_ellipsoidal_head_to_int, R-h, R, args=(c1, R2, R4))[0]
    SA = 2.0/R*quad_val
    return SA


def _SA_partial_horiz_guppy_head_to_int(x, a, R):
    x0 = a*a
    x1 = R - x
    x2 = x1*x1
    x3 = 1.0/x2
    x4 = x0*x3 + 1.0
    x5 = R*R
    x6 = x*x
    x7 = x5 - x6
    x8 = sqrt(x7)
    x9 = x4*x8
    x10 = x0 + 4.0*x5
    x17 = sqrt(sqrt(x10))
    x11 = x17*x17
    x12 = 1.0/x11
    x13 = a*x7
    x14 = x12*x13*x3 + 1.0
    x15 = a*x12
    x16 = sqrt(a)
    x18 = 2*atan(x16*x8/(x1*x17))
    x19 = 0.5 - 0.5*x15
    x100 = (-2.0*R*x*x11 + x11*x5 + x11*x6 + x13)
    x20 = x1*x14*sqrt(x2*x5*(x0 + x2)/(x100*x100))/x5
    return 0.08333333333333333*(
             (-4.0*x10**0.75*x16*x20*ellipeinc(x18, x19) + 4.0*x9
             + 2.0*x17*x20*(a*x11 + x10)*ellipkinc(x18, x19)/x16
             + 8.0*x15*x9/x14)*1.0/sqrt(x4))


def SA_partial_horiz_guppy_head(D, a, h):
    r'''Calculates the partial area of a guppy tank head in the context of
    a horizontal vessel and liquid partially
    filling it. This computes the wetted surface area of one of the guppy
    heads only.

    .. math::
        \text{SA} = 2\int_{-R}^{h-R}\int_{0}^{\sqrt{R^2 - x^2}}
        \sqrt{1 + \left(\frac{a}{2R}\left(1 - \frac{y^2}{(R-x)^2}  \right)
        \right)^2
        + \left(\frac{ay}{R(R-x)}    \right)^2 } dy dx

    After extensive manipulation, the first integral was solved analytically,
    extending the result of [1]_. Even with the special functions, this
    form has somewhat greater performance (and improved precision).

    .. math::
        \text{SA} = 2 \int_{-R}^{h-R} \frac{\frac{2 a \left(4 + \frac{a^{2}
        \left(2 R^{2} - 2 R y\right)^{2}}{R^{2} \left(R - y\right)^{4}}\right)
        \sqrt{R^{2} - y^{2}}}{\sqrt{4 R^{2} + a^{2}} \left(\frac{a \left(R^{2}
        - y^{2}\right)}{\left(R - y\right)^{2} \sqrt{4 R^{2} + a^{2}}}
        + 1\right)} + \left(4 + \frac{a^{2} \left(2 R^{2} - 2 R y\right)
        ^{2}}{R^{2} \left(R - y\right)^{4}}\right) \sqrt{R^{2} - y^{2}}
        - \frac{2 \sqrt{a} \sqrt{\frac{4 R^{2} \left(R - y\right)^{4}
        + a^{2} \left(2 R^{2} - 2 R y\right)^{2}}{\left(R^{2} \sqrt{4 R^{2}
        + a^{2}} - 2 R y \sqrt{4 R^{2} + a^{2}} + a \left(R^{2}
        - y^{2}\right) + y^{2} \sqrt{4 R^{2} + a^{2}}\right)^{2}}}
        \left(R - y\right) \left(4 R^{2} + a^{2}\right)^{0.75}
        \left(\frac{a \left(R^{2} - y^{2}\right)}{\left(R - y\right)^{2}
        \sqrt{4 R^{2} + a^{2}}} + 1\right) \operatorname{ellipeinc}{\left(2
        \operatorname{atan}{\left(\frac{\sqrt{a} \sqrt{R^{2} - y^{2}}}
        {\left(R - y\right) \left(4 R^{2} + a^{2}\right)^{0.25}} \right)},
        - \frac{a}{2 \sqrt{4 R^{2} + a^{2}}} + 0.5 \right)}}{R^{2}}
        + \frac{1.0 \sqrt{\frac{4 R^{2} \left(R - y\right)^{4} + a^{2}
        \left(2 R^{2} - 2 R y\right)^{2}}{\left(R^{2} \sqrt{4 R^{2}
        + a^{2}} - 2 R y \sqrt{4 R^{2} + a^{2}} + a \left(R^{2} - y^{2}\right)
            + y^{2} \sqrt{4 R^{2} + a^{2}}\right)^{2}}} \left(R - y\right)
        \left(4 R^{2} + a^{2}\right)^{0.25} \left(\frac{a \left(R^{2}
        - y^{2}\right)}{\left(R - y\right)^{2} \sqrt{4 R^{2} + a^{2}}}
        + 1\right) \left(4 R^{2} + a^{2} + a \sqrt{4 R^{2} + a^{2}}\right)
        \operatorname{ellipkinc}{\left(2 \operatorname{atan}{\left(\frac{\sqrt{a}
        \sqrt{R^{2} - y^{2}}}{\left(R - y\right) \left(4 R^{2}
        + a^{2}\right)^{0.25}} \right)},- \frac{a}{2 \sqrt{4 R^{2} + a^{2}}}
        + 0.5 \right)}}{R^{2} \sqrt{a}}}{6 \sqrt{4 + \frac{a^{2} \left(2 R^{2}
        - 2 R y\right)^{2}}{R^{2} \left(R - y\right)^{4}}}}

    Where ellipeinc is the incomplete elliptic integral of the second kind,
    and ellipkinc is the incomplete elliptic integral of the first kind,
    both calculated with SciPy's link to the cephes library.

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the guppy head extends on one side, [m]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one guppy tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    The analytical integral was derived with Rubi.

    Examples
    --------
    >>> SA_partial_horiz_guppy_head(D=72., a=48.0, h=24.0)
    1467.8949

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    R = 0.5*D
    if a == R:
        return pi*R*h
    elif h < 0.0:
        return 0.0
    elif h > D:
        h = D
    if -R == h-R:
        return 0.0

#    c1 = a/(2.0*R)
#    c2 = c1*c1
#    a_R_ratio = a/R
#    a_R_ratio2 = a_R_ratio*a_R_ratio
#    from scipy.integrate import dblquad
#    def to_quad(y, x):
#        t1 = y/(R-x)
#        t1 *= t1
#        t2 = (1.0 - t1)
#        return (1.0 + c2*t2*t2 + a_R_ratio2*t1)**0.5
#    quad_val = dblquad(to_quad, -R, h-R, lambda x: 0.0, lambda x: (R*R - x*x)**0.5)[0]
    quad_val = quad(_SA_partial_horiz_guppy_head_to_int, -R, h-R, args=(a, R))[0]
    SA = 2.0*quad_val
    return SA

def _SA_partial_horiz_torispherical_head_int_1(x, b, c):
    x0 = x*x
    x1 = b - x0
    x2 = sqrt(x1)
    x3 = -b + x0
    x4 = c*c
    try:
        x5 = 1.0/sqrt(x1 - x4)
    except:
        x5 = 1.0/csqrt(x1 - x4)
    x6 = x3 + x4
    x7 = sqrt(b)
    try:
        x3_pow = x3**(-1.5)
    except:
        x3_pow = (x3+0j)**(-1.5)
    ans = (x*cacos(c/x2) + x3_pow*x5*(-c*x1*csqrt(-x6*x6)*catan(x*x2/(csqrt(x3)*csqrt(x6)))
        + x6*x7*csqrt(-x1*x1)*catan(c*x*x5/x7))/csqrt(-x6/x1))
    return abs(ans.real)

def _SA_partial_horiz_torispherical_head_int_2(y, t2, s, c1):
#    from mpmath import mp, mpf, atanh as catanh
#    mp.dps=30
#    y, t2, s, c1 = mpf(y), mpf(t2), mpf(s), mpf(c1)
    y2 = y*y
    try:
        x10 = sqrt(t2 - y2)
        try:
            # Some tiny heights make the square root slightly under 0
            x = (sqrt(c1 - y2 + (s+s)*x10)).real
        except:
            # Python 2 compat - don't take the square root of a negative number with no complex part
            x = (csqrt(c1 - y2 + (s+s)*x10 + 0.0j)).real

    except:
        x10 = csqrt(t2 - y2+0.0j)
        x = (csqrt(c1 - y2 + (s+s)*x10 + 0.0j)).real
    try:
        x0 = t2 - y2
        x1 = s*x10.real
        t10 = x1 + x1 + s*s + x0


        # x3, x4 present a very nasty numerical problem.
        # issue occurs when h == R, x3 is really equal to R**2 - 2*R*h + h**2
        x3 = t10 - x*x
        x4 = sqrt(x3)
        # One solution is to use higher precision everywhere



        ans = x4*sqrt(t2*t10/(x0*x3))*catan(x/x4).real
    except:
        ans = 0.0
#     ans = sqrt((t2* (s**2+t2-x**2+2.0*s* sqrt(t2-x**2)))/((t2-x**2)* (s**2+t2-x**2+2 *s* sqrt(t2-x**2)-y**2)))* sqrt(s**2+t2-x**2+2 *s* sqrt(t2-x**2)-y**2) *atan(y/sqrt(s**2+t2-x**2+2 *s* sqrt(t2-x**2)-y**2))
#    print(float(y), float(t2), float(s), float(c1), float(ans.real))
#    return float(ans.real)
    return ans.real

def _SA_partial_horiz_torispherical_head_int_3(y, x, s, t2):
    x2 = x*x
    y2 = y*y
    x10 = sqrt(t2 - x2)
    num = (s + x10)*(s + x10)*x2 + (t2 - x2)*y2
    den = (t2 - x2)*(s*s + t2 - x2 - y2 + 2.0*s*x10)
    f = sqrt(1.0 + num/den)
    return f

def SA_partial_horiz_torispherical_head(D, f, k, h):
    r'''Calculates the partial area of a torispherical tank head in the context of
    a horizontal vessel and liquid partially
    filling it. This computes the wetted surface area of one of the torispherical
    heads only.

    The expressions used are quite complicated; see [1]_ for more details.


    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]
    h : float
        Height, as measured up to where the fluid ends, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one torispherical tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    One integral:

    .. math::
        \int_{R-h}^{fD\sin \alpha} \cos^{-1} \frac{fD\cos \alpha}{\sqrt{f^2 D^2
        - x^2}} dx

    Can be computed as follows, using WolframAlpha.

    .. math::
        x \operatorname{acos}{\left(\frac{c}{\sqrt{b - x^{2}}} \right)} + \frac{\sqrt{b}
        \sqrt{- \left(b - x^{2}\right)^{2}} \left(- b + c^{2} + x^{2}\right)
        \operatorname{atan}{\left(\frac{c x}{\sqrt{b} \sqrt{b - c^{2} - x^{2}}}
        \right)} + c \sqrt{- \left(- b + c^{2} + x^{2}\right)^{2}} \left(- b
        + x^{2}\right) \operatorname{atan}{\left(\frac{x \sqrt{b - x^{2}}}
        {\sqrt{- b + x^{2}} \sqrt{- b + c^{2} + x^{2}}} \right)}}{\sqrt{
        \frac{- b + c^{2} + x^{2}}{- b + x^{2}}} \left(- b + x^{2}\right)^{1.5}
        \sqrt{b - c^{2} - x^{2}}}

    With the following constants:

    .. math::
        c = fD\cos \alpha

    .. math::
        b = f^2 D^2

    The other integral is a double integral. There is an analytical integral
    available for the first integral, which takes the form:

    .. math::
        2 \sqrt{\frac{R^{2} k^{2} \left(4 R^{2} k^{2} - y^{2} + \left(- 2 R k
        + R\right)^{2} + 2 \left(- 2 R k + R\right) \sqrt{4 R^{2} k^{2} - y^{2}}
        \right)}{\left(4 R^{2} k^{2} - y^{2}\right) \left(\left(R
        - h\right)^{2} - \left(- 4 R k + 2 R\right) \sqrt{4 R^{2} k^{2}
        - y^{2}} + 2 \left(- 2 R k + R\right) \sqrt{4 R^{2} k^{2} - y^{2}}
        \right)}} \sqrt{\left(R - h\right)^{2} - \left(- 4 R k + 2 R\right)
        \sqrt{4 R^{2} k^{2} - y^{2}} + 2 \left(- 2 R k + R\right)
        \sqrt{4 R^{2} k^{2} - y^{2}}} \operatorname{atan}{\left(\frac{
        \sqrt{4 R^{2} k^{2} - y^{2} - \left(R - h\right)^{2} + \left(- 4 R k
        + 2 R\right) \sqrt{4 R^{2} k^{2} - y^{2}} + \left(- 2 R k + R\right)
        ^{2}}}{\sqrt{\left(R - h\right)^{2} - \left(- 4 R k + 2 R\right)
        \sqrt{4 R^{2} k^{2} - y^{2}} + 2 \left(- 2 R k + R\right)
        \sqrt{4 R^{2} k^{2} - y^{2}}}} \right)}

    Examples
    --------
    >>> SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=24.0)
    1471.201832459

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if h <= 0.0:
        return 0.0
    elif h > D:
        h = D
    R = D/2.
    r = f*D
    alpha = asin((1.0 - 2.0*k)/(2.*(f-k)))
    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)
    s = R - k*D
    t = k*D

    s2 = s*s
    t2 = t*t
    a1 = r*(1.0 - cos_alpha)
    a2 = k*D*cos_alpha

    c = f*D*cos_alpha

    b = f*f*D*D

    c1 = s2 + t2  - (R - h)**2
    def G_lim(x): # numba: delete
        x2 = x*x # numba: delete
        try: # numba: delete
            G = sqrt(c1 - x2 + (s+s)*sqrt(t2 - x2)) # numba: delete
        except: # numba: delete
            # Python 2 compat - don't take the square root of a negative number with no complex part # numba: delete
            G = sqrt(c1 - x2 + (s+s)*sqrt(t2 - x2+0.0j) + 0.0j) # numba: delete
        return G.real # Some tiny heights make the square root slightly under 0 # numba: delete

    limit_1 = k*D*(1.0 - sin_alpha)

    if h < limit_1:
        SA = quad(_SA_partial_horiz_torispherical_head_int_2, 0.0, sqrt(2*k*D*h - h*h), args=(t2, s, c1))[0]
        return 2.0*SA
    elif limit_1 < h <= R:
        if (D*.499 < h < D*.501): # numba: delete
            from scipy.integrate import dblquad # numba: delete
            SA = 2.0*dblquad(_SA_partial_horiz_torispherical_head_int_3, 0.0, a2, lambda x: 0, G_lim, args=(s, t2))[0] # numba: delete
        else: # numba: delete
            # Numerical issues
            SA = 2.0*quad(_SA_partial_horiz_torispherical_head_int_2, 0.0, a2, args=(t2, s, c1))[0] # numba: delete
#        SA = 2.0*quad(_SA_partial_horiz_torispherical_head_int_2, 0.0, a2, args=(t2, s, c1))[0] # numba: uncomment
        try:
            high = _SA_partial_horiz_torispherical_head_int_1(f*D*sin_alpha, b, c)
        except:
            # Expression with the substitution is equally complicated
            high = _SA_partial_horiz_torispherical_head_int_1(f*D*sin_alpha*(1.0 + 1e-14), b, c)
        int_1_term1 = high - _SA_partial_horiz_torispherical_head_int_1(R-h, b, c)
        SA += 2.0*f*D*int_1_term1
    else:
        SA = 2.0*pi*f*D*a1 + 2*pi*k*D*(a2 + (R - k*D)*asin(a2/(k*D)))
        SA -= SA_partial_horiz_torispherical_head(D, f, k, h=D-h)
    return SA


def SA_partial_vertical_conical_head(D, a, h):
    r'''Calculates the partial area of a conical tank head in the context of
    a vertical vessel and liquid partially
    filling it. This computes the wetted surface area of one of the conical
    heads only, and is valid for `h` up to `a` only.

    .. math::
        \text{SA} = \frac{\pi R h^2 \sqrt{a^2 + R^2}}{a^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the cone head extends beneath the vertical tank, [m]
    h : float
        Height, as measured up to where the fluid ends or the top of the
        conical head, whichever is less, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one conical tank head extending
        beneath the vessel, [m^2]

    Notes
    -----
    This method is undefined for :math:`h < 0`, but this is handled
    by returning zero.

    Examples
    --------
    >>> SA_partial_vertical_conical_head(D=72., a=48.0, h=24.0)
    1696.4600329384882

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if a == 0.0:
        return 0.25*pi*D*D
    elif h <= 0.0:
        return 0.0
    R = 0.5*D
    SA = pi*R*h*h*sqrt(a*a + R*R)/(a*a)
    return SA


def SA_partial_vertical_ellipsoidal_head(D, a, h):
    r'''Calculates the partial area of a ellipsoidal tank head in the context of
    a vertical vessel and liquid partially
    filling it. This computes the wetted surface area of one of the ellipsoidal
    heads only, and is valid for `h` up to `a` only.

    If :math:`a > R`:

    .. math::
        \text{SA} = \pi R^2 - \frac{\pi (a - h)R\sqrt{a^4 - (a-h)^2(a^2-R^2)}}{a^2}
        + \frac{\pi a^2 R}{\sqrt{a^2 - R^2}}\left(
        \cos^{-1} \frac{R}{a} - \sin^{-1} \frac{(a-h)\sqrt{a^2-R^2}}{a^2}
        \right)

    Otherwise for :math:`0 < a < R`:

    .. math::
        \text{SA} = \pi R^2 - \frac{\pi (a - h)R\sqrt{a^4 - (a-h)^2(a^2-R^2)}}{a^2}
        + \frac{\pi a^2 R}{\sqrt{a^2 - R^2}}\ln \left(\frac{a(\sqrt{R^2 - a^2} + R)}
        {(a-h)\sqrt{R^2 - a^2} + \sqrt{a^4 + (a-h)^2(R^2 - a^2)}}
        \right)


    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the ellipsoidal head extends beneath the vertical tank, [m]
    h : float
        Height, as measured up to where the fluid ends or the top of the
        ellipsoidal head, whichever is less, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one ellipsoidal tank head extending
        beneath the vessel, [m^2]

    Notes
    -----
    This method is undefined for :math:`h < 0`, but this is handled
    by returning zero.

    Examples
    --------
    >>> SA_partial_vertical_ellipsoidal_head(D=72., a=48.0, h=24.0)
    4675.23789137632

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if a == 0.0:
        return 0.25*pi*D*D
    elif h <= 0.0:
        return 0.0
    # h should be less than a
    R = 0.5*D
    SA = pi*R*R
    a2 = a*a
    a_inv = 1.0/a
    R2 = R*R
    SA -= pi*(a - h)*R*sqrt(a2*a2 - (a-h)*(a-h)*(a2 - R2))*a_inv*a_inv
    if a > R:
        # This one has issues around a == R
        SA += pi*a2*R/sqrt(a2 - R2)*(acos(R*a_inv) - asin((a-h)*sqrt(a2 - R2)*a_inv*a_inv))
    elif a == R:
        # Special case avoids zero division
        return pi*D*h
    else:
        x1 = sqrt(R2 - a2)
        num = a*(x1 + R)
        den = (a-h)*x1 + sqrt(a2*a2 + (a-h)*(a-h)*(R2 - a2))
        SA += pi*a2*R/x1*log(num/den)
    return SA

def SA_partial_vertical_spherical_head(D, a, h):
    r'''Calculates the partial area of a spherical tank head in the context of
    a vertical vessel and liquid partially
    filling it. This computes the wetted surface area of one of the conical
    heads only, and is valid for `h` up to `a` only.

    .. math::
        \text{SA} = \pi h \left(\frac{a^2 + R^2}{a}\right)

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    a : float
        Distance the spherical head extends beneath the vertical tank, [m]
    h : float
        Height, as measured up to where the fluid ends or the top of the
        spherical head, whichever is less, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one spherical tank head extending
        beneath the vessel, [m^2]

    Notes
    -----
    This method is undefined for :math:`h < 0`, but this is handled
    by returning zero.

    Examples
    --------
    >>> SA_partial_vertical_spherical_head(72, a=24, h=12)
    2940.5307237600464

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if a == 0.0:
        return 0.25*pi*D*D
    elif h <= 0.0:
        return 0.0
    R = 0.5*D
    SA = pi*h*((a*a + R*R)/a)
    return SA


def SA_partial_vertical_torispherical_head(D, f, k, h):
    r'''Calculates the partial area of a torispherical tank head in the context of
    a vertical vessel and liquid partially
    filling it. This computes the wetted surface area of one of the torispherical
    heads only.

    if :math:`a_1 <= h`:

    .. math::
        \text{SA} = 2\pi f D h

    if :math:`a_1 \le h \le a`:

    .. math::
        \text{SA} = 2\pi f D a_1 + 2\pi k D\left(
        h - a_1 + (R - kD) \left(
        \sin^{-1} \frac{a_2}{kD} - \sin^{-1} \frac{a-h}{kD} \right) \right)

    .. math::
        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

    .. math::
        a_1 = fD(1-\cos\alpha)

    .. math::
        a_2 = kD\cos\alpha

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]
    h : float
        Height, as measured up to where the fluid ends or the top of the
        torispherical head, whichever is less, [m]

    Returns
    -------
    SA_partial : float
        Partial (wetted) surface area of one torispherical tank head, [m^2]

    Notes
    -----
    This method is undefined for :math:`h > D` and :math:`h < 0`, but those
    cases are handled by returning the full surface area and the zero
    respectively.

    Examples
    --------
    This method is undefined for :math:`h < 0`, but this is handled
    by returning zero.

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
    '''
    if h <= 0.0:
        return 0.0
    R = 0.5*D
    alpha = asin((1.0 - 2.0*k)/(2.0*(f-k)))
    cos_alpha = cos(alpha)
    a1 = f*D*(1.0 - cos_alpha)
    a2 = k*D*cos_alpha
    a = a1 + a2
    if h < a1:
        SA = 2.0*pi*f*D*h
    elif a1 <= h <= a:
        SA = 2.0*pi*f*D*a1
        kD_inv = 1.0/(k*D)
        SA += 2.0*pi*k*D*(h - a1 + (R - k*D)*(asin(a2*kD_inv) - asin((a-h)*kD_inv)))
    return SA


def a_torispherical(D, f, k):
    r'''Calculates depth of a torispherical head according to [1]_.

    .. math::
        a = a_1 + a_2

    .. math::
        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

    .. math::
        a_1 = fD(1-\cos\alpha)

    .. math::
        a_2 = kD\cos\alpha

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dimensionless dish-radius parameter; also commonly given as the
        product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    k : float
        Dimensionless knuckle-radius parameter; also commonly given as the
        product of `k` and `D` (`kD`), which is called the knuckle radius
        and has units of length, [-]

    Returns
    -------
    a : float
        Depth of head [m]

    Examples
    --------
    Example from [1]_.

    >>> a_torispherical(D=96., f=0.9, k=0.2)
    25.684268924767125

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    alpha = asin((1-2*k)/(2*(f-k)))
    a1 = f*D*(1 - cos(alpha))
    a2 = k*D*cos(alpha)
    return a1 + a2


def V_from_h(h, D, L, horizontal=True, sideA=None, sideB=None, sideA_a=0,
             sideB_a=0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None):
    r'''Calculates partially full volume of a vertical or horizontal tank with
    different head types according to [1]_.

    Parameters
    ----------
    h : float
        Height of the liquid in the tank, [m]
    D : float
        Diameter of the cylindrical section of the tank, [m]
    L : float
        Length of the main cylindrical section of the tank, [m]
    horizontal : bool, optional
        Whether or not the tank is a horizontal or vertical tank
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section, [m]
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section, [m]
    sideA_f : float, optional
        Dimensionless dish-radius parameter for side A; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideA_k : float, optional
        Dimensionless knuckle-radius parameter for side A; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]
    sideB_f : float, optional
        Dimensionless dish-radius parameter for side B; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideB_k : float, optional
        Dimensionless knuckle-radius parameter for side B; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]

    Returns
    -------
    V : float
        Volume up to h [m^3]

    Examples
    --------
    >>> V_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical',
    ... sideB='conical', sideA_a=2., sideB_a=1.)
    10.013826583317465

    References
    ----------
    .. [1] Jones, D. "Compute Fluid Volumes in Vertical Tanks." Chemical
       Processing. December 18, 2003.
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if sideA is not None and sideA not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side A')
    if sideB is not None and sideB not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side B')
    R = 0.5*D
    V = 0.0
    if horizontal:
        # Conical case
        if sideA == 'conical' and sideB == 'conical' and sideA_a == sideB_a:
            V += 2.0*V_horiz_conical(D, L, sideA_a, h, headonly=True)
        else:
            if sideA == 'conical':
                V += V_horiz_conical(D, L, sideA_a, h, headonly=True)
            if sideB == 'conical':
                V += V_horiz_conical(D, L, sideB_a, h, headonly=True)
        # Elliosoidal case
        if sideA == 'ellipsoidal' and sideB == 'ellipsoidal' and sideA_a == sideB_a:
            V += 2.0*V_horiz_ellipsoidal(D, L, sideA_a, h, headonly=True)
        else:
            if sideA == 'ellipsoidal':
                V += V_horiz_ellipsoidal(D, L, sideA_a, h, headonly=True)
            if sideB == 'ellipsoidal':
                V += V_horiz_ellipsoidal(D, L, sideB_a, h, headonly=True)
        # Guppy case
        if sideA == 'guppy' and sideB == 'guppy' and sideA_a == sideB_a:
            V += 2.0*V_horiz_guppy(D, L, sideA_a, h, headonly=True)
        else:
            if sideA == 'guppy':
                V += V_horiz_guppy(D, L, sideA_a, h, headonly=True)
            if sideB == 'guppy':
                V += V_horiz_guppy(D, L, sideB_a, h, headonly=True)
        # Spherical case
        if sideA == 'spherical' and sideB == 'spherical' and sideA_a == sideB_a:
            V += 2.0*V_horiz_spherical(D, L, sideA_a, h, headonly=True)
        else:
            if sideA == 'spherical':
                V += V_horiz_spherical(D, L, sideA_a, h, headonly=True)
            if sideB == 'spherical':
                V += V_horiz_spherical(D, L, sideB_a, h, headonly=True)
        # Torispherical case
        if (sideA == 'torispherical' and sideB == 'torispherical'
            and (sideA_f == sideB_f) and (sideA_k == sideB_k)):
            V += 2.0*V_horiz_torispherical(D, L, sideA_f, sideA_k, h, headonly=True)
        else:
            if sideA == 'torispherical':
                V += V_horiz_torispherical(D, L, sideA_f, sideA_k, h, headonly=True)
            if sideB == 'torispherical':
                V += V_horiz_torispherical(D, L, sideB_f, sideB_k, h, headonly=True)
        if h > D: # Must be before Af, which will raise a domain error
            raise ValueError('Input height is above top of tank')
        Af = R*R*acos((R-h)/R) - (R-h)*sqrt(2.0*R*h - h*h)
        V += L*Af
    else:
        # Bottom head
        if sideA in ('conical', 'ellipsoidal', 'torispherical', 'spherical'):
            if sideA == 'conical':
                V += V_vertical_conical(D, sideA_a, h=min(sideA_a, h))
            if sideA == 'ellipsoidal':
                V += V_vertical_ellipsoidal(D, sideA_a, h=min(sideA_a, h))
            if sideA == 'spherical':
                V += V_vertical_spherical(D, sideA_a, h=min(sideA_a, h))
            if sideA == 'torispherical':
                V += V_vertical_torispherical(D, sideA_f, sideA_k, h=min(sideA_a, h))
        # Cylindrical section
        if h >= sideA_a + L:
            V += pi/4*D**2*L # All middle
        elif h > sideA_a:
            V += pi/4*D**2*(h - sideA_a) # Partial middle
        # Top head
        if h > sideA_a + L:
            h2 = sideB_a - (h - sideA_a - L)
            if sideB == 'conical':
                V += V_vertical_conical(D, sideB_a, h=sideB_a)
                V -= V_vertical_conical(D, sideB_a, h=h2)
            if sideB == 'ellipsoidal':
                V += V_vertical_ellipsoidal(D, sideB_a, h=sideB_a)
                V -= V_vertical_ellipsoidal(D, sideB_a, h=h2)
            if sideB == 'spherical':
                V += V_vertical_spherical(D, sideB_a, h=sideB_a)
                V -= V_vertical_spherical(D, sideB_a, h=h2)
            if sideB == 'torispherical':
                V += V_vertical_torispherical(D, sideB_f, sideB_k, h=sideB_a)
                V -= max(0.0, V_vertical_torispherical(D, sideB_f, sideB_k, h=h2))
        if h > L + sideA_a + sideB_a:
            raise ValueError('Input height is above top of tank')
    return V


def SA_from_h(h, D, L, horizontal=True, sideA=None, sideB=None, sideA_a=0.0,
             sideB_a=0.0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None):
    r'''Calculates partially full wetted surface area of a vertical or horizontal tank with
    different head types according to [1]_.

    Parameters
    ----------
    h : float
        Height of the liquid in the tank, [m]
    D : float
        Diameter of the cylindrical section of the tank, [m]
    L : float
        Length of the main cylindrical section of the tank, [m]
    horizontal : bool, optional
        Whether or not the tank is a horizontal or vertical tank
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section, [m]
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section, [m]
    sideA_f : float, optional
        Dimensionless dish-radius parameter for side A; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideA_k : float, optional
        Dimensionless knuckle-radius parameter for side A; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]
    sideB_f : float, optional
        Dimensionless dish-radius parameter for side B; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideB_k : float, optional
        Dimensionless knuckle-radius parameter for side B; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]

    Returns
    -------
    SA : float
        Wetted wall surface area up to h [m^3]

    Examples
    --------
    >>> SA_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical',
    ... sideB='conical', sideA_a=2., sideB_a=1.)
    28.59477853914843

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Wetted Area." Text. Chemical Processing.
       April 2017. https://www.chemicalprocessing.com/assets/Uploads/calculating-tank-wetted-area.pdf
       http://www.chemicalprocessing.com/articles/2003/193/
    '''
    if sideA is not None and sideA not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side A')
    if sideB is not None and sideB not in ('conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy'):
        raise ValueError('Unspoorted head type for side B')
    R = 0.5*D
    SA = 0.0
    if horizontal:
        # Conical case
        if sideA == 'conical':
            SA += SA_partial_horiz_conical_head(D, sideA_a, h)
        if sideB == 'conical':
            SA += SA_partial_horiz_conical_head(D, sideB_a, h)
        # Elliosoidal case
        if sideA == 'ellipsoidal':
            SA += SA_partial_horiz_ellipsoidal_head(D, sideA_a, h)
        if sideB == 'ellipsoidal':
            SA += SA_partial_horiz_ellipsoidal_head(D, sideB_a, h)
        # Guppy case
        if sideA == 'guppy':
            SA += SA_partial_horiz_guppy_head(D, sideA_a, h)
        if sideB == 'guppy':
            SA += SA_partial_horiz_guppy_head(D, sideB_a, h)
        # Spherical case
        if sideA == 'spherical':
            SA += SA_partial_horiz_spherical_head(D, sideA_a, h)
        if sideB == 'spherical':
            SA += SA_partial_horiz_spherical_head(D, sideB_a, h)
        # Torispherical case
        if sideA == 'torispherical':
            if sideA_f is not None and sideA_k is not None:
                SA += SA_partial_horiz_torispherical_head(D, sideA_f, sideA_k, h)
            else:
                raise ValueError("Torispherical sideA but no `f` and `k` provided")
        if sideB == 'torispherical':
            if sideB_f is not None and sideB_k is not None:
                SA += SA_partial_horiz_torispherical_head(D, sideB_f, sideB_k, h)
            else:
                raise ValueError("Torispherical sideB but no `f` and `k` provided")
        # Flat case
        if sideA is None:
            SA += A_partial_circle(D, h)
        if sideB is None:
            SA += A_partial_circle(D, h)
        if h > D: # Must be before Af, which will raise a domain error
            raise ValueError('Input height is above top of tank')
        SA += L*D*acos((D - h - h)/D)
    else:
        # Bottom head
        if sideA in ('conical', 'ellipsoidal', 'torispherical', 'spherical'):
            if sideA == 'conical':
                SA += SA_partial_vertical_conical_head(D, sideA_a, h=min(sideA_a, h))
            elif sideA == 'ellipsoidal':
                SA += SA_partial_vertical_ellipsoidal_head(D, sideA_a, h=min(sideA_a, h))
            elif sideA == 'spherical':
                SA += SA_partial_vertical_spherical_head(D, sideA_a, h=min(sideA_a, h))
            elif sideA == 'torispherical':
                if sideA_f is not None and sideA_k is not None:
                    SA += SA_partial_vertical_torispherical_head(D, sideA_f, sideA_k, h=min(sideA_a, h))
                else:
                    raise ValueError("Torispherical sideA but no `f` and `k` provided")
        elif sideA is None:
                SA += 0.25*pi*D*D
        # Cylindrical section
        if h >= sideA_a + L:
            SA += pi*D*L # All middle
        elif h > sideA_a:
            SA += pi*D*(h - sideA_a) # Partial middle
        # Top head
        if h >= sideA_a + L: # greater or equals is needed! Flat head on top adds lots of area.
            h2 = sideB_a - (h - sideA_a - L)
            if sideB == 'conical':
                if sideB_a == 0.0:
                    SA += 0.25*pi*D*D
                else:
                    SA += SA_partial_vertical_conical_head(D, sideB_a, h=sideB_a)
                    SA -= SA_partial_vertical_conical_head(D, sideB_a, h=h2)
            elif sideB == 'ellipsoidal':
                if sideB_a == 0.0:
                    SA += 0.25*pi*D*D
                else:
                    SA += SA_partial_vertical_ellipsoidal_head(D, sideB_a, h=sideB_a)
                    SA -= SA_partial_vertical_ellipsoidal_head(D, sideB_a, h=h2)
            elif sideB == 'spherical':
                if sideB_a == 0.0:
                    SA += 0.25*pi*D*D
                else:
                    SA += SA_partial_vertical_spherical_head(D, sideB_a, h=sideB_a)
                    SA -= SA_partial_vertical_spherical_head(D, sideB_a, h=h2)
            elif sideB == 'torispherical':
                if sideB_a == 0.0:
                    SA += 0.25*pi*D*D
                else:
                    if sideB_f is not None and sideB_k is not None:
                        SA += SA_partial_vertical_torispherical_head(D, sideB_f, sideB_k, h=sideB_a)
                        SA -= max(0.0, SA_partial_vertical_torispherical_head(D, sideB_f, sideB_k, h=h2))
                    else:
                        raise ValueError("Torispherical sideB but no `f` and `k` provided")
            elif sideB is None and h == sideA_a + L:
                # End cap if flat
                SA += 0.25*pi*D*D
        if h > L + sideA_a + sideB_a:
            raise ValueError('Input height is above top of tank')
    return SA

def tank_from_two_specs_err(guess, spec0, spec1, spec0_name, spec1_name,
                            h, horizontal, sideA, sideB, sideA_a, sideB_a,
                            sideA_f, sideA_k, sideB_f, sideB_k,
                            sideA_a_ratio, sideB_a_ratio):
    D, L_over_D = float(guess[0]), float(guess[1])
    obj = TANK(D=D, L_over_D=L_over_D, horizontal=horizontal,
         sideA=sideA, sideB=sideB, sideA_a=sideA_a, sideB_a=sideB_a,
         sideA_f=sideA_f, sideA_k=sideA_k, sideB_f=sideB_f, sideB_k=sideB_k,
         sideA_a_ratio=sideA_a_ratio, sideB_a_ratio=sideB_a_ratio)
    # ensure h is always under the top
    h = min(h, obj.h_max)

    if spec0_name == 'V':
        err0 = obj.V_total - spec0
    elif spec0_name == 'SA':
        err0 = obj.A - spec0
    elif spec0_name == 'V_partial':
        err0 = obj.V_from_h(h) - spec0
    elif spec0_name == 'SA_partial':
        err0 = obj.SA_from_h(h) - spec0
    elif spec0_name == 'A_cross':
        err0 = obj.A_cross_sectional(h) - spec0

    if spec1_name == 'V':
        err1 = obj.V_total - spec1
    elif spec1_name == 'SA':
        err1 = obj.A - spec1
    elif spec1_name == 'V_partial':
        err1 = obj.V_from_h(h) - spec1
    elif spec1_name == 'SA_partial':
        err1 = obj.SA_from_h(h) - spec1
    elif spec1_name == 'A_cross':
        err1 = obj.A_cross_sectional(h) - spec1
#    print(err0, err1, D, L_over_D, h)
    return [err0, err1]

class TANK(object):
    """Class representing tank volumes and levels. All parameters are also
    attributes.

    Parameters
    ----------
    D : float
        Diameter of the cylindrical section of the tank, [m]
    L : float
        Length of the main cylindrical section of the tank, [m]
    horizontal : bool, optional
        Whether or not the tank is a horizontal or vertical tank
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical',
        'same'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical',
        'same'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section, [m]
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section, [m]
    sideA_f : float, optional
        Dimensionless dish-radius parameter for side A; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideA_k : float, optional
        Dimensionless knuckle-radius parameter for side A; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]
    sideB_f : float, optional
        Dimensionless dish-radius parameter for side B; also commonly given as
        the product of `f` and `D` (`fD`), which is called dish radius and
        has units of length, [-]
    sideB_k : float, optional
        Dimensionless knuckle-radius parameter for side B; also commonly given
        as the product of `k` and `D` (`kD`), which is called the knuckle
        radius and has units of length, [-]
    sideA_a_ratio : float, optional
        Ratio for `a` parameter; can be used instead of specifying an absolute
        value, [-]
    sideB_a_ratio : float, optional
        Ratio for `a` parameter; can be used instead of specifying an absolute
        value, [-]
    L_over_D : float, optional
        Ratio of length over diameter, used only when D and L are both
        unspecified but V is, [-]
    V : float, optional
        Volume of the tank; solved for if specified, using
        sideA_a_ratio/sideB_a_ratio, sideA, sideB, horizontal, and one
        of L_over_D, L, or D, [m^3]

    Attributes
    ----------
    h_max : float
        Height of the tank, [m]
    V_total : float
        Total volume of the tank as calculated [m^3]
    sideA_V : float
        Volume of only `sideA` [m^3]
    sideB_V : float
        Volume of only `sideB` [m^3]
    lateral_V : float
        Volume of cylindrical section of tank [m^3]
    A : float
        Total surface area of the tank, [m^2]
    A_sideA : float
        Surface area of sideA, [m^2]
    A_sideB : float
        Surface area of sideB, [m^2]
    A_lateral : float
        Surface area of the lateral side, [m^2]
    A_sideA_extra : float
        Additional surface area of sideA beyond that of a flat disk, [m^2]
    A_sideB_extra : float
        Additional surface area of sideB beyond that of a flat disk, [m^2]
    table : bool
        Whether or not a table of heights-volumes has been generated
    heights : ndarray
        Array of heights between 0 and h_max, [m]
    volumes : ndarray
        Array of volumes calculated from the heights, [m^3]
    c_forward : ndarray
        Coefficients for the Chebyshev approximations in calculating V from h,
        [-]
    c_backward : ndarray
        Coefficients for the Chebyshev approximations in calculating h from V,
        [-]

    Notes
    -----
    For torpsherical tank heads, the following `f` and `k` parameters are used
    in standards. The default is ASME F&D.


    +----------------------+-----+-------+
    |                      | f   | k     |
    +======================+=====+=======+
    |  2:1 semi-elliptical | 0.9 | 0.17  |
    +----------------------+-----+-------+
    |  ASME F&D            | 1   | 0.06  |
    +----------------------+-----+-------+
    |  ASME 80/6           | 0.8 | 0.06  |
    +----------------------+-----+-------+
    |  ASME 80/10 F&D      | 0.8 | 0.1   |
    +----------------------+-----+-------+
    |  DIN 28011           | 1   | 0.1   |
    +----------------------+-----+-------+
    |  DIN 28013           | 0.8 | 0.154 |
    +----------------------+-----+-------+

    For the following cases, numerical integrals are used.

    V_horiz_spherical
    V_horiz_torispherical
    SA_partial_horiz_spherical_head
    SA_partial_horiz_ellipsoidal_head
    SA_partial_horiz_guppy_head
    SA_partial_horiz_torispherical_head



    Examples
    --------
    Total volume of a tank:

    >>> TANK(D=1.2, L=4, horizontal=False).V_total
    4.523893421169302

    Volume of a tank at a given height:

    >>> TANK(D=1.2, L=4, horizontal=False).V_from_h(.5)
    0.5654866776461628

    Height of liquid for a given volume:

    >>> TANK(D=1.2, L=4, horizontal=False).h_from_V(.5)
    0.4420970641441539

    Surface area of a tank with a conical head:

    >>> T1 = TANK(V=10, L_over_D=0.7, sideB='conical', sideB_a=0.5)
    >>> T1.A, T1.A_sideA, T1.A_sideB, T1.A_lateral
    (24.94775907657148, 5.118555935958284, 5.497246519930003, 14.331956620683192)

    Solving for tank volumes, first horizontal, then vertical:

    >>> TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.699531057009147
    >>> TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D
    9.999999999999408
    >>> TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.699531057009791

    >>> TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.699531057009147
    >>> TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D
    9.999999999999407
    >>> TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.699531057009791
    """
    table = False
    chebyshev = False
    __full_path__ = __module__  + '.TANK'
    def __repr__(self): # pragma: no cover
        orient = 'Horizontal' if self.horizontal else 'Vertical'
        if self.sideA is None and self.sideB is None:
            sides = 'no heads'
        elif self.sideA == self.sideB:
            if self.sideA_a == self.sideB_a:
                sides = self.sideA + (' heads, a=%f m' %(self.sideA_a))
            else:
                sides = self.sideA + ' heads, sideA a=%f m, sideB a=%f m' % (self.sideA_a, self.sideB_a)
        else:
            if self.sideA:
                A = '%s head on sideA with a=%f m' % (self.sideA, self.sideA_a)
            else:
                A = 'no head on sideA'
            if self.sideB:
                B = ' and %s head on sideB with a=%f m' % (self.sideB, self.sideB_a)
            else:
                B = ' and no head on sideB'
            sides = A + B

        return '<%s tank, V=%f m^3, D=%f m, L=%f m, %s.>' %(orient, self.V_total, self.D, self.L, sides)


    def __init__(self, D=None, L=None, horizontal=True,
                 sideA=None, sideB=None, sideA_a=None, sideB_a=None,
                 sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None,
                 sideA_a_ratio=None, sideB_a_ratio=None, L_over_D=None, V=None):
        self.D = D
        self.L = L
        self.L_over_D = L_over_D
        self.V = V
        self.horizontal = horizontal

        sideA_same, sideB_same = sideA == 'same', sideB == 'same'

        if sideA_same and not sideB_same:
            sideA, sideA_a, sideA_a_ratio, sideA_f, sideA_k = sideB, sideB_a, sideB_a_ratio, sideB_f, sideB_k
        elif sideB_same and not sideA_same:
            sideB, sideB_a, sideB_a_ratio, sideB_f, sideB_k = sideA, sideA_a, sideA_a_ratio, sideA_f, sideA_k
        elif sideA_same and sideB_same:
            raise ValueError("Cannot specify both sides as same")

        self.sideA = sideA
        if sideA is None and sideA_a is None:
            sideA_a = 0.0

        self.sideA_a = sideA_a
        if sideA_a is None and sideA_a_ratio is None and (sideA is not None and sideA != 'torispherical'):
            sideA_a_ratio = 0.25
        self.sideA_a_ratio = sideA_a_ratio

        if sideA_a is None and sideA == 'torispherical':
            if sideA_f is None:
                sideA_f = 1.0
            if sideA_k is None:
                sideA_k = 0.06

        self.sideA_f = sideA_f
        self.sideA_k = sideA_k

        self.sideB = sideB
        if sideB is None and sideB_a is None:
            sideB_a = 0.0
        self.sideB_a = sideB_a

        if sideB_a is None and sideB_a_ratio is None and (sideB is not None and sideB != 'torispherical'):
            sideB_a_ratio = 0.25
        self.sideB_a_ratio = sideB_a_ratio

        if sideB_a is None and sideB == 'torispherical':
            if sideB_f is None:
                sideB_f = 1.0
            if sideB_k is None:
                sideB_k = 0.06

        self.sideB_f = sideB_f
        self.sideB_k = sideB_k

        if self.horizontal:
            self.vertical = False
            self.orientation = 'horizontal'
            self.angle = 0
        else:
            self.vertical = True
            self.orientation = 'vertical'
            self.angle = 90

        # If V is specified and either L or D are known, solve for L, D, L_over_D
        if self.V:
            self._solve_tank_for_V()
        self.set_misc()

    def set_misc(self):
        """Set more parameters, after the tank is better defined than in the
        __init__ function.

        Notes
        -----
        Two of D, L, and L_over_D must be known when this function runs.
        The other one is set from the other two first thing in this function.
        a_ratio parameters are used to calculate a values for the heads here,
        if applicable.
        Radius is calculated here.
        Maximum tank height is calculated here.
        V_total is calculated here.
        """
        if self.D is not None and self.L is not None:
            # If L and D are known, get L_over_D
            self.L_over_D = self.L/self.D
        elif self.D is not None and self.L_over_D is not None:
            # Otherwise, if L_over_D and D are provided, get L
            self.L = self.D*self.L_over_D
        elif self.L is not None and self.L_over_D is not None:
            # Otherwise, if L_over_D and L are provided, get D
            self.D = self.L/self.L_over_D

        D = self.D
        # Calculate diameter
        self.R = self.D/2.

        # If a_ratio is provided for either heads, use it.
        if self.sideA is not None and D is not None:
            if self.sideA_a is None and self.sideA in ('conical', 'ellipsoidal', 'guppy', 'spherical'):
                self.sideA_a = D*self.sideA_a_ratio
        if self.sideB is not None and D is not None:
            if self.sideB_a is None and self.sideB in ('conical', 'ellipsoidal', 'guppy', 'spherical'):
                self.sideB_a = D*self.sideB_a_ratio

        # Calculate a for torispherical heads
        if self.sideA == 'torispherical' and self.sideA_f is not None and self.sideA_k is not None:
            self.sideA_a = a_torispherical(D, self.sideA_f, self.sideA_k)
        if self.sideB == 'torispherical' and self.sideB_f is not None and self.sideB_k is not None:
            self.sideB_a = a_torispherical(D, self.sideB_f, self.sideB_k)

        # Ensure the correct a_ratios are set, whether there is a default being used or not
        if self.sideA_a_ratio is None and self.sideA_a is not None:
            self.sideA_a_ratio = self.sideA_a/D
        elif self.sideA_a_ratio is not None and self.sideA_a is not None and self.sideA_a != D*self.sideA_a_ratio:
            self.sideA_a_ratio = self.sideA_a/D

        if self.sideB_a_ratio is None and self.sideB_a is not None:
            self.sideB_a_ratio = self.sideB_a/D
        elif self.sideB_a_ratio is not None and self.sideB_a is not None and self.sideB_a != D*self.sideB_a_ratio:
            self.sideB_a_ratio = self.sideB_a/D


        # Calculate maximum tank height, h_max
        if self.horizontal:
            self.h_max = D
        else:
            self.h_max = self.L
            if self.sideA_a:
                self.h_max += self.sideA_a
            if self.sideB_a:
                self.h_max += self.sideB_a

        # Set maximum height
#        self.V_total = self.V_from_h(self.h_max)

        self.V_total, self.V_sideA, self.V_sideB, self.V_lateral = V_tank(
        D=D, L=self.L, sideA=self.sideA, sideB=self.sideB, sideA_a=self.sideA_a,
        sideB_a=self.sideB_a, sideA_f=self.sideA_f, sideA_k=self.sideA_k,
        sideB_f=self.sideB_f, sideB_k=self.sideB_k, horizontal=self.horizontal)


        # Set surface areas
        self.A, self.A_sideA, self.A_sideB, self.A_lateral = SA_tank(
        D=D, L=self.L, sideA=self.sideA, sideB=self.sideB, sideA_a=self.sideA_a,
        sideB_a=self.sideB_a, sideA_f=self.sideA_f, sideA_k=self.sideA_k,
        sideB_f=self.sideB_f, sideB_k=self.sideB_k)

        A_circular_plate = 0.25*pi*D*D
        self.A_sideA_extra = self.A_sideA - A_circular_plate
        self.A_sideB_extra = self.A_sideB - A_circular_plate

    @staticmethod
    def from_two_specs(spec0, spec1, spec0_name='V', spec1_name='A_cross',
                       h=None, horizontal=True,
                       sideA=None, sideB=None, sideA_a=None, sideB_a=None,
                       sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None,
                       sideA_a_ratio=None, sideB_a_ratio=None):
        r'''Method to create a new tank instance according to two
        specifications which are not direct geometry parameters.

        The allowable options are 'V', 'SA', 'V_partial', 'SA_partial',
        and 'A_cross', the later three of which require `h` to be specified.


        Parameters
        ----------
        spec0 : float
            Goal for `spec0_name`, [-]
        spec1 : float
            Goal for `spec1_name`, [-]
        spec0_name : str
            One of  'V', 'SA', 'V_partial', 'SA_partial', and 'A_cross' [-]
        spec1_name : str
            One of  'V', 'SA', 'V_partial', 'SA_partial', and 'A_cross' [-]
        h : float
            Height at which to calculate the specs, [m]
        horizontal : bool, optional
            Whether or not the tank is a horizontal or vertical tank
        sideA : string, optional
            The left (or bottom for vertical) head of the tank's type; one of
            [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical',
            'same'].
        sideB : string, optional
            The right (or top for vertical) head of the tank's type; one of
            [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical',
            'same'].
        sideA_a : float, optional
            The distance the head as specified by sideA extends down or to the left
            from the main cylindrical section, [m]
        sideB_a : float, optional
            The distance the head as specified by sideB extends up or to the right
            from the main cylindrical section, [m]
        sideA_f : float, optional
            Dimensionless dish-radius parameter for side A; also commonly given as
            the product of `f` and `D` (`fD`), which is called dish radius and
            has units of length, [-]
        sideA_k : float, optional
            Dimensionless knuckle-radius parameter for side A; also commonly given
            as the product of `k` and `D` (`kD`), which is called the knuckle
            radius and has units of length, [-]
        sideB_f : float, optional
            Dimensionless dish-radius parameter for side B; also commonly given as
            the product of `f` and `D` (`fD`), which is called dish radius and
            has units of length, [-]
        sideB_k : float, optional
            Dimensionless knuckle-radius parameter for side B; also commonly given
            as the product of `k` and `D` (`kD`), which is called the knuckle
            radius and has units of length, [-]


        Returns
        -------
        TANK : TANK
            Tank object at solved specifications, [-]

        Notes
        -----
        Limited testing has been done on this method. The bounds are D between
        0.1 mm and 10 km, with L_D ratios of 1e-4 to 1e4.
        '''

        args = (spec0, spec1, spec0_name, spec1_name,
                h, horizontal, sideA, sideB, sideA_a, sideB_a,
                sideA_f, sideA_k, sideB_f, sideB_k,
                sideA_a_ratio, sideB_a_ratio)

        new_f, translate_into, translate_outof = translate_bound_func(tank_from_two_specs_err,
                                                                      bounds=[(1e-4, 1e4), (1e-4, 1e4)])
        # Diameter and length/diameter as iteration variables
        guess = translate_into([1.0, 3.0])
        from scipy.optimize import fsolve

        ans = fsolve(new_f, guess, args=args, xtol=1e-10, factor=.1)
        val0, val1 = translate_outof(ans)

        return TANK(D=float(val0), L_over_D=float(val1), horizontal=horizontal,
                    sideA=sideA, sideB=sideB, sideA_a=sideA_a, sideB_a=sideB_a,
                    sideA_f=sideA_f, sideA_k=sideA_k, sideB_f=sideB_f, sideB_k=sideB_k,
                    sideA_a_ratio=sideA_a_ratio, sideB_a_ratio=sideB_a_ratio,)


    def add_thickness(self, thickness, sideA_thickness=None,
                      sideB_thickness=None):
        r'''Method to create a new tank instance with the same parameters as
        itself, except with an added thickness to it. This is useful to obtain
        ex. the inside of a tank and the outside; their different in volumes is
        the volume of the shell, and could be used to determine weight.

        Parameters
        ----------
        thickness : float
            Thickness to add to the tank diameter, [m]
        sideA_thickness : float, optional
            The thickness to add to the sideA head; if not specified,
            it will be `thickness`, [m]
        sideB_thickness : float, optional
            The thickness to add to the sideB head; if not specified,
            it will be `thickness`, [m]

        Returns
        -------
        TANK : TANK
            Tank object, [-]

        Notes
        -----
        Be careful not to specify a negative thickness larger than the heads'
        lengths, or the head will become concave! The same applies to adding
        a thickness to convex heads - they can become convex.

        '''
        kwargs = dict(D=self.D, L=self.L, horizontal=self.horizontal,
                 sideA=self.sideA, sideB=self.sideB, sideA_a=self.sideA_a,
                 sideB_a=self.sideB_a, sideA_f=self.sideA_f,
                 sideA_k=self.sideA_k, sideB_f=self.sideB_f, sideB_k=self.sideB_k)
        if sideA_thickness is None:
            sideA_thickness = thickness
        if sideB_thickness is None:
            sideB_thickness = thickness

        # Do not transfer a_ratios or volume or L_over_D
        kwargs['D'] += 2.0*thickness
        kwargs['L'] += sideA_thickness + sideB_thickness

        # For torispherical vessels, the heads are defined from the `f` and `k`
        # parameters which are already functions of diameter, and so will be
        # fixed automatically; if the `a` parameters are specified they would
        # not be corrected
        if self.sideA != 'torispherical':
            kwargs['sideA_a'] += sideA_thickness
        else:
            del kwargs['sideA_a']

        if self.sideB != 'torispherical':
            kwargs['sideB_a'] += sideB_thickness
        else:
            del kwargs['sideB_a']
        return TANK(**kwargs)

    def SA_from_h(self, h, method='full'):
        r'''Method to calculate the volume of liquid in a fully defined tank
        given a specified height `h`. `h` must be under the maximum height.

        Parameters
        ----------
        h : float
            Height specified, [m]
        method : str, optional
            'full' (calculated rigorously) ; nothing else is implemented

        Returns
        -------
        SA : float
            Surface area of liquid in the tank up to the specified height, [m^2]

        Notes
        -----
        '''
        if method == 'full':
            return SA_from_h(h, self.D, self.L, self.horizontal, self.sideA,
                            self.sideB, self.sideA_a, self.sideB_a,
                            self.sideA_f, self.sideA_k, self.sideB_f,
                            self.sideB_k)
        else:
            raise ValueError("Allowable methods are 'full' .")

    def V_from_h(self, h, method='full'):
        r'''Method to calculate the volume of liquid in a fully defined tank
        given a specified height `h`. `h` must be under the maximum height.
        If the method is 'chebyshev', and the coefficients have not yet been
        calculated, they are created by calling `set_chebyshev_approximators`.

        Parameters
        ----------
        h : float
            Height specified, [m]
        method : str
            One of 'full' (calculated rigorously) or 'chebyshev'

        Returns
        -------
        V : float
            Volume of liquid in the tank up to the specified height, [m^3]

        Notes
        -----
        '''
        if method == 'full':
            return V_from_h(h, self.D, self.L, self.horizontal, self.sideA,
                            self.sideB, self.sideA_a, self.sideB_a,
                            self.sideA_f, self.sideA_k, self.sideB_f,
                            self.sideB_k)
        elif method == 'chebyshev':
            if not self.chebyshev:
                self.set_chebyshev_approximators()
            return self.V_from_h_cheb(h)
        else:
            raise ValueError("Allowable methods are 'full' or 'chebyshev'.")

    def h_from_V(self, V, method='spline'):
        r'''Method to calculate the height of liquid in a fully defined tank
        given a specified volume of liquid in it `V`. `V` must be under the
        maximum volume. If the method is 'spline', and the interpolation table
        is not yet defined, creates it by calling the method set_table. If the
        method is 'chebyshev', and the coefficients have not yet been
        calculated, they are created by calling `set_chebyshev_approximators`.

        Parameters
        ----------
        V : float
            Volume of liquid in the tank up to the desired height, [m^3]
        method : str
            One of 'spline', 'chebyshev', or 'brenth'

        Returns
        -------
        h : float
            Height of liquid at which the volume is as desired, [m]
        '''
        if method == 'spline':
            try:
                if not self.table:
                    self.set_table()
                return float(self.interp_h_from_V(V))
            except:
                # Missing scipy
                return self.h_from_V(V, 'brenth')
        elif method == 'chebyshev':
            if not self.chebyshev:
                self.set_chebyshev_approximators()
            return self.h_from_V_cheb(V)
        elif method == 'brenth':
            to_solve = lambda h : self.V_from_h(h, method='full') - V
            return brenth(to_solve, self.h_max, 0)
        else:
            raise ValueError("Allowable methods are 'full' or 'chebyshev', "
                            "or 'brenth'.")

    def A_cross_sectional(self, h, method='full'):
        r'''Method to calculate the cross-sectional liquid surface area
        from which gas can evolve in a fully defined tank
        given a specified height `h`. `h` must be under the maximum height.
        This is calculated by numeric differentiation for most cases.

        Parameters
        ----------
        h : float
            Height specified, [m]
        method : str, optional
            'full' (calculated rigorously) or 'chebyshev', [-]

        Returns
        -------
        A_cross : float
            Surface area of liquid in the tank up to the specified height, [m^2]

        Notes
        -----
        '''
        # The derivative will give bad values in some cases, when right up against boundaries
        # Analytical formulations can be done, but will be lots of code

        return derivative(lambda h: self.V_from_h(h), h, dx=1e-7*h, order=3, n=1)

    def set_table(self, n=100, dx=None):
        r'''Method to set an interpolation table of liquids levels versus
        volumes in the tank, for a fully defined tank. Normally run by the
        h_from_V method, this may be run prior to its use with a custom
        specification. Either the number of points on the table, or the
        vertical distance between steps may be specified.

        Parameters
        ----------
        n : float, optional
            Number of points in the interpolation table, [-]
        dx : float, optional
            Vertical distance between steps in the interpolation table, [m]
        '''
        if dx:
            self.heights = linspace(0.0, self.h_max, int(self.h_max/dx)+1)
        else:
            self.heights = linspace(0.0, self.h_max, n)
        self.volumes = [self.V_from_h(h) for h in self.heights]
        from scipy.interpolate import UnivariateSpline
        self.interp_h_from_V = UnivariateSpline(self.volumes, self.heights, ext=3, s=0.0)
        self.table = True

    def set_chebyshev_approximators(self, deg_forward=50, deg_backwards=200):
        r'''Method to derive and set coefficients for chebyshev polynomial
        function approximation of the height-volume and volume-height
        relationship.

        A single set of chebyshev coefficients is used for the entire height-
        volume and volume-height relationships respectively.

        The forward relationship, `V_from_h`, requires
        far fewer coefficients in its fit than the reverse to obtain the same
        relative accuracy.

        Optionally, deg_forward or deg_backwards can be set to None to try to
        automatically fit the series to machine precision.

        Parameters
        ----------
        deg_forward : int, optional
            The degree of the chebyshev polynomial to be created for the
            `V_from_h` curve, [-]
        deg_backwards : int, optional
            The degree of the chebyshev polynomial to be created for the
            `h_from_V` curve, [-]
        '''
        from fluids.optional.pychebfun import Chebfun
        import numpy as np
        to_fit = lambda h: self.V_from_h(h, 'full')

        # These high-degree polynomials cannot safety be evaluated using Horner's methods
        # chebval is 2.5x as slow but 100% required; around 40 coefficients results are junk
        self.c_forward = Chebfun.from_function(np.vectorize(to_fit),
                                               [0.0, self.h_max], N=deg_forward).coefficients().tolist()

        self.V_from_h_cheb = lambda x : chebval((2.0*x-self.h_max)/(self.h_max), self.c_forward)

        to_fit = lambda h: self.h_from_V(h, 'brenth')
        self.c_backward = Chebfun.from_function(np.vectorize(to_fit), [0.0, self.V_total], N=deg_backwards).coefficients().tolist()
        self.h_from_V_cheb = lambda x : chebval((2.0*x-self.V_total)/(self.V_total), self.c_backward)
        self.chebyshev = True

    def _V_solver_error(self, Vtarget, D, L, horizontal, sideA, sideB, sideA_a,
                       sideB_a, sideA_f, sideA_k, sideB_f, sideB_k,
                       sideA_a_ratio, sideB_a_ratio):
        """Function which uses only the variables given, and the TANK class
        itself, to determine how far from the desired volume, Vtarget, the
        volume produced by the specified parameters in a new TANK instance is.

        Should only be used by _solve_tank_for_V method.
        """
        a = TANK(D=float(D), L=float(L), horizontal=horizontal, sideA=sideA, sideB=sideB,
                 sideA_a=sideA_a, sideB_a=sideB_a, sideA_f=sideA_f,
                 sideA_k=sideA_k, sideB_f=sideB_f, sideB_k=sideB_k,
                 sideA_a_ratio=sideA_a_ratio, sideB_a_ratio=sideB_a_ratio)
        error = (Vtarget - a.V_total)
        return error


    def _solve_tank_for_V(self):
        """Method which is called to solve for tank geometry when a certain
        volume is specified. Will be called by the __init__ method if V is set.

        Notes
        -----
        Raises an error if L and either of sideA_a or sideB_a are specified;
        these can only be set once D is known.
        Raises an error if more than one of D, L, or L_over_D are specified.
        Raises an error if the head ratios are not provided.

        Calculates initial guesses assuming no heads are present, and then uses
        fsolve to determine the correct dimensions for the tank.

        Tested, but bugs and limitations are expected here.
        """
        if self.L and (self.sideA_a or self.sideB_a):
            raise ValueError('Cannot specify head sizes when solving for V')
        if (self.D and self.L) or (self.D and self.L_over_D) or (self.L and self.L_over_D):
            raise ValueError('Only one of D, L, or L_over_D can be specified\
            when solving for V')
        if ((self.sideA is not None and (self.sideA_a_ratio is None and self.sideA_a is None) and self.sideA != 'torispherical')
             or (self.sideB is not None and (self.sideB_a_ratio is None and self.sideB_a is None) and self.sideB != 'torispherical')):
            raise ValueError('When heads are specified, head parameter ratios are required')

        if self.D:
            # Iterate until L is appropriate
            solve_L = lambda L: self._V_solver_error(self.V, self.D, L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            Lguess = self.V/(pi/4*self.D**2)
            self.L = float(secant(solve_L, Lguess, xtol=1e-13))
        elif self.L:
            # Iterate until D is appropriate
            solve_D = lambda D: self._V_solver_error(self.V, D, self.L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            Dguess = sqrt(4*self.V/pi/self.L)
            self.D = float(secant(solve_D, Dguess, xtol=1e-13))
        else:
            # Use L_over_D until L and D are appropriate
            Lguess = (4*self.V*self.L_over_D**2/pi)**(1/3.)
            solve_L_D = lambda L: self._V_solver_error(self.V, L/self.L_over_D, L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            self.L = float(secant(solve_L_D, Lguess, xtol=1e-13))
            self.D = self.L/self.L_over_D




class HelicalCoil(object):
    r'''Class representing a helical coiled tube, as are found in many heated
    tanks and some small nuclear reactors. All parameters are also attributes.

    One set of the following parameters is required; inner tube diameter is
    optional.

        * Tube outer diameter, coil outer diameter, pitch, number of coil turns
        * Tube outer diameter, coil outer diameter, pitch, height
        * Tube outer diameter, coil outer diameter, number of coil turns, height

    Parameters
    ----------
    Dt : float
        Outer diameter of the tube wound to make up the helical spiral, [m]
    Do : float
        Diameter of the spiral as measured from the center of the coil on one
        side to the center of the coil on the other side, [m]
    Do_total : float, optional
        Diameter of the spiral as measured from one edge of the tube to the
        other edge; equal to Do + Dt; either `Do` or `Do_total` may be
        specified and the other will be calculated [m]
    pitch : float, optional
        Height change from one coil to the next as measured from the middles
        of the tube, [m]
    H : float, optional
        Height of the spiral, as measured from the middle of the bottom of the
        tube to the middle of the top of the tube, [m]
    H_total : float, optional
        Height of the spiral as measured from one edge of the tube to the other
        edge; equal to `H_total` + `Dt`; either may be specified and the other
        will be calculated [m]
    N : float, optional
        Number of coil turns; may be specified along with `pitch` instead of
        specifying `H` or `H_total`, [-]
    Di : float, optional
        Inner diameter of the tube; if specified, inside and annulus properties
        will be calculated, [m]

    Attributes
    ----------
    tube_circumference : float
        Circumference of the tube as measured though its center, not inner or
        outer edges;  :math:`C = \pi D_o`, [m]
    tube_length : float
        Length of tube used to make the helical coil;
        :math:`L = \sqrt{(\pi D_o\cdot N)^2 + H^2}`, [m]
    surface_area : float
        Surface area of the outer surface of the helical coil;
        :math:`A_t = \pi D_t L`, [m^2]
    inner_surface_area : float
        Surface area of the inner surface of the helical coil; calculated if
        `Di` is supplied; :math:`A_{inside} = \pi D_i L`, [m^2]
    inlet_area : float
        Area of the inlet to the helical coil; calculated if
        `Di` is supplied; :math:`A_{inlet} = \frac{\pi}{4} D_i^2`, [m^2]
    inner_volume : float
        Volume of the tube as would be filled by a fluid, useful for weight
        calculations; calculated if `Di` is supplied;
        :math:`V_{inside} = A_i L`, [m^3]
    annulus_area : float
        Area of the annulus (wall of the pipe); calculated if `Di` is supplied;
        :math:`A_a = \frac{\pi}{4} (D_t^2 - D_i^2)`, [m^2]
    annulus_volume : float
        Volume of the annulus (wall of the pipe); calculated if `Di`
        is supplied, useful for weight calculations; :math:`V_a = A_a L`, [m^3]
    total_volume : float
        Total volume occupied by the pipe and the fluid inside it;
        :math:`V = D_t L`, [m^3]
    helix_angle : float
        Angle between the pitch and coil diameter; used in some calculations;
        :math:`\alpha = \arctan \left(\frac{p_t}{\pi D_o}\right)`, [radians]
    curvature : float
        Coil curvature, useful in some calculations;
        :math:`\delta = \frac{D_t}{D_o[1 + 4\pi^2 \tan^2(\alpha)]}`, [-]

    Notes
    -----
    `Do` must be larger than `Dt`.

    Examples
    --------
    >>> C1 = HelicalCoil(Do=30, H=20, pitch=5, Dt=2)
    >>> C1.N, C1.tube_length, C1.surface_area
    (4.0, 377.5212621504738, 2372.0360474917497)

    Same coil, with the inputs one would physically measure from the coil,
    and a specified inlet diameter:

    >>> C1 = HelicalCoil(Do_total=32, H_total=22, pitch=5, Dt=2, Di=1.8)
    >>> C1.N, C1.tube_length, C1.surface_area
    (4.0, 377.5212621504738, 2372.0360474917497)
    >>> C1.inner_surface_area, C1.inlet_area, C1.inner_volume, C1.total_volume, C1.annulus_volume
    (2134.832442742575, 2.5446900494077327, 960.6745992341587, 1186.0180237458749, 225.3434245117162)

    References
    ----------
    .. [1] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    def __repr__(self): # pragma : no cover
        s = '<Helical coil, total height=%s m, total outer diameter=%s m, tube \
outer diameter=%s m, number of turns=%s, pitch=%s m' % (self.H_total, self.Do_total, self.Dt, self.N, self.pitch)
        if self.Di:
             s += ', inside diameter %s m' %(self.Di)
        s += '>'
        return s

    def __init__(self, Dt, Do=None, pitch=None, H=None, N=None, H_total=None,
                 Do_total=None, Di=None):
        # H goes from center of tube in bottom of coil to center of tube in top of coil
        # Do goes from the center of the spiral to the center of the outer tube

        if H_total is not None:
            H = H_total - Dt
        if Do_total is not None:
            Do = Do_total - Dt
        self.Do = Do
        self.Dt = Dt
        self.Do_total = self.Do+self.Dt
        if N is not None and pitch is not None:
            self.N = N
            self.pitch = pitch
            self.H = N*pitch
        elif N is not None and H is not None:
            self.N = N
            self.H = H
            self.pitch = self.H/N
            if self.pitch < self.Dt:
                raise ValueError('Pitch is too small - tubes are colliding')#; maximum number of spirals is %f.'%(self.H/self.Dt))
        elif H is not None and pitch is not None:
            self.pitch = pitch
            self.H = H
            self.N = self.H/self.pitch
            if self.pitch < self.Dt:
                raise ValueError('Pitch is too small - tubes are colliding; pitch must be larger than tube diameter.')
        if self.H is not None: # numba
            self.H_total = self.Dt + self.H

        if self.Dt > self.Do:
            raise ValueError('Tube diameter is larger than helix outer diameter - not feasible.')

        self.tube_circumference = pi*self.Do
        self.tube_length = sqrt((self.tube_circumference*self.N)**2 + self.H**2)
        self.surface_area = self.tube_length*pi*self.Dt
        #print(pi*self.tube_length*self.Dt) == surface_area
        self.helix_angle = atan(self.pitch/(pi*self.Do))
        self.curvature = self.Dt/self.Do/(1. + 4*pi**2*tan(self.helix_angle)**2)
        #print(self.N*pi*self.Do/cos(self.helix_angle)) # Confirms the length with another formula
        self.total_inlet_area = pi/4.*self.Dt**2
        self.total_volume = self.total_inlet_area*self.tube_length

        if Di is not None:
            self.Di = Di
            self.inner_surface_area = self.tube_length*pi*self.Di
            self.inlet_area = pi/4.*self.Di**2
            self.inner_volume = self.inlet_area*self.tube_length
            self.annulus_area = self.total_inlet_area - self.inlet_area
            self.annulus_volume = self.total_volume - self.inner_volume


def plate_enlargement_factor(amplitude, wavelength):
    r'''Calculates the enhancement factor of the sinusoidal waves of the
    plate heat exchanger. This is the multiplier for the flat plate area
    to obtain the actual area available for heat transfer. Obtained from
    the following integral:

    .. math::
        \phi = \frac{\text{Effective area}}{\text{Projected area}}
        = \frac{\int_0^\lambda\sqrt{1 + \left(\frac{\gamma\pi}{2}\right)^2
        \cos^2\left(\frac{2\pi}{\lambda}x\right)}dx}{\lambda}

    .. math::
        \gamma = \frac{4a}{\lambda}

    The solution to the integral is:

    .. math::
        \phi = \frac{2E\left(\frac{-4a^2\pi^2}{\lambda^2}\right)}{\pi}

    where E is the complete elliptic integral of the second kind,
    calculated with SciPy.

    Parameters
    ----------
    amplitude : float
        Half the height of the wave of the ridges, [m]
    wavelength : float
        Distance between the bottoms of two of the ridges (sometimes called
        pitch), [m]

    Returns
    -------
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]

    Notes
    -----
    This is the exact analytical integral, obtained via Mathematica, Maple,
    and quite a bit of trial and error. It is confirmed via numerical
    integration. The expression normally given is an
    approximation as follows:

    .. math::
        \phi = \frac{1}{6}\left(1+\sqrt{1+A^2} + 4\sqrt{1+A^2/2}\right)

        A = \frac{2\pi a}{\lambda}

    Most plate heat exchangers approximate a sinusoidal geometry only.

    Examples
    --------
    >>> plate_enlargement_factor(amplitude=5E-4, wavelength=3.7E-3)
    1.1611862034509677
    '''
    b = 2.*amplitude
    return 2.*float(ellipe(-b*b*pi*pi/(wavelength*wavelength)))/pi

class PlateExchanger(object):
    r'''Class representing a plate heat exchanger with sinusoidal ridges.
    All parameters are also attributes.

    Parameters
    ----------
    amplitude : float
        Half the height of the wave of the ridges, [m]
    wavelength : float
        Distance between the bottoms of two of the ridges (sometimes called
        pitch), [m]
    chevron_angle : float, optional
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90, [degrees]
    chevron_angles : tuple(2), optional
        Many plate exchangers use two alternating patterns; for those cases
        provide tuple of the two angles for that situation and the argument
        `chevron_angle` is ignored, [degrees]
    width : float, optional
        Width of the plates in the heat exchanger, between the gaskets, [m]
    length : float, optional
        Length of the heat exchanger as measured from one port to the other,
        excluding the diameter of the ports themselves (little useful heat
        transfer happens there), [m]
    thickness : float, optional
        Thickness of the metal making up the plates, [m]
    d_port : float, optional
        The diameter of the ports in the plates, [m]
    plates : int, optional
        The number of plates in the heat exchanger, including the two not
        used for heat transfer at the beginning and end [-]

    Attributes
    ----------
    chevron_angles : tuple(2)
        The two specified angles (repeated value if only one specified), [degrees]
    chevron_angle : float
        The averaged angle of the chevrons, [degrees]
    inclination_angle : float
        90 - `chevron_angle`, used in many publications instead of `chevron_angle`,
        [degrees]
    plate_corrugation_aspect_ratio : float
        The aspect ratio of the corrugations
        :math:`\gamma = \frac{4a}{\lambda}`, [-]
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]
    D_eq : float
        Equivalent diameter of the channels, :math:`D_{eq} = 4a` [m]
    D_hydraulic : float
        Hydraulic diameter of the channels, :math:`D_{hyd} = \frac{4a}{\phi}` [m]
    length_port : float
        Port center to port center along the direction of flow, [m]
    A_plate_surface : float
        The surface area of one plate in the heat exchanger, including the
        extra due to corrugations (excluding the bit between the ports),
        :math:`A_p = L\cdot W\cdot \phi` [m^2]
    A_heat_transfer : float
        The total surface area available for heat transfer in the exchanger,
        the multiple of `A_plate_surface` by the number of plates after
        removing the two on the edges, [m^2]
    A_channel_flow : float
        The area for the fluid to flow in one channel, :math:`W\cdot b` [m^2]
    channels : int
        The number of plates minus one, [-]
    channels_per_fluid : int
        Half the number of total channels, [-]

    Notes
    -----
    Only wavelength and amplitude are required as inputs to this function.

    Examples
    --------
    >>> PlateExchanger(amplitude=5E-4, wavelength=3.7E-3, length=1.2, width=.3,
    ... d_port=.05, plates=51)
    <Plate heat exchanger, amplitude=0.0005 m, wavelength=0.0037 m, chevron_angles=45/45 degrees, area enhancement factor=1.16119, width=0.3 m, length=1.2 m, port diameter=0.05 m, heat transfer area=20.4833 m^2, 51 plates>

    References
    ----------
    .. [1] Amalfi, Raffaele L., Farzad Vakili-Farahani, and John R. Thome.
       "Flow Boiling and Frictional Pressure Gradients in Plate Heat Exchangers.
       Part 1: Review and Experimental Database." International Journal of
       Refrigeration 61 (January 2016): 166-84. doi:10.1016/j.ijrefrig.2015.07.010.
    '''
    def __repr__(self):  # pragma : no cover
        s = '<Plate heat exchanger, amplitude=%g m, wavelength=%g m, \
chevron_angles=%s degrees, area enhancement factor=%g' %(self.a, self.wavelength, '/'.join([str(i) for i in self.chevron_angles]), self.plate_enlargement_factor)
        if self.width and self.length:
            s += ', width=%g m, length=%g m' %(self.width, self.length)
        if self.d_port:
            s += ', port diameter=%g m' %(self.d_port)
        if self.plates:
            s += ', heat transfer area=%g m^2, %g plates>' %(self.A_heat_transfer, self.plates)
        else:
            s += '>'
        return s

    @property
    def plate_exchanger_identifier(self):
        """Method to create an identifying string in format 'L' + wavelength +
        'A' + amplitude + 'B' + chevron angle-chevron angle.

        Wavelength and amplitude are specified in units of mm and rounded to two
        decimal places.
        """
        wave_rounded = round(self.wavelength*1000, 2)
        amplitude_rounded = round(self.amplitude*1000, 2)
        a1 = self.chevron_angles[0]
        a2 = self.chevron_angles[1]
        s = ('L{0}A{1}B{2}-{3}'.format(wave_rounded, amplitude_rounded, a1, a2))
        return s


    def __init__(self, amplitude, wavelength, chevron_angle=45,
                 chevron_angles=None, width=None, length=None, thickness=None,
                 d_port=None, plates=None):
        self.amplitude = self.a = amplitude # half a sine wave's height
        self.b = 2*self.amplitude # Used in some models. From a flat plate, a press goes down this far into the plate. Also called the hot and cold gap
        self.wavelength = self.pitch = wavelength # self.lambda
        if chevron_angles is not None:
            self.chevron_angles = chevron_angles
            self.chevron_angle = self.beta = 0.5*(chevron_angles[0] + chevron_angles[1])
        else:
            self.chevron_angle = self.beta = chevron_angle # between 0 and 90
            self.chevron_angles = (chevron_angle, chevron_angle)

        self.inclination_angle = 90 - self.chevron_angle # Used in some definitions instead

        self.plate_corrugation_aspect_ratio = self.gamma = 4*self.a/self.wavelength
        self.plate_enlargement_factor = plate_enlargement_factor(self.amplitude, self.wavelength)

        self.D_eq = 4*self.amplitude # Equivalent diameter for inter-plate spacing
        self.D_hydraulic = 4*self.amplitude/self.plate_enlargement_factor # Get better results when correlations use this

        if width is not None:
            self.width = width
        if length is not None:
            self.length = length
        if thickness is not None:
            self.thickness = thickness
        if d_port is not None:
            self.d_port = d_port
        if plates is not None:
            self.plates = plates

        if d_port is not None and length is not None:
            self.length_port = length + d_port # port center to port center along the direction of flow
            # There is another larger length as well, including both port diameters
        if width is not None and length is not None:
            self.A_plate_surface = length*width*self.plate_enlargement_factor # use this in Q = UAdT
            if plates is not None:
                self.A_heat_transfer = (plates-2)*self.A_plate_surface # the two outermost sides aren't used
        if width is not None:
            self.A_channel_flow = self.width*self.b # Use this to get G, kg/s/m^2
        if plates is not None:
            self.channels = self.plates - 1
            self.channels_per_fluid = 0.5*self.channels


class RectangularFinExchanger(object):
    r'''Class representing a plate-fin heat exchanger with straight rectangular
    fins. All parameters are also attributes.

    Parameters
    ----------
    fin_height : float
        The total distance between the two metal plates sandwiching the fins
        and holding them together (abbreviated `h`), [m]
    fin_thickness : float
        The thickness of the material the fins were formed from
        (abbreviated `t`), [m]
    fin_spacing : float
        The unit cell spacing from one fin to the next; the space between the
        sides of two fins plus one thickness (abbreviated `s`), [m]
    length : float, optional
        The total length of the flow passage of the plate-fin exchanger
        (abbreviated `L`), [m]
    width : float, optional
        The total width of the space the fins are in; this is also
        :math:`N_{fins}\times s` (abbreviated `W`), [m]
    layers : int, optional
        The number of layers in the plate-fin exchanger; note these HX almost
        always single-pass only, [-]
    plate_thickness : float, optional
        The thickness of the metal separator between layers, [m]
    flow : str, optional
        One of 'counterflow', 'crossflow', or 'parallelflow'

    Attributes
    ----------
    channel_height : float
        The height of the channel the fluid flows in
        :math:`\text{channel height } = \text{fin height} - \text{fin thickness}`, [m]
    channel_width : float
        The width of the channel the fluid flows in
        :math:`\text{channel width } = \text{fin spacing} - \text{fin thickness}`, [m]
    fin_count : int
        The number of fins per unit length of the layer,
        :math:`\text{fin count} = \frac{1}{\text{fin spacing}}`, [1/m]
    blockage_ratio : float
        The fraction of the layer which is blocked to flow by the fins,
        :math:`\text{blockage ratio} = \frac{s\cdot h - s\cdot t - t(h-t)}{s\cdot h}`,
        [m]
    A_channel : float
        Flow area of a single channel in a single layer,
        :math:`\text{channel area} = (s-t)(h-t)`, [m]
    P_channel : float
        Wetted perimeter of a single channel in a single layer,
        :math:`\text{channel perimeter} = 2(s-t) + 2(h-t)`, [m]
    Dh : float
        Hydraulic diameter of a single channel in a single layer,
        :math:`D_{hydraulic} = \frac{4 A_{channel}}{P_{channel}}`, [m]
    layer_thickness : float
        The thickness of a single layer - the sum of a fin height and
        a plate thickness, [m]
    layer_fin_count : int
        The number of fins in a layer; rounded to the nearest whole fin, [-]
    A_HX_layer : float
        The surface area including fins for heat transfer in one layer of the
        HX, [m^2]
    A_HX : float
        The total surface area of the heat exchanger with all layers combined,
        [m^2]
    height : float
        The height of all the layers of the heat exchanger combined, plus one
        extra plate thickness, [m]
    volume : float
        The product of the height, width, and length of the HX, [m^3]
    A_specific_HX : float
        The specific surface area of the heat exchanger - square meters per
        meter cubed, [m^3]

    Notes
    -----
    The only required parameters are the fin geometry itself; `fin_height`,
    `fin_thickness`, and `fin_spacing`.

    Examples
    --------
    >>> PFE = RectangularFinExchanger(0.03, 0.001, 0.012)
    >>> PFE.Dh
    0.01595

    References
    ----------
    .. [1] Yang, Yujie, and Yanzhong Li. "General Prediction of the Thermal
       Hydraulic Performance for Plate-Fin Heat Exchanger with Offset Strip
       Fins." International Journal of Heat and Mass Transfer 78 (November 1,
       2014): 860-70. doi:10.1016/j.ijheatmasstransfer.2014.07.060.
    .. [2] Sheik Ismail, L., R. Velraj, and C. Ranganayakulu. "Studies on
       Pumping Power in Terms of Pressure Drop and Heat Transfer
       Characteristics of Compact Plate-Fin Heat Exchangers-A Review."
       Renewable and Sustainable Energy Reviews 14, no. 1 (January 2010):
       478-85. doi:10.1016/j.rser.2009.06.033.
    '''
    def __init__(self, fin_height, fin_thickness, fin_spacing, length=None, width=None, layers=None, plate_thickness=None, flow='crossflow'):
        self.h = self.fin_height = fin_height # including 2x thickness
        self.t = self.fin_thickness = fin_thickness
        self.s = self.fin_spacing = fin_spacing

        self.L = self.length = length
        self.W = self.width = width

        self.layers = layers
        self.flow = flow
        self.plate_thickness = plate_thickness

        self.channel_height = self.fin_height - self.fin_thickness
        self.channel_width = self.fin_spacing - self.fin_thickness
        self.fin_count = 1./self.fin_spacing

        self.blockage_ratio = (self.s*self.h - self.s*self.t - (self.h-self.t)*self.t)/(self.s*self.h)

        self.A_channel = (self.s-self.t)*(self.h-self.t)
        self.P_channel = 2*(self.s-self.t) + 2*(self.h-self.t)
        self.Dh = 4*self.A_channel/self.P_channel
        self.set_overall_geometry()


    def set_overall_geometry(self):
        if self.plate_thickness:
            self.layer_thickness = self.plate_thickness + self.fin_height

        if self.length and self.width:
            self.layer_fin_count = round(self.fin_count*self.width, 0)
            if hasattr(self, 'SA_fin'):
                self.A_HX_layer = self.layer_fin_count*self.SA_fin*self.length
            else:
                self.A_HX_layer = self.P_channel*self.length*self.layer_fin_count

            if self.layers:
                self.A_HX = self.layers*self.A_HX_layer
                if self.plate_thickness:
                    self.height = self.layer_thickness*self.layers + self.plate_thickness
                    self.volume = (self.length*self.width*self.height)
                    self.A_specific_HX = self.A_HX/self.volume


class RectangularOffsetStripFinExchanger(RectangularFinExchanger):
    def __init__(self, fin_length, fin_height, fin_thickness, fin_spacing, length=None, width=None, layers=None, plate_thickness=None, flow='crossflow'):
        self.l = self.fin_length = fin_length
        self.h = self.fin_height = fin_height
        self.t = self.fin_thickness = fin_thickness
        self.s = self.fin_spacing = fin_spacing

        self.blockage_ratio = self.omega = 2*self.t/self.s*(1. - self.t/self.h) + self.t/self.h*(1 - 2*self.t/self.s)
        # Kim blockage ratio beta
        self.blockage_ratio_Kim = self.t/self.h + self.t/self.s - self.t**2/(self.h*self.s)

        # Definitions as in the paper with the most common correlation
        self.alpha = self.s/self.h # "General prediction" uses t/h here
        self.delta = self.t/self.l
        self.gamma = self.t/self.s

        # free flow area
        self.A_channel = (self.h  - self.t)*(self.s - self.t)
        self.A = 2.*(self.l*(self.h-self.t) + self.l*(self.s-self.t) + self.t*(self.h-self.t)) + self.t*(self.s-2*self.t)
        self.Dh = 4.*self.l*self.A_channel/self.A # not the standard definition

        self.P_channel = 2*(self.s-self.t) + 2*(self.h-self.t)

        self.Dh_Kays_London = 4*self.A_channel/(2*(self.h -self.t)+ 2*(self.s -self.t))
        # Does not consider the fronts of backs of the fins, only the 2d shape

        self.Dh_Joshi_Webb = 2*self.l*(self.h - self.t)*(self.s - 2*self.t)/(self.l*(self.h-self.t) + self.l*(self.s - self.t) + self.t*(self.h - self.t))

        self.L = self.length = length
        self.W = self.width = width
        self.layers = layers
        self.flow = flow
        self.plate_thickness = plate_thickness
        self.fin_count = 1./self.fin_spacing
        self.set_overall_geometry()


class HyperbolicCoolingTower(object):
    r'''Class representing the geometry of a hyperbolic cooling tower, as used
    in many industries especially the poewr industry.  All parameters are also
    attributes.

    `H_inlet`, `D_outlet`, and `H_outlet` are always required. Additionally,
    one set of the following parameters is required; `H_support`, `D_support`,
    `n_support`, and `inlet_rounding` are all optional as well.

        * Inlet diameter
        * Inlet diameter and throat diameter
        * Inlet diameter and throat height
        * Inlet diameter, throat diameter, and throat height
        * Base diameter, throat diameter, and throat height

    If the inlet diameter is provided but the throat diameter and/or the throat
    height are missing, two heuristics are used to estimate them (to avoid
    these heuristics simply specify the values):

        * Assume the throat elevation is 2/3 the elevation of the tower.
        * Assume the throat diameter is 63% the diameter of the inlet.

    Parameters
    ----------
    H_inlet : float
        Height of the inlet zone of the cooling tower (also called rain zone),
        [m]
    D_outlet : float
        The inside diameter of the cooling tower outlet (top of the tower; the
        elevation the concrete section ends), [m]
    H_outlet : float
        The height of the cooling tower outlet (top of the tower;the
        elevation the concrete section ends), [m]
    D_inlet : float, optional
        The inside diameter of the cooling tower inlet at the elevation the
        concrete section begins, [m]
    D_base : float, optional
        The diameter of the cooling tower at the very base of the tower (the
        bottom of the inlet zone, at the elevation of the ground), [m]
    D_throat : float, optional
        The diameter of the cooling tower at its minimum section, called its
        throat; where the two hyperbolas meet, [m]
    h_throat : float, optional
        The elevation of the cooling tower's throat (its minimum section; where
        the two hyperbolas meet), [m]
    inlet_rounding : float, optional
        Radius of an optional rounded protrusion from the lip of the cooling
        tower shell base, which curves upwards from the lip (used to reduce
        the dead zone area rather than having a flat lip), [m]
    H_support : float, optional
        The height of each support column, [m]
    D_support : float, optional
        The diameter of each support column, [m]
    n_support : int, optional
        The number of support columns of the cooling tower, [m]

    Attributes
    ----------
    b_lower : float
        The `b` parameter in the hyperbolic equation for the lower section of
        the cooling tower, [m]
    b_upper : float
        The `b` parameter in the hyperbolic equation for the upper section of
        the cooling tower, [m]

    Notes
    -----
    Note there are two hyperbolas in a hyperbolic cooling tower - one under the
    throat and one above it; they are not necessarily the same.

    A hyperbolic cooling tower is not the absolute optimal design, but is is
    close. The optimality is determined by the amount of material required to
    build it while maintaining its rigidity. For thermal design purposes,
    a hyperbolic model covers any minor variation quite well.

    Examples
    --------
    >>> ct = HyperbolicCoolingTower(D_outlet=89.0, H_outlet=200, D_inlet=136.18, H_inlet=14.5)
    >>> ct
    <Hyperbolic cooling tower, inlet diameter=136.18 m, outlet diameter=89 m, inlet height=14.5 m, outlet height=200 m, throat diameter=85.7934 m, throat height=133.333 m, base diameter=146.427 m>
    >>> ct.diameter(5)
    142.84514486126062

    References
    ----------
    .. [1] Chen, W. F., and E. M. Lui, eds. Handbook of Structural Engineering,
       Second Edition. Boca Raton, Fla: CRC Press, 2005.
    .. [2] Ansary, A. M. El, A. A. El Damatty, and A. O. Nassef. Optimum Shape
       and Design of Cooling Towers, 2011.
    '''
    def __repr__(self):  # pragma : no cover
        s = '''<Hyperbolic cooling tower, inlet diameter=%g m, outlet diameter=%g m, inlet height=%g m, \
outlet height=%g m, throat diameter=%g m, throat height=%g m, base diameter=%g m>'''
        s = s%(self.D_inlet, self.D_outlet, self.H_inlet, self.H_outlet, self.D_throat, self.H_throat, self.D_base)
        return s

    def __init__(self, H_inlet, D_outlet, H_outlet, D_inlet=None, D_base=None,
                 D_throat=None, H_throat=None,

                 H_support=None, D_support=None, n_support=None,
                 inlet_rounding=None):
        self.D_outlet = D_outlet
        self.H_inlet = H_inlet
        self.H_outlet = H_outlet

        if H_throat is None:
            H_throat = 2/3.0*H_outlet
        self.H_throat = H_throat

        if D_throat is None:
            if D_inlet is not None:
                D_throat = 0.63*D_inlet
            else:
                raise ValueError('Provide either `D_throat`, or `D_inlet` so it may be estimated.')
        self.D_throat = D_throat

        if D_inlet is None and D_base is None:
            raise ValueError('Need `D_inlet` or `D_base`')
        if D_base is not None:
            b = self.D_throat*self.H_throat/sqrt(D_base**2 - self.D_throat**2)
            D_inlet = 2*self.D_throat*sqrt((self.H_throat-H_inlet)**2 + b**2)/(2*b)
        elif D_inlet is not None:
            b = self.D_throat*(self.H_throat-H_inlet)/sqrt(D_inlet**2 - self.D_throat**2)
            D_base = 2*self.D_throat*sqrt(self.H_throat**2 + b**2)/(2*b)

        self.D_inlet = D_inlet
        self.D_base = D_base
        self.b_lower = b

        # Upper b parameter
        self.b_upper = self.D_throat*(self.H_outlet - self.H_throat)/sqrt((self.D_outlet)**2 - self.D_throat**2)

        # May or may not be specified
        self.H_support = H_support
        self.D_support = D_support
        self.n_support = n_support
        self.inlet_rounding = inlet_rounding

    def plot(self, pts=100):  # pragma: no cover
        import matplotlib.pyplot as plt

        Zs = linspace(0, self.H_outlet, pts)
        Rs = [self.diameter(Z)*0.5 for Z in Zs]
        plt.plot(Zs, Rs)
        plt.plot(Zs, [-v for v in Rs])
        plt.show()

    def diameter(self, H):
        r'''Calculates cooling tower diameter at a specified height, using
        the formulas for either hyperbola, depending on the height specified.

        .. math::
            D = D_{throat}\frac{\sqrt{H^2 + b^2}}{b}

        The value of `H` and `b` used in the above equation is as follows:

            * `H_throat` - H  and `b_lower` if under the throat
            * `H` - `H_throat` and `b_upper`, if above the throat

        Parameters
        ----------
        H : float
            Height at which to calculate the cooling tower diameter, [m]

        Returns
        -------
        D : float
            Diameter of the cooling tower at the specified height, [m]
        '''
        # Compute the diameter at H
        if H <= self.H_throat:
            # Height relative to throat height
            H = self.H_throat - H
            b = self.b_lower
        else:
            H = H - self.H_throat
            b = self.b_upper
        R = self.D_throat*sqrt(H*H + b*b)/(2.0*b)
        return R*2.0


class AirCooledExchanger(object):
    r'''Class representing the geometry of an air cooled heat exchanger with
    one or more tube bays, fans, or bundles.
    All parameters are also attributes.

    The minimum information required to describe an air cooler is as follows:

    * `tube_rows`
    * `tube_passes`
    * `tubes_per_row`
    * `tube_length`
    * `tube_diameter`
    * `fin_thickness`

    Two of `angle`, `pitch`, `pitch_parallel`, and `pitch_normal`
    (`pitch_ratio` may take the place of `pitch`)

    Either `fin_diameter` or `fin_height`.
    Either `fin_density` or `fin_interval`.

    Parameters
    ----------
    tube_rows : int
        Number of tube rows per bundle, [-]
    tube_passes : int
        Number of tube passes (times the fluid travels across one tube length),
        [-]
    tubes_per_row : float
        Number of tubes per row per bundle, [-]
    tube_length : float
        Total length of the tube bundle tubes, [m]
    tube_diameter : float
        Diameter of the bare tube, [m]
    fin_thickness : float
        Thickness of the fins, [m]
    angle : float, optional
        Angle of the tube layout, [degrees]
    pitch : float, optional
        Shortest distance between tube centers; defined in relation to the
        flow direction only, [m]
    pitch_parallel : float, optional
        Distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float, optional
        Distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]
    pitch_ratio : float, optional
        Ratio of the pitch to bare tube diameter, [-]
    fin_diameter : float, optional
        Outer diameter of each tube after including the fin on both sides,
        [m]
    fin_height : float, optional
        Height above bare tube of the tube fins, [m]
    fin_density : float, optional
        Number of fins per meter of tube, [1/m]
    fin_interval : float, optional
        Space between each fin, including the thickness of one fin at its
        base, [m]
    parallel_bays : int, optional
        Number of bays in the unit, [-]
    bundles_per_bay : int, optional
        Number of tube bundles per bay, [-]
    fans_per_bay : int, optional
        Number of fans per bay, [-]
    corbels : bool, optional
        Whether or not the air cooler has corbels, which increase the air
        velocity by adding half a tube to the sides for the case of
        non-rectangular tube layouts, [-]
    tube_thickness : float, optional
        Thickness of the bare metal tubes, [m]
    fan_diameter : float, optional
        Diameter of air cooler fan, [m]

    Attributes
    ----------
    bare_length : float
        Length of bare tube between two fins
        :math:`\text{bare length} = \text{fin interval} - t_{fin}`, [m]
    tubes_per_bundle : float
        Total number of tubes per bundle
        :math:`N_{tubes/bundle} = N_{tubes/row} \cdot N_{rows}`, [-]
    tubes_per_bay : float
        Total number of tubes per bay
        :math:`N_{tubes/bay} = N_{tubes/bundle} \cdot N_{bundles/bay}`, [-]
    tubes : float
        Total number of tubes in all bundles in all bays combined
        :math:`N_{tubes} = N_{tubes/bay} \cdot N_{bays}`, [-]

    pitch_diagonal : float
        Distance between tube centers in a diagonal line between one normal
        tube and one parallel tube;
        :math:`s_D = \left[s_L^2 + \left(\frac{s_T}{2}\right)^2\right]^{0.5}`,
        [m]

    A_bare_tube_per_tube : float
        Area of the bare tube including the portion hidden by the fin per
        tube :math:`A_{bare,total/tube} = \pi D_{tube} L_{tube}`, [m^2]
    A_bare_tube_per_row : float
        Area of the bare tube including the portion hidden by the fin per
        tube row
        :math:`A_{bare,total/row} = \pi D_{tube} L_{tube} N_{tubes/row}`, [m^2]
    A_bare_tube_per_bundle : float
        Area of the bare tube including the portion hidden by the fin per
        bundle :math:`A_{bare,total/bundle} = \pi D_{tube} L_{tube}
        N_{tubes/bundle}`, [m^2]
    A_bare_tube_per_bay : float
        Area of the bare tube including the portion hidden by the fin per
        bay :math:`A_{bare,total/bay} = \pi D_{tube} L_{tube} N_{tubes/bay}`,
        [m^2]
    A_bare_tube : float
        Area of the bare tube including the portion hidden by the fin per
        in all bundles and bays combined :math:`A_{bare,total} = \pi D_{tube}
        L_{tube} N_{tubes}`, [m^2]

    A_tube_showing_per_tube : float
        Area of the bare tube which is exposed per tube :math:`A_{bare,
        showing/tube} = \pi D_{tube} L_{tube}  \left(1 - \frac{t_{fin}}
        {\text{fin interval}} \right)`, [m^2]
    A_tube_showing_per_row : float
        Area of the bare tube which is exposed per tube row, [m^2]
    A_tube_showing_per_bundle : float
        Area of the bare tube which is exposed per bundle, [m^2]
    A_tube_showing_per_bay : float
        Area of the bare tube which is exposed per bay, [m^2]
    A_tube_showing : float
        Area of the bare tube which is exposed in all bundles and bays
        combined, [m^2]

    A_per_fin : float
        Surface area per fin :math:`A_{fin} = 2 \frac{\pi}{4} (D_{fin}^2 -
        D_{tube}^2) + \pi D_{fin} t_{fin}`, [m^2]
    A_fin_per_tube : float
        Surface area of all fins per tube
        :math:`A_{fin/tube} = N_{fins/m} L_{tube} A_{fin}`, [m^2]
    A_fin_per_row : float
        Surface area of all fins per row, [m^2]
    A_fin_per_bundle : float
        Surface area of all fins per bundle, [m^2]
    A_fin_per_bay : float
        Surface area of all fins per bay, [m^2]
    A_fin : float
        Surface area of all fins in all bundles and bays combined, [m^2]

    A_per_tube : float
        Surface area of combined finned and non-fined area exposed for heat
        transfer per tube :math:`A_{tube} = A_{bare, showing/tube}
        + A_{fin/tube}`, [m^2]
    A_per_row : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer per tube row, [m^2]
    A_per_bundle : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer per tube bundle, [m^2]
    A_per_bay : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer per bay, [m^2]
    A : float
        Surface area of combined finned and non-finned area exposed for heat
        transfer in all bundles and bays combined, [m^2]
    A_increase : float
        Ratio of actual surface area to bare tube surface area
        :math:`A_{increase} = \frac{A_{tube}}{A_{bare, total/tube}}`, [-]

    A_tube_flow : float
        The area for the fluid to flow in one tube, :math:`\pi/4\cdot D_i^2`,
        [m^2]
    channels : int
        The number of tubes the fluid flows through at the inlet header, [-]

    tube_volume_per_tube : float
        Fluid volume per tube inside :math:`V_{tube, flow} = \frac{\pi}{4}
        D_{i}^2 L_{tube}`, [m^3]
    tube_volume_per_row : float
        Fluid volume of tubes per row, [m^3]
    tube_volume_per_bundle : float
        Fluid volume of tubes per bundle, [m^3]
    tube_volume_per_bay : float
        Fluid volume of tubes per bay, [m^3]
    tube_volume : float
        Fluid volume of tubes in all bundles and bays combined, [m^3]


    A_diagonal_per_bundle : float
        Air flow area along the diagonal plane per bundle
        :math:`A_d = 2 N_{tubes/row} L_{tube} (P_d - D_{tube} - 2 N_{fins/m} h_{fin} t_{fin}) + A_\text{extra,side}`, [m^2]
    A_normal_per_bundle : float
        Air flow area along the normal (transverse) plane; this is normally
        the minimum flow area, except for some staggered configurations
        :math:`A_t = N_{tubes/row} L_{tube} (P_t - D_{tube} - 2 N_{fins/m} h_{fin} t_{fin}) + A_\text{extra,side}`, [m^2]
    A_min_per_bundle : float
        Minimum air flow area per bundle; this is the characteristic area for
        velocity calculation in most finned tube convection correlations
        :math:`A_{min} = min(A_d, A_t)`, [m^2]
    A_min_per_bay : float
        Minimum air flow area per bay, [m^2]
    A_min : float
        Minimum air flow area, [m^2]

    A_face_per_bundle : float
        Face area per bundle :math:`A_{face} = P_{T} (1+N_{tubes/row})
        L_{tube}`; if corbels are used, add 0.5 to tubes/row instead of 1,
        [m^2]
    A_face_per_bay : float
        Face area per bay, [m^2]
    A_face : float
        Total face area, [m^2]
    flow_area_contraction_ratio : float
        Ratio of `A_min` to `A_face`, [-]


    Notes
    -----

    Examples
    --------
    >>> AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=10.9728,
    ... tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
    ... angle=30, pitch=2.5*inch, fin_height=0.625*inch, tube_thickness=0.00338,
    ... bundles_per_bay=2, parallel_bays=3, corbels=True)


    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    def __repr__(self):
        s = '<Air Cooler Geometry, %s>'
        t = ''
        for k, v in self.__dict__.items():
            try:
                t += '%s=%g, ' %(k, v)
            except:
                t += '%s=%s, ' %(k, v)
        t = t[0:-2]
        return s%t

    def __init__(self, tube_rows, tube_passes, tubes_per_row, tube_length,
                 tube_diameter, fin_thickness,

                 angle=None, pitch=None, pitch_parallel=None, pitch_normal=None,
                 pitch_ratio=None,

                 fin_diameter=None, fin_height=None,

                 fin_density=None, fin_interval=None,

                 parallel_bays=1, bundles_per_bay=1, fans_per_bay=1,
                 corbels=False, tube_thickness=None, fan_diameter=None):
        # TODO: fin types

        self.tube_rows = tube_rows
        self.tube_passes = tube_passes
        self.tubes_per_row = tubes_per_row
        self.tube_length = tube_length
        self.tube_diameter = tube_diameter
        self.fin_thickness = fin_thickness
        self.fan_diameter = fan_diameter

        if pitch_ratio is not None:
            if pitch is not None:
                pitch = self.tube_diameter*pitch_ratio
            else:
                raise ValueError('Specify only one of `pitch_ratio` or `pitch`')


        angle, pitch, pitch_parallel, pitch_normal = pitch_angle_solver(
            angle=angle, pitch=pitch, pitch_parallel=pitch_parallel,
            pitch_normal=pitch_normal)
        self.angle = angle
        self.pitch = pitch
        self.pitch_parallel = pitch_parallel
        self.pitch_normal = pitch_normal

        self.pitch_diagonal = sqrt(pitch_parallel**2 + (0.5*pitch_normal)**2)


        if fin_diameter is None and fin_height is None:
            raise ValueError('Specify only one of `fin_diameter` or `fin_height`')
        elif fin_diameter is not None:
            fin_height = 0.5*(fin_diameter - tube_diameter)
        elif fin_height is not None:
            fin_diameter = tube_diameter + 2.0*fin_height
        self.fin_height = fin_height
        self.fin_diameter = fin_diameter

        if fin_density is None and fin_interval is None:
            raise ValueError('Specify only one of `fin_density` or `fin_interval`')
        elif fin_density is not None:
            fin_interval = 1.0/fin_density
        elif fin_interval is not None:
            fin_density = 1.0/fin_interval
        self.fin_interval = fin_interval
        self.fin_density = fin_density

        self.parallel_bays = parallel_bays
        self.bundles_per_bay = bundles_per_bay
        self.fans_per_bay = fans_per_bay

        self.corbels = corbels
        self.tube_thickness = tube_thickness


        if self.fin_interval:
            self.bare_length = self.fin_interval - self.fin_thickness
        else:
            self.bare_length = None

        self.tubes_per_bundle = self.tubes_per_row*self.tube_rows
        self.tubes_per_bay = self.tubes_per_bundle*self.bundles_per_bay
        self.tubes = self.tubes_per_bay*self.parallel_bays


        self.A_bare_tube_per_tube = pi*self.tube_diameter*self.tube_length
        self.A_bare_tube_per_row = self.A_bare_tube_per_tube*self.tubes_per_row
        self.A_bare_tube_per_bundle = self.A_bare_tube_per_tube*self.tubes_per_bundle
        self.A_bare_tube_per_bay = self.A_bare_tube_per_tube*self.tubes_per_bay
        self.A_bare_tube = self.A_bare_tube_per_tube*self.tubes

        self.A_tube_showing_per_tube = pi*self.tube_diameter*self.tube_length*(1.0 - self.fin_thickness/self.fin_interval)
        self.A_tube_showing_per_row = self.A_tube_showing_per_tube*self.tubes_per_row
        self.A_tube_showing_per_bundle = self.A_tube_showing_per_tube*self.tubes_per_bundle
        self.A_tube_showing_per_bay = self.A_tube_showing_per_tube*self.tubes_per_bay
        self.A_tube_showing = self.A_tube_showing_per_tube*self.tubes

        self.A_per_fin = (2.0*pi/4.0*(self.fin_diameter**2 - self.tube_diameter**2)
                     + pi*self.fin_diameter*self.fin_thickness) # pi*D*L(fin)
        self.A_fin_per_tube = self.fin_density*self.tube_length*self.A_per_fin
        self.A_fin_per_row = self.A_fin_per_tube*self.tubes_per_row
        self.A_fin_per_bundle = self.A_fin_per_tube*self.tubes_per_bundle
        self.A_fin_per_bay = self.A_fin_per_tube*self.tubes_per_bay
        self.A_fin = self.A_fin_per_tube*self.tubes

        self.A_per_tube = self.A_tube_showing_per_tube + self.A_fin_per_tube
        self.A_per_row = self.A_tube_showing_per_row + self.A_fin_per_row
        self.A_per_bundle = self.A_tube_showing_per_bundle + self.A_fin_per_bundle
        self.A_per_bay = self.A_tube_showing_per_bay + self.A_fin_per_bay
        self.A = self.A_tube_showing + self.A_fin

        self.A_increase = self.A/self.A_bare_tube


        # TODO A_extra could be calculated based on a fixed width and height of the bay
        A_extra = 0.0
        self.A_diagonal_per_bundle = 2.0*self.tubes_per_row*self.tube_length*(self.pitch_diagonal - self.tube_diameter - 2.0*fin_density*self.fin_height*self.fin_thickness) + A_extra
        self.A_normal_per_bundle = self.tubes_per_row*self.tube_length*(self.pitch_normal - self.tube_diameter - 2.0*fin_density*self.fin_height*self.fin_thickness) + A_extra
        self.A_min_per_bundle = min(self.A_diagonal_per_bundle, self.A_normal_per_bundle)
        self.A_min_per_bay = self.A_min_per_bundle*self.bundles_per_bay
        self.A_min = self.A_min_per_bay*self.parallel_bays

        i = 0.5 if self.corbels else 1.0
        self.A_face_per_bundle = self.pitch_normal*self.tube_length*(self.tubes_per_row + i)
        self.A_face_per_bay = self.A_face_per_bundle*self.bundles_per_bay
        self.A_face = self.A_face_per_bay*self.parallel_bays

        self.flow_area_contraction_ratio = self.A_min/self.A_face

        if self.tube_thickness is not None:
            self.Di = self.tube_diameter - self.tube_thickness*2.0
            self.A_tube_flow = pi/4.0*self.Di*self.Di

            self.tube_volume_per_tube = self.A_tube_flow*self.tube_length
            self.tube_volume_per_row = self.tube_volume_per_tube*self.tubes_per_row
            self.tube_volume_per_bundle = self.tube_volume_per_tube*self.tubes_per_bundle
            self.tube_volume_per_bay = self.tube_volume_per_tube*self.tubes_per_bay
            self.tube_volume = self.tube_volume_per_tube*self.tubes
        else:
            self.Di = None
            self.A_tube_flow = None

            self.tube_volume_per_tube = None
            self.tube_volume_per_row = None
            self.tube_volume_per_bundle = None
            self.tube_volume_per_bay = None
            self.tube_volume = None

        # TODO: Support different numbers of tube rows per pass - maybe pass
        # a list of rows per pass to tube_passes?
        if self.tube_rows % self.tube_passes == 0:
            self.channels = self.tubes_per_bundle/self.tube_passes
        else:
            self.channels = self.tubes_per_row

        if self.angle == 30:
            self.pitch_str = 'triangular'
            self.pitch_class = 'staggered'
        elif self.angle == 60:
            self.pitch_str = 'rotated triangular'
            self.pitch_class = 'staggered'
        elif self.angle == 45:
            self.pitch_str = 'rotated square'
            self.pitch_class = 'in-line'
        elif self.angle == 90:
            self.pitch_str = 'square'
            self.pitch_class = 'in-line'
        else:
            self.pitch_str = 'custom'
            self.pitch_class = 'custom'




def pitch_angle_solver(angle=None, pitch=None, pitch_parallel=None,
                       pitch_normal=None):
    r'''Utility to take any two of `angle`, `pitch`, `pitch_parallel`, and
    `pitch_normal` and calculate the other two. This is useful for applications
    with tube banks, as in shell and tube heat exchangers or air coolers and
    allows for a wider range of user input.

    .. math::
        \text{pitch normal} = \text{pitch} \cdot \sin(\text{angle})

    .. math::
        \text{pitch parallel} = \text{pitch} \cdot \cos(\text{angle})

    Parameters
    ----------
    angle : float, optional
        The angle of the tube layout, [degrees]
    pitch : float, optional
        The shortest distance between tube centers; defined in relation to the
        flow direction only, [m]
    pitch_parallel : float, optional
        The distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float, optional
        The distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]

    Returns
    -------
    angle : float
        The angle of the tube layout, [degrees]
    pitch : float
        The shortest distance between tube centers; defined in relation to the
        flow direction only, [m]
    pitch_parallel : float
        The distance between tube center along a line parallel to the flow;
        has been called `longitudinal` pitch, `pp`, `s2`, `SL`, and `p2`, [m]
    pitch_normal : float
        The distance between tube centers in a line 90° to the line of flow;
        has been called the `transverse` pitch, `pn`, `s1`, `ST`, and `p1`, [m]

    Notes
    -----
    For the 90 and 0 degree case, the normal or parallel pitches can be zero;
    given the angle and the zero value, obviously is it not possible to
    calculate the pitch and a math error will be raised.

    No exception will be raised if three or four inputs are provided; the other
    two will simply be calculated according to the list of if statements used.

    An exception will be raised if only one input is provided.

    Examples
    --------
    >>> pitch_angle_solver(pitch=1, angle=30)
    (30, 1, 0.8660254037844387, 0.49999999999999994)

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    if angle is not None and pitch is not None:
        pitch_normal = pitch*sin(radians(angle))
        pitch_parallel = pitch*cos(radians(angle))
    elif angle is not None and pitch_normal is not None:
        pitch = pitch_normal/sin(radians(angle))
        pitch_parallel = pitch*cos(radians(angle))
    elif angle is not None and pitch_parallel is not None:
        pitch = pitch_parallel/cos(radians(angle))
        pitch_normal = pitch*sin(radians(angle))
    elif pitch_normal is not None and pitch is not None:
        angle = degrees(asin(pitch_normal/pitch))
        pitch_parallel = pitch*cos(radians(angle))
    elif pitch_parallel is not None and pitch is not None:
        angle = degrees(acos(pitch_parallel/pitch))
        pitch_normal = pitch*sin(radians(angle))
    elif pitch_parallel is not None and pitch_normal is not None:
        angle = degrees(asin(pitch_normal/sqrt(pitch_normal**2 + pitch_parallel**2)))
        pitch = sqrt(pitch_normal**2 + pitch_parallel**2)
    else:
        raise ValueError('Two of the arguments are required')
    return angle, pitch, pitch_parallel, pitch_normal


def sphericity(A, V):
    r'''Returns the sphericity of a particle of surface area `A` and volume
    `V`. Sphericity is the ratio of the surface area of a sphere with the same
    volume as the particle (equivalent diameter) to the actual surface area of
    the particle.

    .. math::
        \Psi = \frac{\text{A of sphere with } V_p }  {{A}_p}
        = \frac{\pi^{\frac{1}{3}}(6V_p)^{\frac{2}{3}}}{A_p}

    Parameters
    ----------
    A : float
        Surface area of particle, [m^2]
    V : float
        Volume of particle, [m^3]

    Returns
    -------
    Psi : float
        Sphericity [-]

    Notes
    -----
    All non-spherical particles have spericities less than 1 but greater than 0.
    Many common geometrical shapes have their results calculated exactly in [2]_.

    Examples
    --------
    >>> sphericity(10., 2.)
    0.767663317071005

    For a cube of side length a=3, the surface area is 6*a^2=54 and volume a^3=27.
    Its sphericity is then:

    >>> sphericity(A=54, V=27)
    0.8059959770082346

    References
    ----------
    .. [1] Rhodes, Martin J., ed. Introduction to Particle Technology. 2E.
       Chichester, England ; Hoboken, NJ: Wiley, 2008.
    .. [2] "Sphericity." Wikipedia, March 8, 2017.
       https://en.wikipedia.org/w/index.php?title=Sphericity&oldid=769183043
    '''
    return pi**(1/3.)*(6*V)**(2/3.)/A


def aspect_ratio(Dmin, Dmax):
    r'''Returns the aspect ratio of a shape with minimum and maximum dimension,
    `Dmin` and `Dmax`.

    .. math::
        A_R = \frac{D_{min}}{D_{max}}

    Parameters
    ----------
    Dmin : float
        Minimum dimension, [m]
    Dmax : float
        Maximum dimension, [m]

    Returns
    -------
    a_r : float
        Aspect ratio [-]

    Examples
    --------
    >>> aspect_ratio(.2, 2)
    0.1
    '''
    return Dmin/Dmax


def circularity(A, P):
    r'''Returns the circularity of a shape with area `A` and perimeter `P`.

    .. math::
        f_{circ} = \frac {4 \pi A} {P^2}

    Defined to be 1 for a circle. Used to characterize particles. Any
    non-circular shape must have a circularity less than one.

    Parameters
    ----------
    A : float
        Area of the shape, [m^2]
    P : float
        Perimeter of the shape, [m]

    Returns
    -------
    f_circ : float
        Circularity of the shape [-]

    Examples
    --------
    Square, side length = 2 (all squares are the same):

    >>> circularity(A=(2*2), P=4*2)
    0.7853981633974483

    Rectangle, one side length = 1, second side length = 100

    >>> D1 = 1
    >>> D2 = 100
    >>> A = D1*D2
    >>> P = 2*D1 + 2*D2
    >>> circularity(A, P)
    0.030796908671598795
    '''
    return 4*pi*A/P**2


def A_cylinder(D, L):
    r'''Returns the surface area of a cylinder.

    .. math::
        A = \pi D L + 2\cdot \frac{\pi D^2}{4}

    Parameters
    ----------
    D : float
        Diameter of the cylinder, [m]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    A : float
        Surface area [m^2]

    Examples
    --------
    >>> A_cylinder(0.01, .1)
    0.0032986722862692833
    '''
    cap = pi*D**2/4*2
    side = pi*D*L
    return cap + side


def V_cylinder(D, L):
    r'''Returns the volume of a cylinder.

    .. math::
        V = \frac{\pi D^2}{4}L

    Parameters
    ----------
    D : float
        Diameter of the cylinder, [m]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    >>> V_cylinder(0.01, .1)
    7.853981633974484e-06
    '''
    return pi*D**2/4*L


def A_hollow_cylinder(Di, Do, L):
    r'''Returns the surface area of a hollow cylinder.

    .. math::
        A = \pi D_o L + \pi D_i L  + 2\cdot \frac{\pi D_o^2}{4}
        - 2\cdot \frac{\pi D_i^2}{4}

    Parameters
    ----------
    Di : float
        Diameter of the hollow in the cylinder, [m]
    Do : float
        Diameter of the exterior of the cylinder, [m]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    A : float
        Surface area [m^2]

    Examples
    --------
    >>> A_hollow_cylinder(0.005, 0.01, 0.1)
    0.004830198704894308
    '''
    side_o = pi*Do*L
    side_i = pi*Di*L
    cap_circle = pi*Do**2/4*2
    cap_removed = pi*Di**2/4*2
    return side_o + side_i + cap_circle - cap_removed


def V_hollow_cylinder(Di, Do, L):
    r'''Returns the volume of a hollow cylinder.

    .. math::
        V = \frac{\pi D_o^2}{4}L - L\frac{\pi D_i^2}{4}

    Parameters
    ----------
    Di : float
        Diameter of the hollow in the cylinder, [m]
    Do : float
        Diameter of the exterior of the cylinder, [m]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    >>> V_hollow_cylinder(0.005, 0.01, 0.1)
    5.890486225480862e-06
    '''
    return pi*Do**2/4*L - pi*Di**2/4*L


def A_multiple_hole_cylinder(Do, L, holes):
    r'''Returns the surface area of a cylinder with multiple holes.
    Calculation will naively return a negative value or other impossible
    result if the number of cylinders added is physically impossible.
    Holes may be of different shapes, but must be perpendicular to the
    axis of the cylinder.

    .. math::
        A = \pi D_o L + 2\cdot \frac{\pi D_o^2}{4} +
        \sum_{i}^n \left( \pi D_i L  - 2\cdot \frac{\pi D_i^2}{4}\right)

    Parameters
    ----------
    Do : float
        Diameter of the exterior of the cylinder, [m]
    L : float
        Length of the cylinder, [m]
    holes : list
        List of tuples containing (diameter, count) pairs of descriptions for
        each of the holes sizes.

    Returns
    -------
    A : float
        Surface area [m^2]

    Examples
    --------
    >>> A_multiple_hole_cylinder(0.01, 0.1, [(0.005, 1)])
    0.004830198704894308
    '''
    side_o = pi*Do*L
    cap_circle = pi*Do**2/4*2
    A = cap_circle + side_o
    for Di, n in holes:
        side_i = pi*Di*L
        cap_removed = pi*Di**2/4*2
        A = A + side_i*n - cap_removed*n
    return A


def V_multiple_hole_cylinder(Do, L, holes):
    r'''Returns the solid volume of a cylinder with multiple cylindrical holes.
    Calculation will naively return a negative value or other impossible
    result if the number of cylinders added is physically impossible.

    .. math::
        V = \frac{\pi D_o^2}{4}L - L\frac{\pi D_i^2}{4}

    Parameters
    ----------
    Do : float
        Diameter of the exterior of the cylinder, [m]
    L : float
        Length of the cylinder, [m]
    holes : list
        List of tuples containing (diameter, count) pairs of descriptions for
        each of the holes sizes.

    Returns
    -------
    V : float
        Volume [m^3]

    Examples
    --------
    >>> V_multiple_hole_cylinder(0.01, 0.1, [(0.005, 1)])
    5.890486225480862e-06
    '''
    V = pi*Do**2/4*L
    for Di, n in holes:
        V -= pi*Di*Di/4*L*n
    return V

