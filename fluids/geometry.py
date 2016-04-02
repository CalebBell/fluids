# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from __future__ import division
from math import pi, sin, cos, asin, acos, atan, acosh, log
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve

__all__ = ['TANK', 'SA_partial_sphere', 'V_partial_sphere', 'V_horiz_conical',
           'V_horiz_ellipsoidal', 'V_horiz_guppy', 'V_horiz_spherical',
           'V_horiz_torispherical', 'V_vertical_conical',
           'V_vertical_ellipsoidal', 'V_vertical_spherical',
           'V_vertical_torispherical', 'V_vertical_conical_concave',
           'V_vertical_ellipsoidal_concave', 'V_vertical_spherical_concave',
           'V_vertical_torispherical_concave', 'a_torispherical',
           'SA_ellipsoidal_head', 'SA_conical_head', 'SA_guppy_head',
           'SA_torispheroidal', 'V_from_h', 'SA_tank']


### Spherical Vessels, partially filled


def SA_partial_sphere(D, h):
    r'''Calculates surface area of a partial sphere according to [1]_.
    If h is half of D, the shape is half a sphere. No bottom is considered in
    this function. Valid inputs are positive values of D and h, with h always
    smaller or equal to D.

    .. math::
        a = \sqrt{h(2r - h)}

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
    r = D/2
    a = (h*(2*r - h))**0.5
    A = pi*(a**2 + h**2)
    return A


def V_partial_sphere(D, h):
    r'''Calculates volume of a partial sphere according to [1]_.
    If h is half of D, the shape is half a sphere. No bottom is considered in
    this function. Valid inputs are positive values of D and h, with h always
    smaller or equal to D.

    .. math::
        a = \sqrt{h(2r - h)}

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
    r = D/2
    a = (h*(2*r - h))**0.5
    V = 1/6.*pi*h*(3*a**2 + h**2)
    return V



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

        V_f = A_fL + \frac{2aR^2}{3}\pi/2,\;\; h = R\\

        V_f = A_fL + \frac{2aR^2}{3}(\pi-K), \;\; R< h \le 2R

        K = \cos^{-1} M + M^3\cosh^{-1} \frac{1}{M} - 2M\sqrt{1 - M^2}

        M = \left|\frac{R-h}{R}\right|

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
    R = D/2.
    Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5
    M = abs((R-h)/R)
    if h == R:
        Vf = 2*a*R**2/3*pi/2
    else:
        K = acos(M) + M**3*acosh(1./M) - 2*M*(1-M**2)**0.5
        if 0 <= h < R:
            Vf = 2*a*R**2/3*K
        elif R < h <= 2*R:
            Vf = 2*a*R**2/3*(pi-K)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


def V_horiz_ellipsoidal(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with ellipsoidal ends, according to [1]_.

    .. math::
        V_f = A_fL + \pi a h^2\left(1 - \frac{h}{3R}\right)

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
    R = D/2.
    Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5
    Vf = pi*a*h**2*(1 - h/3./R)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


def V_horiz_guppy(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with guppy heads, according to [1]_.

    .. math::
        V_f = A_fL + \frac{2aR^2}{3}\cos^{-1}\left(1 - \frac{h}{R}\right)
        +\frac{2a}{9R}\sqrt{2Rh - h^2}(2h-3R)(h+R)

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
    R = D/2.
    Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5
    Vf = 2*a*R**2/3*acos(1 - h/R) + 2*a/9./R*(2*R*h - h**2)**0.5*(2*h - 3*R)*(h + R)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


def V_horiz_spherical(D, L, a, h, headonly=False):
    r'''Calculates volume of a tank with spherical heads, according to [1]_.

    .. math::
        V_f = A_fL + \frac{\pi a}{6}(3R^2 + a^2),\;\; h = R, |a|\le R

        V_f = A_fL + \frac{\pi a}{3}(3R^2 + a^2),\;\; h = D, |a|\le R

        V_f = A_fL + \pi a h^2\left(1 - \frac{h}{3R}\right),\;\; h = 0,
        \text{ or } |a| = 0, R, -R

        V_f = A_fL + \frac{a}{|a|}\left\{\frac{2r^3}{3}\left[\cos^{-1}
        \frac{R^2 - rw}{R(w-r)} + \cos^{-1}\frac{R^2 + rw}{R(w+r)}
        - \frac{z}{r}\left(2 + \left(\frac{R}{r}\right)^2\right)
        \cos^{-1}\frac{w}{R}\right] - 2\left(wr^2 - \frac{w^3}{3}\right)
        \tan^{-1}\frac{y}{z} + \frac{4wyz}{3}\right\}
        ,\;\; h \ne R, D; a \ne 0, R, -R, |a| \ge 0.01D

        V_f = A_fL + \frac{a}{|a|}\left[2\int_w^R(r^2 - x^2)\tan^{-1}
        \sqrt{\frac{R^2-x^2}{r^2-R^2}}dx - A_f z\right]
        ,\;\; h \ne R, D; a \ne 0, R, -R, |a| < 0.01D

        Af = R^2\cos^{-1}\frac{R-h}{R} - (R-h)\sqrt{2Rh - h^2}

        r = \frac{a^2 + R^2}{2|a|}

        w = R - h

        y = \sqrt{2Rh-h^2}

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
    R = D/2.
    r = (a**2 + R**2)/2./abs(a)
    w = R - h
    y = (2*R*h - h**2)**0.5
    z = (r**2 - R**2)**0.5
    Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5

    if h == R and abs(a) <= R:
        Vf = pi*a/6*(3*R**2 + a**2)
    elif h == D and abs(a) <= R:
        Vf = pi*a/3*(3*R**2 + a**2)
    elif h == 0 or a == 0 or a == R or a == -R:
        Vf = pi*a*h**2*(1 - h/3./R)
    elif abs(a) >= 0.01*D:
        Vf = a/abs(a)*(
        2*r**3/3.*(acos((R**2 - r*w)/(R*(w-r))) + acos((R**2+r*w)/(R*(w+r)))
        - z/r*(2+(R/r)**2)*acos(w/R))
        - 2*(w*r**2 - w**3/3)*atan(y/z) + 4*w*y*z/3)
    else:
        def V_horiz_spherical_toint(x):
            return (r**2 - x**2)*atan(((R**2 - x**2)/(r**2 - R**2))**0.5)
        integrated = quad(V_horiz_spherical_toint, w, R)[0]
        Vf = a/abs(a)*(2*integrated - Af*z)
    if headonly:
        Vf = Vf/2.
    else:
        Vf += Af*L
    return Vf


def V_horiz_torispherical(D, L, f, k, h, headonly=False):
    r'''Calculates volume of a tank with torispherical heads, according to [1]_.

    .. math::
        V_f  = A_fL + 2V_1, \;\; 0 \le h \le h_1\\
        V_f  = A_fL + 2(V_{1,max} + V_2 + V_3), \;\; h_1 < h < h_2\\
        V_f  = A_fL + 2[2V_{1,max} - V_1(h=D-h) + V_{2,max} + V_{3,max}]
        , \;\; h_2 \le h \le D

        V_1 = \int_0^{\sqrt{2kDh - h^2}} \left[n^2\sin^{-1}\frac{\sqrt
        {n^2-w^2}}{n} - w\sqrt{n^2-w^2}\right]dx

        V_2 = \int_0^{kD\cos\alpha}\left[n^2\left(\cos^{-1}\frac{w}{n}
        - \cos^{-1}\frac{g}{n}\right) - w\sqrt{n^2 - w^2} + g\sqrt{n^2
        - g^2}\right]dx

        V_3 = \int_w^g(r^2 - x^2)\tan^{-1}\frac{\sqrt{g^2 - x^2}}{z}dx
        - \frac{z}{2}\left(g^2\cos^{-1}\frac{w}{g} - w\sqrt{2g(h-h_1)
        - (h-h_1)^2}\right)

        V_{1,max} = v_1(h=h_1)

        v_{2,max} = v_2(h=h_2)

        v_{3,max} = \frac{\pi a_1}{6}(3g^2 + a_1^2)

        a_1 = fD(1-\cos\alpha)

        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

        n = R - kD + \sqrt{k^2D^2-x^2}

        g = r\sin\alpha

        r = fD

        h_2 = D - h_1

        w = R - h

        z = \sqrt{r^2- g^2}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    L : float
        Length of the main cylindrical section, [m]
    f : float
        Dish-radius parameter; fD  = dish radius []
    k : float
        knucle-radius parameter ; kD = knucle radius []
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
    2028.626670842139

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    R = D/2.
    Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5
    r = f*D
    alpha = asin((1 - 2*k)/(2.*(f-k)))
    a1 = r*(1-cos(alpha))
    g = r*sin(alpha)
    z = r*cos(alpha)
    h1 = k*D*(1-sin(alpha))
    h2 = D - h1

    def V1_toint(x, w):
        n = R - k*D + (k**2*D**2 - x**2)**0.5
        ans = n**2*asin((n**2-w**2)**0.5/n) - w*(n**2 - w**2)**0.5
        return ans
    def V2_toint(x, w):
        n = R - k*D + (k**2*D**2 - x**2)**0.5
        ans = n**2*(acos(w/n) - acos(g/n)) - w*(n**2 - w**2)**0.5 + g*(n**2-g**2)**0.5
        return ans
    def V3_toint(x):
        ans = (r**2-x**2)*atan((g**2-x**2)**0.5/z)
        return ans

    if 0 <= h <= h1:
        w = R - h
        Vf = 2*quad(V1_toint, 0, (2*k*D*h-h**2)**0.5, w)[0]
    elif h1 < h < h2:
        w = R - h
        wmax1 = R - h1
        V1max = quad(V1_toint, 0, (2*k*D*h1-h1**2)**0.5, wmax1)[0]
        V2 = quad(V2_toint, 0, k*D*cos(alpha), w)[0]
        V3 = quad(V3_toint, w, g)[0] - z/2.*(g**2*acos(w/g) -w*(2*g*(h-h1) - (h-h1)**2)**0.5)
        Vf = 2*(V1max + V2 + V3)
    else:
        w = R - h
        wmax1 = R - h1
        wmax2 = R - h2
        wwerird = R - (D - h)

        V1max = quad(V1_toint, 0, (2*k*D*h1-h1**2)**0.5, wmax1)[0]
        V1weird = quad(V1_toint, 0, (2*k*D*(D-h)-(D-h)**2)**0.5, wwerird)[0]
        V2max = quad(V2_toint, 0, k*D*cos(alpha), wmax2)[0]
        V3max = pi*a1/6.*(3*g**2 + a1**2)
        Vf = 2*(2*V1max - V1weird + V2max + V3max)
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

        V_f = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\pi u\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        + \frac{\pi tu^2}{2} - \frac{\pi u^3}{3} + \pi D(1-2k)\left[
        \frac{2u-t}{4}\sqrt{s+tu-u^2} + \frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\left(\cos^{-1}\frac{t-2u}{2kD}-\alpha\right)\right]
        ,\; a_1 < h \le a_1 + a_2

        V_f = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\frac{\pi t}{2}\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        +\frac{\pi  t^3}{12} + \pi D(1-2k)\left[\frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\sin^{-1}(\cos\alpha)\right]
        + \frac{\pi D^2}{4}[h-(a_1+a_2)] ,\;  a_1 + a_2 < h

        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

        a_1 = fD(1-\cos\alpha)

        a_2 = kD\cos\alpha

        D_1 = 2fD\sin\alpha

        s = (kD\sin\alpha)^2

        t = 2a_2

        u = h - fD(1-\cos\alpha)

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dish-radius parameter; fD  = dish radius []
    k : float
        knucle-radius parameter ; kD = knucle radius []
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
    904.0688283793511

    References
    ----------
    .. [1] Jones, D. "Calculating Tank Volume." Text. Accessed December 22, 2015.
       http://www.webcalc.com.br/blog/Tank_Volume.PDF'''
    alpha = asin((1-2*k)/(2*(f-k)))
    a1 = f*D*(1 - cos(alpha))
    a2 = k*D*cos(alpha)
    D1 = 2*f*D*sin(alpha)
    s = (k*D*sin(alpha))**2
    t = 2*a2
    u = h - f*D*(1 - cos(alpha))

    if 0 <= h <= a1:
        Vf = pi*h**2/4*(2*a1 + D1**2/2/a1 - 4*h/3)
    elif a1 < h <= a1 + a2:
        Vf = (pi/4*(2*a1**3/3 + a1*D1**2/2.) + pi*u*((D/2. - k*D)**2 + s)
        + pi*t*u**2/2. - pi*u**3/3. + pi*D*(1 - 2*k)*((2*u-t)/4.*(s + t*u
        - u**2)**0.5 + t*s**0.5/4. + k**2*D**2/2*(acos((t-2*u)/(2*k*D))-alpha)))
    else:
        Vf = pi/4*(2*a1**3/3. + a1*D1**2/2.) + pi*t/2.*((D/2 - k*D)**2
        + s) + pi*t**3/12. + pi*D*(1 - 2*k)*(t*s**0.5/4
        + k**2*D**2/2*asin(cos(alpha))) + pi*D**2/4*(h - (a1 + a2))
    return Vf


### Begin vertical tanks with concave heads

def V_vertical_conical_concave(D, a, h):
    r'''Calculates volume of a vertical tank with a concave conical bottom,
    according to [1]_. No provision for the top of the tank is made here.

    .. math::
        V = \frac{\pi D^2}{12} \left(3h + a - \frac{(a+h)^3}{a^2}\right)
        ,\;\; 0 \le h < |a|

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

        V = \frac{\pi D^2 h}{4} - v_1(h=a_1+a_2) + v_2(h=a_1 + a_2 -h),\; a_2 \le h < a_1 + a_2

        V = \frac{\pi D^2 h}{4} - v_1(h=a_1+a_2) + 0,\; h \ge a_1 + a_2

        v_1 = \frac{\pi}{4}\left(\frac{2a_1^3}{3} + \frac{a_1D_1^2}{2}\right)
        +\pi u\left[\left(\frac{D}{2}-kD\right)^2 +s\right]
        + \frac{\pi tu^2}{2} - \frac{\pi u^3}{3} + \pi D(1-2k)\left[
        \frac{2u-t}{4}\sqrt{s+tu-u^2} + \frac{t\sqrt{s}}{4}
        + \frac{k^2D^2}{2}\left(\cos^{-1}\frac{t-2u}{2kD}-\alpha\right)\right]

        v_2 = \frac{\pi h^2}{4}\left(2a_1 + \frac{D_1^2}{2a_1} - \frac{4h}{3}\right)

        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

        a_1 = fD(1-\cos\alpha)

        a_2 = kD\cos\alpha

        D_1 = 2fD\sin\alpha

        s = (kD\sin\alpha)^2

        t = 2a_2

        u = h - fD(1-\cos\alpha)

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dish-radius parameter; fD  = dish radius []
    k : float
        knucle-radius parameter ; kD = knucle radius []
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
        v1 += pi*D*(1-2*k)*((2*u-t)/4.*(s+t*u-u**2)**0.5 + t*s**0.5/4.
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
    r'''Calculates the surface area of an ellipsoidal head according to [1]_.
    Formula below is for the full shape, the result of which is halved. The
    formula also does not support `D` being larger than `a`; this is ensured
    by simply swapping the variables if necessary, as geometrically the result
    is the same. In the equations

    .. math::
        SA = 2\pi a^2 + \frac{\pi c^2}{e_1}\ln\left(\frac{1+e_1}{1-e_1}\right)

        e_1 = \sqrt{1 - \frac{c^2}{a^2}}

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

    References
    ----------
    .. [1] Weisstein, Eric W. "Spheroid." Text. Accessed March 14, 2016.
       http://mathworld.wolfram.com/Spheroid.html.
    '''
    if D == a*2:
        return pi*D**2/2 # necessary to avoid a division by zero when D == a
    D = D/2.
    D, a = min((D, a)), max((D, a))
    e1 = (1 - D**2/a**2)**0.5
    SA = (2*pi*a**2 + pi*D**2/e1*log((1+e1)/(1-e1)))/2.
    return SA


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
    SA = pi*D/2*(a**2 + (D/2)**2)**0.5
    return SA


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
    SA = pi*D/4*(a**2 + D**2)**0.5 + pi*D/2*a
    return SA


def SA_torispheroidal(D, fd, fk):
    r'''Calculates surface area of a torispherical head according to [1]_.
    Somewhat involved. Equations are adapted to be used for a full head.

    .. math::
        SA = S_1 + S_2

        S_1 = 2\pi D^2 f_d \alpha

        S_2 = 2\pi D^2 f_k\left(\alpha - \alpha_1 + (0.5 - f_k)\left(\sin^{-1}
        \left(\frac{\alpha-\alpha_2}{f_k}\right) - \sin^{-1}\left(\frac{
        \alpha_1-\alpha_2}{f_k}\right)\right)\right)

        \alpha_1 = f_d\left(1 - \sqrt{1 - \left(\frac{0.5 - f_k}{f_d-f_k}
        \right)^2}\right)

        \alpha_2 = f_d - \sqrt{f_d^2 - 2f_d f_k + f_k - 0.25}

        \alpha = \frac{a}{D_i}

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    fd : float
        Dish-radius parameter = f; fD  = dish radius []
    fk : float
        knucle-radius parameter = k; kD = knucle radius []

    Returns
    -------
    SA : float
        Surface area [m^2]

    Examples
    --------
    Example from [1]_.

    >>> SA_torispheroidal(D=2.54, fd=1.039370079, fk=0.062362205)
    6.00394283477063

    References
    ----------
    .. [1] Honeywell. "Calculate Surface Areas and Cross-sectional Areas in
       Vessels with Dished Heads". https://www.honeywellprocess.com/library/marketing/whitepapers/WP-VesselsWithDishedHeads-UniSimDesign.pdf
       Whitepaper. 2014.
    '''
    alpha_1 = fd*(1 - (1 - ((0.5-fk)/(fd-fk))**2)**0.5)
    alpha_2 = fd - (fd**2 - 2*fd*fk + fk - 0.25)**0.5
    alpha = alpha_1 # Up to top of dome
    S1 = 2*pi*D**2*fd*alpha_1
    alpha = alpha_2 # up to top of torus
    S2_sub = asin((alpha-alpha_2)/fk) - asin((alpha_1-alpha_2)/fk)
    S2 = 2*pi*D**2*fk*(alpha - alpha_1 + (0.5-fk)*S2_sub)
    SA = S1 + S2
    return SA


def SA_tank(D, L, sideA=None, sideB=None, sideA_a=0,
             sideB_a=0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None,
             full_output=False):
    r'''Calculates the surface are of a cylindrical tank with optional heads.
    In the degenerate case of being provided with only `D` and `L`, provides
    the surface area of a cylinder.

    Parameters
    ----------
    D : float
        Diameter of the cylindrical section of the tank.
    L : float
        Length of the main cylindrical section of the tank.
    sideA : string, optional
        The left (or bottom for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideB : string, optional
        The right (or top for vertical) head of the tank's type; one of
        [None, 'conical', 'ellipsoidal', 'torispherical', 'guppy', 'spherical'].
    sideA_a : float, optional
        The distance the head as specified by sideA extends down or to the left
        from the main cylindrical section
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section
    sideA_f : float, optional
        Dish-radius parameter for side A; fD  = dish radius []
    sideA_k : float, optional
        knucle-radius parameter for side A; kD = knucle radius []
    sideB_f : float, optional
        Dish-radius parameter for side B; fD  = dish radius []
    sideB_k : float, optional
        knucle-radius parameter for side B; kD = knucle radius []

    Returns
    -------
    SA : float
        Surface area of the tank [m^2]
    areas : tuple, only returned if full_output == True
        (sideA_SA, sideB_SA, lateral_SA)

    Other Parameters
    ----------------
    full_output : bool, optional
        Returns a tuple of (sideA_SA, sideB_SA, lateral_SA) if True

    Examples
    --------
    Cylinder, Spheroid, Long Cones, and spheres. All checked.

    >>> SA_tank(D=2, L=2)
    18.84955592153876
    >>> SA_tank(D=1., L=0, sideA='ellipsoidal', sideA_a=2, sideB='ellipsoidal',
    ... sideB_a=2)
    28.480278854014387
    >>> SA_tank(D=1., L=5, sideA='conical', sideA_a=2, sideB='conical',
    ... sideB_a=2)
    22.18452243965656
    >>> SA_tank(D=1., L=5, sideA='spherical', sideA_a=0.5, sideB='spherical',
    ... sideB_a=0.5)
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
        sideA_SA = SA_partial_sphere(D=D, h=sideA_a)
    elif sideA == 'torispherical':
        sideA_SA = SA_torispheroidal(D=D, fd=sideA_f, fk=sideA_k)
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
        sideB_SA = SA_partial_sphere(D=D, h=sideB_a)
    elif sideB == 'torispherical':
        sideB_SA = SA_torispheroidal(D=D, fd=sideB_f, fk=sideB_k)
    else:
        sideB_SA = pi/4*D**2 # Circle

    lateral_SA = pi*D*L

    SA = sideA_SA + sideB_SA + lateral_SA
    if full_output:
        return SA, (sideA_SA, sideB_SA, lateral_SA)
    else:
        return SA


def a_torispherical(D, f, k):
    r'''Calculates depth of a torispherical head according to [1]_.

    .. math::
        a = a_1 + a_2

        \alpha = \sin^{-1}\frac{1-2k}{2(f-k)}

        a_1 = fD(1-\cos\alpha)

        a_2 = kD\cos\alpha

    Parameters
    ----------
    D : float
        Diameter of the main cylindrical section, [m]
    f : float
        Dish-radius parameter; fD  = dish radius []
    k : float
        knucle-radius parameter ; kD = knucle radius []

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
    a = a1 + a2
    return a


def V_from_h(h, D, L, horizontal=True, sideA=None, sideB=None, sideA_a=0,
             sideB_a=0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None):
    r'''Calculates partially full volume of a vertical or horizontal tank with
    different head types according to [1]_.

    Parameters
    ----------
    h : float
        Heifht of the liquid in the tank
    D : float
        Diameter of the cylindrical section of the tank.
    L : float
        Length of the main cylindrical section of the tank.
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
        from the main cylindrical section
    sideB_a : float, optional
        The distance the head as specified by sideB extends up or to the right
        from the main cylindrical section
    sideA_f : float, optional
        Dish-radius parameter for side A; fD  = dish radius []
    sideA_k : float, optional
        knucle-radius parameter for side A; kD = knucle radius []
    sideB_f : float, optional
        Dish-radius parameter for side B; fD  = dish radius []
    sideB_k : float, optional
        knucle-radius parameter for side B; kD = knucle radius []

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
    R = D/2.
    V = 0
    if horizontal:
        # Conical case
        if sideA == 'conical':
            V += V_horiz_conical(D, L, sideA_a, h, headonly=True)
        if sideB == 'conical':
            V += V_horiz_conical(D, L, sideB_a, h, headonly=True)
        # Elliosoidal case
        if sideA == 'ellipsoidal':
            V += V_horiz_ellipsoidal(D, L, sideA_a, h, headonly=True)
        if sideB == 'ellipsoidal':
            V += V_horiz_ellipsoidal(D, L, sideB_a, h, headonly=True)
        # Guppy case
        if sideA == 'guppy':
            V += V_horiz_guppy(D, L, sideA_a, h, headonly=True)
        if sideB == 'guppy':
            V += V_horiz_guppy(D, L, sideB_a, h, headonly=True)
        # Spherical case
        if sideA == 'spherical':
            V += V_horiz_spherical(D, L, sideA_a, h, headonly=True)
        if sideB == 'spherical':
            V += V_horiz_spherical(D, L, sideB_a, h, headonly=True)
        # Torispherical case
        if sideA == 'torispherical':
            V += V_horiz_torispherical(D, L, sideA_f, sideA_k, h, headonly=True)
        if sideB == 'torispherical':
            V += V_horiz_torispherical(D, L, sideB_f, sideB_k, h, headonly=True)
        if h > D: # Must be before Af, which will raise a domain error
            raise Exception('Input height is above top of tank')
        Af = R**2*acos((R-h)/R) - (R-h)*(2*R*h - h**2)**0.5
        V += L*Af
    else:
        # Bottom head
        if sideA in ['conical', 'ellipsoidal', 'torispherical', 'spherical']:
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
                V -= V_vertical_torispherical(D, sideB_f, sideB_k, h=h2)
        if h > L + sideA_a + sideB_a:
            raise Exception('Input height is above top of tank')
    return V


class TANK(object):
    '''
    Class representing tank volumes and levels. All parameters are also
    attributes.

    Attributes
    ----------
    table : bool
        Whether or not a table of heights-volumes has been generated
    h_max : float
        Height of the tank, [m]
    V_total : float
        Total volume of the tank as calculated [m^3]
    heights : ndarray
        Array of heights between 0 and h_max, [m]
    volumes : ndarray
        Array of volumes calculated from the heights [m^3]
    A : float
        Total surface area of the tank
    A_sideA : float
        Surface area of sideA
    A_sideB : float
        Surface area of sideB
    A_lateral : float
        Surface area of the lateral side

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
    0.44209706414415373

    Surface area of a tank with a conical head:

    >>> T1 = TANK(V=10, L_over_D=0.7, sideB='conical', sideB_a=0.5)
    >>> T1.A, T1.A_sideA, T1.A_sideB, T1.A_lateral
    (24.94775907657148, 5.118555935958284, 5.497246519930003, 14.331956620683194)

    Solving for tank volumes, first horizontal, then vertical:

    >>> TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.699531057009146
    >>> TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D
    9.999999999999407
    >>> TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.69953105700979

    >>> TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.699531057009147
    >>> TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D
    9.999999999999407
    >>> TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.69953105700979
    '''
    table = False

    def __init__(self, D=None, L=None, horizontal=True,
                 sideA=None, sideB=None, sideA_a=0, sideB_a=0,
                 sideA_f=1., sideA_k=0.06, sideB_f=1., sideB_k=0.06,
                 sideA_a_ratio=0.25, sideB_a_ratio=0.25, L_over_D=None, V=None):
        '''Initilization method for a new tank. All parameters are attributes.

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
            from the main cylindrical section
        sideB_a : float, optional
            The distance the head as specified by sideB extends up or to the right
            from the main cylindrical section
        sideA_f : float, optional
            Dish-radius parameter for side A; fD  = dish radius []
        sideA_k : float, optional
            knucle-radius parameter for side A; kD = knucle radius []
        sideB_f : float, optional
            Dish-radius parameter for side B; fD  = dish radius []
        sideB_k : float, optional
            knucle-radius parameter for side B; kD = knucle radius []
        L_over_D: float, optional
            Ratio of length over diameter, used only when D and L are both
            unspecified but V is, []
        V : float, optional
            Volume of the tank; solved for if specified, using
            sideA_a_ratio/sideB_a_ratio, sideA, sideB, horizontal, and one
            of L_over_D, L, or D, [m^3]
        '''
        self.D = D
        self.L = L
        self.L_over_D = L_over_D
        self.V = V
        self.horizontal = horizontal

        self.sideA = sideA
        self.sideA_a = sideA_a
        self.sideA_f = sideA_f
        self.sideA_k = sideA_k
        self.sideA_a_ratio = sideA_a_ratio

        self.sideB = sideB
        self.sideB_a = sideB_a
        self.sideB_f = sideB_f
        self.sideB_k = sideB_k
        self.sideB_a_ratio = sideB_a_ratio

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
            self.solve_tank_for_V()
        self.set_misc()

    def set_misc(self):
        '''Set more parameters, after the tank is better defined than in the
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
        '''
        if self.D and self.L:
            # If L and D are known, get L_over_D
            self.L_over_D = self.L/self.D
        elif self.D and self.L_over_D:
            # Otherwise, if L_over_D and D are provided, get L
            self.L = self.D*self.L_over_D
        elif self.L and self.L_over_D:
            # Otherwise, if L_over_D and L are provided, get D
            self.D = self.L/self.L_over_D

        # Calculate diameter
        self.R = self.D/2.

        # If a_ratio is provided for either heads, use it.
        if self.sideA and self.D:
            if not self.sideA_a and self.sideA in ['conical', 'ellipsoidal', 'guppy', 'spherical']:
                self.sideA_a = self.D*self.sideA_a_ratio
        if self.sideB and self.D:
            if not self.sideB_a and self.sideB in ['conical', 'ellipsoidal', 'guppy', 'spherical']:
                self.sideB_a = self.D*self.sideB_a_ratio

        # Calculate a for torispherical heads
        if self.sideA == 'torispherical' and self.sideA_f and self.sideA_k:
            self.sideA_a = a_torispherical(self.D, self.sideA_f, self.sideA_k)
        if self.sideB == 'torispherical' and self.sideB_f and self.sideB_k:
            self.sideB_a = a_torispherical(self.D, self.sideB_f, self.sideB_k)

        # Calculate maximum tank height, h_max
        if self.horizontal:
            self.h_max = self.D
        else:
            self.h_max = self.L
            if self.sideA_a:
                self.h_max += self.sideA_a
            if self.sideB_a:
                self.h_max += self.sideB_a

        # Set maximum height
        self.V_total = self.V_from_h(self.h_max)

        # Set surface areas
        self.A, (self.A_sideA, self.A_sideB, self.A_lateral) = SA_tank(
        D=self.D, L=self.L, sideA=self.sideA, sideB=self.sideB, sideA_a=self.sideA_a,
        sideB_a=self.sideB_a, sideA_f=self.sideA_f, sideA_k=self.sideA_k,
        sideB_f=self.sideB_f, sideB_k=self.sideB_k,
             full_output=True)


    def V_from_h(self, h):
        r'''Method to calculate the volume of liquid in a fully defined tank
        given a specified height `h`. `h` must be under the maximum height.

        Parameters
        ----------
        h : float
            Height specified, [m]

        Returns
        -------
        V : float
            Volume of liquid in the tank up to the specified height, [m^3]
        '''
        V = V_from_h(h, self.D, self.L, self.horizontal, self.sideA, self.sideB,
                 self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k,
                 self.sideB_f, self.sideB_k)
        return V

    def h_from_V(self, V):
        r'''Method to calculate the height of liquid in a fully defined tank
        given a specified volume of liquid in it `V`. `V` must be under the
        maximum volume. If interpolation table is not yet defined, creates it
        by calling the method set_table.

        Parameters
        ----------
        V : float
            Volume of liquid in the tank up to the desired height, [m^3]

        Returns
        -------
        h : float
            Height of liquid at which the volume is as desired, [m]
        '''
        if not self.table:
            self.set_table()
        h = float(self.interp_h_from_V(V))
        return h

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
            self.heights = np.linspace(0, self.h_max, int(self.h_max/dx)+1)
        else:
            self.heights = np.linspace(0, self.h_max, n)
        self.volumes = [self.V_from_h(h) for h in self.heights]
        self.interp_h_from_V = interp1d(self.volumes, self.heights)
        self.table = True


    def _V_solver_error(self, Vtarget, D, L, horizontal, sideA, sideB, sideA_a,
                       sideB_a, sideA_f, sideA_k, sideB_f, sideB_k,
                       sideA_a_ratio, sideB_a_ratio):
        '''Function which uses only the variables given, and the TANK
        class itself, to determine how far from the desired volume, Vtarget,
        the volume produced by the specified parameters in a new TANK instance
        is. Should only be used by solve_tank_for_V method.
        '''
        a = TANK(D=float(D), L=float(L), horizontal=horizontal, sideA=sideA, sideB=sideB,
                 sideA_a=sideA_a, sideB_a=sideB_a, sideA_f=sideA_f,
                 sideA_k=sideA_k, sideB_f=sideB_f, sideB_k=sideB_k,
                 sideA_a_ratio=sideA_a_ratio, sideB_a_ratio=sideB_a_ratio)
        error = abs(Vtarget - a.V_total)
        return error


    def solve_tank_for_V(self):
        '''Method which is called to solve for tank geometry when a certain
        volume is specified. Will be called by the __init__ method if V is set.

        Notes
        -----
        Raises an error if L and either of sideA_a or sideB_a are specified;
        these can only be set once D is known.
        Raises an error if more than one of D, L, or L_over_D are specified.
        Raises an error if the head ratios are not provided.

        Calculates initial guesses assuming no heads are present, and then uses
        fsolve to determine the correct dimentions for the tank.

        Tested, but bugs and limitations are expected here.
        '''
        if self.L and (self.sideA_a or self.sideB_a):
            raise Exception('Cannot specify head sizes when solving for V')
        if (self.D and self.L) or (self.D and self.L_over_D) or (self.L and self.L_over_D):
            raise Exception('Only one of D, L, or L_over_D can be specified\
            when solving for V')
        if ((self.sideA and not self.sideA_a_ratio) or (self.sideB and not self.sideB_a_ratio)):
            raise Exception('When heads are specified, head parameter ratios are required')

        if self.D:
            # Iterate until L is appropriate
            solve_L = lambda L: self._V_solver_error(self.V, self.D, L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            Lguess = self.V/(pi/4*self.D**2)
            self.L = float(fsolve(solve_L, Lguess))
        elif self.L:
            # Iterate until D is appropriate
            solve_D = lambda D: self._V_solver_error(self.V, D, self.L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            Dguess = (4*self.V/pi/self.L)**0.5
            self.D = float(fsolve(solve_D, Dguess))
        else:
            # Use L_over_D until L and D are appropriate
            Lguess = (4*self.V*self.L_over_D**2/pi)**(1/3.)
            solve_L_D = lambda L: self._V_solver_error(self.V, L/self.L_over_D, L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            self.L = float(fsolve(solve_L_D, Lguess))
            self.D = self.L/self.L_over_D

#test = TANK(D=10., L=100., horizontal=True, sideA='spherical', sideB='ellipsoidal',
#            sideA_a=2., sideB_a=2.)
#print test.V_total
#print test.V_from_h(9.2)
#
#test = TANK(D=10., L=100., horizontal=True, sideA='conical', sideB='conical',
#            sideA_a=2., sideB_a=2.)
##
#import matplotlib.pyplot as plt
#print test.heights
##
#plt.plot(test.heights, test.volumes)
#plt.show()
#print [TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L]
#print [TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D]
#print [TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L]
##
#print 'hi'
#print [TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L]
#print [TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D]
#print [TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L]
#
#print [TANK(D=1.2, L=4, horizontal=False).h_from_V(.5)]