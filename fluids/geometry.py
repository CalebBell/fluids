# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016-2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import pi, sin, cos, tan, asin, acos, atan, acosh, log
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.special import ellipe

__all__ = ['TANK', 'HelicalCoil', 'PlateExchanger', 'SA_partial_sphere', 
           'V_partial_sphere', 'V_horiz_conical',
           'V_horiz_ellipsoidal', 'V_horiz_guppy', 'V_horiz_spherical',
           'V_horiz_torispherical', 'V_vertical_conical',
           'V_vertical_ellipsoidal', 'V_vertical_spherical',
           'V_vertical_torispherical', 'V_vertical_conical_concave',
           'V_vertical_ellipsoidal_concave', 'V_vertical_spherical_concave',
           'V_vertical_torispherical_concave', 'a_torispherical',
           'SA_ellipsoidal_head', 'SA_conical_head', 'SA_guppy_head',
           'SA_torispheroidal', 'V_from_h', 'SA_tank', 'sphericity', 
           'aspect_ratio', 'circularity', 'A_cylinder', 'V_cylinder', 
           'A_hollow_cylinder', 'V_hollow_cylinder', 
           'A_multiple_hole_cylinder', 'V_multiple_hole_cylinder']


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
    r = D*0.5
    a = (h*(2.*r - h))**0.5
    return pi*(a*a + h*h)


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
    r = 0.5*D
    a = (h*(2.*r - h))**0.5
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
    Af = R*R*acos((R-h)/R) - (R-h)*(2*R*h - h*h)**0.5
    M = abs((R-h)/R)
    if h == R:
        Vf = a*R*R/3.*pi
    else:
        K = acos(M) + M*M*M*acosh(1./M) - 2.*M*(1.-M*M)**0.5
        if 0. <= h < R:
            Vf = 2.*a*R*R/3*K
        elif R < h <= 2*R:
            Vf = 2.*a*R*R/3*(pi - K)
    if headonly:
        Vf = 0.5*Vf
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
    R = 0.5*D
    Af = R*R*acos((R-h)/R) - (R-h)*(2*R*h - h*h)**0.5
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
    R = 0.5*D
    Af = R*R*acos((R-h)/R) - (R-h)*(2.*R*h - h*h)**0.5
    Vf = 2.*a*R*R/3.*acos(1. - h/R) + 2.*a/9./R*(2*R*h - h**2)**0.5*(2*h - 3*R)*(h + R)
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
        Dish-radius parameter; fD  = dish radius [1/m]
    k : float
        knuckle-radius parameter ; kD = knuckle radius [1/m]
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
        # No analytical integral available in MP
        n = R - k*D + (k**2*D**2 - x**2)**0.5
        ans = n**2*asin((n**2-w**2)**0.5/n) - w*(n**2 - w**2)**0.5
        return ans
    def V2_toint(x, w):
        # No analytical integral available in MP
        n = R - k*D + (k**2*D**2 - x**2)**0.5
        ans = n**2*(acos(w/n) - acos(g/n)) - w*(n**2 - w**2)**0.5 + g*(n**2-g**2)**0.5
        return ans
    def V3_toint(x):
        # There is an analytical integral in MP, but for all cases we seem to 
        # get ZeroDivisionError: 0.0 cannot be raised to a negative power
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
        Dish-radius parameter; fD  = dish radius [1/m]
    k : float
        knuckle-radius parameter ; kD = knuckle radius [1/m]
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
        Dish-radius parameter; fD  = dish radius [1/m]
    k : float
        knuckle-radius parameter ; kD = knuckle radius [1/m]
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
    return (2*pi*a**2 + pi*D**2/e1*log((1+e1)/(1-e1)))/2.


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
    return pi*D/2*(a**2 + (D/2)**2)**0.5


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
    return pi*D/4*(a**2 + D**2)**0.5 + pi*D/2*a


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
        Dish-radius parameter = f; fD  = dish radius [1/m]
    fk : float
        knuckle-radius parameter = k; kD = knuckle radius [1/m]

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
    return S1 + S2


def SA_tank(D, L, sideA=None, sideB=None, sideA_a=0,
             sideB_a=0, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None,
             full_output=False):
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
        Dish-radius parameter; fD  = dish radius [1/m]
    k : float
        knuckle-radius parameter ; kD = knuckle radius [1/m]

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
        Dish-radius parameter for side A; fD  = dish radius [1/m]
    sideA_k : float, optional
        knuckle-radius parameter for side A; kD = knuckle radius [1/m]
    sideB_f : float, optional
        Dish-radius parameter for side B; fD  = dish radius [1/m]
    sideB_k : float, optional
        knuckle-radius parameter for side B; kD = knuckle radius [1/m]

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
    if sideA not in [None, 'conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy']:
        raise Exception('Unspoorted head type for side A')
    if sideB not in [None, 'conical', 'ellipsoidal', 'torispherical', 'spherical', 'guppy']:
        raise Exception('Unspoorted head type for side B')
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
    '''Class representing tank volumes and levels. All parameters are also
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
    L_over_D : float, optional
        Ratio of length over diameter, used only when D and L are both
        unspecified but V is, [-]
    V : float, optional
        Volume of the tank; solved for if specified, using
        sideA_a_ratio/sideB_a_ratio, sideA, sideB, horizontal, and one
        of L_over_D, L, or D, [m^3]

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
        Array of volumes calculated from the heights, [m^3]
    A : float
        Total surface area of the tank, [m^2]
    A_sideA : float
        Surface area of sideA, [m^2]
    A_sideB : float
        Surface area of sideB, [m^2]
    A_lateral : float
        Surface area of the lateral side, [m^2]
        
    Notes
    -----
    For torpsherical tank heads, the following `f` and `k` parameters are used
    in standards. The default is ASME F&D	.
        
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
    (24.94775907657148, 5.118555935958284, 5.497246519930003, 14.331956620683192)

    Solving for tank volumes, first horizontal, then vertical:

    >>> TANK(D=10., horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.699531057009147
    >>> TANK(L=4.69953105701, horizontal=True, sideA='conical', sideB='conical', V=500).D
    9.999999999999407
    >>> TANK(L_over_D=0.469953105701, horizontal=True, sideA='conical', sideB='conical', V=500).L
    4.69953105700979

    >>> TANK(D=10., horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.699531057009147
    >>> TANK(L=4.69953105701, horizontal=False, sideA='conical', sideB='conical', V=500).D
    9.999999999999407
    >>> TANK(L_over_D=0.469953105701, horizontal=False, sideA='conical', sideB='conical', V=500).L
    4.699531057009791
    '''
    table = False

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
                 sideA=None, sideB=None, sideA_a=0, sideB_a=0,
                 sideA_f=1., sideA_k=0.06, sideB_f=1., sideB_k=0.06,
                 sideA_a_ratio=0.25, sideB_a_ratio=0.25, L_over_D=None, V=None):
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
            self.L = float(newton(solve_L, Lguess))
        elif self.L:
            # Iterate until D is appropriate
            solve_D = lambda D: self._V_solver_error(self.V, D, self.L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            Dguess = (4*self.V/pi/self.L)**0.5
            self.D = float(newton(solve_D, Dguess))
        else:
            # Use L_over_D until L and D are appropriate
            Lguess = (4*self.V*self.L_over_D**2/pi)**(1/3.)
            solve_L_D = lambda L: self._V_solver_error(self.V, L/self.L_over_D, L, self.horizontal, self.sideA, self.sideB, self.sideA_a, self.sideB_a, self.sideA_f, self.sideA_k, self.sideB_f, self.sideB_k, self.sideA_a_ratio, self.sideB_a_ratio)
            self.L = float(newton(solve_L_D, Lguess))
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
    Do : float, optional
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
        :math:`\alpha = \arctan \left(\frac{p_t}{\pi D_o}\right)`, [-]
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
        if H_total:
            H = H_total - Dt
        if Do_total:
            Do = Do_total - Dt
        self.Do = Do
        self.Dt = Dt
        self.Do_total = self.Do+self.Dt
        if N and pitch:
            self.N = N
            self.pitch = pitch
            self.H = N*pitch
        elif N and H:
            self.N = N
            self.H = H
            self.pitch = self.H/N
            if self.pitch < self.Dt:
                raise Exception('Pitch is too small - tubes are colliding; maximum number of spirals is %f.'%(self.H/self.Dt))
        elif H and pitch:
            self.pitch = pitch
            self.H = H
            self.N = self.H/self.pitch
            if self.pitch < self.Dt:
                raise Exception('Pitch is too small - tubes are colliding; pitch must be larger than tube diameter.')
        self.H_total = self.Dt + self.H
        
        if self.Dt > self.Do:
            raise Exception('Tube diameter is larger than helix outer diameter - not feasible.')
        
        self.tube_circumference = pi*self.Do
        self.tube_length = ((self.tube_circumference*self.N)**2 + self.H**2)**0.5
        self.surface_area = self.tube_length*pi*self.Dt
        #print(pi*self.tube_length*self.Dt) == surface_area
        self.helix_angle = atan(self.pitch/(pi*self.Do))
        self.curvature = self.Dt/self.Do/(1. + 4*pi**2*tan(self.helix_angle)**2)
        #print(self.N*pi*self.Do/cos(self.helix_angle)) # Confirms the length with another formula
        self.total_inlet_area = pi/4.*self.Dt**2
        self.total_volume = self.total_inlet_area*self.tube_length
        
        if Di:
            self.Di = Di
            self.inner_surface_area = self.tube_length*pi*self.Di
            self.inlet_area = pi/4.*self.Di**2
            self.inner_volume = self.inlet_area*self.tube_length
            self.annulus_area = self.total_inlet_area - self.inlet_area
            self.annulus_volume = self.total_volume - self.inner_volume


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
    chevron_angle : float or tuple(2), optional
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use a tuple
        of the two angles for that situation [degrees]
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
        The area for the fluid to flow in, :math:`W\cdot b` [m^2]
    channels : int
        The number of plates minus one, [-]
    channels_per_fluid : int
        Half the number of total channels, [-]
    plate_exchanger_identifier : str
        Identifying string in format 'L' + wavelength + 'A' + amplitude + 'B'
        + chevron angle-chevron angle

    Notes
    -----
    Only wavelength and amplitude are required as inputs to this function.

    Examples
    --------
    >>> PlateExchanger(amplitude=5E-4, wavelength=3.7E-3, length=1.2, width=.3,
    ... d_port=.05, plates=51)
    <Plate heat exchanger, amplitude=0.0005 m, wavelength=0.0037 m, chevron_angles=45/45 degrees, area enhancement factor=1.16118620345, width=0.3 m, length=1.2 m, port diameter=0.05 m, heat transfer area=20.4833246289 m^2, 51 plates>

    References
    ----------
    .. [1] Amalfi, Raffaele L., Farzad Vakili-Farahani, and John R. Thome. 
       "Flow Boiling and Frictional Pressure Gradients in Plate Heat Exchangers.
       Part 1: Review and Experimental Database." International Journal of 
       Refrigeration 61 (January 2016): 166-84. doi:10.1016/j.ijrefrig.2015.07.010.
    '''
    def __repr__(self):  # pragma : no cover
        s = '<Plate heat exchanger, amplitude=%s m, wavelength=%s m, \
chevron_angles=%s degrees, area enhancement factor=%s' %(self.a, self.wavelength, '/'.join([str(i) for i in self.chevron_angles]), self.plate_enlargement_factor)
        if self.width and self.length:
            s += ', width=%s m, length=%s m' %(self.width, self.length)
        if self.d_port:
            s += ', port diameter=%s m' %(self.d_port)
        if self.plates:
            s += ', heat transfer area=%s m^2, %s plates>' %(self.A_heat_transfer, self.plates)
        else: 
            s += '>'
        return s
        
    @property
    def plate_exchanger_identifier(self):
        '''Method to create an identifying string in format 'L' + wavelength + 
        'A' + amplitude + 'B' + chevron angle-chevron angle. Wavelength and 
        amplitude are specified in units of mm and rounded to two decimal places.
        '''
        s = ('L' + str(round(self.wavelength*1000, 2))
             + 'A' + str(round(self.amplitude*1000, 2))
             + 'B' + '-'.join([str(i) for i in self.chevron_angles]))
        return s
    
    @staticmethod
    def plate_enlargement_factor_analytical(amplitude, wavelength):
        r'''Calculates the enhancement factor of the sinusoidal waves of the
        plate heat exchanger. This is the multiplier for the flat plate area
        to obtain the actual area available for heat transfer. Obtained from
        the following integral:
    
        .. math::
            \phi = \frac{\text{Effective area}}{\text{Projected area}} 
            = \frac{\int_0^\lambda\sqrt{1 + \left(\frac{\gamma\pi}{2}\right)^2
            \cos^2\left(\frac{2\pi}{\lambda}x\right)}dx}{\lambda}
            
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
        >>> PlateExchanger.plate_enlargement_factor_analytical(amplitude=5E-4, wavelength=3.7E-3)
        1.1611862034509677
        '''
        b = 2.*amplitude
        return 2.*ellipe(-b*b*pi*pi/(wavelength*wavelength))/pi
    
    def __init__(self, amplitude, wavelength, chevron_angle=45, width=None,
                 length=None, thickness=None, d_port=None, plates=None):
        self.amplitude = self.a = amplitude # half a sine wave's height
        self.b = 2*self.amplitude # Used in some models. From a flat plate, a press goes down this far into the plate. Also called the hot and cold gap
        self.wavelength = self.pitch = wavelength # self.lambda
        if isinstance(chevron_angle, tuple):
            self.chevron_angles = chevron_angle
            self.chevron_angle = self.beta = 0.5*(chevron_angle[0]+chevron_angle[1])
        else:
            self.chevron_angle = self.beta = chevron_angle # between 0 and 90
            self.chevron_angles = (chevron_angle, chevron_angle)
            
        self.inclination_angle = 90 - self.chevron_angle # Used in some definitions instead

        self.plate_corrugation_aspect_ratio = self.gamma = 4*self.a/self.wavelength
        self.plate_enlargement_factor = self.plate_enlargement_factor_analytical(self.amplitude, self.wavelength)
        
        self.D_eq = 4*self.amplitude # Equivalent diameter for inter-plate spacing
        self.D_hydraulic = 4*self.amplitude/self.plate_enlargement_factor # Get better results when correlations use this
        
        self.width = width
        self.length = length
        self.thickness = thickness
        self.d_port = d_port
        self.plates = plates
                
        if d_port and length:
            self.length_port = self.length + self.d_port # port center to port center along the direction of flow
            # There is another larger length as well, including both port diameters
        if width and length:
            self.A_plate_surface = self.length*self.width*self.plate_enlargement_factor # use this in Q = UAdT
            if plates:
                self.A_heat_transfer = (self.plates-2)*self.A_plate_surface # the two outermost sides aren't used
        if width: 
            self.A_channel_flow = self.width*self.b # Use this to get G, kg/s/m^2
        if plates:
            self.channels = self.plates - 1
            self.channels_per_fluid = 0.5*self.channels




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

