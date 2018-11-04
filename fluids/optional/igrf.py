# -*- coding: utf-8 -*-
"""
igrf module
==============
Vendorized version from:
https://github.com/zzyztyy/pyIGRF

Copyright (c) 2018 zzyztyy

The rational for not including this library as a strict dependency is that
the module is small and straightforward and various changes were made to make 
it run without NumPy and fast under PyPy.

.. moduleauthor :: zzyztyy <2375672032@qq.com>

The copyright notice (MIT) is as follows:
    
MIT License

Copyright (c) 2018 zzyztyy

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

"""

from __future__ import division
from math import pi, sin, cos, atan2, sqrt
import os


FACT = 180./pi

def loadCoeffs(filename):
    """
    load igrf12 coeffs from file
    :param filename: file which save coeffs (str)
    :return: g and h list one by one (list(float))
    """
    gh = []
    gh2arr = []
    with open(filename) as f:
        text = f.readlines()
        for a in text:
            if a[:2] == 'g ' or a[:2] == 'h ':
                b = a.split()[3:]
                b = [float(x) for x in b]
                gh2arr.append(b)
        # Transpose the matrix
        gh2arr = [[gh2arr[j][i] for j in range(len(gh2arr))] for i in range(len(gh2arr[0])) ]
        
        N = len(gh2arr)
        for i in range(N):
            if i < 19:
                for j in range(120):
                    gh.append(gh2arr[i][j])
            else:
                for p in gh2arr[i]:
                    gh.append(p)
        gh.append(0)
        return gh


coeff_path = os.path.join(os.path.dirname(__file__), 'igrf12coeffs.txt')

gh = loadCoeffs(coeff_path)


def getCoeffs(date):  # pragma: no cover
    """
    Not used.
    
    :param gh: list from loadCoeffs
    :param date: float
    :return: list: g, list: h
    """
    if date < 1900.0 or date > 2025.0:
        raise ValueError("Date must be in the range 1900.0 <= date <= 2025.0; date was %s" %(date))
        return [], []
    elif date >= 2015.0:
        if date > 2020.0:
            # not adapt for the model but can calculate
            print('This version of the IGRF is intended for use up to 2020.0.')
            print('values for ' + str(date) + ' will be computed but may be of reduced accuracy')
        t = date - 2015.0
        tc = 1.0
        #     pointer for last coefficient in pen-ultimate set of MF coefficients...
        ll = 3060
        nmx = 13
        nc = nmx * (nmx + 2)
    else:
        t = 0.2 * (date - 1900.0)
        ll = int(t)
        t = t - ll
        #     SH models before 1995.0 are only to degree 10
        if date < 1995.0:
            nmx = 10
            nc = nmx * (nmx + 2)
            ll = nc * ll
        else:
            nmx = 13
            nc = nmx * (nmx + 2)
            ll = round(0.2 * (date - 1995.0))
            #     19 is the number of SH models that extend to degree 10
            ll = 120 * 19 + nc * ll
        tc = 1.0 - t

    g, h = [], []
    temp = ll-1
    for n in range(nmx+1):
        g.append([])
        h.append([])
        if n == 0:
            g[0].append(None)
        for m in range(n+1):
            if m != 0:
                g[n].append(tc*gh[temp] + t*gh[temp+nc])
                h[n].append(tc*gh[temp+1] + t*gh[temp+nc+1])
                temp += 2
            else:
                g[n].append(tc*gh[temp] + t*gh[temp+nc])
                h[n].append(None)
                temp += 1
    return g, h
    

def geodetic2geocentric(theta, alt):
    """
    Conversion from geodetic to geocentric coordinates by using the WGS84 spheroid.
    :param theta: colatitude (float, rad)
    :param alt: altitude (float, km)
    :return gccolat: geocentric colatitude (float, rad)
            d: gccolat minus theta (float, rad)
            r: geocentric radius (float, km)
    """
    ct = cos(theta)
    st = sin(theta)
    a2 = 40680631.6
    b2 = 40408296.0
    one = a2 * st * st
    two = b2 * ct * ct
    three = one + two
    rho = sqrt(three)
    r = sqrt(alt * (alt + 2.0 * rho) + (a2 * one + b2 * two) / three)
    cd = (alt + rho) / r
    sd = (a2 - b2) / rho * ct * st / r
    one = ct
    ct = ct * cd - st * sd
    st = st * cd + one * sd
    gccolat = atan2(st, ct)
    d = atan2(sd, cd)
    return gccolat, d, r


def igrf12syn(isv, date, itype, alt, lat, elong):
    """Commenf from NASA (public domain)
    https://www.ngdc.noaa.gov/IAGA/vmod/igrf12.f
    
     This is a synthesis routine for the 12th generation IGRF as agreed
     in December 2014 by IAGA Working Group V-MOD. It is valid 1900.0 to
     2020.0 inclusive. Values for dates from 1945.0 to 2010.0 inclusive are
     definitive, otherwise they are non-definitive.
   INPUT
     isv   = 0 if main-field values are required
     isv   = 1 if secular variation values are required
     date  = year A.D. Must be greater than or equal to 1900.0 and
             less than or equal to 2025.0. Warning message is given
             for dates greater than 2020.0. Must be double precision.
     itype = 1 if geodetic (spheroid)
     itype = 2 if geocentric (sphere)
     alt   = height in km above sea level if itype = 1
           = distance from centre of Earth in km if itype = 2 (>3485 km)
     lat = latitude (-90~90)
     elong = east-longitude (0-360)
     alt, colat and elong must be double precision.
   OUTPUT
     x     = north component (nT) if isv = 0, nT/year if isv = 1
     y     = east component (nT) if isv = 0, nT/year if isv = 1
     z     = vertical component (nT) if isv = 0, nT/year if isv = 1
     f     = total intensity (nT) if isv = 0, rubbish if isv = 1
     To get the other geomagnetic elements (D, I, H and secular
     variations dD, dH, dI and dF) use routines ptoc and ptocsv.
     Adapted from 8th generation version to include new maximum degree for
     main-field models for 2000.0 and onwards and use WGS84 spheroid instead
     of International Astronomical Union 1966 spheroid as recommended by IAGA
     in July 2003. Reference radius remains as 6371.2 km - it is NOT the mean
     radius (= 6371.0 km) but 6371.2 km is what is used in determining the
     coefficients. Adaptation by Susan Macmillan, August 2003 (for
     9th generation), December 2004, December 2009  \ December 2014.
     Coefficients at 1995.0 incorrectly rounded (rounded up instead of
     to even) included as these are the coefficients published in Excel
     spreadsheet July 2005.
    """

    p, q, cl, sl = [0.] * 105, [0.] * 105, [0.] * 13, [0.] * 13

    # set initial values
    x, y, z = 0., 0., 0.

    if date < 1900.0 or date > 2025.0:
        f = 1.0
#         print('This subroutine will not work with a date of ' + str(date))
#         print('Date must be in the range 1900.0 <= date <= 2025.0')
#         print('On return f = 1.0, x = y = z = 0')
        return x, y, z, f
    elif date >= 2015.0:
        if date > 2020.0:
            pass
            # not adapt for the model but can calculate
#             print('This version of the IGRF is intended for use up to 2020.0.')
#             print('values for ' + str(date) + ' will be computed but may be of reduced accuracy')
        t = date - 2015.0
        tc = 1.0
        if isv == 1:
            t = 1.0
            tc = 0.0
        #     pointer for last coefficient in pen-ultimate set of MF coefficients...
        ll = 3060
        nmx = 13
        nc = nmx * (nmx + 2)
        kmx = (nmx + 1) * (nmx + 2) / 2
    else:
        t = 0.2 * (date - 1900.0)
        ll = int(t)
        t = t - ll
        #     SH models before 1995.0 are only to degree 10
        if date < 1995.0:
            nmx = 10
            nc = nmx * (nmx + 2)
            ll = nc * ll
            kmx = (nmx + 1) * (nmx + 2) / 2
        else:
            nmx = 13
            nc = nmx * (nmx + 2)
            ll = round(0.2 * (date - 1995.0))
            #     19 is the number of SH models that extend to degree 10
            ll = 120 * 19 + nc * ll
            kmx = (nmx + 1) * (nmx + 2) / 2
        tc = 1.0 - t
        if isv == 1:
            tc = -0.2
            t = 0.2

    colat = 90-lat
    r = alt
    one = colat / FACT
    ct = cos(one)
    st = sin(one)
    one = elong / FACT
    cl[0] = cos(one)
    sl[0] = sin(one)
    cd = 1.0
    sd = 0.0
    l = 1
    m = 1
    n = 0
    if itype != 2:
        gclat, gclon, r = geodetic2geocentric(atan2(st, ct), alt)
        ct, st = cos(gclat), sin(gclat)
        cd, sd = cos(gclon), sin(gclon)
    ratio = 6371.2 / r
    rr = ratio * ratio

    #     computation of Schmidt quasi-normal coefficients p and x(=q)
    p[0] = 1.0
    p[2] = st
    q[0] = 0.0
    q[2] = ct

    fn, gn = n, n-1
    for k in range(2, int(kmx)+1):
        if n < m:
            m = 0
            n = n + 1
            rr = rr * ratio
            fn = n
            gn = n - 1

        fm = m
        if m != n:
            gmm = m * m
            one = sqrt(fn * fn - gmm)
            two = sqrt(gn * gn - gmm) / one
            three = (fn + gn) / one
            i = k - n
            j = i - n + 1
            p[k - 1] = three * ct * p[i - 1] - two * p[j - 1]
            q[k - 1] = three * (ct * q[i - 1] - st * p[i - 1]) - two * q[j - 1]
        else:
            if k != 3:
                one = sqrt(1.0 - 0.5 / fm)
                j = k - n - 1
                p[k-1] = one * st * p[j-1]
                q[k-1] = one * (st * q[j-1] + ct * p[j-1])
                cl[m-1] = cl[m - 2] * cl[0] - sl[m - 2] * sl[0]
                sl[m-1] = sl[m - 2] * cl[0] + cl[m - 2] * sl[0]
        #     synthesis of x, y and z in geocentric coordinates
        lm = ll + l
        # print('g', n, m, k, gh[int(lm-1)], gh[int(lm + nc-1)])
        one = (tc * gh[int(lm-1)] + t * gh[int(lm + nc-1)]) * rr
        if m == 0:
            x = x + one * q[k - 1]
            z = z - (fn + 1.0) * one * p[k - 1]
            l = l + 1
        else:
            # print('h', n, m, k, gh[int(lm)], gh[int(lm + nc)])
            two = (tc * gh[int(lm)] + t * gh[int(lm + nc)]) * rr
            three = one * cl[m-1] + two * sl[m-1]
            x = x + three * q[k-1]
            z = z - (fn + 1.0) * three * p[k-1]
            if st == 0.0:
                y = y + (one * sl[m - 1] - two * cl[m - 1]) * q[k - 1] * ct
            else:
                y = y + (one * sl[m-1] - two * cl[m-1]) * fm * p[k-1] / st
            l = l + 2
        m = m+1

    #     conversion to coordinate system specified by itype
    one = x
    x = x * cd + z * sd
    z = z * cd - one * sd
    f = sqrt(x * x + y * y + z * z)
    #
    return x, y, z, f
    



def igrf_value(lat, lon, alt=0., year=2005.):  # pragma: no cover
    """
    :return
         D is declination (+ve east)
         I is inclination (+ve down)
         H is horizontal intensity
         X is north component
         Y is east component
         Z is vertical component (+ve down)
         F is total intensity
    """
    X, Y, Z, F = igrf12syn(0, year, 1, alt, lat, lon)
    D = FACT * atan2(Y, X)
    H = sqrt(X * X + Y * Y)
    I = FACT * atan2(Z, H)
    return D, I, H, X, Y, Z, F


def igrf_variation(lat, lon, alt=0., year=2005):
    """
         Annual variation
         D is declination (+ve east)
         I is inclination (+ve down)
         H is horizontal intensity
         X is north component
         Y is east component
         Z is vertical component (+ve down)
         F is total intensity
    """
    X, Y, Z, F = igrf12syn(0, year, 1, alt, lat, lon)
    H = sqrt(X * X + Y * Y)
    DX, DY, DZ, DF = igrf12syn(1, year, 1, alt, lat, lon)
    DD = (60.0 * FACT * (X * DY - Y * DX)) / (H * H)
    DH = (X * DX + Y * DY) / H
    DS = (60.0 * FACT * (H * DZ - Z * DH)) / (F * F)
    DF = (H * DH + Z * DZ) / F
    return DD, DS, DH, DX, DY, DZ, DF


def igrf_values(lat, lon, alt=0., year=2005.):  # pragma: no cover
    '''Combination of "igrf_value" and "igrf_variation" into one file.
    '''
    # from igrf_value
    X, Y, Z, F = igrf12syn(0, year, 1, alt, lat, lon)
    D = FACT * atan2(Y, X)
    H = sqrt(X * X + Y * Y)
    I = FACT * atan2(Z, H)

    DX, DY, DZ, DF = igrf12syn(1, year, 1, alt, lat, lon)
    DD = (60.0 * FACT * (X * DY - Y * DX)) / (H * H)
    DH = (X * DX + Y * DY) / H
    DS = (60.0 * FACT * (H * DZ - Z * DH)) / (F * F)
    DF = (H * DH + Z * DZ) / F
    
    return (D, I, H, X, Y, Z, F), (DD, DS, DH, DX, DY, DZ, DF)

