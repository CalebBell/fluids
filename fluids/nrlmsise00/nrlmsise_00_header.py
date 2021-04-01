# -*- coding: utf-8 -*-
"""
12/19/2013
Author: Joshua Milas
Python Version: 3.3.2

The NRLMSISE-00 model 2001 ported to python
Based off of Dominik Brodowski 20100516 version available here
http://www.brodo.de/english/pub/nrlmsise/

This is the header of the program that contains all the classes

The MIT License (MIT)

Copyright (c) 2016 Joshua Milas <Josh.Milas@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/* -------------------------------------------------------------------- */
/* ---------  N R L M S I S E - 0 0    M O D E L    2 0 0 1  ---------- */
/* -------------------------------------------------------------------- */

/* This file is part of the NRLMSISE-00  C source code package - release
 * 20041227
 *
 * The NRLMSISE-00 model was developed by Mike Picone, Alan Hedin, and
 * Doug Drob. They also wrote a NRLMSISE-00 distribution package in
 * FORTRAN which is available at
 * http://uap-www.nrl.navy.mil/models_web/msis/msis_home.htm
 *
 * Dominik Brodowski implemented and maintains this C version. You can
 * reach him at mail@brodo.de. See the file "DOCUMENTATION" for details,
 * and check http://www.brodo.de/english/pub/nrlmsise/index.html for
 * updated releases of this package.
 */
"""
from __future__ import division
__all__ = ['nrlmsise_flags', 'ap_array', 'nrlmsise_input', 'nrlmsise_output']


#/* ------------------------------------------------------------------- */
#/* ------------------------------- INPUT ----------------------------- */
#/* ------------------------------------------------------------------- */


class nrlmsise_flags(object):
    """
 *   Switches: to turn on and off particular variations use these switches.
 *   0 is off, 1 is on, and 2 is main effects off but cross terms on.
 *
 *   Standard values are 0 for switch 0 and 1 for switches 1 to 23. The
 *   array "switches" needs to be set accordingly by the calling program.
 *   The arrays sw and swc are set internally.
 *
 *   switches[i]:
 *    i - explanation
 *   -----------------
 *    0 - output in centimeters instead of meters
 *    1 - F10.7 effect on mean
 *    2 - time independent
 *    3 - symmetrical annual
 *    4 - symmetrical semiannual
 *    5 - asymmetrical annual
 *    6 - asymmetrical semiannual
 *    7 - diurnal
 *    8 - semidiurnal
 *    9 - daily ap [when this is set to -1 (!) the pointer
 *                  ap_a in struct nrlmsise_input must
 *                  point to a struct ap_array]
 *   10 - all UT/long effects
 *   11 - longitudinal
 *   12 - UT and mixed UT/long
 *   13 - mixed AP/UT/LONG
 *   14 - terdiurnal
 *   15 - departures from diffusive equilibrium
 *   16 - all TINF var
 *   17 - all TLB var
 *   18 - all TN1 var
 *   19 - all S var
 *   20 - all TN2 var
 *   21 - all NLB var
 *   22 - all TN3 var
 *   23 - turbo scale height var
 """
    def __init__(self):
        self.switches = [0.0]*24
        self.sw = [0.0]*24
        self.swc = [0.0]*24


class ap_array(object):
    """
 * Array containing the following magnetic values:
 *   0 : daily AP
 *   1 : 3 hr AP index for current time
 *   2 : 3 hr AP index for 3 hrs before current time
 *   3 : 3 hr AP index for 6 hrs before current time
 *   4 : 3 hr AP index for 9 hrs before current time
 *   5 : Average of eight 3 hr AP indices from 12 to 33 hrs
 *           prior to current time
 *   6 : Average of eight 3 hr AP indices from 36 to 57 hrs
 *           prior to current time
 """
    def __init__(self):
        self.a = [0.0]*7


class nrlmsise_input(object):
    """
/*
 *   NOTES ON INPUT VARIABLES:
 *      UT, Local Time, and Longitude are used independently in the
 *      model and are not of equal importance for every situation.
 *      For the most physically realistic calculation these three
 *      variables should be consistent (lst=sec/3600 + g_long/15).
 *      The Equation of Time departures from the above formula
 *      for apparent local time can be included if available but
 *      are of minor importance.
 *
 *      f107 and f107A values used to generate the model correspond
 *      to the 10.7 cm radio flux at the actual distance of the Earth
 *      from the Sun rather than the radio flux at 1 AU. The following
 *      site provides both classes of values:
 *      ftp://ftp.ngdc.noaa.govS/STP/SOLAR_DATA/SOLAR_RADIO/FLUX/
 *
 *      f107, f107A, and ap effects are neither large nor well
 *      established below 80 km and these parameters should be set to
 *      150., 150., and 4. respectively.
 */
 """
    def __init__(self, year=0, doy=0, sec=0.0, alt=0.0, g_lat=0.0, g_long=0.0,
                 lst=0.0, f107A=0.0, f107=0.0, ap=0.0, ap_a=None):
        self.year = year #/* year, currently ignored */
        self.doy = doy #/* day of year */
        self.sec = sec #/* seconds in day (UT) */
        self.alt = alt #/* altitude in kilometes */
        self.g_lat = g_lat #/* geodetic latitude */
        self.g_long = g_long #/* geodetic longitude */
        self.lst = lst #/* local apparent solar time (hours), see note above */
        self.f107A = f107A #/* 81 day average of F10.7 flux (centered on doy) */
        self.f107 = f107 #/* daily F10.7 flux for previous day */
        self.ap = ap #/* magnetic index(daily) */
        self.ap_a = ap_a #/* see above */ Set as none for an idiot check
                            #set flags.switches[9] = -1 to use this


#/* ------------------------------------------------------------------- */
#/* ------------------------------ OUTPUT ----------------------------- */
#/* ------------------------------------------------------------------- */

class nrlmsise_output(object):
    """
/*
 *   OUTPUT VARIABLES:
 *      d[0] - HE NUMBER DENSITY(CM-3)
 *      d[1] - O NUMBER DENSITY(CM-3)
 *      d[2] - N2 NUMBER DENSITY(CM-3)
 *      d[3] - O2 NUMBER DENSITY(CM-3)
 *      d[4] - AR NUMBER DENSITY(CM-3)
 *      d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]
 *      d[6] - H NUMBER DENSITY(CM-3)
 *      d[7] - N NUMBER DENSITY(CM-3)
 *      d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)
 *      t[0] - EXOSPHERIC TEMPERATURE
 *      t[1] - TEMPERATURE AT ALT
 *
 *
 *      O, H, and N are set to zero below 72.5 km
 *
 *      t[0], Exospheric temperature, is set to global average for
 *      altitudes below 120 km. The 120 km gradient is left at global
 *      average value for altitudes below 72 km.
 *
 *      d[5], TOTAL MASS DENSITY, is NOT the same for subroutines GTD7
 *      and GTD7D
 *
 *        SUBROUTINE GTD7 -- d[5] is the sum of the mass densities of the
 *        species labeled by indices 0-4 and 6-7 in output variable d.
 *        This includes He, O, N2, O2, Ar, H, and N but does NOT include
 *        anomalous oxygen (species index 8).
 *
 *        SUBROUTINE GTD7D -- d[5] is the "effective total mass density
 *        for drag" and is the sum of the mass densities of all species
 *        in this model, INCLUDING anomalous oxygen.
 */
 """
    def __init__(self):
        self.d = [0.0]*9 #/* densities */
        self.t = [0.0]*2 #/* temperatures */


#/* ------------------------------------------------------------------- */
#/* --------------------------- PROTOTYPES ---------------------------- */
#/* ------------------------------------------------------------------- */
# No prototypes are used here, these are here for reference
'''
/* GTD7 */
/*   Neutral Atmosphere Empircial Model from the surface to lower
 *   exosphere.
 */

/* GTD7D */
/*   This subroutine provides Effective Total Mass Density for output
 *   d[5] which includes contributions from "anomalous oxygen" which can
 *   affect satellite drag above 500 km. See the section "output" for
 *   additional details.
 */

 /* GTS7 */
/*   Thermospheric portion of NRLMSISE-00
 */

 /* GHP7 */
/*   To specify outputs at a pressure level (press) rather than at
 *   an altitude.
 */
 '''
