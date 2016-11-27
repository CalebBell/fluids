# -*- coding: utf-8 -*-
"""
12/19/2013
Author: Joshua Milas
Python Version: 3.3.2

The NRLMSISE-00 model 2001 ported to python
Based off of Dominik Brodowski 20100516 version available here
http://www.brodo.de/english/pub/nrlmsise/

This is the main program that contains all the functions

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
from .nrlmsise_00_header import *
from .nrlmsise_00_data import *
from math import sin, cos, pow, exp, log, sqrt
__all__ = ['gtd7']

"""
/* ------------------------------------------------------------------- */
/* ------------------------- SHARED VARIABLES ------------------------ */
/* ------------------------------------------------------------------- */
"""

#/* PARMB */
gsurf = [0.0];
re = [0.0];

#/* GTS3C */
dd = 0.0;

#/* DMIX */
dm04 = 0.0
dm16 = 0.0
dm28 = 0.0
dm32 = 0.0
dm40 = 0.0
dm01 = 0.0
dm14 = 0.0

#/* MESO7 */
meso_tn1 = [0.0]*5;
meso_tn2 = [0.0]*4;
meso_tn3 = [0.0]*5;
meso_tgn1 = [0.0, 0.0];
meso_tgn2 = [0.0, 0.0];
meso_tgn3 = [0.0, 0.0];

#/* POWER7 */
#/* LOWER7 */
#Dont to need to do anyt of the externs, they are all here


#/* LPOLY */
dfa = 0.0
plg = [[0.0 for _ in range(9)] for _ in range(4)] 
ctloc = 0.0
stloc = 0.0
c2tloc = 0.0
s2tloc = 0.0
s3tloc = 0.0
c3tloc = 0.0
apdf = 0.0
apt = [0.0]*4


#since rgas is used eerywehre usignthe same variable, ill make it glboal
#rgas = 831.44621
#rgas = 831.4
"""
/* ------------------------------------------------------------------- */
/* ------------------------------ TSELEC ----------------------------- */
/* ------------------------------------------------------------------- */
"""
def tselec(flags):

    for i in range(24):
        if(i != 9):
            if(flags.switches[i]==1):
                flags.sw[i]=1
            else:
                flags.sw[i]=0

            if(flags.switches[i]>0):
                flags.swc[i]=1
            else:
                flags.swc[i]=0
        else:
            flags.sw[i]=flags.switches[i]
            flags.swc[i]=flags.switches[i]
    return


"""
/* ------------------------------------------------------------------- */
/* ------------------------------ GLATF ------------------------------ */
/* ------------------------------------------------------------------- */
"""
def glatf(lat, gv, reff):
    dgtr = 1.74533E-2
    c2 = cos(2.0*dgtr*lat)

    #Need to return these since the c program wants pointers to these
    gv[0] = 980.616 * (1.0 - 0.0026373 * c2)
    reff[0] = 2.0 * (gv[0]) / (3.085462E-6 + 2.27E-9 * c2) * 1.0E-5 #The may-be troubled line
    return #gv, rref

"""
/* ------------------------------------------------------------------- */
/* ------------------------------ CCOR ------------------------------- */
/* ------------------------------------------------------------------- */
"""
def ccor(alt, r, h1, zh):
    """
/*        CHEMISTRY/DISSOCIATION CORRECTION FOR MSIS MODELS
 *         ALT - altitude
 *         R - target ratio
 *         H1 - transition scale length
 *         ZH - altitude of 1/2 R
 */
 """
    e = (alt - zh) / h1
    if(e>70):
        return exp(0) # pragma: no cover
    if (e<-70):
        return exp(r)
    ex = exp(e)
    e = r / (1.0 + ex)
    return exp(e)

'''
/* ------------------------------------------------------------------- */
/* ------------------------------ CCOR ------------------------------- */
/* ------------------------------------------------------------------- */
'''
def ccor2(alt, r, h1, zh, h2):
    '''
/*        CHEMISTRY/DISSOCIATION CORRECTION FOR MSIS MODELS
 *         ALT - altitude
 *         R - target ratio
 *         H1 - transition scale length
 *         ZH - altitude of 1/2 R
 *         H2 - transition scale length #2 ?
 */
 '''
    e1 = (alt - zh) / h1;
    e2 = (alt - zh) / h2;
    if ((e1 > 70) or (e2 > 70)): # pragma: no cover
        return exp(0)
    if ((e1 < -70) and (e2 < -70)): # pragma: no cover
        return exp(r)
    ex1 = exp(e1);
    ex2 = exp(e2);
    ccor2v = r / (1.0 + 0.5 * (ex1 + ex2));
    return exp(ccor2v);



"""
/* ------------------------------------------------------------------- */
/* ------------------------------- SCALH ----------------------------- */
/* ------------------------------------------------------------------- */
"""
def scalh(alt, xm, temp):
    #rgas = 831.44621    #maybe make this a global constant?
    rgas = 831.4
    g = gsurf[0] / (pow((1.0 + alt/re[0]),2.0))
    g = rgas * temp / (g * xm)
    return g


'''
/* ------------------------------------------------------------------- */
/* -------------------------------- DNET ----------------------------- */
/* ------------------------------------------------------------------- */
'''
def dnet(dd, dm, zhm, xmm, xm):
    '''
/*       TURBOPAUSE CORRECTION FOR MSIS MODELS
 *        Root mean density
 *         DD - diffusive density
 *         DM - full mixed density
 *         ZHM - transition scale length
 *         XMM - full mixed molecular weight
 *         XM  - species molecular weight
 *         DNET - combined density
 */
 '''
    a  = zhm / (xmm-xm)
    if( not((dm>0) and (dd>0))): # pragma: no cover
        if((dd==0) and (dm==0)):
            dd=1
        if(dm==0):
            return dd
        if(dd==0):
            return dm

    ylog = a * log(dm/dd)
    if(ylog < -10):
        return dd
    if(ylog>10): # pragma: no cover
        return dm
    a = dd*pow((1.0 + exp(ylog)),(1.0/a))
    return a


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- SPLINI ---------------------------- */
/* ------------------------------------------------------------------- */
'''
def splini(xa, ya, y2a, n, x, y):
    '''
/*      INTEGRATE CUBIC SPLINE FUNCTION FROM XA(1) TO X
 *       XA,YA: ARRAYS OF TABULATED FUNCTION IN ASCENDING ORDER BY X
 *       Y2A: ARRAY OF SECOND DERIVATIVES
 *       N: SIZE OF ARRAYS XA,YA,Y2A
 *       X: ABSCISSA ENDPOINT FOR INTEGRATION
 *       Y: OUTPUT VALUE
 */
 '''
    yi = 0
    klo = 0
    khi = 1
    while((x>xa[klo]) and (khi<n)):
        xx = x
        if(khi < (n-1)):
            if (x < xa[khi]):
                xx = x
            else:
                xx = xa[khi]
        h = xa[khi] - xa[klo]
        a = (xa[khi] - xx)/h
        b = (xx - xa[klo])/h
        a2 = a*a
        b2 = b*b
        yi += ((1.0 - a2) * ya[klo] / 2.0 + b2 * ya[khi] / 2.0 + ((-(1.0+a2*a2)/4.0 + a2/2.0) * y2a[klo] + (b2*b2/4.0 - b2/2.0) * y2a[khi]) * h * h / 6.0) * h;
        klo += 1;
        khi += 1;
    y[0] = yi
    return #yi


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- SPLINT ---------------------------- */
/* ------------------------------------------------------------------- */
'''
def splint(xa, ya, y2a, n, x, y):
    '''
/*      CALCULATE CUBIC SPLINE INTERP VALUE
 *       ADAPTED FROM NUMERICAL RECIPES BY PRESS ET AL.
 *       XA,YA: ARRAYS OF TABULATED FUNCTION IN ASCENDING ORDER BY X
 *       Y2A: ARRAY OF SECOND DERIVATIVES
 *       N: SIZE OF ARRAYS XA,YA,Y2A
 *       X: ABSCISSA FOR INTERPOLATION
 *       Y: OUTPUT VALUE
 */
 '''
    klo = 0
    khi = n-1

    while((khi-klo)>1):
        k=int((khi+klo)/2);
        if (xa[k]>x):
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo];
    a = (xa[khi] - x)/h;
    b = (x - xa[klo])/h;
    yi = a * ya[klo] + b * ya[khi] + ((a*a*a - a) * y2a[klo] + (b*b*b - b) * y2a[khi]) * h * h/6.0;
    y[0] = yi #may not need this
    return #yi #or this


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- SPLINE ---------------------------- */
/* ------------------------------------------------------------------- */
'''
def spline(x, y, n, yp1, ypn, y2):
    '''
/*       CALCULATE 2ND DERIVATIVES OF CUBIC SPLINE INTERP FUNCTION
 *       ADAPTED FROM NUMERICAL RECIPES BY PRESS ET AL
 *       X,Y: ARRAYS OF TABULATED FUNCTION IN ASCENDING ORDER BY X
 *       N: SIZE OF ARRAYS X,Y
 *       YP1,YPN: SPECIFIED DERIVATIVES AT X[0] AND X[N-1]; VALUES
 *                >= 1E30 SIGNAL SIGNAL SECOND DERIVATIVE ZERO
 *       Y2: OUTPUT ARRAY OF SECOND DERIVATIVES
 */
 '''
    u = [0.0]*n #I think this is the same as malloc

    #no need for the out of memory

    if (yp1 > 0.99E30): # pragma: no cover
        y2[0] = 0
        u[0] = 0
    else:
        y2[0]=-0.5
        u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1)

    for i in range(1, n-1):
        sig = (x[i]-x[i-1])/(x[i+1] - x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (6.0 * ((y[i+1] - y[i])/(x[i+1] - x[i]) -(y[i] - y[i-1]) / (x[i] - x[i-1]))/(x[i+1] - x[i-1]) - sig * u[i-1])/p;

    if (ypn>0.99E30): # pragma: no cover
        qn = 0;
        un = 0
    else:
        qn = 0.5
        un = (3.0 / (x[n-1] - x[n-2])) * (ypn - (y[n-1] - y[n-2])/(x[n-1] - x[n-2]));

    y2[n-1] = (un - qn * u[n-2]) / (qn * y2[n-2] + 1.0);

    #it uses a for loop here, but its not something python can do (I dont think)
    k = n-2
    while(k >= 0):
        y2[k] = y2[k] * y2[k+1] + u[k]
        k -= 1
    #This for loop might work
    #for k in range(n-2, -1, -1):
    #    y2[k] = y2[k] * y2[k+1] + u[k]
    #no need to free u here
    return


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- DENSM ----------------------------- */
/* ------------------------------------------------------------------- */
'''
def zeta(zz, zl):
    return ((zz-zl)*(re[0]+zl)/(re[0]+zz))    #re is the global variable

def densm(alt, d0, xm, tz, mn3, zn3, tn3, tgn3, mn2, zn2, tn2, tgn2):
    '''
/*      Calculate Temperature and Density Profiles for lower atmos.  */
'''
    xs = [0.0]*10
    ys = [0.0]*10
    y2out = [0.0]*10
    rgas = 831.4
    #rgas = 831.44621    #maybe make this a global constant?
    densm_tmp=d0
    if (alt>zn2[0]): # pragma: no cover
        if(xm==0.0):
            return tz[0]
        else:
            return d0

    #/* STRATOSPHERE/MESOSPHERE TEMPERATURE */
    if (alt>zn2[mn2-1]):
        z=alt
    else:
        z=zn2[mn2-1];
    mn=mn2;
    z1=zn2[0];
    z2=zn2[mn-1];
    t1=tn2[0];
    t2=tn2[mn-1];
    zg = zeta(z, z1);
    zgdif = zeta(z2, z1);

    #/* set up spline nodes */
    for k in range(mn):
        xs[k]=zeta(zn2[k],z1)/zgdif;
        ys[k]=1.0 / tn2[k];
    yd1=-tgn2[0] / (t1*t1) * zgdif;
    yd2=-tgn2[1] / (t2*t2) * zgdif * (pow(((re[0]+z2)/(re[0]+z1)),2.0));

    #/* calculate spline coefficients */
    spline (xs, ys, mn, yd1, yd2, y2out);   #No need to change this
    x = zg/zgdif;
    y = [0.0]
    splint (xs, ys, y2out, mn, x, y);

    #/* temperature at altitude */
    tz[0] = 1.0 / y[0];
    if (xm!=0.0):
        #/* calaculate stratosphere / mesospehere density */
        glb = gsurf[0] / (pow((1.0 + z1/re[0]),2.0));
        gamm = xm * glb * zgdif / rgas;

        #/* Integrate temperature profile */
        yi = [0.0]
        splini(xs, ys, y2out, mn, x, yi);
        expl=gamm*yi[0];
        if (expl>50.0): # pragma: no cover
            expl=50.0

        #/* Density at altitude */
        densm_tmp = densm_tmp * (t1 / tz[0]) * exp(-expl);

    if (alt>zn3[0]):
        if (xm==0.0):
            return tz[0]
        else:
            return densm_tmp

    #/* troposhere / stratosphere temperature */
    z = alt;
    mn = mn3;
    z1=zn3[0];
    z2=zn3[mn-1];
    t1=tn3[0];
    t2=tn3[mn-1];
    zg=zeta(z,z1);
    zgdif=zeta(z2,z1);



    #/* set up spline nodes */
    for k in range(mn):
        xs[k] = zeta(zn3[k],z1) / zgdif;
        ys[k] = 1.0 / tn3[k];
    
    yd1=-tgn3[0] / (t1*t1) * zgdif;
    yd2=-tgn3[1] / (t2*t2) * zgdif * (pow(((re[0]+z2)/(re[0]+z1)),2.0));

    #/* calculate spline coefficients */
    spline (xs, ys, mn, yd1, yd2, y2out);
    x = zg/zgdif;
    y = [0.0]
    splint (xs, ys, y2out, mn, x, y);

    #/* temperature at altitude */
    tz[0] = 1.0 / y[0];
    if (xm!=0.0):
        #/* calaculate tropospheric / stratosphere density */
        glb = gsurf[0] / (pow((1.0 + z1/re[0]),2.0));
        gamm = xm * glb * zgdif / rgas;

        #/* Integrate temperature profile */
        yi = [0.0]
        splini(xs, ys, y2out, mn, x, yi);
        expl=gamm*yi[0];
        if (expl>50.0): # pragma: no cover
            expl=50.0;

        #/* Density at altitude */
        densm_tmp = densm_tmp * (t1 / tz[0]) * exp(-expl);
    
    if (xm==0.0):
        return tz[0];
    else:
        return densm_tmp;


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- DENSU ----------------------------- */
/* ------------------------------------------------------------------- */
'''
def densu(alt, dlb, tinf, tlb, xm, alpha, tz, zlb, s2, mn1, zn1, tn1, tgn1):
    '''
/*      Calculate Temperature and Density Profiles for MSIS models
 *      New lower thermo polynomial
 */
 tz, zn1, tn1, and tgn1 are simulated pointers
 '''
    rgas = 831.4
    #rgas = 831.44621    #maybe make this a global constant?
    densu_temp = 1.0

    xs = [0.0]*5
    ys = [0.0]*5
    y2out = [0.0]*5

    #/* joining altitudes of Bates and spline */
    za=zn1[0];
    if (alt>za):
        z=alt;
    else:
        z=za;

    #/* geopotential altitude difference from ZLB */
    zg2 = zeta(z, zlb);
    
    #/* Bates temperature */
    tt = tinf - (tinf - tlb) * exp(-s2*zg2);
    ta = tt;
    tz[0] = tt
    densu_temp = tz[0]

    if (alt<za):
        #/* calculate temperature below ZA
        # * temperature gradient at ZA from Bates profile */
        dta = (tinf - ta) * s2 * pow(((re[0]+zlb)/(re[0]+za)),2.0);
        tgn1[0]=dta;
        tn1[0]=ta;
        if (alt>zn1[mn1-1]):
            z=alt;
        else:
            z=zn1[mn1-1];
        mn=mn1;
        z1=zn1[0];
        z2=zn1[mn-1];
        t1=tn1[0];
        t2=tn1[mn-1];
        #/* geopotental difference from z1 */
        zg = zeta (z, z1);
        zgdif = zeta(z2, z1);
        #/* set up spline nodes */
        for k in range(mn):
            xs[k] = zeta(zn1[k], z1) / zgdif;
            ys[k] = 1.0 / tn1[k];
        
        #/* end node derivatives */
        yd1 = -tgn1[0] / (t1*t1) * zgdif;
        yd2 = -tgn1[1] / (t2*t2) * zgdif * pow(((re[0]+z2)/(re[0]+z1)),2.0);
        #/* calculate spline coefficients */
        spline (xs, ys, mn, yd1, yd2, y2out);
        x = zg / zgdif;
        y = [0.0]
        splint (xs, ys, y2out, mn, x, y);
        #/* temperature at altitude */
        tz[0] = 1.0 / y[0];
        densu_temp = tz[0];

    if (xm==0):
        return densu_temp;

    #/* calculate density above za */
    glb = gsurf[0] / pow((1.0 + zlb/re[0]),2.0);
    gamma = xm * glb / (s2 * rgas * tinf);
    expl = exp(-s2 * gamma * zg2);
    if (expl>50.0): # pragma: no cover
        expl=50.0;
    if (tt<=0): # pragma: no cover
        expl=50.0;

    #/* density at altitude */
    densa = dlb * pow((tlb/tt),((1.0+alpha+gamma))) * expl;
    densu_temp=densa;
    if (alt>=za):
        return densu_temp;

    #/* calculate density below za */
    glb = gsurf[0] / pow((1.0 + z1/re[0]),2.0);
    gamm = xm * glb * zgdif / rgas;

    #/* integrate spline temperatures */
    yi = [0]
    splini (xs, ys, y2out, mn, x, yi);
    expl = gamm * yi[0];
    if (expl>50.0): # pragma: no cover
        expl=50.0;
    if (tz[0]<=0): # pragma: no cover
        expl=50.0;

    #/* density at altitude */
    densu_temp = densu_temp * pow ((t1 / tz[0]),(1.0 + alpha)) * exp(-expl);
    return densu_temp;


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- GLOBE7 ---------------------------- */
/* ------------------------------------------------------------------- */
'''

#/*    3hr Magnetic activity functions */
#/*    Eq. A24d */
def g0(a, p):
    return (a - 4.0 + (p[25] - 1.0) * (a - 4.0 + (exp(-sqrt(p[24]*p[24]) * (a - 4.0)) - 1.0) / sqrt(p[24]*p[24])));


#/*    Eq. A24c */
def sumex(ex):
    return (1.0 + (1.0 - pow(ex,19.0)) / (1.0 - ex) * pow(ex,0.5));


#/*    Eq. A24a */
def sg0(ex, p, ap):
    return (g0(ap[1],p) + (g0(ap[2],p)*ex + g0(ap[3],p)*ex*ex + \
                g0(ap[4],p)*pow(ex,3.0)	+ (g0(ap[5],p)*pow(ex,4.0) + \
                g0(ap[6],p)*pow(ex,12.0))*(1.0-pow(ex,8.0))/(1.0-ex)))/sumex(ex);


def globe7(p, Input, flags):
    '''
/*       CALCULATE G(L) FUNCTION 
 *       Upper Thermosphere Parameters */
'''
    t = [0]*15  #modified this, there was a for loop that did this
    sw9 = 1
    sr = 7.2722E-5;
    dgtr = 1.74533E-2;
    dr = 1.72142E-2;
    hr = 0.2618;

    tloc = Input.lst
    #for j in range(14):
    #    t[j] = 0
    if(flags.sw[9] > 0):
        sw9 = 1
    elif(flags.sw[9] < 0): # pragma: no cover
        sw9 = -1
    xlong = Input.g_long

    #/* calculate legendre polynomials */
    c = sin(Input.g_lat * dgtr);
    s = cos(Input.g_lat * dgtr);
    c2 = c*c;
    c4 = c2*c2;
    s2 = s*s;

    plg[0][1] = c;
    plg[0][2] = 0.5*(3.0*c2 -1.0);
    plg[0][3] = 0.5*(5.0*c*c2-3.0*c);
    plg[0][4] = (35.0*c4 - 30.0*c2 + 3.0)/8.0;
    plg[0][5] = (63.0*c2*c2*c - 70.0*c2*c + 15.0*c)/8.0;
    plg[0][6] = (11.0*c*plg[0][5] - 5.0*plg[0][4])/6.0;
#/*      plg[0][7] = (13.0*c*plg[0][6] - 6.0*plg[0][5])/7.0; */
    plg[1][1] = s;
    plg[1][2] = 3.0*c*s;
    plg[1][3] = 1.5*(5.0*c2-1.0)*s;
    plg[1][4] = 2.5*(7.0*c2*c-3.0*c)*s;
    plg[1][5] = 1.875*(21.0*c4 - 14.0*c2 +1.0)*s;
    plg[1][6] = (11.0*c*plg[1][5]-6.0*plg[1][4])/5.0;
#/*      plg[1][7] = (13.0*c*plg[1][6]-7.0*plg[1][5])/6.0; */
#/*      plg[1][8] = (15.0*c*plg[1][7]-8.0*plg[1][6])/7.0; */
    plg[2][2] = 3.0*s2;
    plg[2][3] = 15.0*s2*c;
    plg[2][4] = 7.5*(7.0*c2 -1.0)*s2;
    plg[2][5] = 3.0*c*plg[2][4]-2.0*plg[2][3];
    plg[2][6] =(11.0*c*plg[2][5]-7.0*plg[2][4])/4.0;
    plg[2][7] =(13.0*c*plg[2][6]-8.0*plg[2][5])/5.0;
    plg[3][3] = 15.0*s2*s;
    plg[3][4] = 105.0*s2*s*c; 
    plg[3][5] =(9.0*c*plg[3][4]-7.*plg[3][3])/2.0;
    plg[3][6] =(11.0*c*plg[3][5]-8.*plg[3][4])/3.0;

    if( not (((flags.sw[7]==0) and (flags.sw[8]==0)) and (flags.sw[14] == 0))):
        global stloc
        stloc = sin(hr*tloc);
        global ctloc
        ctloc = cos(hr*tloc);
        global s2tloc
        s2tloc = sin(2.0*hr*tloc);
        global c2tloc
        c2tloc = cos(2.0*hr*tloc);
        global s3tloc
        s3tloc = sin(3.0*hr*tloc);
        global c3tloc
        c3tloc = cos(3.0*hr*tloc);

    cd32 = cos(dr*(Input.doy-p[31]));
    cd18 = cos(2.0*dr*(Input.doy-p[17]));
    cd14 = cos(dr*(Input.doy-p[13]));
    cd39 = cos(2.0*dr*(Input.doy-p[38]));
    p32=p[31];
    p18=p[17];
    p14=p[13];
    p39=p[38];

    #/* F10.7 EFFECT */
    df = Input.f107 - Input.f107A;
    global dfa
    dfa = Input.f107A - 150.0;
    t[0] =  p[19]*df*(1.0+p[59]*dfa) + p[20]*df*df + p[21]*dfa + p[29]*pow(dfa,2.0);
    f1 = 1.0 + (p[47]*dfa +p[19]*df+p[20]*df*df)*flags.swc[1];
    f2 = 1.0 + (p[49]*dfa+p[19]*df+p[20]*df*df)*flags.swc[1];

    #/*  TIME INDEPENDENT */
    t[1] = (p[1]*plg[0][2]+ p[2]*plg[0][4]+p[22]*plg[0][6]) + \
          (p[14]*plg[0][2])*dfa*flags.swc[1] +p[26]*plg[0][1];

    #/*  SYMMETRICAL ANNUAL */
    t[2] = p[18]*cd32;

    #/*  SYMMETRICAL SEMIANNUAL */
    t[3] = (p[15]+p[16]*plg[0][2])*cd18;

    #/*  ASYMMETRICAL ANNUAL */
    t[4] =  f1*(p[9]*plg[0][1]+p[10]*plg[0][3])*cd14;

    #/*  ASYMMETRICAL SEMIANNUAL */
    t[5] =    p[37]*plg[0][1]*cd39;

    #/* DIURNAL */
    if (flags.sw[7]):
        t71 = (p[11]*plg[1][2])*cd14*flags.swc[5];
        t72 = (p[12]*plg[1][2])*cd14*flags.swc[5];
        t[6] = f2*((p[3]*plg[1][1] + p[4]*plg[1][3] + p[27]*plg[1][5] + t71) * \
                   ctloc + (p[6]*plg[1][1] + p[7]*plg[1][3] + p[28]*plg[1][5] \
                            + t72)*stloc);


    #/* SEMIDIURNAL */
    if (flags.sw[8]):
        t81 = (p[23]*plg[2][3]+p[35]*plg[2][5])*cd14*flags.swc[5];
        t82 = (p[33]*plg[2][3]+p[36]*plg[2][5])*cd14*flags.swc[5];
        t[7] = f2*((p[5]*plg[2][2]+ p[41]*plg[2][4] + t81)*c2tloc +(p[8]*plg[2][2] + p[42]*plg[2][4] + t82)*s2tloc);
    

    #/* TERDIURNAL */
    if (flags.sw[14]):
        t[13] = f2 * ((p[39]*plg[3][3]+(p[93]*plg[3][4]+p[46]*plg[3][6])*cd14*flags.swc[5])* s3tloc +(p[40]*plg[3][3]+(p[94]*plg[3][4]+p[48]*plg[3][6])*cd14*flags.swc[5])* c3tloc);


    #/* magnetic activity based on daily ap */
    if (flags.sw[9]==-1):
        ap = Input.ap_a;
        if (p[51]!=0):
            exp1 = exp(-10800.0*sqrt(p[51]*p[51])/(1.0+p[138]*(45.0-sqrt(Input.g_lat*Input.g_lat))));
            if (exp1>0.99999): # pragma: no cover
                    exp1=0.99999;
            if (p[24]<1.0E-4): # pragma: no cover
                    p[24]=1.0E-4;
            apt[0]=sg0(exp1,p,ap.a);
            #/* apt[1]=sg2(exp1,p,ap->a);
            #   apt[2]=sg0(exp2,p,ap->a);
            #   apt[3]=sg2(exp2,p,ap->a);
            #*/
            if (flags.sw[9]):
                t[8] = apt[0]*(p[50]+p[96]*plg[0][2]+p[54]*plg[0][4]+ \
                        (p[125]*plg[0][1]+p[126]*plg[0][3]+p[127]*plg[0][5])*cd14*flags.swc[5]+ \
                        (p[128]*plg[1][1]+p[129]*plg[1][3]+p[130]*plg[1][5])*flags.swc[7]* \
                                       cos(hr*(tloc-p[131])));
                    
            
    else:
        apd=Input.ap-4.0;
        p44=p[43];
        p45=p[44];
        if (p44<0): # pragma: no cover
            p44 = 1.0E-5;
        global apdf
        apdf = apd + (p45-1.0)*(apd + (exp(-p44 * apd) - 1.0)/p44);
        if (flags.sw[9]):
            t[8]=apdf*(p[32]+p[45]*plg[0][2]+p[34]*plg[0][4]+ \
             (p[100]*plg[0][1]+p[101]*plg[0][3]+p[102]*plg[0][5])*cd14*flags.swc[5]+
             (p[121]*plg[1][1]+p[122]*plg[1][3]+p[123]*plg[1][5])*flags.swc[7]*
                cos(hr*(tloc-p[124])));
            
    

    if ((flags.sw[10]) and (Input.g_long>-1000.0)):

            #/* longitudinal */
            if (flags.sw[11]):
                    t[10] = (1.0 + p[80]*dfa*flags.swc[1])* \
                     ((p[64]*plg[1][2]+p[65]*plg[1][4]+p[66]*plg[1][6]\
                      +p[103]*plg[1][1]+p[104]*plg[1][3]+p[105]*plg[1][5]\
                      +flags.swc[5]*(p[109]*plg[1][1]+p[110]*plg[1][3]+p[111]*plg[1][5])*cd14)* \
                          cos(dgtr*Input.g_long) \
                      +(p[90]*plg[1][2]+p[91]*plg[1][4]+p[92]*plg[1][6]\
                      +p[106]*plg[1][1]+p[107]*plg[1][3]+p[108]*plg[1][5]\
                      +flags.swc[5]*(p[112]*plg[1][1]+p[113]*plg[1][3]+p[114]*plg[1][5])*cd14)* \
                      sin(dgtr*Input.g_long));
            

            #/* ut and mixed ut, longitude */
            if (flags.sw[12]):
                    t[11]=(1.0+p[95]*plg[0][1])*(1.0+p[81]*dfa*flags.swc[1])*\
                            (1.0+p[119]*plg[0][1]*flags.swc[5]*cd14)*\
                            ((p[68]*plg[0][1]+p[69]*plg[0][3]+p[70]*plg[0][5])*\
                            cos(sr*(Input.sec-p[71])));
                    t[11]+=flags.swc[11]*\
                            (p[76]*plg[2][3]+p[77]*plg[2][5]+p[78]*plg[2][7])*\
                            cos(sr*(Input.sec-p[79])+2.0*dgtr*Input.g_long)*(1.0+p[137]*dfa*flags.swc[1]);
            

            #/* ut, longitude magnetic activity */
            if (flags.sw[13]):
                if (flags.sw[9]==-1):
                    if (p[51]):
                            t[12]=apt[0]*flags.swc[11]*(1.+p[132]*plg[0][1])*\
                                    ((p[52]*plg[1][2]+p[98]*plg[1][4]+p[67]*plg[1][6])*\
                                     cos(dgtr*(Input.g_long-p[97])))\
                                    +apt[0]*flags.swc[11]*flags.swc[5]*\
                                    (p[133]*plg[1][1]+p[134]*plg[1][3]+p[135]*plg[1][5])*\
                                    cd14*cos(dgtr*(Input.g_long-p[136])) \
                                    +apt[0]*flags.swc[12]* \
                                    (p[55]*plg[0][1]+p[56]*plg[0][3]+p[57]*plg[0][5])*\
                                    cos(sr*(Input.sec-p[58]));
                        
                else:
                    t[12] = apdf*flags.swc[11]*(1.0+p[120]*plg[0][1])*\
                            ((p[60]*plg[1][2]+p[61]*plg[1][4]+p[62]*plg[1][6])*\
                            cos(dgtr*(Input.g_long-p[63])))\
                            +apdf*flags.swc[11]*flags.swc[5]* \
                            (p[115]*plg[1][1]+p[116]*plg[1][3]+p[117]*plg[1][5])* \
                            cd14*cos(dgtr*(Input.g_long-p[118])) \
                            + apdf*flags.swc[12]* \
                            (p[83]*plg[0][1]+p[84]*plg[0][3]+p[85]*plg[0][5])* \
                            cos(sr*(Input.sec-p[75]));
                    			
            
    

    #/* parms not used: 82, 89, 99, 139-149 */
    tinf = p[30];
    for i in range(14):
        tinf = tinf + abs(flags.sw[i+1])*t[i];
    return tinf;


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- GLOB7S ---------------------------- */
/* ------------------------------------------------------------------- */
'''
def glob7s(p, Input, flags):
    '''
/*    VERSION OF GLOBE FOR LOWER ATMOSPHERE 10/26/99 
 */
 '''
    pset = 2.0
    t = [0.0]*14
    dr=1.72142E-2;
    dgtr=1.74533E-2;

    #/* confirm parameter set */
    if (p[99]==0): # pragma: no cover
        p[99]=pset;
    
    #for j in range(14):    #Already taken care of
    #    t[j]=0.0;
    cd32 = cos(dr*(Input.doy-p[31]));
    cd18 = cos(2.0*dr*(Input.doy-p[17]));
    cd14 = cos(dr*(Input.doy-p[13]));
    cd39 = cos(2.0*dr*(Input.doy-p[38]));
    p32=p[31];
    p18=p[17];
    p14=p[13];
    p39=p[38];

    #/* F10.7 */
    t[0] = p[21]*dfa;

    #/* time independent */
    t[1]=p[1]*plg[0][2] + p[2]*plg[0][4] + p[22]*plg[0][6] + p[26]*plg[0][1] + p[14]*plg[0][3] + p[59]*plg[0][5];

    #/* SYMMETRICAL ANNUAL */
    t[2]=(p[18]+p[47]*plg[0][2]+p[29]*plg[0][4])*cd32;

    #/* SYMMETRICAL SEMIANNUAL */
    t[3]=(p[15]+p[16]*plg[0][2]+p[30]*plg[0][4])*cd18;

    #/* ASYMMETRICAL ANNUAL */
    t[4]=(p[9]*plg[0][1]+p[10]*plg[0][3]+p[20]*plg[0][5])*cd14;

    #/* ASYMMETRICAL SEMIANNUAL */
    t[5]=(p[37]*plg[0][1])*cd39;

    #/* DIURNAL */
    if (flags.sw[7]):
        t71 = p[11]*plg[1][2]*cd14*flags.swc[5];
        t72 = p[12]*plg[1][2]*cd14*flags.swc[5];
        t[6] = ((p[3]*plg[1][1] + p[4]*plg[1][3] + t71) * ctloc + (p[6]*plg[1][1] + p[7]*plg[1][3] + t72) * stloc) ;
    

    #/* SEMIDIURNAL */
    if (flags.sw[8]):
        t81 = (p[23]*plg[2][3]+p[35]*plg[2][5])*cd14*flags.swc[5];
        t82 = (p[33]*plg[2][3]+p[36]*plg[2][5])*cd14*flags.swc[5];
        t[7] = ((p[5]*plg[2][2] + p[41]*plg[2][4] + t81) * c2tloc + (p[8]*plg[2][2] + p[42]*plg[2][4] + t82) * s2tloc);
    

    #/* TERDIURNAL */
    if (flags.sw[14]):
            t[13] = p[39] * plg[3][3] * s3tloc + p[40] * plg[3][3] * c3tloc;
    

    #/* MAGNETIC ACTIVITY */
    if (flags.sw[9]):
        if (flags.sw[9]==1):
            t[8] = apdf * (p[32] + p[45] * plg[0][2] * flags.swc[2]);
        if (flags.sw[9]==-1):	
            t[8]=(p[50]*apt[0] + p[96]*plg[0][2] * apt[0]*flags.swc[2]);
    

    #/* LONGITUDINAL */
    if ( not((flags.sw[10]==0) or (flags.sw[11]==0) or (Input.g_long<=-1000.0))):
            t[10] = (1.0 + plg[0][1]*(p[80]*flags.swc[5]*cos(dr*(Input.doy-p[81]))\
                    +p[85]*flags.swc[6]*cos(2.0*dr*(Input.doy-p[86])))\
                    +p[83]*flags.swc[3]*cos(dr*(Input.doy-p[84]))\
                    +p[87]*flags.swc[4]*cos(2.0*dr*(Input.doy-p[88])))\
                    *((p[64]*plg[1][2]+p[65]*plg[1][4]+p[66]*plg[1][6]\
                    +p[74]*plg[1][1]+p[75]*plg[1][3]+p[76]*plg[1][5]\
                    )*cos(dgtr*Input.g_long)\
                    +(p[90]*plg[1][2]+p[91]*plg[1][4]+p[92]*plg[1][6]\
                    +p[77]*plg[1][1]+p[78]*plg[1][3]+p[79]*plg[1][5]\
                    )*sin(dgtr*Input.g_long));
    
    tt=0;
    for i in range(14):
        tt+=abs(flags.sw[i+1])*t[i];
    return tt;


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- GTD7 ------------------------------ */
/* ------------------------------------------------------------------- */
'''
def gtd7(Input, flags, output):
    '''The standard model subroutine (GTD7) always computes the 
    ‘‘thermospheric’’ mass density by explicitly summing the masses of
    the species in equilibrium at the thermospheric temperature T(z).
    '''
    mn3 = 5
    zn3 = [32.5,20.0,15.0,10.0,0.0]
    mn2 = 4
    zn2 = [72.5,55.0,45.0,32.5]
    zmix = 62.5
    soutput = nrlmsise_output()

    tselec(flags);

    #/* Latitude variation of gravity (none for sw[2]=0) */
    xlat=Input.g_lat;
    if (flags.sw[2]==0): # pragma: no cover
        xlat=45.0;
    glatf(xlat, gsurf, re);

    xmm = pdm[2][4];

    #/* THERMOSPHERE / MESOSPHERE (above zn2[0]) */
    if (Input.alt>zn2[0]):
        altt=Input.alt;
    else:
        altt=zn2[0];

    tmp=Input.alt;
    Input.alt=altt;

    gts7(Input, flags, soutput);
    altt=Input.alt;
    Input.alt=tmp;
    if (flags.sw[0]): # pragma: no cover  #/* metric adjustment */
        dm28m= dm28*1.0E6;
    else:
        dm28m = dm28;
    output.t[0]=soutput.t[0];
    output.t[1]=soutput.t[1];
    if (Input.alt>=zn2[0]): 
        for i in range(9):
            output.d[i]=soutput.d[i];
        return


#/*       LOWER MESOSPHERE/UPPER STRATOSPHERE (between zn3[0] and zn2[0])
#*         Temperature at nodes and gradients at end nodes
#*         Inverse temperature a linear function of spherical harmonics
#*/
    meso_tgn2[0]=meso_tgn1[1];
    meso_tn2[0]=meso_tn1[4];
    meso_tn2[1]=pma[0][0]*pavgm[0]/(1.0-flags.sw[20]*glob7s(pma[0], Input, flags));
    meso_tn2[2]=pma[1][0]*pavgm[1]/(1.0-flags.sw[20]*glob7s(pma[1], Input, flags));
    meso_tn2[3]=pma[2][0]*pavgm[2]/(1.0-flags.sw[20]*flags.sw[22]*glob7s(pma[2], Input, flags));
    meso_tgn2[1]=pavgm[8]*pma[9][0]*(1.0+flags.sw[20]*flags.sw[22]*glob7s(pma[9], Input, flags))*meso_tn2[3]*meso_tn2[3]/(pow((pma[2][0]*pavgm[2]),2.0));
    meso_tn3[0]=meso_tn2[3];

    if (Input.alt<zn3[0]):
#/*       LOWER STRATOSPHERE AND TROPOSPHERE (below zn3[0])
#*         Temperature at nodes and gradients at end nodes
#*         Inverse temperature a linear function of spherical harmonics
#*/
        meso_tgn3[0]=meso_tgn2[1];
        meso_tn3[1]=pma[3][0]*pavgm[3]/(1.0-flags.sw[22]*glob7s(pma[3], Input, flags));
        meso_tn3[2]=pma[4][0]*pavgm[4]/(1.0-flags.sw[22]*glob7s(pma[4], Input, flags));
        meso_tn3[3]=pma[5][0]*pavgm[5]/(1.0-flags.sw[22]*glob7s(pma[5], Input, flags));
        meso_tn3[4]=pma[6][0]*pavgm[6]/(1.0-flags.sw[22]*glob7s(pma[6], Input, flags));
        meso_tgn3[1]=pma[7][0]*pavgm[7]*(1.0+flags.sw[22]*glob7s(pma[7], Input, flags)) *meso_tn3[4]*meso_tn3[4]/(pow((pma[6][0]*pavgm[6]),2.0));
    

    #/* LINEAR TRANSITION TO FULL MIXING BELOW zn2[0] */

    dmc=0;
    if (Input.alt>zmix):
        dmc = 1.0 - (zn2[0]-Input.alt)/(zn2[0] - zmix);
    dz28=soutput.d[2];
    
    #/**** N2 density ****/
    dmr=soutput.d[2] / dm28m - 1.0;
    tz = [0.0]
    output.d[2]=densm(Input.alt,dm28m,xmm, tz, mn3, zn3, meso_tn3, meso_tgn3, mn2, zn2, meso_tn2, meso_tgn2);
    output.d[2]=output.d[2] * (1.0 + dmr*dmc);

    #/**** HE density ****/
    dmr = soutput.d[0] / (dz28 * pdm[0][1]) - 1.0;
    output.d[0] = output.d[2] * pdm[0][1] * (1.0 + dmr*dmc);

    #/**** O density ****/
    output.d[1] = 0;
    output.d[8] = 0;

    #/**** O2 density ****/
    dmr = soutput.d[3] / (dz28 * pdm[3][1]) - 1.0;
    output.d[3] = output.d[2] * pdm[3][1] * (1.0 + dmr*dmc);

    #/**** AR density ***/
    dmr = soutput.d[4] / (dz28 * pdm[4][1]) - 1.0;
    output.d[4] = output.d[2] * pdm[4][1] * (1.0 + dmr*dmc);

    #/**** Hydrogen density ****/
    output.d[6] = 0;

    #/**** Atomic nitrogen density ****/
    output.d[7] = 0;

    #/**** Total mass density */
    output.d[5] = 1.66E-24 * (4.0 * output.d[0] + 16.0 * output.d[1] + 28.0 * output.d[2] + 32.0 * output.d[3] + 40.0 * output.d[4] + output.d[6] + 14.0 * output.d[7]);

    if (flags.sw[0]): # pragma: no cover
        output.d[5]=output.d[5]/1000;

    #/**** temperature at altitude ****/
    global dd
    dd = densm(Input.alt, 1.0, 0, tz, mn3, zn3, meso_tn3, meso_tgn3, mn2, zn2, meso_tn2, meso_tgn2);
    output.t[1]=tz[0];
    return


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- GTD7D ----------------------------- */
/* ------------------------------------------------------------------- */
'''
def gtd7d(Input, flags, output): # pragma: no cover
    '''A separate subroutine (GTD7D) computes the ‘‘effec-
    tive’’ mass density by summing the thermospheric mass
    density and the mass density of the anomalous oxygen
    component. Below 500 km, the effective mass density is
    equivalent to the thermospheric mass density, and we drop
    the distinction.
    '''
    gtd7(Input, flags, output)
    output.d[5] = 1.66E-24 * (4.0 * output.d[0] + 16.0 * output.d[1] + 28.0 * output.d[2] + 32.0 * output.d[3] + 40.0 * output.d[4] + output.d[6] + 14.0 * output.d[7] + 16.0 * output.d[8]);
    if (flags.sw[0]):
        output.d[5]=output.d[5]/1000;
    return


'''
/* ------------------------------------------------------------------- */
/* -------------------------------- GHP7 ----------------------------- */
/* ------------------------------------------------------------------- */
'''
def ghp7(Input, flags, output, press): # pragma: no cover
    bm = 1.3806E-19;
    rgas = 831.4;
    #rgas = 831.44621    #maybe make this a global constant?
    test = 0.00043;
    ltest = 12;

    pl = log10(press)

    if (pl >= -5.0):
        if (pl>2.5):
            zi = 18.06 * (3.00 - pl);
        elif ((pl>0.075) and (pl<=2.5)):
            zi = 14.98 * (3.08 - pl);
        elif ((pl>-1) and (pl<=0.075)):
            zi = 17.80 * (2.72 - pl);
        elif ((pl>-2) and (pl<=-1)):
            zi = 14.28 * (3.64 - pl);
        elif ((pl>-4) and (pl<=-2)):
            zi = 12.72 * (4.32 -pl);
        elif (pl<=-4):
            zi = 25.3 * (0.11 - pl);
        cl = Input.g_lat/90.0;
        cl2 = cl*cl;
        if (Input.doy<182):
            cd = (1.0 - float(Input.doy)) / 91.25;
        else:
            cd = (float(Input.doy)) / 91.25 - 3.0;
        ca = 0;
        if ((pl > -1.11) and (pl<=-0.23)):
            ca = 1.0;
        if (pl > -0.23):
            ca = (2.79 - pl) / (2.79 + 0.23);
        if ((pl <= -1.11) and (pl>-3)):
            ca = (-2.93 - pl)/(-2.93 + 1.11);
        z = zi - 4.87 * cl * cd * ca - 1.64 * cl2 * ca + 0.31 * ca * cl;
    else:
        z = 22.0 * pow((pl + 4.0),2.0) + 110.0;

    #/* iteration  loop */
    l = 0;
    while(True):
        l += 1;
        Input.alt = z;
        gtd7(Input, flags, output);
        z = Input.alt;
        xn = output.d[0] + output.d[1] + output.d[2] + output.d[3] + output.d[4] + output.d[6] + output.d[7];
        p = bm * xn * output.t[1];
        if (flags.sw[0]):
            p = p*1.0E-6;
        diff = pl - log10(p);
        if (sqrt(diff*diff)<test):
            return;
        
        xm = output.d[5] / xn / 1.66E-24;
        if (flags.sw[0]):
            xm = xm * 1.0E3;
        g = gsurf[0] / (pow((1.0 + z/re[0]),2.0));
        sh = rgas * output.t[1] / (xm * g);

        #/* new altitude estimate using scale height */
        if (l <  6):
            z = z - sh * diff * 2.302;
        else:
            z = z - sh * diff;
    return


'''
/* ------------------------------------------------------------------- */
/* ------------------------------- GTS7 ------------------------------ */
/* ------------------------------------------------------------------- */
'''
def gts7(Input, flags, output):
    '''
/*     Thermospheric portion of NRLMSISE-00
 *     See GTD7 for more extensive comments
 *     alt > 72.5 km! 
 */
 '''
    zn1 = [120.0, 110.0, 100.0, 90.0, 72.5]
    mn1 = 5
    dgtr=1.74533E-2;
    dr=1.72142E-2;
    alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0]
    altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0]
    za = pdl[1][15];
    zn1[0] = za;

    for j in range(9):
        output.d[j]=0;

    #/* TINF VARIATIONS NOT IMPORTANT BELOW ZA OR ZN1(1) */
    if (Input.alt>zn1[0]):
        tinf = ptm[0]*pt[0] * \
                    (1.0+flags.sw[16]*globe7(pt,Input,flags));
    else:
        tinf = ptm[0]*pt[0];
    output.t[0]=tinf;

    #/*  GRADIENT VARIATIONS NOT IMPORTANT BELOW ZN1(5) */
    if (Input.alt>zn1[4]):
        g0 = ptm[3]*ps[0] * \
            (1.0+flags.sw[19]*globe7(ps,Input,flags));
    else:
        g0 = ptm[3]*ps[0];
    tlb = ptm[1] * (1.0 + flags.sw[17]*globe7(pd[3],Input,flags))*pd[3][0];
    s = g0 / (tinf - tlb);

#/*      Lower thermosphere temp variations not significant for
# *       density above 300 km */
    if (Input.alt<300.0):
        meso_tn1[1]=ptm[6]*ptl[0][0]/(1.0-flags.sw[18]*glob7s(ptl[0], Input, flags));
        meso_tn1[2]=ptm[2]*ptl[1][0]/(1.0-flags.sw[18]*glob7s(ptl[1], Input, flags));
        meso_tn1[3]=ptm[7]*ptl[2][0]/(1.0-flags.sw[18]*glob7s(ptl[2], Input, flags));
        meso_tn1[4]=ptm[4]*ptl[3][0]/(1.0-flags.sw[18]*flags.sw[20]*glob7s(ptl[3], Input, flags));
        meso_tgn1[1]=ptm[8]*pma[8][0]*(1.0+flags.sw[18]*flags.sw[20]*glob7s(pma[8], Input, flags))*meso_tn1[4]*meso_tn1[4]/(pow((ptm[4]*ptl[3][0]),2.0));
    else:
        meso_tn1[1]=ptm[6]*ptl[0][0];
        meso_tn1[2]=ptm[2]*ptl[1][0];
        meso_tn1[3]=ptm[7]*ptl[2][0];
        meso_tn1[4]=ptm[4]*ptl[3][0];
        meso_tgn1[1]=ptm[8]*pma[8][0]*meso_tn1[4]*meso_tn1[4]/(pow((ptm[4]*ptl[3][0]),2.0));
	

    z0 = zn1[3];
    t0 = meso_tn1[3];
    tr12 = 1.0;

    #/* N2 variation factor at Zlb */
    g28=flags.sw[21]*globe7(pd[2], Input, flags);

    #/* VARIATION OF TURBOPAUSE HEIGHT */
    zhf=pdl[1][24]*(1.0+flags.sw[5]*pdl[0][24]*sin(dgtr*Input.g_lat)*cos(dr*(Input.doy-pt[13])));
    output.t[0]=tinf;
    xmm = pdm[2][4];
    z = Input.alt;


    #/**** N2 DENSITY ****/

    #/* Diffusive density at Zlb */
    db28 = pdm[2][0]*exp(g28)*pd[2][0];
    #/* Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[2]=densu(z,db28,tinf,tlb,28.0,alpha[2],RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[2];
    #/* Turbopause */
    zh28=pdm[2][2]*zhf;
    zhm28=pdm[2][3]*pdl[1][5]; 
    xmd=28.0-xmm;
    #/* Mixed density at Zlb */
    tz = [0]
    b28=densu(zh28,db28,tinf,tlb,xmd,(alpha[2]-1.0),tz,ptm[5],s,mn1, zn1,meso_tn1,meso_tgn1);
    if ((flags.sw[15]) and (z<=altl[2])):
        #/*  Mixed density at Alt */
        global dm28
        dm28=densu(z,b28,tinf,tlb,xmm,alpha[2],tz,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        #/*  Net density at Alt */
        output.d[2]=dnet(output.d[2],dm28,zhm28,xmm,28.0);
    


    #/**** HE DENSITY ****/

    #/*   Density variation factor at Zlb */
    g4 = flags.sw[21]*globe7(pd[0], Input, flags);
    #/*  Diffusive density at Zlb */
    db04 = pdm[0][0]*exp(g4)*pd[0][0];
    #/*  Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[0]=densu(z,db04,tinf,tlb, 4.,alpha[0],RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[0];
    if ((flags.sw[15]) and (z<altl[0])):
        #/*  Turbopause */
        zh04=pdm[0][2];
        #/*  Mixed density at Zlb */
        RandomVariable = [output.t[1]]
        b04=densu(zh04,db04,tinf,tlb,4.-xmm,alpha[0]-1.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        #/*  Mixed density at Alt */
        RandomVariable = [output.t[1]]
        global dm04
        dm04=densu(z,b04,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        zhm04=zhm28;
        #/*  Net density at Alt */
        output.d[0]=dnet(output.d[0],dm04,zhm04,xmm,4.);
        #/*  Correction to specified mixing ratio at ground */
        rl=log(b28*pdm[0][1]/b04);
        zc04=pdm[0][4]*pdl[1][0];
        hc04=pdm[0][5]*pdl[1][1];
        #/*  Net density corrected at Alt */
        output.d[0]=output.d[0]*ccor(z,rl,hc04,zc04);
    


    #/**** O DENSITY ****/

    #/*  Density variation factor at Zlb */
    g16= flags.sw[21]*globe7(pd[1],Input,flags);
    #/*  Diffusive density at Zlb */
    db16 =  pdm[1][0]*exp(g16)*pd[1][0];
    #/*   Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[1]=densu(z,db16,tinf,tlb, 16.,alpha[1],RandomVariable,ptm[5],s,mn1, zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[1];
    if ((flags.sw[15]) and (z<=altl[1])):
        #/*   Turbopause */
        zh16=pdm[1][2];
        #/*  Mixed density at Zlb */
        RandomVariable = [output.t[1]]
        b16=densu(zh16,db16,tinf,tlb,16.0-xmm,(alpha[1]-1.0), RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        #/*  Mixed density at Alt */
        RandomVariable = [output.t[1]]
        global dm16
        dm16=densu(z,b16,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        zhm16=zhm28;
        #/*  Net density at Alt */
        output.d[1]=dnet(output.d[1],dm16,zhm16,xmm,16.);
        rl=pdm[1][1]*pdl[1][16]*(1.0+flags.sw[1]*pdl[0][23]*(Input.f107A-150.0));
        hc16=pdm[1][5]*pdl[1][3];
        zc16=pdm[1][4]*pdl[1][2];
        hc216=pdm[1][5]*pdl[1][4];
        output.d[1]=output.d[1]*ccor2(z,rl,hc16,zc16,hc216);
        #/*   Chemistry correction */
        hcc16=pdm[1][7]*pdl[1][13];
        zcc16=pdm[1][6]*pdl[1][12];
        rc16=pdm[1][3]*pdl[1][14];
        #/*  Net density corrected at Alt */
        output.d[1]=output.d[1]*ccor(z,rc16,hcc16,zcc16);
    


    #/**** O2 DENSITY ****/

    #/*   Density variation factor at Zlb */
    g32= flags.sw[21]*globe7(pd[4], Input, flags);
    #/*  Diffusive density at Zlb */
    db32 = pdm[3][0]*exp(g32)*pd[4][0];
    #/*   Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[3]=densu(z,db32,tinf,tlb, 32.,alpha[3],RandomVariable,ptm[5],s,mn1, zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[3];
    if (flags.sw[15]):
        if (z<=altl[3]):
            #/*   Turbopause */
            zh32=pdm[3][2];
            #/*  Mixed density at Zlb */
            RandomVariable = [output.t[1]]
            b32=densu(zh32,db32,tinf,tlb,32.-xmm,alpha[3]-1., RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
            output.t[1] = RandomVariable[0]
            #/*  Mixed density at Alt */
            RandomVariable = [output.t[1]]
            global dm32
            dm32=densu(z,b32,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
            output.t[1] = RandomVariable[0]
            zhm32=zhm28;
            #/*  Net density at Alt */
            output.d[3]=dnet(output.d[3],dm32,zhm32,xmm,32.);
            #/*   Correction to specified mixing ratio at ground */
            rl=log(b28*pdm[3][1]/b32);
            hc32=pdm[3][5]*pdl[1][7];
            zc32=pdm[3][4]*pdl[1][6];
            output.d[3]=output.d[3]*ccor(z,rl,hc32,zc32);
        
        #/*  Correction for general departure from diffusive equilibrium above Zlb */
        hcc32=pdm[3][7]*pdl[1][22];
        hcc232=pdm[3][7]*pdl[0][22];
        zcc32=pdm[3][6]*pdl[1][21];
        rc32=pdm[3][3]*pdl[1][23]*(1.+flags.sw[1]*pdl[0][23]*(Input.f107A-150.));
        #/*  Net density corrected at Alt */
        output.d[3]=output.d[3]*ccor2(z,rc32,hcc32,zcc32,hcc232);
    


    #/**** AR DENSITY ****/

    #/*   Density variation factor at Zlb */
    g40= flags.sw[21]*globe7(pd[5],Input,flags);
    #/*  Diffusive density at Zlb */
    db40 = pdm[4][0]*exp(g40)*pd[5][0];
    #/*   Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[4]=densu(z,db40,tinf,tlb, 40.,alpha[4],RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[4];
    if ((flags.sw[15]) and (z<=altl[4])):
        #/*   Turbopause */
        zh40=pdm[4][2];
        #/*  Mixed density at Zlb */
        RandomVariable = [output.t[1]]
        b40=densu(zh40,db40,tinf,tlb,40.-xmm,alpha[4]-1.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        #/*  Mixed density at Alt */
        RandomVariable = [output.t[1]]
        global dm40
        dm40=densu(z,b40,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        zhm40=zhm28;
        #/*  Net density at Alt */
        output.d[4]=dnet(output.d[4],dm40,zhm40,xmm,40.);
        #/*   Correction to specified mixing ratio at ground */
        rl=log(b28*pdm[4][1]/b40);
        hc40=pdm[4][5]*pdl[1][9];
        zc40=pdm[4][4]*pdl[1][8];
        #/*  Net density corrected at Alt */
        output.d[4]=output.d[4]*ccor(z,rl,hc40,zc40);
      


    #/**** HYDROGEN DENSITY ****/

    #/*   Density variation factor at Zlb */
    g1 = flags.sw[21]*globe7(pd[6], Input, flags);
    #/*  Diffusive density at Zlb */
    db01 = pdm[5][0]*exp(g1)*pd[6][0];
    #/*   Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[6]=densu(z,db01,tinf,tlb,1.,alpha[6],RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[6];
    if ((flags.sw[15]) and (z<=altl[6])):
        #/*   Turbopause */
        zh01=pdm[5][2];
        #/*  Mixed density at Zlb */
        RandomVariable = [output.t[1]]
        b01=densu(zh01,db01,tinf,tlb,1.-xmm,alpha[6]-1., RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        #/*  Mixed density at Alt */
        RandomVariable = [output.t[1]]
        global dm01
        dm01=densu(z,b01,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        zhm01=zhm28;
        #/*  Net density at Alt */
        output.d[6]=dnet(output.d[6],dm01,zhm01,xmm,1.);
        #/*   Correction to specified mixing ratio at ground */
        rl=log(b28*pdm[5][1]*sqrt(pdl[1][17]*pdl[1][17])/b01);
        hc01=pdm[5][5]*pdl[1][11];
        zc01=pdm[5][4]*pdl[1][10];
        output.d[6]=output.d[6]*ccor(z,rl,hc01,zc01);
        #/*   Chemistry correction */
        hcc01=pdm[5][7]*pdl[1][19];
        zcc01=pdm[5][6]*pdl[1][18];
        rc01=pdm[5][3]*pdl[1][20];
        #/*  Net density corrected at Alt */
        output.d[6]=output.d[6]*ccor(z,rc01,hcc01,zcc01);



    #/**** ATOMIC NITROGEN DENSITY ****/

    #/*   Density variation factor at Zlb */
    g14 = flags.sw[21]*globe7(pd[7],Input,flags);
    #/*  Diffusive density at Zlb */
    db14 = pdm[6][0]*exp(g14)*pd[7][0];
    #/*   Diffusive density at Alt */
    RandomVariable = [output.t[1]]
    output.d[7]=densu(z,db14,tinf,tlb,14.,alpha[7],RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    dd=output.d[7];
    if ((flags.sw[15]) and (z<=altl[7])): 
        #/*   Turbopause */
        zh14=pdm[6][2];
        #/*  Mixed density at Zlb */
        RandomVariable = [output.t[1]]
        b14=densu(zh14,db14,tinf,tlb,14.-xmm,alpha[7]-1., RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        #/*  Mixed density at Alt */
        RandomVariable = [output.t[1]]
        global dm14
        dm14=densu(z,b14,tinf,tlb,xmm,0.,RandomVariable,ptm[5],s,mn1,zn1,meso_tn1,meso_tgn1);
        output.t[1] = RandomVariable[0]
        zhm14=zhm28;
        #/*  Net density at Alt */
        output.d[7]=dnet(output.d[7],dm14,zhm14,xmm,14.);
        #/*   Correction to specified mixing ratio at ground */
        rl=log(b28*pdm[6][1]*sqrt(pdl[0][2]*pdl[0][2])/b14);
        hc14=pdm[6][5]*pdl[0][1];
        zc14=pdm[6][4]*pdl[0][0];
        output.d[7]=output.d[7]*ccor(z,rl,hc14,zc14);
        #/*   Chemistry correction */
        hcc14=pdm[6][7]*pdl[0][4];
        zcc14=pdm[6][6]*pdl[0][3];
        rc14=pdm[6][3]*pdl[0][5];
        #/*  Net density corrected at Alt */
        output.d[7]=output.d[7]*ccor(z,rc14,hcc14,zcc14);
    


    #/**** Anomalous OXYGEN DENSITY ****/

    g16h = flags.sw[21]*globe7(pd[8],Input,flags);
    db16h = pdm[7][0]*exp(g16h)*pd[8][0];
    tho = pdm[7][9]*pdl[0][6];
    RandomVariable = [output.t[1]]
    dd=densu(z,db16h,tho,tho,16.,alpha[8],RandomVariable,ptm[5],s,mn1, zn1,meso_tn1,meso_tgn1);
    output.t[1] = RandomVariable[0]
    zsht=pdm[7][5];
    zmho=pdm[7][4];
    zsho=scalh(zmho,16.0,tho);
    output.d[8]=dd*exp(-zsht/zsho*(exp(-(z-zmho)/zsht)-1.));


    #/* total mass density */
    output.d[5] = 1.66E-24*(4.0*output.d[0]+16.0*output.d[1]+28.0*output.d[2]+32.0*output.d[3]+40.0*output.d[4]+ output.d[6]+14.0*output.d[7]);
    db48=1.66E-24*(4.0*db04+16.0*db16+28.0*db28+32.0*db32+40.0*db40+db01+14.0*db14);



    #/* temperature */
    z = sqrt(Input.alt*Input.alt);
    RandomVariable = [output.t[1]]
    ddum = densu(z,1.0, tinf, tlb, 0.0, 0.0, RandomVariable, ptm[5], s, mn1, zn1, meso_tn1, meso_tgn1);
    output.t[1] = RandomVariable[0]
    if (flags.sw[0]): # pragma: no cover
        for i in range(9):
            output.d[i]=output.d[i]*1.0E6;
        output.d[5]=output.d[5]/1000;
    return