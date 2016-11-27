# -*- coding: utf-8 -*-
"""
12/19/2013
Author: Joshua Milas
Python Version: 3.3.2

The NRLMSISE-00 model 2001 ported to python
Based off of Dominik Brodowski 20100516 version available here
http://www.brodo.de/english/pub/nrlmsise/

This is the test program, and the output should be compaired to

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

from __future__ import division, print_function
import time
from fluids.nrlmsise00.nrlmsise_00 import gtd7
from fluids.nrlmsise00.nrlmsise_00_header import nrlmsise_output, nrlmsise_input, nrlmsise_flags , ap_array

def test_gtd7():
    output = [nrlmsise_output() for _ in range(17)]
    Input = [nrlmsise_input() for _ in range(17)]
    flags = nrlmsise_flags()
    aph = ap_array()

    for i in range(7):
        aph.a[i]=100
    flags.switches[0] = 0
    for i in range(1, 24):
        flags.switches[i]=1
        
    for i in range(17):
        Input[i].doy=172;
        Input[i].year=0; #/* without effect */
        Input[i].sec=29000;
        Input[i].alt=400;
        Input[i].g_lat=60;
        Input[i].g_long=-70;
        Input[i].lst=16;
        Input[i].f107A=150;
        Input[i].f107=150;
        Input[i].ap=4;
	
    Input[1].doy=81;
    Input[2].sec=75000;
    Input[2].alt=1000;
    Input[3].alt=100;
    Input[10].alt=0;
    Input[11].alt=10;
    Input[12].alt=30;
    Input[13].alt=50;
    Input[14].alt=70;
    Input[16].alt=100;
    Input[4].g_lat=0;
    Input[5].g_long=0;
    Input[6].lst=4;
    Input[7].f107A=70;
    Input[8].f107=180;
    Input[9].ap=40;

    Input[15].ap_a = aph
    Input[16].ap_a = aph

    #evaluate 0 to 14
    for i in range(15):
        gtd7(Input[i], flags, output[i])

    #/* evaluate 15 and 16 */
    flags.switches[9] = -1
    for i in range(15, 17):
        gtd7(Input[i], flags, output[i])

    #/* output type 1 */
    for i in range(17):
        print('\n', end='')
        for j in range(9):
            print('%E ' % output[i].d[j], end='')
        print('%E ' % output[i].t[0], end='')
        print('%E ' % output[i].t[1])
        #/* DL omitted */

    #/* output type 2 */
    for i in range(3):
        print('\n', end='')
        print("\nDAY   ", end='')
        for j in range(5):
            print("         %3i" % Input[i*5+j].doy, end='')
        print("\nUT    ", end='')
        for j in range(5):
            print("       %5.0f" % Input[i*5+j].sec, end='')
        print("\nALT   ", end='')
        for j in range(5):
            print("        %4.0f" % Input[i*5+j].alt, end='')
        print("\nLAT   ", end='')
        for j in range(5):
            print("         %3.0f" % Input[i*5+j].g_lat, end='')
        print("\nLONG  ", end='')
        for j in range(5):
            print("         %3.0f" % Input[i*5+j].g_long, end='')
        print("\nLST   ", end='')
        for j in range(5):
            print("       %5.0f" % Input[i*5+j].lst, end='')
        print("\nF107A ", end='')
        for j in range(5):
            print("         %3.0f" % Input[i*5+j].f107A, end='')
        print("\nF107  ", end='')
        for j in range(5):
            print("         %3.0f" % Input[i*5+j].f107, end='')

        print('\n\n', end='')
        
        print("\nTINF  ", end='')
        for j in range(5):
            print("     %7.2f" % output[i*5+j].t[0], end='')
        print("\nTG    ", end='')
        for j in range(5):
            print("     %7.2f" % output[i*5+j].t[1], end='')
        print("\nHE    ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[0], end='')
        print("\nO     ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[1], end='')
        print("\nN2    ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[2], end='')
        print("\nO2    ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[3], end='')
        print("\nAR    ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[4], end='')
        print("\nH     ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[6], end='')
        print("\nN     ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[7], end='')
        print("\nANM   ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[8], end='')
        print("\nRHO   ", end='')
        for j in range(5):
            print("   %1.3e" % output[i*5+j].d[5], end='')
        print('\n')







if __name__ == '__main__':
#    start = time.clock()
    test_gtd7()
#    print(time.clock() - start)
