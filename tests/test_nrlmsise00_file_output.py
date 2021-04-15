# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

The `gtd7_file_output` test is copyright:

12/19/2013
Author: Joshua Milas
Python Version: 3.3.2

The NRLMSISE-00 model 2001 ported to python
Based off of Dominik Brodowski 20100516 version available here
http://www.brodo.de/english/pub/nrlmsise/

This is the test program, and the output should be compared to

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

'''

from __future__ import division
import time
from fluids.nrlmsise00.nrlmsise_00 import gtd7
from fluids.nrlmsise00.nrlmsise_00_header import nrlmsise_output, nrlmsise_input, nrlmsise_flags , ap_array
import hashlib
import os
import pytest


base = ''

def build_file(txt, end=''):
    # TODO: replace print with this call. Return a giant string, not a plain text output.
    # It is important to not have to start a new python process - slow
    global base
    base += txt + end

def gtd7_file_output():
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
        build_file('\n', end='')
        for j in range(9):
            build_file('%E ' % output[i].d[j], end='')
        build_file('%E ' % output[i].t[0], end='')
        build_file('%E ' % output[i].t[1], end='\n')
        #/* DL omitted */

    #/* output type 2 */
    for i in range(3):
        build_file('\n', end='')
        build_file("\nDAY   ", end='')
        for j in range(5):
            build_file("         %3i" % Input[i*5+j].doy, end='')
        build_file("\nUT    ", end='')
        for j in range(5):
            build_file("       %5.0f" % Input[i*5+j].sec, end='')
        build_file("\nALT   ", end='')
        for j in range(5):
            build_file("        %4.0f" % Input[i*5+j].alt, end='')
        build_file("\nLAT   ", end='')
        for j in range(5):
            build_file("         %3.0f" % Input[i*5+j].g_lat, end='')
        build_file("\nLONG  ", end='')
        for j in range(5):
            build_file("         %3.0f" % Input[i*5+j].g_long, end='')
        build_file("\nLST   ", end='')
        for j in range(5):
            build_file("       %5.0f" % Input[i*5+j].lst, end='')
        build_file("\nF107A ", end='')
        for j in range(5):
            build_file("         %3.0f" % Input[i*5+j].f107A, end='')
        build_file("\nF107  ", end='')
        for j in range(5):
            build_file("         %3.0f" % Input[i*5+j].f107, end='')

        build_file('\n\n', end='')

        build_file("\nTINF  ", end='')
        for j in range(5):
            build_file("     %7.2f" % output[i*5+j].t[0], end='')
        build_file("\nTG    ", end='')
        for j in range(5):
            build_file("     %7.2f" % output[i*5+j].t[1], end='')
        build_file("\nHE    ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[0], end='')
        build_file("\nO     ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[1], end='')
        build_file("\nN2    ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[2], end='')
        build_file("\nO2    ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[3], end='')
        build_file("\nAR    ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[4], end='')
        build_file("\nH     ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[6], end='')
        build_file("\nN     ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[7], end='')
        build_file("\nANM   ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[8], end='')
        build_file("\nRHO   ", end='')
        for j in range(5):
            build_file("   %1.3e" % output[i*5+j].d[5], end='')
        build_file('\n', '\n')


known_hash = 'bb504fc1ab541260f13b2d2d89884c4d'

@pytest.mark.slow
def test_NRLMSISE00_against_C_output():
    global base
    # Test results currently match up exactly with those of the C test file.
    script = os.path.join(os.path.dirname(__file__), 'nrlmsise_00_test.py')
    # Load known data
    known = os.path.join(os.path.dirname(__file__), 'data_from_C_version.txt')
    # On a separate process, run the test script, and capture its output
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        # Run the test only on linux; print statements to files have different
        # formats on windows
        gtd7_file_output()
        response = base
        try:
            response = response.encode("utf-8")
        except:
            pass

        #proc = subprocess.Popen(["python", script], stdout=subprocess.PIPE)
        #response = proc.communicate()[0]

        # Hash it, check it is as expected.
        hasher = hashlib.md5()
        hasher.update(response)
        expect = hasher.hexdigest()
        assert expect == known_hash
    base = ''
