# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import hashlib
import os
import subprocess
import pytest

known_hash = 'bb504fc1ab541260f13b2d2d89884c4d'

@pytest.mark.slow
def test_NRLMSISE00_against_C_output():
    # Test results currently match up exactly with those of the C test file.
    script = os.path.join(os.path.dirname(__file__), 'nrlmsise_00_test.py')
    # Load known data
    known = os.path.join(os.path.dirname(__file__), 'data_from_C_version.txt')
    # On a separate process, run the test script, and capture its output
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        # Run the test only on linux; print statements to files have different
        # formats on windows
        proc = subprocess.Popen(["python", script], stdout=subprocess.PIPE)
        response = proc.communicate()[0]  
    
        # Hash it, check it is as expected.
        hasher = hashlib.md5()
        hasher.update(response)
        expect = hasher.hexdigest()
        assert expect == known_hash
