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

import hashlib
import os
import subprocess

known_hash = 'bb504fc1ab541260f13b2d2d89884c4d'


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
