# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import pytest


def swap_funcs_and_test(names, substitutions, test):
    '''Function to replace some globals in another function,
    run that function, and then set the globals back.

    names : list[str]
        object names to switch out
    substitutions : list[obj]
        Objects to put in
    test : function
        Unit test to run in the file
    '''
    originals = {}
    glob = test.__globals__
    for name, sub in zip(names, substitutions):
        if name in glob:
            originals[name] = glob[name]
            glob[name] = sub
    try:
        test()
    except Exception as e:
        glob.update(originals)
        raise e
    glob.update(originals)

try:
    import fluids.numba
    numba_substitutions = []
    numba_func_names = []
    for s in fluids.numba.__all__:
        if hasattr(fluids.numba, s):
            f = getattr(fluids.numba, s)
            numba_substitutions.append(f)
            numba_func_names.append(s)
except:
    pass

def swap_for_numba_test(func):
    return swap_funcs_and_test(fluids.numba.__all__, numba_substitutions, func)


def mark_as_numba(func):
    func = pytest.mark.numba(func)
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(numba is None, reason="Numba is missing")(func)
    return func
