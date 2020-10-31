# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from inspect import isclass
import types

try:
    import importlib.machinery
    import importlib.util
except ImportError: # pragma: no cover
    raise ImportError('The typing functionality requires a more recent version of Python')


__all__ = ['type_module', 'copy_types']


def copy_types(typed_obj, untyped_obj):
    if isinstance(typed_obj, property):
        return
    try:
        an = typed_obj.__annotations__
    except:
        an = {}
    untyped_obj.__annotations__ = an
    if isclass(untyped_obj):
        for f_name in dir(typed_obj):
            if len(f_name) > 2 and '__' == f_name[:2]:
                continue
            typed_fun = getattr(typed_obj, f_name)
            untyped_fun = getattr(untyped_obj, f_name)
            copy_types(typed_fun, untyped_fun)

def type_module(mod):
    loader = importlib.machinery.SourceFileLoader('dummy_module', mod.__file__ + 'i')
    mod_types = types.ModuleType(loader.name)
    loader.exec_module(mod_types)

    for f_name in mod.__all__:
        if hasattr(mod_types, f_name):
            untyped_fun = getattr(mod, f_name)
            typed_fun = getattr(mod_types, f_name)
            copy_types(typed_fun, untyped_fun)
