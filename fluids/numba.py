# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
SOFTWARE.'''

from __future__ import division
import sys
import importlib.util
import types
import numpy as np
import fluids as normal_fluids
import numba

'''Basic module which wraps all fluids functions with numba's jit.
All other object - dicts, classes, etc - are not wrapped. Supports star 
imports; so the same objects exported when importing from the main library
will be imported from here. 

>>> from fluids.numba import *


>>> fluids.numba.friction_factor(Re=100.0, eD=0.0)
array([ 0.64      ,  0.064     ,  0.03088295])

Note that because this needs to import fluids itself, fluids.numba
needs to be imported separately; the following will cause an error:
    
>>> import fluids
>>> fluids.numba # Won't work, has not been imported yet

The correct syntax is as follows:

>>> import fluids.numba # Necessary
>>> from fluids.numba import * # May be used without first importing fluids
'''

__all__ = []

__funcs = {}

#set_signatures = {'Clamond': [numba.float64(numba.float64, numba.float64, numba.boolean),
#                              numba.float64(numba.float64, numba.float64, numba.optional(numba.boolean))
#                              ]
#                    }
set_signatures = {}

#nopython = set(['Clamond'])
#skip = set(['entrance_sharp'])
    
bad_names = set(('__file__', '__name__', '__package__', '__cached__'))

#for name in dir(normal_fluids):
#    obj = getattr(normal_fluids, name)
#    if isinstance(obj, types.FunctionType):
#        nopython = name not in skip
#        obj = numba.jit(set_signatures.get(name, None), nopython=nopython, forceobj=not nopython,
#                        fastmath=nopython, cache=True)(obj)
#    elif isinstance(obj, str):
#        if name in bad_names:
#            continue
#    __all__.append(name)
#    __funcs.update({name: obj})
#    globals()[name] = obj

NUMERICS_SUBMOD_COPY = importlib.util.find_spec('fluids.numerics')
NUMERICS_SUBMOD = importlib.util.module_from_spec(NUMERICS_SUBMOD_COPY)
NUMERICS_SUBMOD_COPY.loader.exec_module(NUMERICS_SUBMOD)


names = list(NUMERICS_SUBMOD.__all__)
try:
    names += NUMERICS_SUBMOD.__numba_additional_funcs__
except:
    pass

import inspect
source = inspect.getsource(NUMERICS_SUBMOD.secant)
source = source.replace(', kwargs={}', '').replace(', **kwargs', '')
source = source.replace('iterations=i, point=p, err=q1', '')
source = source.replace(', q1=q1, p1=p1, q0=q0, p0=p0', '')
exec(source)
NUMERICS_SUBMOD.secant = secant


numerics_forceobj = set(['secant'])
replaced = {}
for name in names:
    obj = getattr(NUMERICS_SUBMOD, name)
    if isinstance(obj, types.FunctionType):
        forceobj = name in numerics_forceobj
        obj = numba.jit(cache=False, forceobj=forceobj)(obj)
        NUMERICS_SUBMOD.__dict__[name] = obj
        replaced[name] = obj
replaced['bisplev'] = replaced['py_bisplev']
replaced['splev'] = replaced['py_splev']
replaced['lambertw'] = replaced['py_lambertw']


#
#from fluids.numerics import (trunc_exp, trunc_log, roots_cubic, roots_quartic, mean, horner, horner_and_der, horner_and_der2, horner_and_der3, quadratic_from_f_ders, polyder, polyint, chebder, horner_log, fit_integral_linear_extrapolation, best_fit_integral_value, fit_integral_over_T_linear_extrapolation, best_fit_integral_over_T_value, evaluate_linear_fits, evaluate_linear_fits_d, evaluate_linear_fits_d2, chebval, interp, py_bisect, py_ridder, py_brenth, secant, py_newton, newton_system, py_bisplev, fpbspl, init_w, cy_bispev, py_splev, binary_search)
#
#to_replace = [trunc_exp, trunc_log, roots_cubic, roots_quartic, mean, horner, horner_and_der, horner_and_der2, horner_and_der3, quadratic_from_f_ders, polyder, polyint, chebder, horner_log, fit_integral_linear_extrapolation, best_fit_integral_value, fit_integral_over_T_linear_extrapolation, best_fit_integral_over_T_value, evaluate_linear_fits, evaluate_linear_fits_d, evaluate_linear_fits_d2, chebval, interp, py_bisect, py_ridder, py_brenth, secant, py_newton, newton_system, py_bisplev, fpbspl, init_w, cy_bispev, py_splev, binary_search]
#replaced = {}
#for v in to_replace:
#    replaced[v.__name__] = numba.jit(cache=True)(v)
#replaced['bisplev'] = replaced['py_bisplev']
#replaced['splev'] = replaced['py_splev']


new_mods = []
    
# Run module-by-module. Expensive, as we need to create module copies
for mod in normal_fluids.submodules:
#    print(mod)
    FLUIDS_SUBMOD_COPY = importlib.util.find_spec(mod.__name__)
    FLUIDS_SUBMOD = importlib.util.module_from_spec(FLUIDS_SUBMOD_COPY)
    FLUIDS_SUBMOD_COPY.loader.exec_module(FLUIDS_SUBMOD)
    
    FLUIDS_SUBMOD.__dict__.update(replaced)
    new_mods.append(FLUIDS_SUBMOD)
    
    __funcs[mod.__name__.split('fluids.')[1]] = FLUIDS_SUBMOD
    
    names = list(FLUIDS_SUBMOD.__all__)
    try:
        names += FLUIDS_SUBMOD.__numba_additional_funcs__
    except:
        pass

    new_objs = []
    for name in names:
        obj = getattr(FLUIDS_SUBMOD, name)
        if isinstance(obj, types.FunctionType):
#            nopython = name not in skip
            obj = numba.jit(#set_signatures.get(name, None), nopython=False, #forceobj=not nopython,
#                            fastmath=nopython,
                            cache=False)(obj)
            FLUIDS_SUBMOD.__dict__[name] = obj
            new_objs.append(obj)
        __funcs.update({name: obj})

    to_do = {}
    for arr_name in FLUIDS_SUBMOD.__dict__.keys():
        obj = getattr(FLUIDS_SUBMOD, arr_name)
        if type(obj) is list and len(obj) and type(obj[0]) in (float, int, complex):
            to_do[arr_name] = np.array(obj)
        elif type(obj) is list and len(obj) and all([
                (type(r) is list and len(r) and type(r[0]) in (float, int, complex)) for r in obj]):
            
            to_do[arr_name] = np.array(obj)
    FLUIDS_SUBMOD.__dict__.update(to_do)
    __funcs.update(to_do)


    for t in new_objs:
        t.py_func.__globals__.update(FLUIDS_SUBMOD.__dict__)
        t.py_func.__globals__.update(to_do)
        t.py_func.__globals__.update(replaced)
    
    # for name in names:
    #     obj = getattr(FLUIDS_SUBMOD, name)
    #     if isinstance(obj, types.FunctionType):
    #         FLUIDS_SUBMOD.__dict__[name] = obj


globals().update(__funcs)
globals().update(replaced)

# Do our best to allow functions to be found
for mod in new_mods:
    mod.__dict__.update(__funcs)






