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
import re
import types
import inspect
import string
import numpy as np
import fluids as normal_fluids
import numba
from numba import int32, float32, int64, float64
from numba.experimental import jitclass
from numba import cfunc
import linecache
import numba.types
from math import pi


'''Basic module which wraps all fluids functions with numba's jit.
All other object - dicts, classes, etc - are not wrapped. Supports star 
imports; so the same objects exported when importing from the main library
will be imported from here. 

>>> from fluids.numba import *

Note that because this needs to import fluids itself, fluids.numba
needs to be imported separately; the following will cause an error:
    
>>> import fluids
>>> fluids.numba # Won't work, has not been imported yet

The correct syntax is as follows:

>>> import fluids.numba # Necessary
>>> from fluids.numba import * # May be used without first importing fluids
'''

caching = False
__all__ = []

__funcs = {}
no_conv_data_names = set(['__builtins__', 'fmethods'])


def numba_exec_cacheable(source, lcs=None, gbls=None, cache_name='cache-safe'):
    filepath = "<ipython-%s>" % cache_name
    lines = [line + '\n' for line in source.splitlines()]
    linecache.cache[filepath] = (len(source), None, lines, filepath)
    if lcs is None:
        lcs = {}
    if gbls is None:
        gbls = globals()
    exec(compile(source, filepath, 'exec'), gbls, lcs)
    return lcs, gbls

# Some unfotrunate code duplication

@numba.njit
def fpbspl(t, n, k, x, l, h, hh):
    h[0] = 1.0
    for j in range(1, k + 1):
        hh[0:j] = h[0:j]
        h[0] = 0.0
        for i in range(j):
            li = l+i
            f = hh[i]/(t[li] - t[li - j])
            h[i] = h[i] + f*(t[li] - x)
            h[i + 1] = f*(x - t[li - j])
    return h, hh

@numba.njit
def init_w(t, k, x, lx, w):
    tb = t[k]
    n = len(t)
    m = len(x)
    h = np.zeros(6, dtype=np.float64)#([0]*6 )
    hh = np.zeros(5, dtype=np.float64)##np.array([0]*5)
    te = t[n - k - 1]
    l1 = k + 1
    l2 = l1 + 1
    for i in range(m):
        arg = x[i]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
        while not (arg < t[l1] or l1 == (n - k - 1)):
            l1 = l2
            l2 = l1 + 1
        h, hh = fpbspl(t, n, k, arg, l1, h, hh)

        lx[i] = l1 - k - 1
        for j in range(k + 1):
            w[i][j] = h[j]
    return w
@numba.njit
def cy_bispev(tx, ty, c, kx, ky, x, y):
    nx = len(tx)
    ny = len(ty)
    mx = 1 # hardcode to one point
    my = 1 # hardcode to one point

    kx1 = kx + 1
    ky1 = ky + 1

    nkx1 = nx - kx1
    nky1 = ny - ky1
    
    wx = np.zeros((mx, kx1))
    wy = np.zeros((my, ky1))
    lx = np.zeros(mx, dtype=np.int32)
    ly = np.zeros(my, dtype=np.int32)

    size_z = mx*my
    
    z = [0.0]*size_z
    wx = init_w(tx, kx, x, lx, wx)
    wy = init_w(ty, ky, y, ly, wy)
    for j in range(my):
        for i in range(mx):
            sp = 0.0
            err = 0.0
            for i1 in range(kx1):
                for j1 in range(ky1):
                    l2 = lx[i]*nky1 + ly[j] + i1*nky1 + j1
                    a = c[l2]*wx[i][i1]*wy[j][j1] - err
                    tmp = sp + a
                    err = (tmp - sp) - a
                    sp = tmp
            z[j*mx + i] += sp
    return z

    
    
@numba.njit
def bisplev(x, y, tck, dx=0, dy=0):
    tx, ty, c, kx, ky = tck
    return cy_bispev(tx, ty, c, kx, ky, np.array([x]), np.array([y]))[0]



to_set_num = ['bisplev', 'cy_bispev', 'init_w', 'fpbspl']





def infer_dictionary_types(d):
    if not d:
        raise ValueError("Empty dictionary cannot infer")
    keys = list(d.keys())
    type_keys = type(keys[0])
    for k in keys:
        if type(k) != type_keys:
            raise ValueError("Inconsistent key types in dictionary")
    values = list(d.values())
    type_values = type(values[0])
    for v in values:
        if type(v) != type_values:
            raise ValueError("Inconsistent value types in dictionary")
            
    return numba.typeof(keys[0]), numba.typeof(values[0])
    
def numba_dict(d):
    key_type, value_type = infer_dictionary_types(d)
    new = numba.typed.Dict.empty(key_type=key_type, value_type=value_type)
    for k, v in d.items():
        new[k] = v
    return new

def return_value_numpy(source):
    ret = re.search(r'return +\[', source)
    if ret:
        start_return, start_bracket = ret.regs[-1]
        enclosing = 1
        for i, v in enumerate(source[start_bracket:]):
            if v == '[':
                enclosing += 1
            if v == ']':
                enclosing -= 1
            if not enclosing:
                break
        return source[:start_bracket-1] + 'np.array([%s)' %source[start_bracket:i+start_bracket+1]        
    return source


# Magic to make a lists into arrays
list_mult_expr = r'\[ *([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) *\] *\* *([a-zA-Z0-9]+)'
numpy_not_list_expr = r'np.full((\4,), \1)'


def transform_lists_to_arrays(module, to_change, __funcs, vec=False):
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit

    for s in to_change:
        mod, func = s.split('.')
        fake_mod = __funcs[mod]
        source = inspect.getsource(getattr(getattr(module, mod), func))
        source = return_value_numpy(source)
        source = re.sub(list_mult_expr, numpy_not_list_expr, source)
        source = remove_for_numba(source)
#        print(source)
        numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching)(new_func)
        __funcs[func] = obj
        fake_mod.__dict__[func] = obj
        obj.__doc__ = ''


#set_signatures = {'Clamond': [numba.float64(numba.float64, numba.float64, numba.boolean),
#                              numba.float64(numba.float64, numba.float64, numba.optional(numba.boolean))
#                              ]
#                    }


set_signatures = {}

remove_comment_line = re.compile(r'''r?(['"])\1\1(.*?)\1{3}''', re.DOTALL)

def remove_for_numba(source):
    source = re.sub(r'''.*# ?numba ?: *(DELETE|delete).*''', '', source)
    source = re.sub(r'''#(.*)# ?(numba|NUMBA) ?: *(UNCOMMENT|uncomment).*''', r'\1', source)
    return source

def remove_branch(source, branch):
    source = re.sub(remove_comment_line, '', source)
    
    ret = re.search(r'if +%s *' %branch, source)
    if ret:
        start_return, start_bracket = ret.regs[-1]
        enclosing_square = enclosing_curley = enclosing_round = 0
        required_line_start = source[0:start_return].replace('\r', '\n').replace('\n\n','\n').split('\n')[-1]
        required_spacing = 4
        search_txt = source[start_bracket:]
#         print(search_txt)
        for i, v in enumerate(search_txt):
            if v == '[':
                enclosing_square += 1
            if v == ']':
                enclosing_square -= 1
            if v == '{':
                enclosing_curley += 1
            if v == '}':
                enclosing_curley -= 1
            if v == '(':
                enclosing_round += 1
            if v == ')':
                enclosing_round -= 1

            if enclosing_round == 0 and enclosing_square == 0 and enclosing_curley == 0:
                if (search_txt[i:i+len(required_line_start)+1] == '\n' + required_line_start):
#                     print([True, search_txt[i:i+len(required_line_start)+2]])
                    
                    if (search_txt[i+len(required_line_start)+1] in string.ascii_letters):
                        end_idx = i
                        break

        return source[:start_return] + search_txt[end_idx:]
    return source


#nopython = set(['Clamond'])
skip = set([])
total_skip = set([])
skip_cache = set(['secant', 'brenth', 'py_solve'])

bad_names = set(('__file__', '__name__', '__package__', '__cached__'))

from fluids.numerics import SamePointError, UnconvergedError, NotBoundedError
def create_numerics(replaced, vec=False):
    
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit
    
    NUMERICS_SUBMOD_COPY = importlib.util.find_spec('fluids.numerics')
    NUMERICS_SUBMOD = importlib.util.module_from_spec(NUMERICS_SUBMOD_COPY)
    NUMERICS_SUBMOD.IS_NUMBA = True
    NUMERICS_SUBMOD.FORCE_PYPY = True
    NUMERICS_SUBMOD.array_if_needed = np.array
    
    NUMERICS_SUBMOD_COPY.loader.exec_module(NUMERICS_SUBMOD)
    
    # So long as the other modules are using the system numerics and being updated with the correct numerics methods later
    # numba wants to make sure these are the same
    same_classes = ['OscillationError', 'UnconvergedError', 'SamePointError', 'NoSolutionError', 'NotBoundedError', 'DiscontinuityError']
    for s in same_classes:
        setattr(NUMERICS_SUBMOD, s, getattr(normal_fluids.numerics, s))

    names = list(NUMERICS_SUBMOD.__all__)
    try:
        names += NUMERICS_SUBMOD.__numba_additional_funcs__
    except:
        pass
    
    NUMERICS_SUBMOD.py_solve = np.linalg.solve
    
    bad_names = set(['tck_interp2d_linear', 'implementation_optimize_tck'])
    bad_names.update(to_set_num)
    
    solvers = ['secant', 'brenth', 'newton', 'ridder', 'newton_system'] # 
    for s in solvers:
        source = inspect.getsource(getattr(NUMERICS_SUBMOD, s))
        source = source.replace(', kwargs={}', '').replace(', **kwargs', '').replace(', kwargs=kwargs', '')
        source = source.replace('iterations=i, point=p, err=q1', '')
        source = source.replace(', q1=q1, p1=p1, q0=q0, p0=p0', '')
        source = source.replace('%d iterations" %maxiter', '"')
        source = source.replace('ytol=None', 'ytol=1e100')
        source = source.replace(', value=%s" %(maxiter, x)', '"')
        source = re.sub(r'''UnconvergedError\(.*''', '''UnconvergedError("Failed to converge")''', source) # Gotta keep errors all one one line
        source = remove_for_numba(source)
#        if s == 'newton_system':
#            print(source)
        numba_exec_cacheable(source, NUMERICS_SUBMOD.__dict__, NUMERICS_SUBMOD.__dict__)


#    numerics_forceobj = set(solvers) # Force the sovlers to compile in object mode
#    numerics_forceobj = []
    for name in names:
        if name not in bad_names:
            obj = getattr(NUMERICS_SUBMOD, name)
            if isinstance(obj, types.FunctionType):
#                forceobj = name in numerics_forceobj
#                forceobj = False
                # cache=not forceobj 
                # cache=name not in skip_cache
                obj = numba.jit(cache=caching, forceobj=False)(obj)
                NUMERICS_SUBMOD.__dict__[name] = obj
                replaced[name] = obj
#                globals()[name] = objs
            
    for name in to_set_num:
        NUMERICS_SUBMOD.__dict__[name] = globals()[name]
            
    replaced['bisplev'] = replaced['py_bisplev'] = NUMERICS_SUBMOD.__dict__['bisplev'] = bisplev
#    replaced['lambertw'] = NUMERICS_SUBMOD.__dict__['lambertw'] = NUMERICS_SUBMOD.__dict__['py_lambertw']
    for s in ('ellipe', 'gammaincc', 'gamma', 'i1', 'i0', 'k1', 'k0', 'iv', 'hyp2f1', 'erf'):
        replaced[s] = NUMERICS_SUBMOD.__dict__[s]
    
#    replaced['splev'] = NUMERICS_SUBMOD.__dict__['splev']  = replaced['py_splev']
#    replaced['lambertw'] = NUMERICS_SUBMOD.__dict__['lambertw'] = replaced['py_lambertw']
    
#    @numba.njit
#    def newton_err(x):
#        return np.abs(np.array(x), dtype=np.float64).sum()
#    replaced['newton_err'] = NUMERICS_SUBMOD.newton_err = newton_err
    return replaced, NUMERICS_SUBMOD

replaced = {'sum': np.sum}
replaced, NUMERICS_SUBMOD = create_numerics(replaced, vec=False)
numerics = NUMERICS_SUBMOD
#old_numerics = sys.modules['fluids.numerics']
#sys.modules['fluids.numerics'] = numerics
normal = normal_fluids


def transform_module(normal, __funcs, replaced, vec=False):
    new_mods = []
    
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit
    mod_name = normal.__name__
    # Run module-by-module. Expensive, as we need to create module copies
    for mod in normal.submodules:
        SUBMOD_COPY = importlib.util.find_spec(mod.__name__)
        SUBMOD = importlib.util.module_from_spec(SUBMOD_COPY)
        SUBMOD.IS_NUMBA = True
        SUBMOD_COPY.loader.exec_module(SUBMOD)
        SUBMOD.np = np
        
        SUBMOD.__dict__.update(replaced)
        new_mods.append(SUBMOD)
        
        __funcs[mod.__name__.split(mod_name + '.')[1]] = SUBMOD
        
        names = list(SUBMOD.__all__)
        try:
            names += SUBMOD.__numba_additional_funcs__
        except:
            pass
    
        new_objs = []
        for name in names:
            obj = getattr(SUBMOD, name)
            if isinstance(obj, types.FunctionType):
                nopython = name not in skip
                if name not in total_skip:
                    obj = conv_fun(#set_signatures.get(name, None), 
                            nopython=nopython,
                            #forceobj=not nopython,
                                    fastmath=nopython,#Parallel=nopython
                                    cache=caching)(obj)
                SUBMOD.__dict__[name] = obj
                new_objs.append(obj)
            __funcs[name] = obj
    
        module_constants_changed_type = {}
        for arr_name in SUBMOD.__dict__.keys():
            if arr_name not in no_conv_data_names:
                obj = getattr(SUBMOD, arr_name)
                obj_type = type(obj)
                if obj_type is list and len(obj) and type(obj[0]) in (float, int, complex):
                    module_constants_changed_type[arr_name] = np.array(obj)
                elif obj_type is list and len(obj) and all([
                        (type(r) is list and len(r) and type(r[0]) in (float, int, complex)) for r in obj]):
                    module_constants_changed_type[arr_name] = np.array(obj)
                elif obj_type in (set, frozenset):
                    module_constants_changed_type[arr_name] = tuple(obj)
                elif obj_type is dict:
                    continue
                    try:
                        print('starting', arr_name)
                        infer_dictionary_types(obj)
                        module_constants_changed_type[arr_name] = numba_dict(obj)
                    except:
                        print(arr_name, 'failed')
                        pass
        
##        print(SUBMOD)
#        if 'h0_Gorenflow_1993' in SUBMOD.__dict__:
##        if hasattr, ''):
#            module_constants_changed_type['h0_Gorenflow_1993'] =  numba_dict(SUBMOD.h0_Gorenflow_1993)
#            #SUBMOD.__dict__['h0_Gorenflow_1993'] = numba_dict(SUBMOD.h0_Gorenflow_1993)
#
                
        SUBMOD.__dict__.update(module_constants_changed_type)
        __funcs.update(module_constants_changed_type)
    
        if not vec:
            for t in new_objs:
                try:
                    glob = t.py_func.__globals__
                except:
                    glob = t.__globals__
                glob.update(SUBMOD.__dict__)
                #glob.update(to_do)
                glob.update(replaced)
    
    # Do our best to allow functions to be found
    for mod in new_mods:
        mod.__dict__.update(__funcs)
        mod.__dict__.update(replaced)
        
    return new_mods


def transform_complete(replaced, __funcs, __all__, normal, vec=False):
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit
    new_mods = transform_module(normal, __funcs, replaced, vec=vec)

    
    # Do some classes by hand
    
    PlateExchanger_spec = [
        ('pitch', float64),
        ('beta', float64),
        ('gamma', float64),
        ('a', float64),
        ('amplitude', float64),               
        ('wavelength', float64),               
        ('b', float64),               
        ('chevron_angle', float64),               
        ('inclination_angle', float64),               
        ('plate_corrugation_aspect_ratio', float64),               
        ('plate_enlargement_factor', float64),               
        ('D_eq', float64),               
        ('D_hydraulic', float64),               
        ('width', float64),               
        ('length', float64),               
        ('thickness', float64),               
        ('d_port', float64),               
        ('plates', float64),               
        ('length_port', float64),               
        ('A_plate_surface', float64),               
        ('A_heat_transfer', float64),               
        ('A_channel_flow', float64),               
        ('channels', float64),               
        ('channels_per_fluid', float64),               
    ]
    
    
    HelicalCoil_spec = [(k, float64) for k in 
                        ('Do', 'Dt', 'Di', 'Do_total', 'N', 'pitch', 'H', 'H_tot', 
                         'tube_circumference', 'tube_length', 'surface_area', 'helix_angle',
                         'curvature', 'total_inlet_area', 'total_volume', 'inner_surface_area',
                         'inlet_area', 'inner_volume', 'annulus_area', 'annulus_volume')]
    
    ATMOSPHERE_1976_spec = [(k, float64) for k in 
                        ('Z', 'dT', 'H', 'T_layer', 'T_increase', 'P_layer', 'H_layer', 'H_above_layer', 
                         'T', 'P', 'rho', 'v_sonic',
                         'mu', 'k', 'g', 'R')]
    
    to_change = ['packed_tower._Stichlmair_flood_f_and_jac', 
                 'packed_tower.Stichlmair_flood']
    transform_lists_to_arrays(normal_fluids, to_change, __funcs, vec=vec)
    
    
    # AvailableMethods  will be removed in the future in favor of non-numba only 
    # calls to method functions
    
    to_change_AvailableMethods = ['friction.friction_factor_curved', 'friction.friction_factor',
     'packed_bed.dP_packed_bed', 'two_phase.two_phase_dP', 'drag.drag_sphere',
     'two_phase_voidage.liquid_gas_voidage', 'two_phase_voidage.gas_liquid_viscosity']
    
    
    to_change_full_output = ['two_phase.Mandhane_Gregory_Aziz_regime',
                             'two_phase.Taitel_Dukler_regime']
    
    to_change = {k: 'AvailableMethods' for k in to_change_AvailableMethods}
    to_change.update({k: 'full_output' for k in to_change_full_output})
    to_change['fittings.Darby3K'] = 'name in Darby: # NUMBA: DELETE'
    to_change['fittings.Hooper2K'] = 'name in Hooper: # NUMBA: DELETE'
    to_change['friction.roughness_Farshad'] = 'ID in _Farshad_roughness'
    
    for s, bad_branch in to_change.items():
        mod, func = s.split('.')
        source = inspect.getsource(getattr(getattr(normal_fluids, mod), func))
        fake_mod = __funcs[mod]
        source = remove_branch(source, bad_branch)
        source = remove_for_numba(source)
        numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching)(new_func)
        __funcs[func] = obj
#        globals()[func] = obj
        obj.__doc__ = ''
        
        
    to_change = ['compressible.isothermal_gas']
    for s in to_change:
        mod, func = s.split('.')
        source = inspect.getsource(getattr(getattr(normal_fluids, mod), func))
        fake_mod = __funcs[mod]
        source = remove_for_numba(source)
        numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching)(new_func)
        __funcs[func] = obj
#        globals()[func] = obj
        obj.__doc__ = ''
    
    
    
    # Almost there but one argument has a variable type
    #PlateExchanger = jitclass(PlateExchanger_spec)(getattr(__funcs['geometry'], 'PlateExchanger'))
    #HelicalCoil = jitclass(HelicalCoil_spec)(getattr(__funcs['geometry'], 'HelicalCoil'))
    ATMOSPHERE_1976 = jitclass(ATMOSPHERE_1976_spec)(getattr(__funcs['atmosphere'], 'ATMOSPHERE_1976'))
    __funcs['ATMOSPHERE_1976'] = __funcs['atmosphere'].ATMOSPHERE_1976 = ATMOSPHERE_1976
    
    
    # Not needed
    __funcs['friction'].Colebrook = __funcs['Colebrook'] = __funcs['Clamond']
    #for k in ('flow_meter', 'fittings', 'two_phase', 'friction'):
    #    __funcs[k].friction_factor = __funcs['friction_factor'] = __funcs['Clamond']
    #__funcs['PlateExchanger'] = __funcs['geometry'].PlateExchanger = PlateExchanger
    #__funcs['HelicalCoil'] = __funcs['geometry'].HelicalCoil = HelicalCoil
    
    # Works but 50% slower
    #__funcs['geometry']._V_horiz_spherical_toint = __funcs['_V_horiz_spherical_toint'] = cfunc("float64(float64, float64, float64, float64)")(normal_fluids.geometry._V_horiz_spherical_toint)
    
    # ex = fluids.numba.geometry.PlateExchanger(amplitude=5E-4, wavelength=3.7E-3, length=1.2, width=.3, d_port=.05, plates=51, thickness=1e-10)
    #fluids.numba.geometry.HelicalCoil(Do_total=32.0, H_total=22.0, pitch=5.0, Dt=2.0, Di=1.8)
    
    for mod in new_mods:
        mod.__dict__.update(__funcs)

transform_complete(replaced, __funcs, __all__, normal, vec=False)


globals().update(__funcs)
globals().update(replaced)

#sys.modules['fluids.numerics'] = old_numerics







