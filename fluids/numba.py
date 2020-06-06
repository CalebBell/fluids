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

caching = False
__all__ = []

__funcs = {}


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










#set_signatures = {'Clamond': [numba.float64(numba.float64, numba.float64, numba.boolean),
#                              numba.float64(numba.float64, numba.float64, numba.optional(numba.boolean))
#                              ]
#                    }


set_signatures = {}

remove_comment_line = re.compile(r'''r?(['"])\1\1(.*?)\1{3}''', re.DOTALL)

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
skip = set(['V_horiz_spherical'])
total_skip = set(['V_horiz_spherical'])
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
    NUMERICS_SUBMOD.FORCE_PYPY = True
    NUMERICS_SUBMOD.array_if_needed = np.array
    
    NUMERICS_SUBMOD_COPY.loader.exec_module(NUMERICS_SUBMOD)

    names = list(NUMERICS_SUBMOD.__all__)
    try:
        names += NUMERICS_SUBMOD.__numba_additional_funcs__
    except:
        pass
    
#    NUMERICS_SUBMOD.py_solve = np.linalg.solve
    
    
    import inspect
    solvers = ['secant', 'brenth'] # newton_system
    for s in solvers:
        source = inspect.getsource(getattr(NUMERICS_SUBMOD, s))
        source = source.replace(', kwargs={}', '').replace(', **kwargs', '')
        source = source.replace('iterations=i, point=p, err=q1', '')
        source = source.replace(', q1=q1, p1=p1, q0=q0, p0=p0', '')
        source = source.replace('%d iterations" %maxiter', '"')
        source = source.replace('ytol=None', 'ytol=1e100')
        source = source.replace(', value=%s" %(maxiter, x)', '"')
        
        numba_exec_cacheable(source, globals(), globals())
        setattr(NUMERICS_SUBMOD, s, globals()[s])


    numerics_forceobj = set(solvers) # Force the sovlers to compile in object mode
    numerics_forceobj = []
    for name in names:
        obj = getattr(NUMERICS_SUBMOD, name)
        if isinstance(obj, types.FunctionType):
            forceobj = name in numerics_forceobj
            # cache=not forceobj 
            # cache=name not in skip_cache
            obj = numba.jit(cache=caching, forceobj=forceobj)(obj)
            NUMERICS_SUBMOD.__dict__[name] = obj
            replaced[name] = obj
            
    for name in to_set_num:
        NUMERICS_SUBMOD.__dict__[name] = globals()[name]
    replaced['py_bisplev'] = globals()['bisplev']
            
    replaced['bisplev'] = NUMERICS_SUBMOD.__dict__['bisplev'] = replaced['py_bisplev']
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
        SUBMOD_COPY.loader.exec_module(SUBMOD)
        
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
            __funcs.update({name: obj})
    
        to_do = {}
        for arr_name in SUBMOD.__dict__.keys():
            obj = getattr(SUBMOD, arr_name)
            if type(obj) is list and len(obj) and type(obj[0]) in (float, int, complex):
                to_do[arr_name] = np.array(obj)
            elif type(obj) is list and len(obj) and all([
                    (type(r) is list and len(r) and type(r[0]) in (float, int, complex)) for r in obj]):
                to_do[arr_name] = np.array(obj)
            elif type(obj) in (set, frozenset):
                to_do[arr_name] = tuple(obj)
                
        SUBMOD.__dict__.update(to_do)
        __funcs.update(to_do)
    
        if not vec:
            for t in new_objs:
                try:
                    glob = t.py_func.__globals__
                except:
                    glob = t.__globals__
                glob.update(SUBMOD.__dict__)
                glob.update(to_do)
                glob.update(replaced)
    
    # Do our best to allow functions to be found
    for mod in new_mods:
        mod.__dict__.update(__funcs)
    return new_mods


new_mods = transform_module(normal, __funcs, replaced, vec=False)


# Do some classes by hand
from numba import int32, float32, int64, float64
from math import pi

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

to_change_AvailableMethods = ['friction.friction_factor_curved', 'friction.friction_factor',
 'packed_bed.dP_packed_bed', 'two_phase.two_phase_dP', 'drag.drag_sphere',
 'two_phase_voidage.liquid_gas_voidage', 'two_phase_voidage.gas_liquid_viscosity']
to_change_full_output = ['two_phase.Mandhane_Gregory_Aziz_regime',
                         'two_phase.Taitel_Dukler_regime']

to_change = {k: 'AvailableMethods' for k in to_change_AvailableMethods}
to_change.update({k: 'full_output' for k in to_change_full_output})

for s, bad_branch in to_change.items():
    mod, func = s.split('.')
    source = inspect.getsource(getattr(getattr(normal_fluids, mod), func))
    fake_mod = __funcs[mod]
    source = remove_branch(source, bad_branch)
#    if s == 'friction.friction_factor_curved':
#        print(source)
    numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
    
    new_func = fake_mod.__dict__[func]
#    print(new_func)
    obj = numba.jit(cache=caching)(new_func)
#    setattr(source, func, obj)
    __funcs[func] = obj
    globals()[func] = obj
    obj.__doc__ = ''


# Almost there but one argument has a variable type
#PlateExchanger = jitclass(PlateExchanger_spec)(getattr(__funcs['geometry'], 'PlateExchanger'))
#HelicalCoil = jitclass(HelicalCoil_spec)(getattr(__funcs['geometry'], 'HelicalCoil'))

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

globals().update(__funcs)
globals().update(replaced)







