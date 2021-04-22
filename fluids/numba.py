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
import fluids.optional.spa
import ctypes
from numba.extending import get_cython_function_address
from numba.extending import overload


caching = True
extra_args_std = {'nogil': True, 'fastmath': True}
extra_args_vec = {}
__all__ = []

__funcs = {}
no_conv_data_names = set(['__builtins__', 'fmethods'])

try:
    import scipy.special as sc
    name_to_numba_signatures = {
        'ellipe': [(float64,)],
        'iv': [(float64, float64,)],
        'gamma': [(float64,)],
        'gammainc': [(float64, float64,)],
        'gammaincc': [(float64, float64,)],
        'i0': [(float64,)],
        'i1': [(float64,)],
        'k0': [(float64,)],
        'k1': [(float64,)],
        'hyp2f1': [(float64, float64, float64, float64,)],
        'ellipkinc': [(float64, float64,)],
        'ellipeinc': [(float64, float64,)],
        'erf': [(float64,)],
    }

    name_and_types_to_pointer = {
        ('ellipe', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'ellipe')),
        ('iv', float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1iv')),
        ('gamma', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1gamma')),
        ('gammainc', float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'gammainc')),
        ('gammaincc', float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'gammaincc')),
        ('i0', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'i0')),
        ('i1', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'i1')),
        ('k0', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'k0')),
        ('k1', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'k1')),
        ('hyp2f1', float64, float64, float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1hyp2f1')),
        ('ellipkinc', float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'ellipkinc')),
        ('ellipeinc', float64, float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', 'ellipeinc')),
        ('erf', float64): ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1erf')),
    }

    def select_kernel(name, signature):
        f2 = name_and_types_to_pointer[(name, *signature)]
        second_lambda = lambda *args: lambda *args: f2(*args)
        return second_lambda

    def add_scipy_special_overloads():
        for name, sigs in name_to_numba_signatures.items():
            sig = sigs[0] # Sig is a tuple of arguments
            func = getattr(sc, name)
            overload(func)(select_kernel(name, sigs[0]))

    add_scipy_special_overloads()

except:
    pass


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

@numba.njit(cache=caching, **extra_args_std)
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

@numba.njit(cache=caching, **extra_args_std)
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
@numba.njit(cache=caching, **extra_args_std)
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


@numba.njit(cache=caching, **extra_args_std)
def normalize(values):
    tot_inv = 1.0/sum(values)
    return np.array([i*tot_inv for i in values])

@numba.njit(cache=caching, **extra_args_std)
def bisplev(x, y, tck, dx=0, dy=0):
    tx, ty, c, kx, ky = tck
    return cy_bispev(tx, ty, c, kx, ky, np.array([x]), np.array([y]))[0]


@numba.njit(cache=caching, **extra_args_std)
def combinations(pool, r):
    n = len(pool)
#    indices = tuple(list(range(r)))
    indices = np.arange(r)
    empty = not (n and (0 < r <= n))

    if not empty:
#        yield [pool[i] for i in indices]
#        yield (pool[i] for i in indices)
        yield np.array([pool[i] for i in indices])

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            result = np.array([pool[i] for i in indices])
            yield result



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
list_mult_expr = r'\[ *([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)) *\] *\* *([a-zA-Z0-9_]+)'
numpy_not_list_expr = r'np.full((\4,), \1)'

match_prange = r'range\( *([a-zA-Z0-9_]+) *\) *: *# * (numba|NUMBA) *: *(prange|PRANGE)'
sub_prange = r'prange(\1):'

def transform_lists_to_arrays(module, to_change, __funcs, vec=False, cache_blacklist=set([])):
    if vec:
        conv_fun = numba.vectorize
        extra_args = extra_args_vec
    else:
        conv_fun = numba.njit
        extra_args = extra_args_std
    for s in to_change:
        func = s.split('.')[-1]
        mod = '.'.join(s.split('.')[:-1])
        fake_mod = __funcs[mod]

        try:
            real_mod = getattr(module, mod)
        except:
            real_mod = module
            for s in mod.split('.'):
                real_mod = getattr(real_mod, s)

        orig_func = getattr(real_mod, func)
        source = inspect.getsource(orig_func)
        source = remove_for_numba(source)  # do before anything else
        if type(orig_func) is not type:
            source = return_value_numpy(source)
            source = re.sub(list_mult_expr, numpy_not_list_expr, source)
            parallel = 'prange' in source
            source = re.sub(match_prange, sub_prange, source)
#        if 'roughness_Farshad' in source:
#            print(source)
#            print(parallel, 'hi', extra_args)
        numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        do_cache = caching and func not in cache_blacklist
        if type(orig_func) is type:
            obj = new_func
        else:
            obj = conv_fun(cache=do_cache, parallel=parallel, **extra_args)(new_func)
#        if 'Wilke_large' in source:
#            print(id(obj), 'id')
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
    source = re.sub(r'''.*# ?(numba|NUMBA) ?: *(DELETE|delete|comment|COMMENT).*''', '', source)
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
bad_names = set(('__file__', '__name__', '__package__', '__cached__', 'solve'))

from fluids.numerics import SamePointError, UnconvergedError, NotBoundedError
def create_numerics(replaced, vec=False):
    cache_unsuported = set(['brenth', 'newton_system', 'quad', 'quad_adaptive', 'fixed_quad_Gauss_Kronrod', 'py_lambertw', 'secant', 'lambertw', 'ridder', 'bisect'])
#    cache_unsuported = set([])
#    if vec:
#        conv_fun = numba.vectorize
#    else:
    # Not part of the public API - do not need to worry about the stricter
    # numba.vectorize interface!
    conv_fun = numba.njit

    NUMERICS_SUBMOD_COPY = importlib.util.find_spec('fluids.numerics')
    NUMERICS_SUBMOD = importlib.util.module_from_spec(NUMERICS_SUBMOD_COPY)
    NUMERICS_SUBMOD.IS_NUMBA = True
    NUMERICS_SUBMOD.FORCE_PYPY = True
    NUMERICS_SUBMOD.numba = numba
    NUMERICS_SUBMOD.jitclass = jitclass
    NUMERICS_SUBMOD.njit = numba.njit
    NUMERICS_SUBMOD.jit = numba.jit
    NUMERICS_SUBMOD.array_if_needed = np.array
    NUMERICS_SUBMOD.sum = np.sum

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

    bad_names = set(['tck_interp2d_linear', 'implementation_optimize_tck', 'py_solve'])
    bad_names.update(to_set_num)

    solvers = ['secant', 'brenth', 'newton', 'halley', 'ridder', 'newton_system', 'solve_2_direct', 'solve_3_direct', 'solve_4_direct', 'basic_damping', 'bisect'] #
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
        source = re.sub(list_mult_expr, numpy_not_list_expr, source)

#        if any(i in s for i in ('bisect', 'solve_2_direct', 'basic_damping')):
#            print(source)
        numba_exec_cacheable(source, NUMERICS_SUBMOD.__dict__, NUMERICS_SUBMOD.__dict__)


#    numerics_forceobj = set(solvers) # Force the sovlers to compile in object mode
#    numerics_forceobj = []
    for name in names:
        if name not in bad_names:
            obj = getattr(NUMERICS_SUBMOD, name)
            if isinstance(obj, types.FunctionType):
                do_cache = caching and name not in cache_unsuported
#                forceobj = name in numerics_forceobj
#                forceobj = False
                # cache=not forceobj
                # cache=name not in skip_cache
                obj = conv_fun(cache=do_cache, **extra_args_std)(obj)
                NUMERICS_SUBMOD.__dict__[name] = obj
                replaced[name] = obj
#                globals()[name] = objs

    for name in to_set_num:
        NUMERICS_SUBMOD.__dict__[name] = globals()[name]

    replaced['bisplev'] = replaced['py_bisplev'] = NUMERICS_SUBMOD.__dict__['bisplev'] = bisplev
#    replaced['lambertw'] = NUMERICS_SUBMOD.__dict__['lambertw'] = NUMERICS_SUBMOD.__dict__['py_lambertw']
    for s in ('ellipe', 'gammaincc', 'gamma', 'i1', 'i0', 'k1', 'k0', 'iv', 'hyp2f1', 'erf', 'ellipkinc', 'ellipeinc'):
        replaced[s] = NUMERICS_SUBMOD.__dict__[s]

    NUMERICS_SUBMOD.normalize = normalize
    replaced['normalize'] = normalize
    return replaced, NUMERICS_SUBMOD

replaced = {'sum': np.sum, 'combinations': combinations, 'np': np}
replaced, NUMERICS_SUBMOD = create_numerics(replaced, vec=False)
numerics_dict = replaced
numerics = NUMERICS_SUBMOD
#old_numerics = sys.modules['fluids.numerics']
#sys.modules['fluids.numerics'] = numerics
normal = normal_fluids


def transform_module(normal, __funcs, replaced, vec=False, blacklist=frozenset([]),
                     cache_blacklist=set([])):
    new_mods = []
    if vec:
        conv_fun = numba.vectorize
        extra_args = extra_args_vec
    else:
        conv_fun = numba.njit
        extra_args = extra_args_std
    # Run module-by-module. Expensive, as we need to create module copies
    try:
        all_submodules = normal.all_submodules()
    except:
        all_submodules = normal.submodules
    numtypes = {float, int, complex}
    settypes = {set, frozenset}
    for mod in all_submodules:
        #print(all_submodules, mod)
        SUBMOD_COPY = importlib.util.find_spec(mod.__name__)
        SUBMOD = importlib.util.module_from_spec(SUBMOD_COPY)
        SUBMOD.IS_NUMBA = True
        SUBMOD.numba = numba
        SUBMOD.jitclass = jitclass
        SUBMOD.njit = numba.njit
        SUBMOD.jit = numba.jit
        SUBMOD.prange = numba.prange

        if vec:
            SUBMOD.IS_NUMBA_VEC = True
        SUBMOD_COPY.loader.exec_module(SUBMOD)
        SUBMOD.np = np
        SUBMOD.sum = np.sum

        SUBMOD.__dict__.update(replaced)
        new_mods.append(SUBMOD)
        mod_split_names = mod.__name__.split('.')
        __funcs[mod_split_names[-1]] = SUBMOD # fluids.numba.optional.spa
        __funcs['.'.join(mod_split_names[:-1])] = SUBMOD # set fluids.optional.spa fluids.numba.spa
        __funcs['.'.join(mod_split_names[-2:])] = SUBMOD # set 'optional.spa' in the dict too

        try:
            names = set(SUBMOD.__all__)
        except:
            names = set()
        for mod_obj_name in dir(SUBMOD):
            obj = getattr(SUBMOD, mod_obj_name)
            if (isinstance(obj, types.FunctionType)
                and mod_obj_name != '__getattr__'
                and not mod_obj_name.startswith('_load')
                and obj.__module__ == SUBMOD.__name__):
                names.add(mod_obj_name)

        # try:
        #     names += SUBMOD.__numba_additional_funcs__
        # except:
        #     pass

        numba_funcs = []
        funcs = []
        for name in names:
            obj = getattr(SUBMOD, name)
            if isinstance(obj, types.FunctionType):
                if name not in total_skip and name not in blacklist:
                    SUBMOD.__dict__[name] = obj = conv_fun(cache=(caching and name not in cache_blacklist), **extra_args)(obj)
                    numba_funcs.append(obj)
                else:
                    funcs.append(obj)
            __funcs[name] = obj

        module_constants_changed_type = {}
        for arr_name in SUBMOD.__dict__:
            if arr_name in no_conv_data_names: continue
            obj = getattr(SUBMOD, arr_name)
            obj_type = type(obj)
            if obj_type is list and obj:
                # Assume all elements have the same general type
                r = obj[0]
                r_type = type(r)
                if r_type in numtypes:
                    arr = np.array(obj)
                    if arr.dtype.char != 'O': module_constants_changed_type[arr_name] = arr
                elif r_type is list and r and type(r[0]) in numtypes:
                    if len(set([len(r) for r in obj])) == 1:
                        # All same size - nice numpy array
                        arr = np.array(obj)
                        if arr.dtype.char != 'O': module_constants_changed_type[arr_name] = arr
                    else:
                        # Tuple of different size numpy arrays
                        module_constants_changed_type[arr_name] = tuple([np.array(v) for v in obj])
            elif obj_type in settypes:
                module_constants_changed_type[arr_name] = tuple(obj)
            # elif obj_type is dict:
            #     try:
            #         print('starting', arr_name)
            #         infer_dictionary_types(obj)
            #         module_constants_changed_type[arr_name] = numba_dict(obj)
            #     except:
            #         print(arr_name, 'failed')
            #         pass

        SUBMOD.__dict__.update(module_constants_changed_type)
        __funcs.update(module_constants_changed_type)

        # if not vec:
            # for t in numba_funcs:
            #     #if normal.__name__ == 'chemicals':
            #     #    if 'iapws' not in all_submodules[-1].__name__:
            #     #        print(new_objs, t)
            #     #        1/0
            #     t.py_func.__globals__.update(SUBMOD.__dict__)
            # for t in funcs:
            #     t.__globals__.update(SUBMOD.__dict__)

    # Do our best to allow functions to be found
    if '__file__' in __funcs:
        del __funcs['__file__']
    if '__file__' in replaced:
        del replaced['__file__']
    for mod in new_mods:
        mod.__dict__.update(__funcs)
    return new_mods


def transform_complete(replaced, __funcs, __all__, normal, vec=False):
    cache_blacklist = set(['Stichlmair_flood', 'airmass',
   'Spitzglass_high', '_to_solve_Spitzglass_high',
   '_to_solve_Spitzglass_low', 'Spitzglass_low',
   'Oliphant', '_to_solve_Oliphant',
   'P_isothermal_critical_flow', 'P_upstream_isothermal_critical_flow',
   'isothermal_gas_err_P1', 'isothermal_gas_err_P2', 'isothermal_gas_err_P2_basis', 'isothermal_gas_err_D', 'isothermal_gas',
   'v_terminal', 'differential_pressure_meter_solver', 'err_dp_meter_solver_P1', 'err_dp_meter_solver_D2',
   'err_dp_meter_solver_P2', 'err_dp_meter_solver_m', 'V_horiz_spherical', 'V_horiz_torispherical',
   'Prandtl_von_Karman_Nikuradse', 'plate_enlargement_factor', 'Stichlmair_wet', 'V_from_h',
   'SA_partial_horiz_spherical_head', '_SA_partial_horiz_spherical_head_to_int',
   '_SA_partial_horiz_ellipsoidal_head_to_int', '_SA_partial_horiz_ellipsoidal_head_limits', 'SA_partial_horiz_ellipsoidal_head',
   '_SA_partial_horiz_guppy_head_to_int', 'SA_partial_horiz_guppy_head', 'SA_partial_horiz_torispherical_head',
   'SA_from_h', 'V_tank'])
#    cache_blacklist = set([])
    if vec:
        conv_fun = numba.vectorize
        extra_args = extra_args_vec
    else:
        conv_fun = numba.njit
        extra_args = extra_args_std
    new_mods = transform_module(normal, __funcs, replaced, vec=vec, cache_blacklist=cache_blacklist)


    to_change = ['packed_tower._Stichlmair_flood_f_and_jac',
                 'packed_tower.Stichlmair_flood', 'compressible.isothermal_gas',
                 'fittings.Darby3K', 'fittings.Hooper2K', 'geometry.SA_partial_horiz_torispherical_head',
                 'optional.spa.solar_position', 'optional.spa.longitude_obliquity_nutation',
                 'optional.spa.transit_sunrise_sunset',
                 'fittings.bend_rounded_Crane', 'geometry.tank_from_two_specs_err',
                 'friction.roughness_Farshad',
                 ]
    transform_lists_to_arrays(normal_fluids, to_change, __funcs, vec=vec, cache_blacklist=cache_blacklist)


    # AvailableMethods  will be removed in the future in favor of non-numba only
    # calls to method functions

    to_change = {}
#    to_change['friction.roughness_Farshad'] = 'ID in _Farshad_roughness'

    for s, bad_branch in to_change.items():
        mod, func = s.split('.')
        source = inspect.getsource(getattr(getattr(normal_fluids, mod), func))
        fake_mod = __funcs[mod]
        source = remove_branch(source, bad_branch)
        source = remove_for_numba(source)
        numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching, **extra_args)(new_func)
        __funcs[func] = obj
        obj.__doc__ = ''


    # Do some classes by hand
    PlateExchanger_spec = [(k, float64) for k in ('pitch', 'beta', 'gamma', 'a', 'amplitude', 'wavelength',
                           'b', 'chevron_angle', 'inclination_angle', 'plate_corrugation_aspect_ratio',
                           'plate_enlargement_factor', 'D_eq', 'D_hydraulic', 'width', 'length', 'thickness',
                           'd_port', 'plates', 'length_port', 'A_plate_surface', 'A_heat_transfer',
                           'A_channel_flow', 'channels', 'channels_per_fluid')]
    PlateExchanger_spec.append(('chevron_angles', numba.types.UniTuple(float64, 2)))

    HelicalCoil_spec = [(k, float64) for k in
                        ('Do', 'Dt', 'Di', 'Do_total', 'N', 'pitch', 'H', 'H_total',
                         'tube_circumference', 'tube_length', 'surface_area', 'helix_angle',
                         'curvature', 'total_inlet_area', 'total_volume', 'inner_surface_area',
                         'inlet_area', 'inner_volume', 'annulus_area', 'annulus_volume')]

    ATMOSPHERE_1976_spec = [(k, float64) for k in
                        ('Z', 'dT', 'H', 'T_layer', 'T_increase', 'P_layer', 'H_layer', 'H_above_layer',
                         'T', 'P', 'rho', 'v_sonic',
                         'mu', 'k', 'g', 'R')]

#    # No string support
#    PlateExchanger = jitclass(PlateExchanger_spec)(getattr(__funcs['geometry'], 'PlateExchanger'))
#    __funcs['PlateExchanger'] = __funcs['geometry'].PlateExchanger = PlateExchanger

    HelicalCoil = jitclass(HelicalCoil_spec)(getattr(__funcs['geometry'], 'HelicalCoil'))
    __funcs['HelicalCoil'] = __funcs['geometry'].HelicalCoil = HelicalCoil

    ATMOSPHERE_1976 = jitclass(ATMOSPHERE_1976_spec)(getattr(__funcs['atmosphere'], 'ATMOSPHERE_1976'))
    __funcs['ATMOSPHERE_1976'] = __funcs['atmosphere'].ATMOSPHERE_1976 = ATMOSPHERE_1976


    # Not needed
    __funcs['friction'].Colebrook = __funcs['Colebrook'] = __funcs['Clamond']

    # Works but 50% slower
    #__funcs['geometry']._V_horiz_spherical_toint = __funcs['_V_horiz_spherical_toint'] = cfunc("float64(float64, float64, float64, float64)")(normal_fluids.geometry._V_horiz_spherical_toint)

    for mod in new_mods:
        mod.__dict__.update(__funcs)
        try:
            __all__.extend(mod.__all__)
        except AttributeError:
            pass

transform_complete(replaced, __funcs, __all__, normal, vec=False)

numbafied_fluids_functions = __funcs
globals().update(__funcs)
globals().update(replaced)

#sys.modules['fluids.numerics'] = old_numerics







