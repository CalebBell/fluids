# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020, 2021, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['wraps_numpydoc', 'u']

import types
import re
import inspect
import sys
from inspect import getsource, cleandoc
import functools
try:
    from collections.abc import Iterable
except:
    from collections import Iterable
from copy import copy
import fluids
import fluids.vectorized
import numpy as np
try:
    import pint
    from pint import _DEFAULT_REGISTRY as u
    from pint import DimensionalityError

except ImportError: # pragma: no cover
    raise ImportError('The unit handling in fluids requires the installation '
                      'of the package pint, available on pypi or from '
                      'https://github.com/hgrecco/pint')


'''See fluids.units.rst for documentation for this module.
'''

try:
    doc_stripped = sys.flags.optimize == 2
except:
    doc_stripped = False
# is_critical_flow is broken

def get_docstring(f):
    '''Returns the docstring of a function, working in -OO mode also.
    '''
    try:
        if f.__doc__ is not None:
            return f.__doc__
    except:
        # micropython
        return None
    if not doc_stripped:
        return None
    src = cleandoc(inspect.getsource(f))
    single_pos = src.find("'''")
    double_pos = src.find('"""')
    if single_pos == -1:
        if double_pos == -1:
            # Neither
            return None
        # double, not single
        return cleandoc(src.split('"""')[1])
    elif double_pos == -1 and single_pos != -1:
        # single, not double
        return cleandoc(src.split("'''")[1])
    else:
        # single and double
        if single_pos < double_pos:
            return cleandoc(src.split("'''")[1])
        return cleandoc(src.split('"""')[1])


def func_args(func):
    """Basic function which returns a tuple of arguments of a function or
    method."""
    try:
        return tuple(inspect.getfullargspec(func).args)
    except:
        return tuple(inspect.getargspec(func).args)

u.autoconvert_offset_to_baseunit = True


expr = re.compile('Parameters *\n *-+\n +')
expr2 = re.compile('Returns *\n *-+\n +')
match_sections = re.compile('\n *[A-Za-z ]+ *\n *-+')
match_section_names = re.compile('\n *[A-Za-z]+ *\n *-+')
variable = re.compile('[a-zA-Z_0-9]* : ')
match_units = re.compile(r'\[[a-zA-Z0-9().\/*^\- ]*\]')


parse_numpydoc_variables_units_cache = {}
def parse_numpydoc_variables_units(func):
    text = get_docstring(func)
    if text is None:
        text = ''
    h = hash(text)
    if h in parse_numpydoc_variables_units_cache:
        return parse_numpydoc_variables_units_cache[h]
    res = parse_numpydoc_variables_units_docstring(text)
    parse_numpydoc_variables_units_cache[h] = res
    return res

def parse_numpydoc_variables_units_docstring(text):
    section_names = [i.replace('-', '').strip() for i in match_sections.findall(text)]
    section_text = match_sections.split(text)

    sections = {}
    for i, j in zip(section_names, section_text[1:]):
        sections[i] = j


    parsed = {}
    for section in ['Parameters', 'Returns', 'Attributes', 'Other Parameters']:
        if section not in sections:
            # Handle the case where the function has nothing in a section
            parsed[section] = {'units': [], 'vars': []}
            continue

        p = sections[section]
        parameter_vars = [i[:-2].strip() for i in variable.findall(p)]
        unit_strings = [i.strip() for i in variable.split(p)[1:]]
        units = []
        for i in unit_strings:
            matches = match_units.findall(i)
            if len(matches) == 0:
                # If there is no unit listed, assume it's dimensionless (probably a string)
                matches = ['[]']
            match = matches[-1] # Assume the last bracketed group listed is the unit group
            match = match.replace('[', '').replace(']', '')
            if len(match) == 1:
                match = match.replace('-', 'dimensionless')
            if match == '':
                match = 'dimensionless'
            if match == 'base SI':
                match = 'dimensionless' # TODO - write special wrappers for these cases
            units.append(match)

        parsed[section] = {'units': units, 'vars': parameter_vars}
    return parsed


def check_args_order(func):
    """Reads a numpydoc function and compares the Parameters and Other
    Parameters with the input arguments of the actual function signature. Raises
    an exception if not correctly defined.

    getargspec is used for Python 2.7 compatibility and is deprecated in Python
    3.

    >>> check_args_order(fluids.core.Reynolds)
    """
    try:
        argspec = inspect.getfullargspec(func)
    except:
        argspec = inspect.getargspec(func)
    parsed_data = parse_numpydoc_variables_units(func)
    # compare the parsed arguments with those actually defined
    parsed_units = copy(parsed_data['Parameters']['units'])
    parsed_parameters = copy(parsed_data['Parameters']['vars'])
    if 'Other Parameters' in parsed_data:
        parsed_parameters += parsed_data['Other Parameters']['vars']
        parsed_units += parsed_data['Other Parameters']['units']

    if argspec.args != parsed_parameters: # pragma: no cover
        raise ValueError('Function %s signature is not the same as the documentation'
                        ' signature = %s; documentation = %s' %(func.__name__, argspec.args, parsed_parameters))


def match_parse_units(doc, i=-1):
    if doc is None:
        matches = ['[]']
    else:
        matches = match_units.findall(doc)
    if len(matches) == 0:
        # If there is no unit listed, assume it's dimensionless (probably a string)
        matches = ['[]']
    match = matches[i] # Assume the last bracketed group listed is the unit group
    match = match.replace('[', '').replace(']', '')
    if len(match) == 1:
        match = match.replace('-', 'dimensionless')
    if match == '':
        match = 'dimensionless'
    if match == 'base SI':
        match = 'dimensionless' # TODO - write special wrappers for these cases
    return match


def convert_input(val, unit, ureg, strict=True):
    if val is None:
        return val # Handle optional units which are given
    if unit != 'dimensionless':
        try:
            return val.to(unit).magnitude
        except AttributeError:
            if strict:
                raise TypeError('%s has no quantity' %(val))
            else:
                return val
        except DimensionalityError as e:
            raise ValueError('Converting %s to units of %s raised DimensionalityError: %s'%(val, unit, str(e)))
    else:
        if type(val) == ureg.Quantity:
            return val.to_base_units().magnitude
        else:
            return val

pint_expression_cache = {}

def parse_expression_cached(unit, ureg):
    if unit in pint_expression_cache:
        return pint_expression_cache[unit]
    ans = ureg.parse_expression(unit)
    pint_expression_cache[unit] = ans
    return ans


def convert_output(result, out_units, out_vars, ureg):
    # Attempt to handle multiple return values
    # Must be able to convert all values to a pint expression
    t = type(result)
    if t == str or t == bool:
        return result
    elif t == dict:
        for key, ans in result.items():
            unit = out_units[out_vars.index(key)]
            result[key] = ans*parse_expression_cached(unit, ureg)
        return result
    elif isinstance(result, Iterable):
        conveted_result = []
        for ans, unit in zip(result, out_units):
            conveted_result.append(ans*parse_expression_cached(unit, ureg))
        return conveted_result
    else:
        return result*parse_expression_cached(out_units[0], ureg)


in_vars_cache = {}
in_units_cache = {}
out_vars_cache = {}
out_units_cache = {}


def wraps_numpydoc(ureg, strict=True):
    def decorator(func):
        assigned = (attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
        updated = (attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))
        parsed_info = parse_numpydoc_variables_units(func)

        in_vars = copy(parsed_info['Parameters']['vars'])
        in_units = copy(parsed_info['Parameters']['units'])
        if 'Other Parameters' in parsed_info:
            in_vars += parsed_info['Other Parameters']['vars']
            in_units += parsed_info['Other Parameters']['units']
        in_vars_to_dict = {}
        for i, j in zip(in_vars, in_units):
            in_vars_to_dict[i] = j

        out_units = parsed_info['Returns']['units']
        out_vars = parsed_info['Returns']['vars']
        # Handle the case of dict answers - require the first line's args to be
        # parsed as 'results'
        if out_vars and 'results' == out_vars[0]:
            out_units.pop(0)
            out_vars.pop(0)

        in_vars_cache[func] = in_vars
        in_units_cache[func] = in_units
        out_vars_cache[func] = out_vars
        out_units_cache[func] = out_units

        @functools.wraps(func, assigned=assigned, updated=updated)
        def wrapper(*values, **kw):
            # Convert input ordered variables to dimensionless form, after converting
            # them to the the units specified by their documentation
            conv_values = []
            for val, unit in zip(values, in_units):
                conv_values.append(convert_input(val, unit, ureg, strict))

            # For keyword arguments, lookup their unit; convert to that;
            # handle dimensionless arguments the same way
            kwargs = {}
            for name, val in kw.items():
                unit = in_vars_to_dict[name]
                kwargs[name] = convert_input(val, unit, ureg, strict)
            if any([type(i.m) == np.ndarray for i in list(kw.values()) + list(values) if type(i) == u.Quantity]):
                result = getattr(fluids.vectorized, func.__name__)(*conv_values, **kwargs)
            else:
                result = func(*conv_values, **kwargs)
            if type(result) == np.ndarray:
                units = convert_output(result, out_units, out_vars, ureg)[0].units
                return result*units
            else:
                return convert_output(result, out_units, out_vars, ureg)

        return wrapper
    return decorator


class UnitAwareClass(object):
    wrapped = None
    ureg = u
    strict = True
    property_units = {} # for properties and attributes only
    method_units = {}

    def __repr__(self):
        '''Called only on the class instance, not any instance - ever.
        https://stackoverflow.com/questions/10376604/overriding-special-methods-on-an-instance
        '''
        return self.wrapped.__repr__()

    def __add__(self, other):
        new_obj = self.wrapped.__add__(other.wrapped)
        new_instance = copy(self)
        new_instance.wrapped = new_obj
        return new_instance

    def __sub__(self, other):
        new_obj = self.wrapped.__sub__(other.wrapped)
        new_instance = copy(self)
        new_instance.wrapped = new_obj
        return new_instance

    def __init__(self, *args, **kwargs):
        args_base, kwargs_base =  self.input_units_to_dimensionless('__init__', *args, **kwargs)
        self.wrapped = self.wrapped(*args_base, **kwargs_base)

    @classmethod
    def wrap(self, wrapped):
        new = super(self, self).__new__(self)
        new.wrapped = wrapped
        return new



    def __getattr__(self, name):
        instance = True
        if name in self.class_methods or name in self.static_methods:
            instance = False
        try:
            value = getattr(self.wrapped, name)
        except Exception as e:
            raise AttributeError('Failed to get property %s with error %s' %(str(name), str(e)))
        if value is not None:
            if name in self.property_units:
                if type(value) == dict:
                    d = {}
                    unit = self.property_units[name]
                    for key, val in value.items():
                        d[key] = val*unit
                    return d
                try:
                    return value*self.property_units[name]
                except:
                    # Not everything is going to work. The most common case here
                    # is returning a list, some of the values being None and so
                    # it cannot be wrapped.
                    return value
            else:
                if hasattr(value, '__call__'):

#                    if not instance:
#                        @functools.wraps(value)
#                        # Special case where self needs to be passed in specifically
#                        def call_func_with_inputs_to_SI(*args, **kwargs):
#                            args_base, kwargs_base = self.input_units_to_dimensionless(self, name, *args, **kwargs)
#                            result = value(*args_base, **kwargs_base)
#                            if name == '__init__':
#                                return result
#                            _, _, _, out_vars, out_units = self.method_units[name]
#                            if not out_units:
#                                return
#                            return convert_output(result, out_units, out_vars, self.ureg)
#
#                    else:
                    @functools.wraps(value)
                    def call_func_with_inputs_to_SI(*args, **kwargs):
                        args_base, kwargs_base = self.input_units_to_dimensionless(name, *args, **kwargs)
                        result = value(*args_base, **kwargs_base)
                        if name == '__init__':
                            return result
                        elif type(result) is self.wrapped:
                            # Creating a new class, wrap it
                            return self.wrap(result)
                        _, _, _, out_vars, out_units = self.method_units[name]
                        if not out_units:
                            return
                        return convert_output(result, out_units, out_vars, self.ureg)

                    return call_func_with_inputs_to_SI
                raise AttributeError('Error: Property does not yet have units attached')
        else:
            return value

    _another_getattr = classmethod(__getattr__)

    @classmethod
    def input_units_to_dimensionless(self, name, *values, **kw):
        in_vars, in_units, in_vars_to_dict, out_vars, out_units = self.method_units[name]
        conv_values = []
        for val, unit in zip(values, in_units):
            conv_values.append(convert_input(val, unit, self.ureg, self.strict))

        # For keyword arguments, lookup their unit; convert to that;
        # handle dimensionless arguments the same way
        kwargs = {}
        for name, val in kw.items():
            unit = in_vars_to_dict[name]
            kwargs[name] = convert_input(val, unit, self.ureg, self.strict)
        return conv_values, kwargs


def clean_parsed_info(parsed_info):
    in_vars = parsed_info['Parameters']['vars']
    in_units = parsed_info['Parameters']['units']
    if 'Other Parameters' in parsed_info:
        in_vars += parsed_info['Other Parameters']['vars']
        in_units += parsed_info['Other Parameters']['units']
    in_vars_to_dict = {}
    for i, j in zip(in_vars, in_units):
        in_vars_to_dict[i] = j

    out_units = parsed_info['Returns']['units']
    out_vars = parsed_info['Returns']['vars']
    # Handle the case of dict answers - require the first line's args to be
    # parsed as 'results'
    if out_vars and 'results' == out_vars[0]:
        out_units.pop(0)
        out_vars.pop(0)

    return in_vars, in_units, in_vars_to_dict, out_vars, out_units


def wrap_numpydoc_obj(obj_to_wrap):
    callable_methods = {}
    property_unit_map = {}
    static_methods = set([])
    class_methods = set([])
    for prop in dir(obj_to_wrap):
        attr = getattr(obj_to_wrap, prop)
        if isinstance(attr, types.FunctionType) or isinstance(attr, types.MethodType) or type(attr) == property:
            try:
                if isinstance(obj_to_wrap.__dict__[prop], staticmethod):
                    static_methods.add(prop)
                if isinstance(obj_to_wrap.__dict__[prop], classmethod):
                    class_methods.add(prop)
            except:
                pass
            if type(attr) is property:
                name = prop
                #name = attr.fget.__name__
            else:
                name = attr.__name__
            if hasattr(attr, '__doc__') and attr.__doc__ is not None:
                if type(attr) is property:
                    try:
                        docstring = attr.__doc__
                        if docstring is None:
                            docstring = attr.fget.__doc__
                        # Is it a full style string?
                        if 'Returns' in docstring and '-------' in docstring:
                                found_unit = parse_expression_cached(parse_numpydoc_variables_units_docstring(docstring)['Returns']['units'][0], u)
                        else:
                            found_unit = parse_expression_cached(match_parse_units(docstring, i=0), u)
                    except Exception as e:
                        if name[0] == '_':
                            found_unit = u.dimensionless
                        else:
                            print('Failed on attribute %s' %name)
                            raise e
                    property_unit_map[name] = found_unit
                else:
                    parsed = parse_numpydoc_variables_units(attr)
                    callable_methods[name] = clean_parsed_info(parsed)
                    if 'Attributes' in parsed:
                        property_unit_map.update(parsed['Attributes'])

    # We need to parse the __doc__ for the main docstring of each of the inherited
    # objects, but in reverse order so older properties get overwritten by newer
    # properties. Ignore the object type as well.
    for inherited in reversed(list(obj_to_wrap.__mro__[0:-1])):
        parsed = parse_numpydoc_variables_units(inherited)
        callable_methods['__init__'] = clean_parsed_info(parsed)

        if 'Attributes' in parsed:
            property_unit_map.update({var:parse_expression_cached(unit, u) for var, unit in zip(parsed['Attributes']['vars'], parsed['Attributes']['units'])} )
        if 'Parameters' in parsed:
            property_unit_map.update({var:parse_expression_cached(unit, u) for var, unit in zip(parsed['Parameters']['vars'], parsed['Parameters']['units'])} )

    name = obj_to_wrap.__name__
    classkwargs = {'wrapped': obj_to_wrap,
            'property_units': property_unit_map, 'method_units': callable_methods,
                   'static_methods': static_methods, 'class_methods': class_methods}

    fun = type(name, (UnitAwareClass,), classkwargs
           )
    for m in static_methods:
        #def a_static_method(*args, the_method=m, **kwargs):
            #return fun._another_getattr(the_method)(*args, **kwargs)
        setattr(fun, m, staticmethod(fun._another_getattr(m)))
    for m in class_methods:
        setattr(fun, m, classmethod(fun._another_getattr(m)))
    return fun

def kwargs_to_args(args, kwargs, signature):
    '''Accepts an *args and **kwargs and a signature
    like ['rho', 'mu', 'nu'] which is an ordered list of
    all accepted arguments.

    Returns a list containing all the arguments, sorted, and
    left as None if not specified
    '''
    argument_number = len(signature)
    arg_number = len(args)
    output = list(args)
    # Extend the list and initialize as None by default
    output.extend([None]*(argument_number - arg_number))
    for i in range(arg_number, argument_number):
        if signature[i] in kwargs:
            output[i] = kwargs[signature[i]]
    return output


__pint_wrapped_functions = {}

for name in dir(fluids):
    if 'RectangularOffsetStripFinExchanger' in name:
        continue
    if 'ParticleSizeDistribution' in name:
        continue
    if name == '__getattr__' or name == '__test__':
        continue
    obj = getattr(fluids, name)
    if isinstance(obj, types.FunctionType):
        obj = wraps_numpydoc(u)(obj)
    elif type(obj) == type:
        obj = wrap_numpydoc_obj(obj)
    elif type(obj) is types.ModuleType:
        # Functions accessed with the namespace like friction.friction_factor
        # would call the original function - leads to user confusion if they are exposed
        continue
    elif isinstance(obj, str):
        continue
    if name == '__all__':
        continue
    __all__.append(name)
    __pint_wrapped_functions.update({name: obj})

globals().update(__pint_wrapped_functions)
__all__.extend(['wraps_numpydoc', 'convert_output', 'convert_input',
                'check_args_order', 'match_parse_units', 'parse_numpydoc_variables_units',
                'wrap_numpydoc_obj', 'UnitAwareClass'])


def A_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.A_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**2

def V_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.V_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**3

variable_output_unit_funcs = {
    # True: arg should be present; False: arg should be None
    'nu_mu_converter': ({(True, False, True): [u.Pa*u.s],
                        (True, True, False): [u.m**2/u.s],
                        }, 3),
    'differential_pressure_meter_solver': ({(True, True, True, True, False, True, True, True): [u.m],
                                            (True, True, True, True, True, False, True, True): [u.Pa],
                                            (True, True, True, True, True, True, False, True): [u.Pa],
                                            (True, True, True, True, True, True, True, False): [u.kg/u.s],
                                            }, 8),
    'isothermal_gas': ({(True, True, False, True, True, True, True): [u.Pa],
                        (True, True, True, False, True, True, True): [u.Pa],
                        (True, True, True, True, False, True, True): [u.m],
                        (True, True, True, True, True, False, True): [u.m],
                        (True, True, True, True, True, True, False): [u.kg/u.s],
                        }, 7)
}

simple_compressible_variable_output = ({(True, True, False, True, True, True, True): [u.m],
                                        (True, True, True, False, True, True, True): [u.m],
                                        (True, True, True, True, False, True, True): [u.Pa],
                                        (True, True, True, True, True, False, True): [u.Pa],
                                        (True, True, True, True, True, True, False): [u.m**3/u.s],
                                        }, 7)
for f in ['Panhandle_A', 'Panhandle_B', 'Weymouth', 'Spitzglass_high', 'Spitzglass_low', 'Oliphant', 'Fritzsche']:
    variable_output_unit_funcs[f] = simple_compressible_variable_output

IGT_Muller_variable_output = ({(True, True, True, False, True, True, True, True): [u.m],
                               (True, True, True, True, False, True, True, True): [u.m],
                               (True, True, True, True, True, False, True, True): [u.Pa],
                               (True, True, True, True, True, True, False, True): [u.Pa],
                               (True, True, True, True, True, True, True, False): [u.m**3/u.s],
                               }, 8)

for f in ['Muller', 'IGT']:
    variable_output_unit_funcs[f] = IGT_Muller_variable_output

def variable_output_wrapper(func, wrapped_basic_func, output_signatures, input_length):
    name = func.__name__
    intput_signature = in_vars_cache[func]

    def thing(*args, **kwargs):
        ans = wrapped_basic_func(*args, **kwargs)
        args_for_sig = kwargs_to_args(args, kwargs, intput_signature)
        args_for_sig = [i is not None for i in args_for_sig]
        if len(args_for_sig) > input_length:
            # Allow other arguments later to not matter
            args_for_sig = args_for_sig[:input_length]

        output_units = output_signatures[tuple(args_for_sig)]
        if type(ans) in (list, tuple):
            return [output_units[i]*ans[i] for i in range(len(ans))]
        return output_units[0]*ans
    return thing

for name, val in variable_output_unit_funcs.items():
    globals()[name] = variable_output_wrapper(getattr(fluids, name),
            __pint_wrapped_functions[name], val[0], val[1])