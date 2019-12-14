# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['wraps_numpydoc', 'u']

import types
import re
import inspect
import functools
import collections
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
# is_critical_flow is broken

def func_args(func):
    '''Basic function which returns a tuple of arguments of a function or
    method.
    '''
    try:
        return tuple(inspect.getargspec(func).args)
    except:
        return tuple(inspect.getfullargspec(func).args)

u.autoconvert_offset_to_baseunit = True


expr = re.compile('Parameters *\n *-+\n +')
expr2 = re.compile('Returns *\n *-+\n +')
match_sections = re.compile('\n *[A-Za-z ]+ *\n +-+')
match_section_names = re.compile('\n *[A-Za-z]+ *\n +-+')
variable = re.compile('[a-zA-Z_0-9]* : ')
match_units = re.compile(r'\[[a-zA-Z0-9().\/*^\- ]*\]')


def parse_numpydoc_variables_units(func):
    try:
        text = func.__doc__
    except:
        text = ''
    if text is None:
        text = ''
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
    '''Reads a numpydoc function and compares the Parameters and
    Other Parameters with the input arguments of the actual function signature.
    Raises an exception if not correctly defined.
    
    getargspec is used for Python 2.7 compatibility and is deprecated in Python
    3.
    
    >>> check_args_order(fluids.core.Reynolds)
    '''
    try:
        argspec = inspect.getfullargspec(func)
    except:
        argspec = inspect.getargspec(func)
    parsed_data = parse_numpydoc_variables_units(func)
    # compare the parsed arguments with those actually defined
    parsed_units = parsed_data['Parameters']['units']
    parsed_parameters = parsed_data['Parameters']['vars']
    if 'Other Parameters' in parsed_data:
        parsed_parameters += parsed_data['Other Parameters']['vars']
        parsed_units += parsed_data['Other Parameters']['units']
    
    if argspec.args != parsed_parameters: # pragma: no cover
        raise Exception('Function %s signature is not the same as the documentation'
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
            raise Exception('Converting %s to units of %s raised DimensionalityError: %s'%(val, unit, str(e)))
    else:
        if type(val) == ureg.Quantity:
            return val.magnitude
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
    elif isinstance(result, collections.Iterable):
        conveted_result = []
        for ans, unit in zip(result, out_units):
            conveted_result.append(ans*parse_expression_cached(unit, ureg))
        return conveted_result
    else:
        return result*parse_expression_cached(out_units[0], ureg)


def wraps_numpydoc(ureg, strict=True):    
    def decorator(func):
        assigned = (attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
        updated = (attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))
        parsed_info = parse_numpydoc_variables_units(func)

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

                
    def __getattr__(self, name):
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
                    @functools.wraps(value)
                    def call_func_with_inputs_to_SI(*args, **kwargs):
                        args_base, kwargs_base = self.input_units_to_dimensionless(name, *args, **kwargs)
                        result = value(*args_base, **kwargs_base)
                        if name == '__init__':
                            return result
                        _, _, _, out_vars, out_units = self.method_units[name]
                        if not out_units:
                            return
                        return convert_output(result, out_units, out_vars, self.ureg)
                        
                    return call_func_with_inputs_to_SI
                raise AttributeError('Error: Property does not yet have units attached')
        else:
            return value
        
        
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
    for i in dir(obj_to_wrap):
        attr = getattr(obj_to_wrap, i)
        if isinstance(attr, types.FunctionType) or isinstance(attr, types.MethodType) or type(attr) == property:
            if type(attr) is property:
                name = attr.fget.__name__
            else:
                name = attr.__name__
            if hasattr(attr, '__doc__'):
                if type(attr) is property:
                    property_unit_map[name] = u(match_parse_units(attr.fget.__doc__, i=0))
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
            property_unit_map.update({var:u(unit) for var, unit in zip(parsed['Attributes']['vars'], parsed['Attributes']['units'])} )
        if 'Parameters' in parsed:
            property_unit_map.update({var:u(unit) for var, unit in zip(parsed['Parameters']['vars'], parsed['Parameters']['units'])} )

    name = obj_to_wrap.__name__
    fun = type(name, (UnitAwareClass,), 
           {'wrapped': obj_to_wrap, #'__doc__': obj_to_wrap.__doc__,
            'property_units': property_unit_map, 'method_units': callable_methods})
    return fun


__funcs = {}


for name in dir(fluids):
    if 'RectangularOffsetStripFinExchanger' in name:
        continue
    if 'ParticleSizeDistribution' in name:
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
    __funcs.update({name: obj})
    
globals().update(__funcs)
__all__.extend(['wraps_numpydoc', 'convert_output', 'convert_input',
                'check_args_order', 'match_parse_units', 'parse_numpydoc_variables_units', 
                'wrap_numpydoc_obj', 'UnitAwareClass'])


def A_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.A_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**2

#A_multiple_hole_cylinder.__doc__ = fluids.geometry.A_multiple_hole_cylinder.__doc__

def V_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.V_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**3

#V_multiple_hole_cylinder.__doc__ = fluids.geometry.V_multiple_hole_cylinder.__doc__

wrapped_isothermal_gas = isothermal_gas
wrapped_Panhandle_A = Panhandle_A
wrapped_Muller = Muller
wrapped_IGT = IGT
wrapped_nu_mu_converter = nu_mu_converter
wrapped_SA_tank= SA_tank

wrapped_differential_pressure_meter_solver = differential_pressure_meter_solver


def nu_mu_converter(rho, mu=None, nu=None):
    ans = wrapped_nu_mu_converter(rho, mu, nu)
    if mu is None:
        return ans*u.Pa*u.s
    return ans*u.m**2/u.s


def SA_tank(D, L, sideA=None, sideB=None, sideA_a=0*u.m,
             sideB_a=0*u.m, sideA_f=None, sideA_k=None, sideB_f=None, sideB_k=None,
             full_output=False):
    ans = wrapped_SA_tank(D, L, sideA, sideB, sideA_a, sideB_a, sideA_f, 
                          sideA_k, sideB_f, sideB_k, full_output)
    if full_output:
        SA, (sideA_SA, sideB_SA, lateral_SA) = ans
    else:
        SA = ans
    if full_output:
        return SA, (sideA_SA*u.m**2, sideB_SA*u.m**2, lateral_SA*u.m**2)
    else:
        return SA


def isothermal_gas(rho, fd, P1=None, P2=None, L=None, D=None, m=None): # pragma: no cover
    '''
    >>> isothermal_gas(rho=11.3*u.kg/u.m**3, fd=0.00185*u.dimensionless, P1=1E6*u.Pa, P2=9E5*u.Pa, L=1000*u.m, D=0.5*u.m)
    <Quantity(145.484757264, 'kilogram / second')>
    '''
    ans = wrapped_isothermal_gas(rho, fd, P1, P2, L, D, m)    
    if m is None and (None not in [P1, P2, L, D]):
        return ans*u.kg/u.s
    elif L is None and (None not in [P1, P2, D, m]):
        return ans*u.m
    elif P1 is None and (None not in [L, P2, D, m]):
        return ans*u.Pa
    elif P2 is None and (None not in [L, P1, D, m]):
        return ans*u.Pa
    elif D is None and (None not in [P2, P1, L, m]):
        return ans*u.m


def Muller(SG, Tavg, mu, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7*u.K,
           Ps=101325.*u.Pa, Zavg=1, E=1): # pragma: no cover
    ans = wrapped_Muller(SG, Tavg, mu, L, D, P1, P2, Q, Ts, Ps, Zavg, E)    
    if Q is None and (None not in [L, D, P1, P2]):
        return ans*u.m**3/u.s
    elif D is None and (None not in [L, Q, P1, P2]):
        return ans*u.m
    elif P1 is None and (None not in [L, Q, D, P2]):
        return ans*u.Pa
    elif P2 is None and (None not in [L, Q, D, P1]):
        return ans*u.Pa
    elif L is None and (None not in [P2, Q, D, P1]):
        return ans*u.m


def IGT(SG, Tavg, mu, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7*u.K,
        Ps=101325.*u.Pa, Zavg=1, E=1): # pragma: no cover
    ans = wrapped_IGT(SG, Tavg, mu, L, D, P1, P2, Q, Ts, Ps, Zavg, E)    
    if Q is None and (None not in [L, D, P1, P2]):
        return ans*u.m**3/u.s
    elif D is None and (None not in [L, Q, P1, P2]):
        return ans*u.m
    elif P1 is None and (None not in [L, Q, D, P2]):
        return ans*u.Pa
    elif P2 is None and (None not in [L, Q, D, P1]):
        return ans*u.Pa
    elif L is None and (None not in [P2, Q, D, P1]):
        return ans*u.m


funcs = ['Panhandle_A', 'Panhandle_B', 'Weymouth', 'Spitzglass_high', 'Spitzglass_low', 'Oliphant', 'Fritzsche']
Es = [.92, .92, .92, 1, 1, .92, 1]

for wrapper, E in zip(funcs, Es):
    wrapper_name = wrapper + '_wrapper'
    globals()[wrapper_name] = globals()[wrapper]
    
    def compressible_flow_wrapper(SG, Tavg, L=None, D=None, P1=None, P2=None, Q=None, Ts=288.7*u.K,
                Ps=101325.*u.Pa, Zavg=1, E=E, _=wrapper_name): # pragma: no cover
        '''
        >>> Panhandle_A(SG=0.693, D=0.340*u.m, P1=90E5*u.Pa, P2=20E5*u.Pa, L=160E3*u.m, Tavg=277.15*u.K)
        <Quantity(42.560820512, 'meter ** 3 / second')>
        '''
        ans = globals()[_](SG, Tavg, L, D, P1, P2, Q, Ts, Ps, Zavg, E)    
        if Q is None and (None not in [L, D, P1, P2]):
            return ans*u.m**3/u.s
        elif D is None and (None not in [L, Q, P1, P2]):
            return ans*u.m
        elif P1 is None and (None not in [L, Q, D, P2]):
            return ans*u.Pa
        elif P2 is None and (None not in [L, Q, D, P1]):
            return ans*u.Pa
        elif L is None and (None not in [P2, Q, D, P1]):
            return ans*u.m
    globals()[wrapper] = compressible_flow_wrapper


# NOTE: class support can't do static methods unless a class is already instantiated

def differential_pressure_meter_solver(D, rho, mu, k, D2=None, P1=None, P2=None, 
                                       m=None, meter_type=None, 
                                       taps=None): # pragma: no cover
    ans = wrapped_differential_pressure_meter_solver(D, rho, mu, k, D2=D2, P1=P1, P2=P2, 
                                       m=m, meter_type=meter_type, 
                                       taps=taps)  
    if m is None and (None not in [D, D2, P1, P2]):
        return ans*u.kg/u.s
    elif D2 is None and (None not in [D, m, P1, P2]):
        return ans*u.m
    elif P2 is None and (None not in [D, D2, P1, m]):
        return ans*u.Pa
    elif P1 is None and (None not in [D, D2, m, P2]):
        return ans*u.Pa

