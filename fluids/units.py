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
import types
import re
import functools
import collections
import fluids
try:
    import pint
    from pint import _DEFAULT_REGISTRY as u
    from pint import DimensionalityError
    
except ImportError:
    raise ImportError('The unit handling in fluids requires the installation'
                      'of the package pint, available on pypi or from '
                      'https://github.com/hgrecco/pint')


expr = re.compile('Parameters *\n *-+\n +')
expr2 = re.compile('Returns *\n *-+\n +')
match_sections = re.compile('\n *[A-Za-z ]+ *\n +-+')
match_section_names = re.compile('\n *[A-Za-z]+ *\n +-+')
variable = re.compile('[a-zA-Z_0-9]* : ')
match_units = re.compile('\[[a-zA-Z0-9()./*^\- ]*\]')


def parse_numpydoc_variables_units(func):
    text = func.__doc__
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
            match = match.replace('[', '').replace(']', '').replace('-', 'dimensionless')
            if match == '':
                match = 'dimensionless'
            if match == 'base SI':
                match = 'dimensionless' # TODO - write special wrappers for these cases
            units.append(match)

        parsed[section] = {'units': units, 'vars': parameter_vars}
    return parsed


def convert_input(val, unit, ureg, strict=True):
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
                
            result = func(*conv_values, **kwargs)
            
            # Attempt to handle multiple return values
            # Must be able to convert all values to a pint expression
            if isinstance(result, collections.Iterable):
                conveted_result = []
                for ans, unit in zip(result, out_units):
                    conveted_result.append(ans*ureg.parse_expression(unit))
                return conveted_result
            else:
                return result*ureg.parse_expression(out_units[0])

        return wrapper
    return decorator



__all__ = ['wraps_numpydoc', 'u']

__funcs = {}

for name in dir(fluids):
    obj = getattr(fluids, name)
    if isinstance(obj, types.FunctionType):
        obj = wraps_numpydoc(u)(obj)
    elif isinstance(obj, str):
        continue
    __all__.append(name)
    __funcs.update({name: obj})
globals().update(__funcs)


'''
Known unsupported functions:
* A_multiple_hole_cylinder
* V_multiple_hole_cylinder
* SA_tank
isentropic_work_compression, polytropic_exponent, isothermal_gas, Panhandle_A, Panhandle_B, Weymouth, Spitzglass_high, Spitzglass_low, Oliphant, Fritzsche, Muller, IGT
roughness_Farshad
nu_mu_converter

All the classes

'''

def A_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.A_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**2

A_multiple_hole_cylinder.__doc__ = fluids.geometry.A_multiple_hole_cylinder.__doc__

def V_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.V_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**3

V_multiple_hole_cylinder.__doc__ = fluids.geometry.V_multiple_hole_cylinder.__doc__
