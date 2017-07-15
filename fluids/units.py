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
import functools
import collections
import fluids
try:
    import pint
    from pint import _DEFAULT_REGISTRY as u
    from pint import DimensionalityError
    
except ImportError: # pragma: no cover
    raise ImportError('The unit handling in fluids requires the installation '
                      'of the package pint, available on pypi or from '
                      'https://github.com/hgrecco/pint')


'''Basic module which wraps all fluids functions to be compatible with the
`pint <https://github.com/hgrecco/pint>`_ unit handling library.
All other object - dicts, classes, etc - are not wrapped. Supports star 
imports; so the same objects exported when importing from the main library
will be imported from here. 

>>> from fluids.units import *

There is no global unit registry in pint, and each registry must be a singleton.
However, there is a default registry which is suitable for use in multiple
modules at once. 

This defualt registry should be imported in one of the following ways (it does
not need to be called `u`; it can be imported from pint as `ureg` or any other
name):

>>> from pint import _DEFAULT_REGISTRY as u

Note that if the star import convention is used, it will be imported as `u`
for you. Unlike the normal convention, this registry is already initialized. To repeat
it again, you CANNOT do the following in your project and work with 
fluids.units.

>>> from pint import UnitRegistry
>>> u = UnitRegistry() # NO

All dimensional arguments to functions in fluids.units must be provided as Quantity objects.

>>> Reynolds(V=3.5*u.m/u.s, D=2*u.m, rho=997.1*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
<Quantity(6979700.0, 'dimensionless')>

The result is always one or more Quantity objects, depending on the signature
of the function called. 

For arguments whose documentation specify they are dimensionless, they can
optionaly be passed in without making them dimensionless numbers with pint.

>>> speed_synchronous(50*u.Hz, poles=12)
<Quantity(1500.0, 'revolutions_per_minute')>
>>> speed_synchronous(50*u.Hz, poles=12*u.dimensionless)
<Quantity(1500.0, 'revolutions_per_minute')>

It is good practice to use dimensionless quantities as follows, but it is 
optional.
    
>>> K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True)
<Quantity(0.0794470406403, 'meter / second')>
 
Like all pint registries, the default unit system can be changed. However, all
functions will still return the unit their documentation says they do. To
convert to the new base units, use the method .to_base_units(). 

>>> u.default_system = 'imperial'
>>> K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True).to_base_units()
<Quantity(0.0868843401578, 'yard / second')>

The order of the arguments to a function is the same as it is in the regular 
library; it won't try to infer argument position from their units, an 
exception will be raised.

>>> K_separator_Watkins(985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, 0.88*u.dimensionless, horizontal=True)
Exception: Converting 0.88 dimensionless to units of kg/m^3 raised DimensionalityError: Cannot convert from 'dimensionless' (dimensionless) to 'kilogram / meter ** 3' ([mass] / [length] ** 3)

Unsupported functions
---------------------
Some functions are too "clever" to be wrapped and are not surrently supported
in fluids.units:

* SA_tank
* isentropic_work_compression
* polytropic_exponent
* isothermal_gas
* Panhandle_A
* Panhandle_B
* Weymouth
* Spitzglass_high
* Spitzglass_low
* Oliphant
* Fritzsche
* Muller
* IGT
* roughness_Farshad
* nu_mu_converter
'''


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




__funcs = {}

for name in dir(fluids):
    obj = getattr(fluids, name)
    if isinstance(obj, types.FunctionType):
        obj = wraps_numpydoc(u)(obj)
    elif isinstance(obj, str):
        continue
    if name == '__all__':
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

#A_multiple_hole_cylinder.__doc__ = fluids.geometry.A_multiple_hole_cylinder.__doc__

def V_multiple_hole_cylinder(Do, L, holes):
    Do = Do.to(u.m).magnitude
    L = L.to(u.m).magnitude
    holes = [(i.to(u.m).magnitude, N) for i, N in holes]
    A = fluids.geometry.V_multiple_hole_cylinder(Do, L, holes)
    return A*u.m**3

#V_multiple_hole_cylinder.__doc__ = fluids.geometry.V_multiple_hole_cylinder.__doc__

u.autoconvert_offset_to_baseunit = True