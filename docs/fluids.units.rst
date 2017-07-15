Support for pint Quantities (fluids.units)
==========================================

Basic module which wraps all fluids functions to be compatible with the
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
