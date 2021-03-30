Support for pint Quantities (fluids.units)
==========================================

Basic module which wraps all fluids functions and classes to be compatible with the
`pint <https://github.com/hgrecco/pint>`_ unit handling library.
All other object - dicts, lists, etc - are not wrapped. 

>>> import fluids
>>> fluids.units.friction_factor # doctest: +ELLIPSIS
<function friction_factor at 0x...>

The fluids.units module also supports star imports; the same objects exported when importing from the main library
will be imported from fluids.units.

>>> from fluids.units import *

It is also possible to use `fluids.units` without the star import:


There is no global unit registry in pint, and each registry must be a singleton.
However, there is a default registry which is suitable for use in multiple
modules at once. 

This default registry should be imported in one of the following ways (it does
not need to be called `u`; it can be imported from pint as `ureg` or any other
name):

>>> from pint import _DEFAULT_REGISTRY as u

Note that if the star import convention is used, the default unit registry be imported as `u`
for you. Unlike the normal convention, this registry is already initialized. To repeat
it again, you CANNOT do the following in your project and work with 
fluids.units.

>>> from pint import UnitRegistry
>>> u = UnitRegistry() # doctest: +SKIP

All dimensional arguments to functions in fluids.units must be provided as Quantity objects.

>>> Reynolds(V=3.5*u.m/u.s, D=2*u.m, rho=997.1*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
<Quantity(6979700.0, 'dimensionless')>

The result is always one or more Quantity objects, depending on the signature
of the function called. 

For arguments whose documentation specify they are dimensionless, they can
optionally be passed in without making them dimensionless numbers with pint.

>>> speed_synchronous(50*u.Hz, poles=12*u.dimensionless) # doctest: +SKIP
<Quantity(1500.0, 'revolutions_per_minute')>
>>> speed_synchronous(50*u.Hz, poles=12)
<Quantity(1500.0, 'revolutions_per_minute')>

It is good practice to use dimensionless quantities as follows, but it is 
optional.
    
>>> K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True)
<Quantity(0.079516136, 'meter / second')>
 
Like all pint registries, the default unit system can be changed. However, all
functions will still return the unit their documentation says they do. To
convert to the new base units, use the method .to_base_units(). 

>>> u.default_system = 'imperial'
>>> K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True).to_base_units()
<Quantity(0.0869599038, 'yard / second')>
>>> u.default_system = 'mks'

The order of the arguments to a function is the same as it is in the regular 
library; it won't try to infer argument position from their units, an 
exception will be raised.

>>> K_separator_Watkins(985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, 0.88*u.dimensionless, horizontal=True) # doctest: +SKIP
Exception: Converting 0.88 dimensionless to units of kg/m^3 raised DimensionalityError: Cannot convert from 'dimensionless' (dimensionless) to 'kilogram / meter ** 3' ([mass] / [length] ** 3)

Support for classes is provided by wrapping each class by a proxy class which reads
the docstrings of each method and the main class to determine the inputs and outputs.
Properties, attributes, inputs, and units are all included.


>>> T1 = TANK(L=3*u.m, D=150*u.cm, horizontal=True)
>>> T1.V_total, T1.h_max
(<Quantity(5.3014376, 'meter ** 3')>, <Quantity(1.5, 'meter')>)
>>> T1.V_from_h(0.1*u.m)
<Quantity(0.151783071, 'meter ** 3')>

>>> atm = ATMOSPHERE_NRLMSISE00(Z=1E3*u.m, latitude=45*u.degrees, longitude=45*u.degrees, day=150*u.day)
>>> atm.rho, atm.O2_density
(<Quantity(1.1019062, 'kilogram / meter ** 3')>, <Quantity(4.80470351e+24, 'count / meter ** 3')>)