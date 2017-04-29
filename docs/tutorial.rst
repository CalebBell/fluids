fluids tutorial
===============

Importing
---------

Fluids can be imported as a standalone library, or all of its functions
and classes may be imported with star imports:

>>> import fluids # Good practice
>>> from fluids import * # Bad practice but convenient

All functions are available from either the main fluids module or the 
submodule; i.e. both fluids.friction_factor and 
fluids.friction.friction_factor are valid ways of accessing a function.

Dimentionless numbers
---------------------

More than 30 dimentionless numbers are available in :ref:`fluids.core <fluids.core>`:

Calculation of Reynolds and Prandtl number for water flowing in a 0.01 m 
diameter pipe at 1.5 m/s:

>>> fluids.core.Reynolds(D=0.01, rho=1000, V=1.5, mu=1E-3)
15000.0
>>> fluids.core.Prandtl(rho=1000, mu=1E-3, Cp=4200, k=0.6)
7.000000000000001




Friction factors
----------------

>>> epsilon = 1.5E-6 # m, clean steel
>>> fluids.friction.friction_factor(Re=15000, eD=epsilon/0.01)
0.02808790938573186

The transition to laminar flow is implemented abruptly at Re=2320.
Friction factor in curved pipes in available as friction_factor_curved.

ASME/ANSI pipe tables from B36.10M-2004 and B36-19M-2004 are implemented 
in fluids.piping.

Piping can be looked up based on nominal pipe size, outer diameter, or
inner diameter.

>>> nearest_pipe(NPS=2) # returns NPS, inside diameter, outer diameter, wall thickness
(2, 0.05248, 0.0603, 0.00391) 

When looking up by actual diameter, the nearest pipe as large or larger 
then requested is returned:

>>> NPS, Di, Do, t = nearest_pipe(Di=0.5)
>>> Di
0.57504

Tank geometry
-------------

Sizing of vessels and storage tanks is implemented in an object-oriented way 
as TANK in fluids.geometry. All results use the exact equations; all are
documented in the many functions in :ref:`<fluids.geometry>`

>>> T1 = TANK(D=1.2, L=4, horizontal=False)
>>> T1.V_total, T1.A # Total volume of the tank and its surface area
4.523893421169302, 17.34159144781566

By default, tanks are cylinders without heads. Tank heads can be specified
to be conical, ellipsoidal, torispherical, guppy, or spherical. The heads can 
be specified independently. The diameter and length are not required;
the total volume desired can be specified along with the length to 
diameter ratio.

>>> T1 = TANK(V=10, L_over_D=0.7, sideB='conical', horizontal=False)
>>> T1.L, T1.D
(1.7731788548899077, 2.5331126498427254)

Conical, ellipsoidal, guppy and spherical heads are all governed only
by one parameter, `a`, the distance the head extends out from the main
tank body. Torispherical heads are governed by two parameters `k` and `f`.
If these parameters are not provided, the distance the head extends out
will be 25% of the size of the tank's diameter. For torispherical heads, the
distance is similar but more complicated.

>>> TANK(D=10., V=500, horizontal=False, sideA='ellipsoidal', sideB='ellipsoidal', sideA_a=1, sideB_a=1)
<Vertical tank, V=500.000000 m^3, D=10.000000 m, L=5.032864 m, ellipsoidal heads, a=1.000000 m.>

Each TANK has __repr__ implemented, to describe the tank when printed.

Torispherical tanks default to the ratios specified as ASME F&D. Other 
standard ratios can also be used; the documentation for :ref:`<TANK>` lists
their values. Here we implement DIN 28011's ratios.

>>> TANK(D=0.01, V=0.25, horizontal=False, sideA='torispherical', sideB='torispherical')
<Vertical tank, V=0.250000 m^3, D=0.010000 m, L=3183.096137 m, torispherical heads, a=0.001693 m.>
>>> DIN = TANK(L=3, D=5, horizontal=False, sideA='torispherical', sideB='torispherical', sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
>>> print(DIN)
<Vertical tank, V=90.299352 m^3, D=5.000000 m, L=3.000000 m, torispherical heads, a=0.968871 m.>

Partial volume lookups are also useful. This is useful when the height of fluid
in the tank is known, but not the volume. The reverse calculation is also
implemented, and useful when doing dynamic simulation and to calculate the new
height after a specified volume of liquid is removed.

>>> DIN.h_max
4.937742251701451
>>> DIN.h_from_V(40)
2.3760173045849315
>>> DIN.V_from_h(4.1)
73.83841540117238

Surface areas of the heads and the main body are available as well as the total
surface area of the tank.

>>> DIN.A_sideA, DIN.A_sideB, DIN.A_lateral, DIN.A
(24.7496775831724, 24.7496775831724, 47.12388980384689, 96.62324497019169)




Atmospheric properties
----------------------
Four main classes are available to model the atmosphere. They are the
US Standard Atmosphere 1976 (ATMOSPHERE_1976), a basic
but very quick model; the NRLMSISE 00 model, substantially more powerful and
accurate and still the standard to this day (ATMOSPHERE_NRLMSISE00); and two
models for wind speed only, Horizontal Wind Model 1993 (hwm93) and 
Horizontal Wind Model 2014 (hwm14). The two horizontal wind models are actually
fortran codes, and are not compilled automatically on installation.

ATMOSPHERE_1976 is the simplest model, and very suitable for basic engineering
purposes. It supports atmospheric temperature, density, and pressure as a 
function of elevation. Optionally, a local temperature difference from earth's
average can be specified to correct the model to local conditions but this is 
only a crude approximation.

Conditions 5 km into the air:

>>> atm = ATMOSPHERE_1976(Z=5000)
>>> atm.T, atm.P, atm.rho
(255.67554322180348, 54048.28614576141, 0.7364284207799743)

The standard also specifies simplistic formulas for calculating the thermal 
conductivity, viscosity, speed of sound, and gravity at a given elevation:

>>> atm.g, atm.mu, atm.k, atm.v_sonic
(9.791241076982665, 1.628248135362207e-05, 0.02273190295142526, 320.5455196704035)

Those property routines are static methods, and can be used without instantiating
an atmosphere object:

>>> ATMOSPHERE_1976.gravity(Z=1E5)
9.505238763515356
>>> ATMOSPHERE_1976.sonic_velocity(T=300)
347.22080908230015
>>> ATMOSPHERE_1976.viscosity(T=400)
2.285266457680251e-05
>>> ATMOSPHERE_1976.thermal_conductivity(T=400)
0.033657148617592114

ATMOSPHERE_NRLMSISE00 is the recommended model, and calculates atmospheric density,
temperature, and pressure as a function of height, latitude/longitude, day of year, 
and seconds since start of day. The model can also take into account solar and 
geomagnetic disturbances which effect the atmosphere at very high elevations
if more parameters are provided. It is valid up to 1000 km. This model
is somewhat slow; it is a Python port of the fortran version, created by Joshua 
Milas. It does not support gravity profiles or transport properties, but does 
calculate the composition of the atmosphere (He, O, N2, O2, Ar, H2, N2 as 
constituents).

1000 m elevation, 45 degrees latitude and longitude, 150th day of year, 0 seconds in:

>>> atm = ATMOSPHERE_NRLMSISE00(Z=1E3, latitude=45, longitude=45, day=150)
>>> atm.T, atm.P, atm.rho
(285.54408606237405, 90394.40851588511, 1.1019062026405517)

The composition of the atmosphere is specified in terms of individual molecules/m^3:

>>> atm.N2_density, atm.O2_density
(1.7909954550444606e+25, 4.8047035072477747e+24)

This model uses the ideal gas law to convert particle counts to mass density.
Mole fractions of each species are available as well.

>>> atm.components
['N2', 'O2', 'Ar', 'He', 'O', 'H', 'N']
>>> atm.zs
[0.7811046347676225, 0.2095469403691101, 0.009343183088772914, 5.241774494627779e-06, 0.0, 0.0, 0.0]

The horizontal wind models have almost the same API, and calculate wind speed
and direction as a function of elevation, latitude, longitude, day of year and
time of day. hwm93 can also take as an argument local geomagnetic conditions 
and solar activity, but this effect was found to be so negligible it was removed
from future versions of the model such as hwm14.

Calculation of wind velocity, meridional (m/sec Northward) and zonal (m/sec
Eastward) for 1000 m elevation, 45 degrees latitude and longitude, 150th day
of year, 0 seconds in, with both models:

>>> hwm93(Z=1000, latitude=45, longitude=45, day=150)
[-0.0038965975400060415, 3.8324742317199707]
>>> hwm14(Z=1000, latitude=45, longitude=45, day=150)
[-0.9920163154602051, 0.4105832874774933]

These wind velocities are only historical normals; conditions may vary year to 
year. 


Compressor sizing
-----------------
Both isothermal and isentropic/polytropic compression models are implemented in
fluids.compressible. Isothermal compression calculates the work required to compress a gas from
one pressure to another at a specified temperature. This is the best possible case 
for compression; all actual compresssors require more work to do the compression.
By making the compression take a large number of stages and cooling the gas
between stages, this can be approached reasonable closely. Integrally 
geared compressors are often used for this purpose.

>>> isothermal_work_compression(P1=1E5, P2=1E6, T=300)
5743.425357533477

Work is calculated on a J/mol basis. If the second pressure is lower than the
first, a negative work will result and you are modeling an expander instead
of a compressor. Gas compressibility factor can also be specified. The lower
the gas's compressibility factor, the less power required to compress it.

>>> isothermal_work_compression(P1=1E6, P2=1E5, T=300)
-5743.425357533475
>>> isothermal_work_compression(P1=1E5, P2=1E6, T=300, Z=0.95)
5456.2540896568025

There is only one function implemented to model both isentropic and polytropic
compressors, as the only difference is that a polytropic exponent `n` is used
instead of the gas's isentropic exponent Cp/Cv `k` and the type of efficiency
is changed. The model requires initial temperature, inlet and outlet pressure,
isentropic exponent or polytropic exponent, and optionally an efficiency.

Compressing air from 1 bar to 10 bar, with inlet temperature of 300 K and
efficiency of 78%:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78) # work, J/mol
10416.873455626454

The model allows for the inlet or outlet pressure or efficiency to be calculated
instead of the work:

>>> isentropic_work_compression(T1=300, P1=1E5, P2=1E6, k=1.4, W=10416) # Calculate efficiency
0.7800654085434559
>>> isentropic_work_compression(T1=300, P1=1E5, k=1.4, W=10416, eta=0.78) # Calculate P2
999858.5366533266
>>> isentropic_work_compression(T1=300, P2=1E6, k=1.4, W=10416, eta=0.78) # Calculate P1
100014.14833613831

The approximate temperature rise can also be calculated with the function
isentropic_T_rise_compression.

>>> T2 = isentropic_T_rise_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78)
>>> T2, T2-300 # outlet temperature and temperature rise, K
(657.960664955096, 357.96066495509604)

It is more accurate to use an enthalpy-based model which incorporates departure
functions.

Polytropic exponents and efficiencies are convertible to isentropic exponents and
efficiencies.  For the above example, with k=1.4 and `eta_s`=0.78:

>>> eta_p = isentropic_efficiency(P1=1E5, P2=1E6, k=1.4, eta_s=0.78) # with eta_s specified, returns polytropic efficiency
>>> n = polytropic_exponent(k=1.4, eta_p=eta_p)
>>> eta_p, n
(0.8376785349411107, 1.517631868575738)

With those results, we can prove the calculation worked by calculating the
work required using these polytropic inputs:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=n, eta=eta_p)
10416.873455626452

The work is the same as calculated with the original inputs. Note that the 
conversion is specific to three inputs: Inlet pressure; outlet pressure;
and isentropic exponent `k`. If any of those change, then the calculated
polytropic exponent and efficiency will be different as well.

To go in the reverse direction, we take the case of isentropic exponent 
k =Cp/Cv=1.4, eta_p=0.83 The power is calculated to be:

We first need to calculate the polytropic exponent from the polytropic
efficiency:

>>> n = polytropic_exponent(k=1.4, eta_p=0.83)
>>> print(n)
1.5249343832

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=n, eta=0.83)
10556.494602042329

Converting polytropic efficiency to isentropic efficiency:

>>> eta_s = isentropic_efficiency(P1=1E5, P2=1E6, k=1.4, eta_p=0.83)
>>> print(eta_s)
0.7588999047069671

Checking the calculated power is the same:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=eta_s)
10556.494602042327

Gas pipeline sizing
-------------------

The standard isothermal compressible gas flow is fully implemented, and through
a variety of numerical and analytical expressions, can solve for any of the
following parameters:

    * Mass flow rate
    * Upstream pressure (numerical)
    * Downstream pressure (analytical or numerical if an overflow occurs)
    * Diameter of pipe (numerical)
    * Length of pipe

Solve for the mass flow rate of gas (kg/s) flowing through a 1 km long 0.5 m
inner diameter pipeline, initially at 10 bar with a density of 11.3 kg/m^3
going downstream to a pressure of 9 bar.

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, D=0.5)
145.4847572636031

The same case, but sizing the pipe to take 100 kg/s of gas:

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, m=100)
0.42971708911060613

The same case, but determining what the outlet pressure will be if 200 kg/s
flow in the 0.5 m diameter pipe:

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, D=0.5, L=1000, m=200)
784701.0681827427

Determining pipe length from known diameter, pressure drop, and mass flow
(possible but not necessarily useful):

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, D=0.5, m=150)
937.3258027759333

Not all specified mass flow rates are possible. At a certain downstream
pressure, chocked flow will develop - that downstream pressure is that
at which the mass flow rate reaches a maximum. An exception will be
raised if such an input is specified:

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, L=1000, D=0.5, m=260)
Exception: The desired mass flow rate cannot be achieved with the specified upstream pressure; the maximum flowrate is 257.216733 at an downstream pressure of 389699.731765
>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=3E5, L=1000, D=0.5)
Exception: Given outlet pressure is not physically possible due to the formation of choked flow at P2=389699.731765, specified outlet pressure was 300000.000000

A number of limitations exist with respect to the accuracy of this model:
    
* Density dependence is that of an ideal gas.
* If calculating the pressure drop, the average gas density cannot
  be known immediately; iteration must be used to correct this.
* The friction factor depends on both the gas density and velocity,
  so it should be solved for iteratively as well. It changes throughout
  the pipe as the gas expands and velocity increases.
* The model is not easily adapted to include elevation effects due to 
  the acceleration term included in it.
* As the gas expands, it will change temperature slightly, further
  altering the density and friction factor.
  
We can explore how the gas density and friction factor effect the model using
the `thermo library <https://github.com/CalebBell/thermo>`_ for chemical properties.

Compute the downstream pressure of 50 kg/s of natural gas flowing in a 0.5 m 
diameter pipeline for 1 km, roughness = 5E-5 m:
 
>>> from thermo import *
>>> from fluids import *
>>> D = 0.5
>>> L = 1000
>>> epsilon = 5E-5
>>> S1 = Stream('natural gas', P=1E6, m=50)
>>> V = S1.Q/(pi/4*D**2)
>>> Re = S1.Reynolds(D=D, V=V)
>>> fd = friction_factor(Re=Re, eD=epsilon/D)
>>> P2 = isothermal_gas(rho=S1.rho, fd=fd, P1=S1.P, D=D, L=L, m=S1.m)
>>> 877852.8365849017

In the above example, the friction factor was calculated using the density
and velocity of the gas when it enters the stream. However, the average values,
at the middle pressure, and more representative. We can iterate to observe
the effect of using the average values:

>>> for i in range(10):
>>>     S2 = Stream('natural gas', P=0.5*(P2+S1.P), m=50)
>>>     V = S2.Q/(pi/4*D**2)
>>>     Re = S2.Reynolds(D=D, V=V)
>>>     fd = friction_factor(Re=Re, eD=epsilon/D)
>>>     P2 = isothermal_gas(rho=S2.rho, fd=fd, P1=S1.P, D=D, L=L, m=S1.m)
>>>     print(P2)
868992.832357
868300.621412
868246.236225
868241.961444
868241.625427
868241.599014
868241.596938
868241.596775
868241.596762
868241.596761

As can be seen, the system converges very quickly. The difference in calculated
pressure drop is approximately 1%.




