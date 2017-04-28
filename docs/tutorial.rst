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


Tank Geometry
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

Helical Coils
-------------
As coils are often used in fluid dynamics calculations, a convenience class 
to construct them is available, HelicalCoil.





