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

Design philosophy
-----------------
Like all libraries, this was developed to scratch my own itches. Since its
public release it has been found useful by many others, from students across 
the world to practicing engineers at some of the world's largest companies.

The bulk of this library's API is considered stable; enhancements to 
functions and classes will still happen, and default methods when using a generic 
correlation interface may change to newer and more accurate correlations as
they are published and reviewed.

To the extent possible, correlations are implemented depending on the highest
level parameters. The friction_factor correlation does not accept pipe diameter,
velocity, viscosity, density, and roughness - it accepts Reynolds number and
relative roughness. This makes the API cleaner and encourages modular design.

All functions are designed to accept inputs in base SI units. However, any 
set of consistent units given to a function will return a consistent result;
for instance, a function calculating volume doesn't care if given an input in
inches or meters; the output units will be the cube of those given to it.
The user is directed to unit conversion libraries such as 
`pint <https://github.com/hgrecco/pint>`_ to perform unit conversions if they
prefer not to work in SI units. The tutorial for using it with fluids is
at :doc:`fluids.units <fluids.units>`.

There are two ways to use numpy arrays with fluids. Easiest to use is a `vectorized` module,
which wraps all of the fluids functions with np.vectorize. Instead of importing
from fluids, the user can import from :doc:`fluids.vectorized <fluids.vectorized>`:

>>> from fluids.vectorized import * # doctest: +SKIP
>>> friction_factor(Re=[100, 1000, 10000], eD=0) # doctest: +SKIP
array([0.64      , 0.064     , 0.03088295])

It is possible to switch back and forth between the namespaces with a subsequent
import:

>>> from fluids import * 

The second way is `Numba <https://github.com/numba/numba>`_. This
optional dependency provides the speed you expect from NumPy arrays -
or better. In some cases, much better. The tutorial for using it
is at :doc:`fluids.numba <fluids.numba>`, but in general use it the same way but
with a different import.

>>> from fluids.numba_vectorized import * # doctest: +SKIP

Dimensionless numbers
---------------------

More than 30 Dimensionless numbers are available in :py:mod:`fluids.core` :

Calculation of :py:func:`~.Reynolds` and :py:func:`~.Prandtl` number for
water flowing in a 0.01 m diameter pipe at 1.5 m/s:

>>> fluids.core.Reynolds(D=0.01, rho=1000, V=1.5, mu=1E-3)
15000.0
>>> fluids.core.Prandtl(rho=1000, mu=1E-3, Cp=4200, k=0.6)
7.000000000000001

Where different parameters may be used with a dimensionless number, either
a separate function is created for each or both sets of parameters are can
be specified. For example, instead of specifying viscosity and density for the
Reynolds number calculation, kinematic viscosity could have been used instead:

>>> Reynolds(D=0.01, V=1.5, nu=1E-6)
15000.0

In the case of groups like the Fourier number, used in both heat and mass
transfer, two separate functions are available, :py:func:`~.Fourier_heat` and 
:py:func:`~.Fourier_mass`. The heat transfer version supports specifying either the 
density, heat capacity, and thermal conductivity - or just the thermal 
diffusivity. There is no equivalent set of three parameters for the mass
transfer version; it always requires mass diffusivity.

>>> Fourier_heat(t=1.5, L=2, rho=1000., Cp=4000., k=0.6)
5.625e-08
>>> Fourier_heat(1.5, 2, alpha=1E-7)
3.75e-08
>>> Fourier_mass(t=1.5, L=2, D=1E-9)
3.7500000000000005e-10

Among the coded dimensionless numbers are :py:func:`~.Archimedes`, :py:func:`~.Bejan_L`, :py:func:`~.Bejan_p`, :py:func:`~.Biot`, :py:func:`~.Boiling`, :py:func:`~.Bond`, :py:func:`~.Capillary`, :py:func:`~.Cavitation`, :py:func:`~.Confinement`, :py:func:`~.Dean`, :py:func:`~.Drag`, :py:func:`~.Eckert`, :py:func:`~.Euler`, :py:func:`~.Fourier_heat`, :py:func:`~.Fourier_mass`, :py:func:`~.Froude_densimetric`, :py:func:`~.Froude`, :py:func:`~.Graetz_heat`, :py:func:`~.Grashof`, :py:func:`~.Hagen`, :py:func:`~.Jakob`, :py:func:`~.Knudsen`, :py:func:`~.Lewis`, :py:func:`~.Mach`, :py:func:`~.Nusselt`, :py:func:`~.Ohnesorge`, :py:func:`~.Peclet_heat`, :py:func:`~.Peclet_mass`, :py:func:`~.Power_number`, :py:func:`~.Prandtl`, :py:func:`~.Rayleigh`, :py:func:`~.Reynolds`, :py:func:`~.Schmidt`, :py:func:`~.Sherwood`, :py:func:`~.Stanton`, :py:func:`~.Stokes_number`, :py:func:`~.Strouhal`, :py:func:`~.Suratman`, :py:func:`~.Weber`, :py:func:`~.Morton`.

Miscellaneous utilities
-----------------------
More than just dimensionless groups are implemented in :py:mod:`fluids.core`.

Converters between loss coefficient, L/D equivalent, length of pipe, and
pressure drop are available.
It is recommended to convert length/diameter equivalents and lengths of pipe
at specified friction factors to loss coefficients using the 
:py:func:`~.K_from_L_equiv` and :py:func:`~.K_from_f` functions respectively. 
They can all be summed easily afterwards.

>>> K_from_f(fd=0.018, L=100., D=.3)
6.0
>>> K_from_L_equiv(L_D=240, fd=0.02)
4.8

Either head loss or pressure drop can be calculated once the total loss 
coefficient K is known using :py:func:`~.head_from_K` or :py:func:`~.dP_from_K`
respectively. Head loss does not require knowledge of the fluid's
density, but pressure drop does.

>>> head_from_K(K=(6+4.8), V=3)
4.955820795072732

>>> dP_from_K(K=(6+4.8), rho=1000, V=3)
48600.0

If a K value is known and desired to be converted to a L/D ratio or to an
equivalent length of pipe, that calculation is available as well with
:py:func:`~.L_from_K` or :py:func:`~.L_equiv_from_K` respectively:

>>> L_from_K(K=6, fd=0.018, D=.3)
100.0
>>> L_equiv_from_K(3.6, fd=0.02)
180.0

Pressure and head are also convertible with the :py:func:`~.head_from_P`
and :py:func:`~.P_from_head` functions:

>>> head_from_P(P=98066.5, rho=1000)
10.000000000000002
>>> P_from_head(head=5., rho=800.)
39226.6

Also implemented in :py:mod:`fluids.core`. are the following:

:py:func:`~.thermal_diffusivity`:

>>> thermal_diffusivity(k=0.02, rho=1., Cp=1000.)
2e-05

Speed of sound in an ideal gas :py:func:`~.c_ideal_gas`: 
(requires temperature, isentropic exponent Cp/Cv):

>>> c_ideal_gas(T=303, k=1.4, MW=28.96)
348.9820953185441

A converter between dynamic and kinematic viscosity :py:func:`~.nu_mu_converter`:

>>> nu_mu_converter(rho=998., nu=1.0E-6)
0.000998
>>> nu_mu_converter(998., mu=0.000998)
1e-06

Calculation of :py:func:`~.gravity` on earth as a function of height 
and latitude (input in degrees and height in meters):

>>> gravity(latitude=55, H=1E6)
6.729011976863571

    
Friction factors
----------------
Friction factor is easily calculable with :py:func:`~.friction_factor`.

>>> epsilon = 1.5E-6 # m, clean steel
>>> fluids.friction.friction_factor(Re=15000, eD=epsilon/0.01)
0.028087909385731864

The transition to laminar flow is implemented abruptly at Re=2040,
one of the latest experimental results which is accurate to +/- 10. 
If the Reynolds number is in the laminar regime, the transition to a
different correlation happens  automatically and the well-known 
solution fd = 64/Re is given.

>>> fluids.friction.friction_factor(Re=150)
0.4266666666666667

Friction factor in curved pipes in available as :py:func:`~.friction_factor_curved`.
The curved friction factor is applicable for helices and coils, and to a
lesser extent curved bends.

>>> friction_factor_curved(Re=15000, Di=.01, Dc=2.5, roughness=1.5E-6)
0.029846229072776263

The critical Reynolds number for curved pipes
is increased compared to straight pipe flow, and is a function of the 
curvature of the pipe. The preferred method to calculate the transition 
(used by default for the automatic regime transition)
is the method of Schmidt (1967) :py:func:`~.helical_transition_Re_Schmidt`.

>>> helical_transition_Re_Schmidt(Di=.01, Dc=2.5)
3948.7442097768603

Although roughness is a hard value to know without measuring it for a pipe,
several hundred pipe conditions have had their roughness values measured in the
literature, and they can be searched through using fuzzy matching and the
functions :py:func:`~.nearest_material_roughness` and :py:func:`~.material_roughness`.

>>> nearest_material_roughness('Used water piping', clean=False)
'Seamless steel tubes, Used water piping'
>>> material_roughness('Seamless steel tubes, Used water piping')
0.0015

The material_roughness function can also be used directly, but in that case
there is no feedback about the material which was found.

>>> material_roughness('glass')
1e-05

As fuzzy string matching is a pretty terrible solution, it is encouraged to find the
desired string in the `actual source code of fluids <https://github.com/CalebBell/fluids/blob/master/fluids/friction.py#L2766>`_.

There is one more way of obtaining the roughness of a clean pipe, developed by
Farshad and Rieke (2006) :py:func:`~.roughness_Farshad`. It has been established 
that in commercial pipe, the larger the diameter, the larger the roughness. 

>>> roughness_Farshad('Carbon steel, bare', D=0.05)
3.529128126365038e-05

Only the following types of clean, new pipe have data available:

* 'Plastic coated'
* 'Carbon steel, honed bare'
* 'Cr13, electropolished bare'
* 'Cement lining'
* 'Carbon steel, bare'
* 'Fiberglass lining'
* 'Cr13, bare'


There is also a term called `Transmission factor`, used in many pipeline applications.
It is effectively a variant on friction factor. They can be inter-converted 
with the :py:func:`~.transmission_factor` function.

>>> transmission_factor(fd=0.0185) # calculate transmission factor
14.704292441876154
>>> transmission_factor(F=20) # calculate Darcy friction factor
0.01


Pipe schedules
--------------
ASME/ANSI pipe tables from B36.10M-2004 and B36-19M-2004 are implemented 
in fluids.piping.

Piping can be looked up based on nominal pipe size, outer diameter, or
inner diameter with the :py:func:`~.nearest_pipe` function.

>>> nearest_pipe(NPS=2) # returns NPS, inside diameter, outer diameter, wall thickness
(2, 0.05248, 0.0603, 0.00391)

When looking up by actual diameter, the nearest pipe as large or larger 
then requested is returned:

>>> NPS, Di, Do, t = nearest_pipe(Di=0.5)
>>> Di
0.57504
>>> nearest_pipe(Do=0.5)
(20, 0.47781999999999997, 0.508, 0.01509)

By default, the pipe schedule used for the lookup is schedule 40. Other schedules 
that are available are: '5', '10', '20', '30', '40', '60', '80', '100',
'120', '140', '160', 'STD', 'XS', 'XXS', '5S', '10S', '40S', '80S'.

>>> nearest_pipe(Do=0.5, schedule='40S')
(20, 0.48894, 0.508, 0.009529999999999999)
>>> nearest_pipe(Do=0.5, schedule='80')
(20, 0.45562, 0.508, 0.02619)

If a diameter which is larger than any pipe in the schedule is input, an
exception is raised:

>>> nearest_pipe(Do=1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "fluids/piping.py", line 276, in nearest_pipe
    raise ValueError('Pipe input is larger than max of selected schedule')
ValueError: Pipe input is larger than max of selected schedule


Wire gauges
-----------

The construction of mechanical systems often uses the "gauge" systems, a variety
of old imperial conversions between plate or wire thickness and a dimensionless
number. Conversion from and to the gauge system is done by the :py:func:`~.gauge_from_t` 
and :py:func:`~.t_from_gauge` functions.

Looking up the gauge from a wire of known diameter approximately 1.2 mm:

>>> gauge_from_t(.0012)
18

The reverse conversion:

>>> t_from_gauge(18)
0.001245

Other schedules are also supported: 

* Birmingham Wire Gauge (BWG) ranges from 0.2 (0.5 inch) to 36 (0.004 inch).
* American Wire Gauge (AWG) ranges from 0.167 (0.58 inch) to 51 (0.00099
  inch). These are used for electrical wires.
* Steel Wire Gauge (SWG) ranges from 0.143 (0.49 inch) to 51 (0.0044 inch).
  Also called Washburn & Moen wire gauge, American Steel gauge, Wire Co.
  gauge, and Roebling wire gauge.
* Music Wire Gauge (MWG) ranges from 0.167 (0.004 inch) to 46 (0.18
  inch). Also called Piano Wire Gauge.
* British Standard Wire Gage (BSWG) ranges from 0.143 (0.5 inch) to
  51 (0.001 inch). Also called Imperial Wire Gage (IWG).
* Stub's Steel Wire Gage (SSWG) ranges from 1 (0.227 inch) to 80 (0.013 inch)

>>> t_from_gauge(18, schedule='AWG')
0.00102362


Tank geometry
-------------

Sizing of vessels and storage tanks is implemented in an object-oriented way 
as :py:class:`~.TANK` in :py:mod:`fluids.geometry`. All results use the exact equations; all are
documented in the many functions in :py:mod:`fluids.geometry`.

>>> T1 = TANK(D=1.2, L=4, horizontal=False)
>>> T1.V_total, T1.A # Total volume of the tank and its surface area
(4.523893421169302, 17.34159144781566)

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
standard ratios can also be used; the documentation for :py:class:`~.TANK` lists
their values. Here we implement DIN 28011's ratios.

>>> TANK(D=0.01, V=0.25, horizontal=False, sideA='torispherical', sideB='torispherical')
<Vertical tank, V=0.250000 m^3, D=0.010000 m, L=3183.096799 m, torispherical heads, a=0.001693 m.>
>>> DIN = TANK(L=3, D=5, horizontal=False, sideA='torispherical', sideB='torispherical', sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
>>> print(DIN)
<Vertical tank, V=83.646361 m^3, D=5.000000 m, L=3.000000 m, torispherical heads, a=0.968871 m.>

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
>>> DIN.SA_from_h(2.1)
42.51740838962569

Surface areas of the heads and the main body are available as well as the total
surface area of the tank.

>>> DIN.A_sideA, DIN.A_sideB, DIN.A_lateral, DIN.A
(24.7496775831724, 24.7496775831724, 47.12388980384689, 96.62324497019169)

Miscellaneous geometry
----------------------
In addition to sizing all sorts of tanks, helical coils are supported and so are 
a number of other simple calculations.

Sphericity is implemented as :py:func:`~.sphericity`, requiring a calculated
surface area and volume. 
For a cube of side length 3, the surface area is 6*a^2=54 and volume a^3=27.
Its sphericity is then:

>>> sphericity(A=54, V=27)
0.8059959770082346

Aspect ratio of is implemented as :py:func:`~.aspect_ratio`; for example,
a rectangle 0.2 m by 2 m:

>>> aspect_ratio(.2, 2)
0.1

Circularity, a parameter used to characterize 2d images of particles, is implemented
as :py:func:`~.circularity`.
For a rectangle, one side length = 1, second side length = 100:

>>> D1 = 1
>>> D2 = 100
>>> A = D1*D2
>>> P = 2*D1 + 2*D2
>>> circularity(A, P)
0.030796908671598795


Atmospheric properties
----------------------
Various main classes are available to model the atmosphere, of varying accuracy. They are the
US Standard Atmosphere 1976 (:py:class:`~.ATMOSPHERE_1976`), a basic
but very quick model; the NRLMSISE 00 model, substantially more powerful and
accurate and still the standard to this day (:py:class:`~.ATMOSPHERE_NRLMSISE00`); and two
models for wind speed only, Horizontal Wind Model 1993 (:py:func:`~.hwm93`) and 
Horizontal Wind Model 2014 (:py:func:`~.hwm14`). The two horizontal wind models are actually
fortran codes, and are not compiled automatically on installation. Solar models are :py:func:`~.earthsun_distance`,
:py:func:`~.solar_position`, :py:func:`~.sunrise_sunset` and :py:func:`~.solar_irradiation`.

:py:class:`~.ATMOSPHERE_1976` is the simplest model, and very suitable for basic engineering
purposes. It supports atmospheric temperature, density, and pressure as a 
function of elevation. Optionally, a local temperature difference from earth's
average can be specified to correct the model to local conditions but this is 
only a crude approximation.

Conditions 5 km into the air:

>>> atm = ATMOSPHERE_1976(Z=5000)
>>> atm.T, atm.P, atm.rho
(255.6755432218, 54048.2861457, 0.736428420779)

The standard also specifies simplistic formulas for calculating the thermal 
conductivity, viscosity, speed of sound, and gravity at a given elevation:

>>> atm.g, atm.mu, atm.k, atm.v_sonic
(9.79124107698, 1.62824813536e-05, 0.0227319029514, 320.545519670)

Those property routines are static methods, and can be used without instantiating
an atmosphere object:

>>> ATMOSPHERE_1976.gravity(Z=1E5)
9.50523876351
>>> ATMOSPHERE_1976.sonic_velocity(T=300)
347.2208090823
>>> ATMOSPHERE_1976.viscosity(T=400)
2.28526645768e-05
>>> ATMOSPHERE_1976.thermal_conductivity(T=400)
0.03365714861759

:py:class:`~.ATMOSPHERE_NRLMSISE00` is the recommended model, and calculates atmospheric density,
temperature, and pressure as a function of height, latitude/longitude, day of year, 
and seconds since start of day. The model can also take into account solar and 
geomagnetic disturbances which effect the atmosphere at very high elevations
if more parameters are provided. It is valid up to 1000 km. This model
is somewhat slow; it is a Python port of the Fortran version, created by Joshua 
Milas. It does not support gravity profiles or transport properties, but does 
calculate the composition of the atmosphere (He, O, N2, O2, Ar, H2, N2 as 
constituents).

1000 m elevation, 45 degrees latitude and longitude, 150th day of year, 0 seconds in:

>>> atm = ATMOSPHERE_NRLMSISE00(Z=1E3, latitude=45, longitude=45, day=150)
>>> atm.T, atm.P, atm.rho
(285.54408606237, 90394.44061071, 1.1019062026405)

The composition of the atmosphere is specified in terms of individual molecules/m^3:

>>> atm.N2_density, atm.O2_density
(1.7909954550444e+25, 4.8047035072477e+24)

This model uses the ideal gas law to convert particle counts to mass density.
Mole fractions of each species are available as well.

>>> atm.components
['N2', 'O2', 'Ar', 'He', 'O', 'H', 'N']
>>> atm.zs
[0.7811046347676, 0.2095469403691, 0.009343183088772, 5.241774494627e-06, 0.0, 0.0, 0.0]

The horizontal wind models have almost the same API, and calculate wind speed
and direction as a function of elevation, latitude, longitude, day of year and
time of day. hwm93 can also take as an argument local geomagnetic conditions 
and solar activity, but this effect was found to be so negligible it was removed
from future versions of the model such as hwm14.

Calculation of wind velocity, meridional (m/sec Northward) and zonal (m/sec
Eastward) for 1000 m elevation, 45 degrees latitude and longitude, 150th day
of year, 0 seconds in, with both models:

>>> hwm93(Z=1000, latitude=45, longitude=45, day=150)  # doctest: +SKIP
(-0.0038965975400060415, 3.8324742317199707)
>>> hwm14(Z=1000, latitude=45, longitude=45, day=150)  # doctest: +SKIP
(-0.9920163154602051, 0.4105832874774933)

These wind velocities are only historical normals; conditions may vary year to 
year. 

The solar radiation model is based around the Sun Position Algorithm (SPA)
developed by NREL; it can calculate the position of the sun in the sky at
any time for any place on Earth, and can calculate how far away the sun is
from Earth. The python implementation used is a slightly modified version
of the Python implementation written by Tony Lorenzo and released under
the BSD 3-clause license. The algorithm is published with the excellent
`pvlib <https://github.com/pvlib/pvlib-python>`_ library for solar 
energy modelling applications. 

The functions included are 
:py:func:`~.earthsun_distance`, :py:func:`~.sunrise_sunset`, 
:py:func:`~.solar_position` and :py:func:`~.solar_irradiation`.
All take and/or receive datetime instances, which introduces the
nightmare of time zones.

All the functions have no internal way of knowing about what time zone
the latitude/longitude inputs are in. They only calculate the position
of earth, and they need to know what "real" time it is, so it can deal
with leap seconds, etc. There are now two options for how to provide
time inputs. The first is to provide the time in the UTC time zone,
which has replaced Greenwich Mean Time (GMT) as the standard reference time.
The inputs and outputs of this function will look strange, because
unless you happen to be working somewhere with that time zone,
you have to convert the time inputs to that time zone initially.

So to find the solar position at 6 AM in Perth, Australia (offset -8 hours), we would manually 
convert the time zone.

>>> from datetime import datetime, timedelta
>>> solar_position(datetime(2020, 6, 6, 14, 30, 0) - timedelta(hours=8), -31.95265, 115.85742)
[63.4080568623, 63.4400018158, 26.59194313766, 26.5599981841, 325.121376246, 75.7467475485]

This painful, so timezone support has been added to the functions
using the library `pytz`.

>>> import pytz
>>> when = pytz.timezone('Australia/Perth').localize(datetime(2020, 6, 6, 14, 30, 0))
>>> solar_position(when, -31.95265, 115.85742)
[63.4080568623, 63.4400018158, 26.59194313766, 26.55999818417, 325.121376246, 75.7467475485]


To determine the distance of earth and the sun, use the 
:py:func:`~.earthsun_distance` function which accepts a single datetime
object and returns the distance in meters. This is still impacted by timezones.

>>> earthsun_distance(pytz.timezone('America/Edmonton').localize(datetime(2003, 10, 17, 13, 30, 30)))
149080606927.6

To determine when the sun rises, sets, and is at solar noon, use the
:py:func:`~.sunrise_sunset` function, which accepts a datetime 
instance, a latitude, and a longitude in degrees. 

>>> import pytz
>>> sunrise, sunset, transit = sunrise_sunset(pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 17)), 51.0486, -114.07)
>>> sunrise
datetime.datetime(2018, 4, 16, 6, 39, 1, 570479, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>)
>>> sunset
datetime.datetime(2018, 4, 16, 20, 32, 25, 778162, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>)
>>> transit
datetime.datetime(2018, 4, 16, 13, 36, 0, 386341, tzinfo=<DstTzInfo 'America/Edmonton' MDT-1 day, 18:00:00 DST>)

To determine where in the sky the sun appears at any location and 
time, use the :py:func:`~.solar_position` function, which requires 
a datetime instance, a latitude, and a longitude.

>>> apparent_zenith, _, _, _, azimuth, _ = solar_position(pytz.timezone('America/Edmonton').localize(datetime(2003, 10, 17, 13, 30, 30)), 51.0486, -114.07)
>>> apparent_zenith, azimuth
(60.3674252872, 182.513677566)

The function returns several other properties which may be of interest.
Its first return value, apparent_zenith, is the zenith which an observer
on the ground would see the sun at after accounting for atmospheric
refraction. To more accurately calculate the solar position, the temperature
and pressure at ground level are required as well - as they impact the 
refraction as well; these arguments are accepted as well by :py:func:`~.solar_position` for more accuracy. 
When specifying pressure, be sure to use the real pressure of the site - not an adjusted to
standard conditions one as reported by weather stations!

>>> solar_position(pytz.timezone('America/Edmonton').localize(datetime(2003, 10, 17, 13, 30, 30)), 51.0486, -114.07, T=290, P=8.9E4)[0]
60.370155603

The primary application of sun position is for calculating the amount of sunlight received
by an object, via the :py:func:`~.solar_irradiation` function. Unlike the previous functions,
it requires an installation of `pvlib <https://github.com/pvlib/pvlib-python>`_ to work.

In addition to the arguments previously discussed, the surface_tilt and surface_azimuth
of the object are required. The object is assumed to be a plane only - other objects 
need to be discretized into planes through finite-element calculations. The elevation
is required, as well as the average albedo of the ground surrounding the object (not
immediately; within several kilometers). The calculation is then straightforward:

>>> solar_irradiation(Z=1100.0, latitude=51.0486, longitude=-114.07, linke_turbidity=3,
... moment=pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 13, 43, 5)), surface_tilt=41.0,
... surface_azimuth=180.0, albedo=0.25)
(1065.762189628, 945.265656450, 120.4965331774, 95.3153534421, 25.18117973531)

The first return value is the solar radiation which hits the object, in W/m^2.
The next two are the components of the radiation that comes 1) directly from
the sun and 2) diffusely, after being reflected from some other object. The final
two return values break up the diffuse light into 3) a component reflected only
in the sky and clouds and 4) a component caused by earth's albedo, bounding off
the surface, then the sky, before hitting the object.

Note that if not provided, the temperature and pressure of the ground
are obtained via the :py:class:`~.ATMOSPHERE_NRLMSISE00` class, but this 
quadruples the time required for the calculation.


Compressor sizing
-----------------
Both isothermal and isentropic/polytropic compression models are implemented in
:py:mod:`fluids.compressible`. Isothermal compression calculates the work required to compress a gas from
one pressure to another at a specified temperature. This is the best possible case 
for compression; all actual compressors require more work to do the compression.
By making the compression take a large number of stages and cooling the gas
between stages, this can be approached reasonable closely. Integrally 
geared compressors are often used for this purpose 

The function :py:func:`~.isothermal_work_compression` provides this calculation.

>>> isothermal_work_compression(P1=1E5, P2=1E6, T=300)
5743.4273042447

Work is calculated on a J/mol basis. If the second pressure is lower than the
first, a negative work will result and you are modeling an expander instead
of a compressor. Gas compressibility factor can also be specified. The lower
the gas's compressibility factor, the less power required to compress it.

>>> isothermal_work_compression(P1=1E6, P2=1E5, T=300)
-5743.4273042447
>>> isothermal_work_compression(P1=1E5, P2=1E6, T=300, Z=0.95)
5456.25593903253

There is only one function implemented to model both isentropic and polytropic
compressors, as the only difference is that a polytropic exponent `n` is used
instead of the gas's isentropic exponent Cp/Cv `k` and the type of efficiency
is changed. The model requires initial temperature, inlet and outlet pressure,
isentropic exponent or polytropic exponent, and optionally an efficiency.

Compressing air from 1 bar to 10 bar, with inlet temperature of 300 K and
efficiency of 78% with the :py:func:`~.isentropic_work_compression` function:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78) # work, J/mol
10416.8769863844

The model allows for the inlet or outlet pressure or efficiency to be calculated
instead of the work:

>>> isentropic_work_compression(T1=300, P1=1E5, P2=1E6, k=1.4, W=10416) # Calculate efficiency
0.7800656729435386
>>> isentropic_work_compression(T1=300, P1=1E5, k=1.4, W=10416, eta=0.78) # Calculate P2
999857.9648950758
>>> isentropic_work_compression(T1=300, P2=1E6, k=1.4, W=10416, eta=0.78) # Calculate P1
100014.20552817611

The approximate temperature rise can also be calculated with the function
:py:func:`~.isentropic_T_rise_compression`.

>>> T2 = isentropic_T_rise_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=0.78)
>>> T2, T2-300 # outlet temperature and temperature rise, K
(657.960664955, 357.9606649550)

It is more accurate to use an enthalpy-based model which incorporates departure
functions.

Polytropic exponents and efficiencies are convertible to isentropic exponents and
efficiencies with :py:func:`~.isentropic_efficiency` and 
:py:func:`~.polytropic_exponent` . For the above example, with `k` = 1.4 and `eta_s` = 0.78:

>>> eta_p = isentropic_efficiency(P1=1E5, P2=1E6, k=1.4, eta_s=0.78) # with eta_s specified, returns polytropic efficiency
>>> n = polytropic_exponent(k=1.4, eta_p=eta_p)
>>> eta_p, n
(0.8376785349411107, 1.517631868575738)

With those results, we can prove the calculation worked by calculating the
work required using these polytropic inputs:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=n, eta=eta_p)
10416.87698638448

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
1.5249343832020996

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=n, eta=0.83)
10556.49818012439

Converting polytropic efficiency to isentropic efficiency:

>>> eta_s = isentropic_efficiency(P1=1E5, P2=1E6, k=1.4, eta_p=0.83)
>>> print(eta_s)
0.7696836498942261

Checking the calculated power is the same:

>>> isentropic_work_compression(P1=1E5, P2=1E6, T1=300, k=1.4, eta=eta_s)
10556.49818012439

Gas pipeline sizing
-------------------

The standard isothermal compressible gas flow is fully implemented as   
:py:func:`~.isothermal_gas`, and through
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
0.42971708911

The same case, but determining what the outlet pressure will be if 200 kg/s
flow in the 0.5 m diameter pipe:

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, D=0.5, L=1000, m=200)
784701.068182

Determining pipe length from known diameter, pressure drop, and mass flow
(possible but not necessarily useful):

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, D=0.5, m=150)
937.325802775

Not all specified mass flow rates are possible. At a certain downstream
pressure, choked flow will develop - that downstream pressure is that
at which the mass flow rate reaches a maximum. An exception will be
raised if such an input is specified:

>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, L=1000, D=0.5, m=260) # doctest: +SKIP
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "fluids/compressible.py", line 886, in isothermal_gas
    'kg/s at a downstream pressure of %f' %(P1, m_max, Pcf))
Exception: The desired mass flow rate cannot be achieved with the specified upstream pressure of 1000000.000000 Pa; the maximum flowrate is 257.216733 kg/s at a downstream pressure of 389699.731765
>>> isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=3E5, L=1000, D=0.5) # doctest: +SKIP
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "fluids/compressible.py", line 821, in isothermal_gas
    due to the formation of choked flow at P2=%f, specified outlet pressure was %f' % (Pcf, P2))
Exception: Given outlet pressure is not physically possible due to the formation of choked flow at P2=389699.731765, specified outlet pressure was 300000.000000

The downstream pressure at which choked flow occurs can be calculated directly
as well:

>>> P_isothermal_critical_flow(P=1E6, fd=0.00185, L=1000., D=0.5)
389699.731764

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
 
>>> from thermo import *  # doctest: +SKIP
>>> from fluids import *  # doctest: +SKIP
>>> from math import pi
>>> D = 0.5  # doctest: +SKIP
>>> L = 1000  # doctest: +SKIP
>>> epsilon = 5E-5  # doctest: +SKIP
>>> S1 = Stream('natural gas', P=1E6, m=50)  # doctest: +SKIP
>>> V = S1.Q/(pi/4*D**2)  # doctest: +SKIP
>>> Re = S1.Reynolds(D=D, V=V)  # doctest: +SKIP
>>> fd = friction_factor(Re=Re, eD=epsilon/D)  # doctest: +SKIP
>>> P2 = isothermal_gas(rho=S1.rho, fd=fd, P1=S1.P, D=D, L=L, m=S1.m)  # doctest: +SKIP
>>> P2  # doctest: +SKIP
877420.071063

In the above example, the friction factor was calculated using the density
and velocity of the gas when it enters the stream. However, the average values,
at the middle pressure, and more representative. We can iterate to observe
the effect of using the average values:

>>> for i in range(10):  # doctest: +SKIP
...     S2 = Stream('natural gas', P=0.5*(P2+S1.P), m=50)  # doctest: +SKIP
...     V = S2.Q/(pi/4*D**2)  # doctest: +SKIP
...     Re = S2.Reynolds(D=D, V=V)  # doctest: +SKIP
...     fd = friction_factor(Re=Re, eD=epsilon/D) # doctest: +SKIP
...     P2 = isothermal_gas(rho=S2.rho, fd=fd, P1=S1.P, D=D, L=L, m=S1.m)  # doctest: +SKIP
...     print('%g' %P2)  # doctest: +SKIP
868535
867840
867786
867781
867781
867781
867781
867781
867781
867781

As can be seen, the system converges very quickly. The difference in calculated
pressure drop is approximately 1%. Please note the values given here may change
as properties are updated in the `thermo` library, they are here to demonstrate
the technique only.

Gas pipeline sizing: Empirical equations
----------------------------------------
In addition to the actual model, many common simplifications used in industry
are implemented as well. These are equally capable of solving for any of the
following inputs:

* Mass flow rate
* Upstream pressure
* Downstream pressure
* Diameter of pipe
* Length of pipe

None of these models include an acceleration term. In addition to reducing 
their accuracy, it allows all solutions for the above variables to be analytical.
These models cannot predict the occurrence of choked flow, and model only
turbulent, not laminar, flow. Most of these models do not depend on the gas's
viscosity.

Rather than using mass flow rate, they use specific gravity and volumetric 
flow rate. The volumetric flow rate is specified with respect to a reference
temperature and pressure. The defaults are 288.7 K and 101325 Pa, dating to
the old imperial standard of 60° F. The specific gravity is with respect to 
air at the reference conditions. As the ideal gas law is used in each of 
these models, in addition to pressure and specific gravity the average 
temperature in the pipeline is required. Average compressibility factor is
an accepted input to all models and corrects the ideal gas law's ideality. 

The full list of approximate models is as follows:

* :py:func:`~.Panhandle_A`
* :py:func:`~.Panhandle_B`
* :py:func:`~.Weymouth`
* :py:func:`~.Oliphant`
* :py:func:`~.Fritzsche`
* :py:func:`~.Muller`
* :py:func:`~.IGT`
* :py:func:`~.Spitzglass_high`
* :py:func:`~.Spitzglass_low`

As an example, calculating flow for a pipe with diameter 0.34 m, upstream 
pressure 90 bar and downstream pressure 20 bar, 160 km long, 0.693 specific
gravity and with an average temperature in the pipeline of 277.15 K:

>>> Panhandle_A(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, Tavg=277.15)
42.56082051195

Each model also includes a pipeline efficiency term, ranging from 0 to 1. These
are just empirical correction factors, Some of the models were developed with 
theory and a correction factor applied always; others are more empirical, and
have a default correction factor. 0.92 is the default for the Panhandle A/B,
Weymouth, and Oliphant models; the rest default to a correction of 1 i.e. no
correction at all.

The Muller and IGT models are the most accurate and recent approximations.
They both depend on viscosity.

>>> Muller(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, mu=1E-5, Tavg=277.15)
60.4579669814
>>> IGT(D=0.340, P1=90E5, P2=20E5, L=160E3, SG=0.693, mu=1E-5, Tavg=277.15)
48.9235178678

These empirical models are included because they are mandated in many industrial
applications regardless of their accuracy, and correction factors have already 
been determined.

A great deal of effort was spent converting these models to base SI units
and checking the coefficients used in each model with multiple sources. 
In many cases multiple sets of coefficients are available for a model;
the most authoritative or common ones were used in those cases.



Drag and terminal velocity
--------------------------
A number of spherical particle drag correlations are implemented.

In the simplest case, consider a spherical particle of diameter D=1 mm,
density=3400 kg/m^3, travelling at 30 m/s in air with viscosity mu=1E-5 Pa*s
and density 1.2 kg/m^3.

We calculate the particle Reynolds number:

>>> Re = Reynolds(V=30, rho=1.2, mu=1E-5, D=1E-3)
>>> Re
3599.9999999999995

The drag coefficient `Cd` can be calculated with no other parameters
from :py:func:`~.drag_sphere`:

>>> drag_sphere(Re)
0.3914804681941151

The terminal velocity of the particle is easily calculated with the 
:py:func:`~.v_terminal` function. 

>>> v_terminal(D=1E-3, rhop=3400, rho=1.2, mu=1E-5)
8.971223953182

Very often, we are not interested in just what the velocity of the particle will
be at terminal conditions, but on the distance it will travel and the particle will
never have time to reach terminal conditions. An integrating function is available 
to do that. Consider that same particle being shot directly down from a helicopter
100 m high. 

The integrating function, :py:func:`~.integrate_drag_sphere`, performs the integral with respect
to time. At one second, we can see the (velocity, distance travelled):

>>> integrate_drag_sphere(D=1E-3, rhop=3400., rho=1.2, mu=1E-5, t=1, V=30, distance=True)
(10.561878111, 15.607904177)

After integrating to 10 seconds, we can see the particle has travelled 97 meters and is
almost on the ground. 

>>> integrate_drag_sphere(D=1E-3, rhop=3400., rho=1.2, mu=1E-5, t=10, V=30, distance=True)
(8.97122398706, 97.13276290361)

For this example simply using the terminal velocity would have given an accurate estimation
of distance travelled:

>>> 8.971223953182939*10
89.7122395318294

Many engineering applications such as direct contact condensers do operate far from terminal
velocity however, and this function is useful there.

Pressure drop through packed beds
---------------------------------

Twelve different packed bed pressure drop correlations are available. A meta
function which allows any of them to be selected and automatically selects
the most accurate correlation for the given parameters.

Pressure drop through a packed bed depends on the density, viscosity and  
velocity of the fluid, as well as the diameter of the particles, the amount
of free space in the bed (voidage), and to a lesser amount the ratio of 
particle to tube diameter and the shape of the particles. 

Consider 0.8 mm pebbles with 40% empty space with water flowing through a 2 m  
column creeping flow at a superficial velocity of 1 mm/s. We can calculate the 
pressure drop in Pascals using the :py:func:`~.dP_packed_bed` function:

>>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=2)
2876.56539176

The method can be specified manually as well, for example the commonly used Ergun equation:

>>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=2, Method='Ergun')
2677.73437

Incorporation of the tube diameter will add wall effects to the model.

>>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=2, Dt=0.01)
2510.325132

Models can be used directly as well. The length of the column is an optional
input; if not provided, the result will be in terms of Pa/m.

>>> KTA(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3) # A correlation standardized for use in pebble reactors
1440.40927

If the column diameter was 0.5 m, the flow rate in m^3/s would be:

>>> .001*(pi/4*0.5**2) # superficial_velocity*A_column
0.0001963495408

The holdup (total volume of the column holding fluid not particles) would be:

>>> (pi/4*0.5**2)*(2)*0.4 # A_column*H_column*voidage
0.15707963267948


Not all particles are spherical. There have been correlations published for 
specific shapes, but what is often performed is simply an adjustment of particle
diameter by its sphericity in the correlation, with the effective `dp` used
as the product of the actual `dp` and the sphericity of the particle. The less
spherical the particles, the higher the pressure drop. This is supported in 
all of the correlations.

>>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=2, Dt=0.01, sphericity=0.9)
3050.419598116

While it is easy to measure the volume of particles added to a given column 
and determine the voidage experimentally, this does not help in the design process.
Several authors have methodically filled columns with particles of different sizes and
created correlations in terms of sphericity and particle to tube diameter ratios.
Three such correlations are implemented in fluids, one generally using sphericity,
one for spheres, and one for cylinders (:py:func:`~.voidage_Benyahia_Oneil`,
:py:func:`~.voidage_Benyahia_Oneil_spherical` and
:py:func:`~.voidage_Benyahia_Oneil_cylindrical` respectively).

1 mm spheres in a 5 cm diameter tube:

>>> voidage_Benyahia_Oneil_spherical(Dp=.001, Dt=.05)
0.390665315744

1 mm diameter cylinder 5 mm long in a 5 cm diameter tube:

>>> V_cyl = V_cylinder(D=0.001, L=0.005)
>>> D_sphere_eq = (6*V_cyl/pi)**(1/3.)
>>> A_cyl = A_cylinder(D=0.001, L=0.005)
>>> sph = sphericity(A=A_cyl, V=V_cyl)
>>> voidage_Benyahia_Oneil_cylindrical(Dpe=D_sphere_eq, Dt=0.05, sphericity=sph)
0.3754895273247

Same calculation, but using the general correlation for all shapes:

>>> voidage_Benyahia_Oneil(Dpe=D_sphere_eq, Dt=0.05, sphericity=sph)
0.4425769555048

Pressure drop through piping
----------------------------
It is straightforward to calculate the pressure drop of fluid flowing in a 
pipeline with any number of fittings using the fluids library's 
:py:mod:`fluids.fittings` submodule.

15 m of piping, with a sharp entrance and sharp exit, two 30 degree miter 
bends, one rounded bend 45 degrees, 1 sharp contraction to half the pipe
diameter and 1 sharp expansion back to the normal pipe diameter (water,
V=3 m/s, Di=0.05, roughness 0.01 mm):

>>> Re = Reynolds(V=3, D=0.05, rho=1000, mu=1E-3)
>>> fd = friction_factor(Re, eD=1E-5/0.05)
>>> K = K_from_f(fd=fd, L=15, D=0.05)
>>> K += entrance_sharp()
>>> K += exit_normal()
>>> K += 2*bend_miter(angle=30)
>>> K += bend_rounded(Di=0.05, angle=45, fd=fd)
>>> K += contraction_sharp(Di1=0.05, Di2=0.025)
>>> K += diffuser_sharp(Di1=0.025, Di2=0.05)
>>> dP_from_K(K, rho=1000, V=3)
37920.5114014

If the diameter of the piping varies, not all of the loss coefficients will be
with respect to the same diameter. Each loss coefficient must be converted to
one standard diameter before the total pressure drop can be calculated. The
following example is solved with the optional `pint` unit compatibility module.

40 m piping, beveled entrance (10 mm length, 30 degrees, into 5 cm ID pipe)
, then a 30 degree miter bend, a sharp contraction to half the pipe diameter (5 m long), 
a 30 degree miter bend, a rounded 45 degree bend, a sharp expansion to 4 cm ID
pipe (15 more meters), and a sharp exit:

>>> from fluids.units import *
>>> from math import *
>>> material = nearest_material_roughness('steel', clean=True)
>>> epsilon = material_roughness(material)
>>> Q = .01*u.m**3/u.s
>>> rho = 1000*u.kg/u.m**3
>>> mu = 1E-4*u.Pa*u.s
>>> D1 = 5*u.cm
>>> D2 = 2.5*u.cm
>>> D3 = 4*u.cm
>>> L1 = 20*u.m
>>> L2 = 5*u.m
>>> L3 = 15*u.m
>>> V1 = Q/(pi/4*D1**2)

>>> Re = Reynolds(V=V1, D=D1, rho=rho, mu=mu)
>>> fd = friction_factor(Re, eD=epsilon/D1)
>>> K = K_from_f(fd=fd, L=L1, D=D1)
>>> K += entrance_beveled(Di=D1, l=10*u.mm, angle=30*u.degrees)
>>> K += bend_miter(angle=30*u.degrees)
>>> K += contraction_sharp(Di1=D1, Di2=D2)

>>> V2 = Q/(pi/4*D2**2)
>>> Re2 = Reynolds(V=V2, D=D2, rho=rho, mu=mu)
>>> fd2 = friction_factor(Re2, eD=epsilon/D2)

>>> K += change_K_basis(K_from_f(fd=fd2, L=L2, D=D2), D1=D2, D2=D1)
>>> K += change_K_basis(K1=bend_miter(angle=30*u.degrees), D1=D2, D2=D1)
>>> K += change_K_basis(K1=bend_rounded(Di=D2, angle=45*u.degrees, fd=fd2), D1=D2, D2=D1)

>>> V3 = Q/(pi/4*D3**2)
>>> Re3 = Reynolds(V=V3, D=D3, rho=rho, mu=mu)
>>> fd3 = friction_factor(Re3, eD=epsilon/D3)

>>> K += change_K_basis(K_from_f(fd=fd3, L=L3, D=D3), D1=D3, D2=D1)
>>> K += diffuser_sharp(Di1=D2, Di2=D3)

>>> dP_from_K(K, rho=rho, V=V1)
<Quantity(734959.105, 'pascal')>



Control valve sizing: Introduction
----------------------------------
The now internationally-standardized methods (IEC 60534) for sizing liquid and 
gas valves have been implemented. Conversion factors among the different types
of valve coefficients are implemented as well.

There are two forms of loss coefficient used for vales, an imperial and a metric
variable called "valve flow coefficient". Both can be converted to the standard
dimensionless loss coefficient.

If one knows the actual loss coefficient of a valve, the valve flow coefficient
can be calculated in either metric or imperial forms as follows. The flow
coefficients are specific to the diameter of the valve. Kv, Cv, and K values
can be converted easily with the functions :py:func:`~.K_to_Kv`,
:py:func:`~.K_to_Cv`, :py:func:`~.Cv_to_K`, :py:func:`~.Kv_to_K`,
:py:func:`~.Cv_to_Kv`, and :py:func:`~.Kv_to_Cv`.

>>> from fluids import *
>>> K_to_Kv(K=16, D=0.016)
2.56
>>> K_to_Cv(K=16, D=0.016)
2.9596140245853

If Kv or Cv are known, they can be converted to each other with the
proportionality constant 1.156, which is derived from a unit conversion only.
This conversion does not require valve diameter.

>>> Cv_to_Kv(12)
10.379731865307
>>> Kv_to_Cv(10.37)
11.988748998027

If a Cv or Kv is obtained from a valve datasheet, it can be converted into a
standard loss coefficient as follows.

>>> Kv_to_K(Kv=2.56, D=0.016)
16.000000000000004
>>> Cv_to_K(Cv=3, D=0.016)
15.57211586581753

For a valve with a specified Kv and pressure drop, the flow rate can be calculated
easily for the case of non-choked non-compressible flow (neglecting other friction 
losses), as illustrated in the example below for a 5 cm valve with a pressure drop
370 kPa and density of 870 kg/m^3:

>>> Kv = 72.5
>>> D = 0.05 
>>> dP = 370E3
>>> K = Kv_to_K(D=D, Kv=Kv)
>>> rho = 870
>>> V = (dP/(.5*rho*K))**0.5 # dP = K*0.5*rho*V^2
>>> A = pi/4*D**2
>>> Q = V*A
>>> Q
0.04151682468778643

Alternatively, the required Kv can be calculated from an assumed diameter and allowable
pressure drop:

>>> Q = .05
>>> D = 0.05 
>>> dP = 370E3
>>> rho = 870
>>> A = pi/4*D**2
>>> V = Q/A
>>> K = dP/(.5*rho*V**2)
>>> K_to_Kv(D=D, K=K)
87.31399925838778

The approach documented above is not an adequate procedure for sizing valves
however because choked flow, compressible flow, the effect of inlet and outlet
reducers, the effect of viscosity and the effect of laminar/turbulent flow all
have large influences on the performance of control valves. 

Historically, valve manufacturers had their own standards for sizing valves, 
but these have been standardized today into the IEC 60534 methods. 

Control valve sizing: Liquid flow
---------------------------------
To rigorously size a control valve for liquid flow, the inlet pressure, 
allowable pressure drop, and desired flow rate must first be known. 
These need to be determined taking into account the entire pipe network
and the various operating conditions it needs to support; sizing the valves
can be performed afterward and only if no valve with the desired performance
is available does the network need to be redesigned. 

To illustrate sizing a valve, we borrow an example from Emerson's
Control Valve Handbook, 4th edition (2005). It involves a flow of 800 gpm of
liquid propane. The inlet and outlet pipe size is 8 inches, but the size of the 
valve itself is unknown. The desired pressure drop is 25 psi. 

Converting this problem to SI units and using the thermo library to calculate
the necessary properties of the fluid, we calculate the necessary Kv of the 
valve based on an assumed valve size of 3 inches:

>>> from scipy.constants import *
>>> from fluids.control_valve import size_control_valve_l  # doctest: +SKIP
>>> from thermo.chemical import Chemical # doctest: +SKIP
>>> P1 = 300*psi + psi # to Pa # doctest: +SKIP
>>> P2 = 275*psi + psi # to Pa # doctest: +SKIP
>>> T = 273.15 + 21 # to K # doctest: +SKIP
>>> propane = Chemical('propane', P=(P1+P2)/2, T=T) # doctest: +SKIP
>>> rho = propane.rho # doctest: +SKIP
>>> Psat = propane.Psat # doctest: +SKIP
>>> Pc = propane.Pc # doctest: +SKIP
>>> mu = propane.mu # doctest: +SKIP
>>> Q = 800*gallon/minute # to m^3/s # doctest: +SKIP
>>> D1 = D2 = 8*inch # to m # doctest: +SKIP
>>> d = 3*inch # to m # doctest: +SKIP

The standard specifies two more parameters specific to a valve:

* FL, Liquid pressure recovery factor of a control valve without attached fittings
* Fd, Valve style modifier

Both of these are factors between 0 and 1. In the Emerson handbook, they are 
not considered in the sizing procedure and set to 1. These factors are also
a function of the diameter of the valve and are normally tabulated next to the
values of Cv or Kv for a valve. Now using :py:func:`~.size_control_valve_l` 
to solve for the flow coefficient:

>>> Kv = size_control_valve_l(rho, Psat, Pc, mu, P1, P2, Q, D1, D2, d, FL=1, Fd=1)  # doctest: +SKIP
>>> Kv # doctest: +SKIP
109.39701927957

The handbook states the Cv of the valve is 121; we convert Kv to Cv:

>>> Kv_to_Cv(Kv=Kv) # doctest: +SKIP
126.4738095733

The example in the book calculated Cv = 125.7, but doesn't actually use the 
full calculation method. Either way, the valve will not carry the desired flow 
rate; we need to try a larger valve size. The 4 inch size is tried next in the 
example, which has a known Cv of 203.

>>> d = 4*inch # to m
>>> Kv = size_control_valve_l(rho, Psat, Pc, mu, P1, P2, Q, D1, D2, d, FL=1, Fd=1)  # doctest: +SKIP
>>> Kv_to_Cv(Kv=Kv)  # doctest: +SKIP
116.17550388277834

The calculated Cv is well under the valve's maximum Cv; we can select it.

This model requires a vapor pressure and a critical pressure of the fluid as
inputs. There is no clarification in the standard about how to handle mixtures,
which do not have these values. It is reasonable
to calculate vapor pressure as the bubble pressure, and the mixture's critical
pressure through a mole-weighted average.

For actual values of Cv, Fl, Fd, and available diameters, an excellent resource
is the `Fisher Catalog 12 <http://www.documentation.emersonprocess.com/groups/public/documents/catalog/cat12_s1.pdf>`_.

Control valve sizing: Gas flow
------------------------------
To rigorously size a control valve for gas flow, the inlet pressure, 
allowable pressure drop, and desired flow rate must first be known. 
These need to be determined taking into account the entire pipe network
and the various operating conditions it needs to support; sizing the valves
can be performed afterward and only if no valve with the desired performance
is available does the network need to be redesigned. 

To illustrate sizing a valve, we borrow an example from Emerson's
Control Valve Handbook, 4th edition (2005). It involves a flow of 6 million ft^3/hour
of natural gas. The inlet and outlet pipe size is 8 inches, but the size of the 
valve itself is unknown. The desired pressure drop is 150 psi. 

Converting this problem to SI units and using the thermo library to calculate
the necessary properties of the fluid, we calculate the necessary Kv of the 
valve based on an assumed valve size of 8 inches.

>>> from scipy.constants import *
>>> from fluids.control_valve import size_control_valve_g
>>> from thermo.chemical import Chemical  # doctest: +SKIP
>>> P1 = 214.7*psi
>>> P2 = 64.7*psi
>>> T = 16 + 273.15
>>> natural_gas = Mixture('natural gas', T=T, P=(P1+P2)/2)  # doctest: +SKIP
>>> Z = natural_gas.Z  # doctest: +SKIP
>>> MW = natural_gas.MW  # doctest: +SKIP
>>> mu = natural_gas.mu  # doctest: +SKIP
>>> gamma = natural_gas.isentropic_exponent  # doctest: +SKIP
>>> Q = 6E6*foot**3/hour  # doctest: +SKIP
>>> D1 = D2 = d = 8*inch #  8-inch Fisher Design V250   # doctest: +SKIP

The standard specifies three more parameters specific to a valve:

* FL, Liquid pressure recovery factor of a control valve without attached fittings
* Fd, Valve style modifier
* xT, Pressure difference ratio factor of a valve without fittings at choked flow

All three of these are factors between 0 and 1. In the Emerson handbook, FL and Fd are 
not considered in the sizing procedure and set to 1. xT is specified as 0.137
at full opening. These factors are also a function of the diameter of the 
valve and are normally tabulated next to the values of Cv or Kv for a valve.
Now using :py:func:`~.size_control_valve_g` to solve for the flow coefficient:

>>> Kv = size_control_valve_g(T, MW, mu, gamma, Z, P1, P2, Q, D1, D2, d, FL=1, Fd=1, xT=.137)  # doctest: +SKIP
>>> Kv_to_Cv(Kv)  # doctest: +SKIP
1563.447724002566

The 8-inch valve is rated with Cv = 2190. The valve is adequate to provide 
the desired flow because the rated Cv is higher. The calculated value in their
example is 1515, differing slightly due to the properties used. 

The example next goes on to determine the actual opening position the valve
should be set at to provide the required flow. Their conclusion is approximately
75% open; we can do better using a numerical solver. The values of opening at
different positions are obtained in this example from the valve's 
`datasheet <http://www.emerson.com/documents/automation/141362.pdf>`_.

Loading the data and creating interpolation functions so FL, Fd, and xT 
are all smooth functions:

>>> from scipy.interpolate import interp1d
>>> from scipy.optimize import newton
>>> openings = [.2, .3, .4, .5, .6, .7, .8, .9]
>>> Fds = [0.59, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99, 0.99]
>>> Fls = [0.9, 0.9, 0.9, 0.85, 0.78, 0.68, 0.57, 0.45]
>>> xTs = [0.92, 0.81, 0.85, 0.63, 0.58, 0.48, 0.29, 0.14]
>>> Kvs = [24.1, 79.4, 153, 266, 413, 623, 1060, 1890]
>>> Fd_interp = interp1d(openings, Fds, kind='cubic')
>>> Fl_interp = interp1d(openings, Fls, kind='cubic')
>>> xT_interp = interp1d(openings, xTs, kind='cubic')
>>> Kv_interp = interp1d(openings, Kvs, kind='cubic')

Creating and solving the objective function:

>>> def to_solve(opening):
...     Fd = float(Fd_interp(opening))
...     Fl = float(Fl_interp(opening))
...     xT = float(xT_interp(opening))
...     Kv_lookup = float(Kv_interp(opening))
...     Kv_calc = size_control_valve_g(T, MW, mu, gamma, Z, P1, P2, Q, D1, D2, d, FL=Fl, Fd=Fd, xT=xT)
...     return Kv_calc - Kv_lookup 
>>> newton(to_solve, .8) # initial guess of 80%  # doctest: +SKIP
0.7500814714947652

We see the valve should indeed be set to almost exactly 75% open to provide 
the desired flow. 

Electric motor sizing
---------------------
Motors are available in standard sizes, mostly as designated by the
National Electrical Manufacturers Association (NEMA). To easily determine what
the power of a motor will actually be once purchased, 
:py:func:`~.motor_round_size` implements
rounding up of a motor power to the nearest size. NEMA standard motors are
specified in terms of horsepower.

>>> motor_round_size(1E5) # 100 kW motor; 11.8% larger than desired
111854.98073734052
>>> from scipy.constants import hp
>>> motor_round_size(1E5)/hp # convert to hp
150.0

Motors are designed to generate a certain amount of power, but they themselves are 
not 100% efficient at doing this and require more power due to efficiency losses.
Many minimum values for motor efficiency are standardized. The Canadian standard
for this is implemented in fluids as :py:func:`~.CSA_motor_efficiency`.

>>> CSA_motor_efficiency(P=5*hp)
0.855

Most motors are not enclosed (the default assumption), but those that are closed
are more efficient. 

>>> CSA_motor_efficiency(P=5*hp, closed=True)
0.875

The number of poles in a motor also affects its efficiency:

>>> CSA_motor_efficiency(P=5*hp, poles=6)
0.875

There is also a schedule of higher efficiency values standardized as well,
normally available at somewhat higher cost:

>>> CSA_motor_efficiency(P=5*hp, closed=True, poles=6, high_efficiency=True)
0.895

A motor will spin at more or less its design frequency, depending on its type.
However, if it does not meet sufficient resistance, it will not be using its
design power. This is good and bad - less power is used, but as a motor 
drops under 50% of its design power, its efficiency becomes terrible. The function
:py:func:`~.motor_efficiency_underloaded`
has been written based on generic performance curves to estimate the underloaded
efficiency of a motor. Just how bad efficiency drops off depends on the design
power of a motor - higher power motors do better operating at low loads than 
small motors.

>>> motor_efficiency_underloaded(P=1E3, load=.9)
1
>>> motor_efficiency_underloaded(P=1E3, load=.2)
0.6639347559654663

This needs to be applied on top of the normal motor efficiency; for example,
that 1 kW motor at 20% load would have a net efficiency of:

>>> motor_efficiency_underloaded(P=1E3, load=.2)*CSA_motor_efficiency(P=1E3)
0.5329404286134798


Many motors have Variable Frequency Drives (VFDs) which allow them to vary the
speed of their rotation. The VFD is another source of inefficiency, but by allowing
the pump or other piece of equipment to vary its speed, a system may be designed to
be less energy intensive. For example, rather than running a pump at a certain
high frequency and controlling the flow with a large control valve, the flow 
rate can be controlled with the VFD directly.

The efficiency of a VFD depends on the maximum power it needs to be able to
generate, and the power it is actually generating at an instant (load).
A table of typical modern VFD efficiencies is implemented in fluids as
:py:func:`~.VFD_efficiency`.

>>> VFD_efficiency(1E5) # 100 kW
0.97
>>> VFD_efficiency(5E3, load=.2) # 5 kW, 20% load
0.8562


Particle Size Distributions
---------------------------
Fluids has means for calculating, fitting, and manipulating particle size 
distributions through the :py:mod:`fluids.particle_size_distribution`
module. In addition to discrete and continuous distributions, there are
also means to create interpolating distributions from discrete 
distributions, and to use any of SciPy's statistical distributions or a
custom distribution for calculations.

The following example particle size distribution shows some calculations. Note there
is one more diameter point than number point - this is how the input should be given
when the analysis provides classes and each bin has a range of sizes representing it.
Also supported is providing as many diameter values as fraction values.

>>> ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
>>> numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
>>> psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)
>>> psd
<Particle Size Distribution, points=14, D[3, 3]=2450.886241 m>

In the above example, the analysis available was the number of particles counted
in each bin. This is equivalent to having normalized the numbers into fractions;
they are normalized inside the :py:class:`~.ParticleSizeDistribution` class.

If masses in each of the different bins had been known instead, then the same
constructor would be given except with `order=3`, representing a mass or volume
distribution (they are the same thing for distributions with the same density for
all particles).

If the data is available as a cumulative distribution, simple add the flag cdf=True
and it will be interpreted correctly.

The probability distribution and cumulative distribution can be plotted with 
:py:meth:`~.plot_pdf` and :py:meth:`~.plot_cdf` respectively.

Important statistical parameters describing the distribution can be calculated
with the methods :py:meth:`fluids.particle_size_distribution.ParticleSizeDistribution.mean_size`
or :py:meth:`fluids.particle_size_distribution.ParticleSizeDistribution.mean_size_ISO`.
The following example shows calculation of the size-weighted mean diameter; 
arithmetic mean diameter; Sauter mean diameter; and De Brouckere diameter.

>>> psd.mean_size(2, 1)
1857.788857205
>>> psd.mean_size(1, 0)
1459.372565067
>>> psd.mean_size(1, 2)
1857.788857205
>>> psd.mean_size(1, 3)
2053.27039773

An interpolated distribution exists underneath the discrete data to allow useful 
properties to be calculated, such as the D10 or D90:

>>> psd.dn(0.1), psd.dn(0.9)
(1437.07139276, 3911.47963636)

Or probability density functions:

>>> psd.pdf(1000)
0.000106323843275
>>> psd.cdf(5000)
0.98974007348

Statistical distributions implemented are :py:class:`~.PSDLognormal`,
:py:class:`~.PSDGatesGaudinSchuhman`, and :py:class:`~.PSDRosinRammler`.
Discrete and continuous distributions share most methods.

>>> psd = PSDLognormal(s=0.5, d_characteristic=5E-6)
>>> psd.pdf(1e-6) # probability density function
4487.8921553
>>> psd.cdf(7e-6) # cumulative distribution function
0.7495086913
>>> psd.dn(0.1) # At what diameter is this fraction of particles smaller than?
2.63441759148e-06
>>> psd.mean_size(3, 2)
4.4124845129e-06
>>> ds = psd.ds_discrete(pts=1000) # Compare calculations with the discrete distribution
>>> fractions = psd.fractions_discrete(ds)
>>> ParticleSizeDistribution(ds=ds, fractions=fractions, order=3).mean_size(3, 2)
4.4257436305e-06

It is straightforward to calculate descriptions of the distribution using the
available routines:

Volume specific surface area:

>>> psd.vssa
1359778.14368

Span (D90 - D10):

>>> psd.dn(.9) - psd.dn(0.1)
6.8553459451e-06

Relative span (D90 - D10)/D50:

>>> (psd.dn(.9) - psd.dn(0.1))/psd.dn(0.5)
1.37106918903

Percentile ratios, D75/D25 and D90/D10:

>>> psd.dn(0.75)/psd.dn(0.25)
1.96303108415
>>> psd.dn(0.9)/psd.dn(0.1)
3.60222447927

Required Resources
------------------

The fluids library is designed to be a low-overhead, lightweight repository
of engineering knowledge and utilities that relate to fluid dynamics.
It occupies ~4 MB of RAM on load and should load in a small fraction of a
second. Fluids does load NumPy if it is present, which takes ~150 ms; fluids
itself loads in approximately 20 ms. No other libraries will become required 
dependencies; anything else, including SciPy, is optional and loaded when
needed.

Fluids was originally tightly integrated with SciPy and NumPy; today they
are optional components used for only a small amount of functionality
which do not have pure-Python numerical methods implemented.
Fluids targets Python 2.7 and up as well as PyPy2 and PyPy3. Additionally,
fluids has been tested by the author to load in IronPython, Jython,
and micropython.
