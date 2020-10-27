# -*- coding: utf-8 -*-
"""Vendorized, partial version of scipy.constants which does not implement the
full codata formulations.

This was implemented to provide a consistent set of constants across scipy
versions; and to prevent the tests from failing when new CODATA formulations
come out.
"""

import math as _math

# mathematical constants
pi = _math.pi
pi_inv = 1.0/pi
golden = golden_ratio = 1.618033988749895

# SI prefixes
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21

# binary prefixes
kibi = 2**10
mebi = 2**20
gibi = 2**30
tebi = 2**40
pebi = 2**50
exbi = 2**60
zebi = 2**70
yobi = 2**80

# physical constants
c = speed_of_light = 299792458.0
mu_0 = 4e-7*pi
epsilon_0 = 1.0 / (mu_0*c*c)
h = Planck = 6.62607004e-34
hbar = h / (2.0 * pi)
G = gravitational_constant = 6.67408e-11
g = 9.80665
g_sqrt = 3.1315571206669692#_math.sqrt(g)
e = elementary_charge = 1.6021766208e-19
alpha = fine_structure = 0.0072973525664
N_A = Avogadro = 6.022140857e+23
k = Boltzmann = 1.38064852e-23
sigma = Stefan_Boltzmann = 5.670367e-08
Wien = 0.0028977729
Rydberg = 10973731.568508

k = 1.380649e-23
N_A = 6.02214076e23
R = gas_constant = N_A*k #8.3144598 # N_A*k
R_inv = 1.0/R
R2 = R*R

# mass in kg
gram = 1e-3
metric_ton = 1e3
grain = 64.79891e-6
lb = pound = 7000 * grain  # avoirdupois
blob = slinch = pound * g / 0.0254  # lbf*s**2/in (added in 1.0.0)
slug = blob / 12  # lbf*s**2/foot (added in 1.0.0)
oz = ounce = pound / 16.0
stone = 14.0 * pound
long_ton = 2240.0 * pound
short_ton = 2000.0 * pound

troy_ounce = 480.0 * grain  # only for metals / gems
troy_pound = 12.0 * troy_ounce
carat = 200e-6

m_e = electron_mass = 9.10938356e-31
m_p = proton_mass = 1.672621898e-27
m_n = neutron_mass = 1.674927471e-27
m_u = u = atomic_mass = 1.66053904e-27

# angle in rad
degree = pi / 180.0
arcmin = arcminute = degree / 60.0
arcsec = arcsecond = arcmin / 60.0

# time in second
minute = 60.0
hour = 60.0 * minute
day = 24.0 * hour
week = 7.0 * day
year = 365.0 * day
Julian_year = 365.25 * day

# length in meter
inch = 0.0254
inch_inv = 1.0/inch
foot = 12 * inch
yard = 3 * foot
mile = 1760 * yard
mil = 0.001*inch 
pt = point = inch / 72  # typography
survey_foot = 1200.0 / 3937
survey_mile = 5280.0 * survey_foot
nautical_mile = 1852.0
fermi = 1e-15
angstrom = 1e-10
micron = 1e-6
au = astronomical_unit = 149597870691.0
light_year = Julian_year * c
parsec = au / arcsec

# pressure in pascal
atm = atmosphere = 101325.0
bar = 1e5
torr = mmHg = atm / 760
psi = pound * g / (inch * inch)

atm_inv = atmosphere_inv = 1.0/atm
torr_inv = mmHg_inv = 1.0/torr
psi_inv = 1.0/psi

# area in meter**2
hectare = 1e4
acre = 43560 * foot*foot

# volume in meter**3
litre = liter = 1e-3
gallon = gallon_US = 231.0 * inch*inch*inch  # US
# pint = gallon_US / 8
fluid_ounce = fluid_ounce_US = gallon_US / 128
bbl = barrel = 42.0 * gallon_US  # for oil

gallon_imp = 4.54609e-3  # UK
fluid_ounce_imp = gallon_imp / 160.0

# speed in meter per second
kmh = 1e3 / hour
mph = mile / hour
mach = speed_of_sound = 340.5  # approx value at 15 degrees in 1 atm. is this a common value?
knot = nautical_mile / hour

# temperature in kelvin
zero_Celsius = 273.15
degree_Fahrenheit = 1.0/1.8  # only for differences

# energy in joule
eV = electron_volt = elementary_charge  # * 1 Volt
calorie = calorie_th = 4.184
calorie_IT = 4.1868
erg = 1e-7
Btu_th = pound * degree_Fahrenheit * calorie_th / gram
Btu = Btu_IT = pound * degree_Fahrenheit * calorie_IT / gram
ton_TNT = 1e9 * calorie_th
# Wh = watt_hour

# power in watt
hp = horsepower = 550.0 * foot * pound * g

# force in newton
dyn = dyne = 1e-5
lbf = pound_force = pound * g
kgf = kilogram_force = g # * 1 kg


deg2rad = 0.017453292519943295769 # Multiple an angle in degrees by this to get radians
rad2deg = 57.295779513082320877# Multiple an angle in radians by this to get degrees
