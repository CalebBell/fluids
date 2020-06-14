Support for Numba (fluids.numba)
================================

Basic module which wraps most of fluids functions and classes to be compatible with the
`Numba <https://github.com/numba/numba>`_ dynamic Python compiler.
Numba is only supported on Python 3, and may require the latest version of Numba.
Numba is rapidly evolving, and hopefully in the future it will support more of
the functionality of fluids.

Using the numba-accelerated version of `fluids` is easy; simply call functions
and classes from the fluids.numba namespace. The fluids.numba module must be
imported separately; it is not loaded automatically as part of fluids.

>>> import fluids
>>> import fluids.numba
>>> fluids.numba.bend_rounded(Di=4.020, rc=4.0*5, angle=30, Re=1E5)
0.11519070808085191

There is a delay while the code is compiled when using Numba;
the speed is not quite free.

It is easy to compare the speed of a function with and without Numba.

>>> %timeit fluids.numba.Stichlmair_flood(Vl=5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
15.9 µs ± 266 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
>>> %timeit fluids.Stichlmair_flood(Vl=5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)
109 µs ± 2.01 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

Not everything is faster in the numba interface. For example, dictionary
lookups have to be turned into slower jump lists:

>>> %timeit fluids.numba.Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
7.04 µs ± 62.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
>>> %timeit fluids.Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
435 ns ± 9.01 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

Functions which take strings as inputs are also known to normally get slower:

>>> %timeit fluids.numba.geometry.V_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)
11.2 µs ± 457 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
>>> %timeit fluids.geometry.V_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)
1.64 µs ± 25.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Nevertheless, using the function from the numba interface may be preferably,
to allow an even larger program to be completely compiled in njit mode.


Today, the list of things known not to work is as follows:

- :py:func:`~.integrate_drag_sphere` (uses SciPys's odeint)
- The geometry class TANK and HelicalCoil, PlateExchanger, RectangularFinExchanger, HyperbolicCoolingTower, AirCooledExchanger in :py:mod:`fluids.geometry`, as well as several methods performing double numerical quadrature
    - :py:func:`~.SA_partial_horiz_ellipsoidal_head`
    - :py:func:`~.SA_partial_horiz_guppy_head`
    - :py:func:`~.SA_partial_horiz_torispherical_head`
- Everything in :py:mod:`fluids.particle_size_distribution`
- Everything in :py:mod:`fluids.atmosphere`
- Everything in :py:mod:`fluids.piping` (uses global lookups)
- In :py:mod:`fluids.friction`, only :py:func:`~.nearest_material_roughness`,  :py:func:`~.material_roughness`,  :py:func:`~.roughness_Farshad` are unsupported as they use global lookups


Numpy Support
-------------
Numba also allows fluids to provide any of its supported functions as a numpy universal
function. Numpy's wonderful broadcasting is implemented, so some arguments can
be arrays and some can not.

>>> import fluids.numba_vectorized
>>> import numpy as np
>>> fluids.numba_vectorized.Moody(np.linspace(1e3, 1e4, 5), 1e-4)
array([0.06053664, 0.04271113, 0.03677223, 0.03343543, 0.03119781])
>>> fluids.numba_vectorized.Moody(np.linspace(1e3, 1e4, 5), np.linspace(1e-4, 1e-5, 5))
array([0.06053664, 0.0426931 , 0.03672111, 0.03333917, 0.03104575])

Unfortunately, keyword-arguments are not supported by Numba.

>>> fluids.numba_vectorized.Moody(Re=np.linspace(1e3, 1e4, 5), eD=np.linspace(1e-4, 1e-5, 5)) 
ValueError: invalid number of arguments

Also default arguments are not presently supported by Numba.

>>> fluids.numba_vectorized.V_horiz_conical(108., 156., 42., np.linspace(0, 4, 4), False)
array([    0.        ,  3333.2359001 ,  9441.84364485, 17370.09634651])
>>> fluids.numba_vectorized.V_horiz_conical(108., 156., 42., np.linspace(0, 4, 4))
ValueError: invalid number of arguments

Despite these limitations is is here that Numba really shines! Arrays are Numba's
strength.

>>> Res = np.linspace(1e4, 1e7, 10000)
>>> %timeit fluids.numba_vectorized.Clamond(Res, 1E-4, False)
797 µs ± 19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

About 80 nanoseconds per friction factor call! As compared to the `fluids.numba`
interface (442 ns) and the normal interface (1440 ns):

>>> %timeit fluids.numba.Clamond(1e4, 1E-4, False)
442 ns ± 7.36 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
>>> %timeit fluids.Clamond(1e4, 1E-4, False)
1.44 µs ± 40.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

Please note this interface is provided, but what works and what doesn't is
mostly up to the numba project. This backend is not quite as polished as
their normal engine.
