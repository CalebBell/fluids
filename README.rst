======
fluids
======

.. image:: http://img.shields.io/pypi/v/fluids.svg?style=flat
   :target: https://pypi.python.org/pypi/fluids
   :alt: Version_status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://fluids.readthedocs.io/
   :alt: Documentation
.. image:: https://github.com/CalebBell/fluids/workflows/Build/badge.svg
   :target: https://github.com/CalebBell/fluids/actions
   :alt: Build_status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/CalebBell/fluids/blob/master/LICENSE.txt
   :alt: license
.. image:: https://img.shields.io/coveralls/CalebBell/fluids.svg?release
   :target: https://coveralls.io/github/CalebBell/fluids
   :alt: Coverage
.. image:: https://img.shields.io/pypi/pyversions/fluids.svg?
   :target: https://pypi.python.org/pypi/fluids
   :alt: Supported_versions
.. image:: http://img.shields.io/appveyor/ci/calebbell/fluids.svg?
   :target: https://ci.appveyor.com/project/calebbell/fluids/branch/master
   :alt: Build_status
.. image:: https://zenodo.org/badge/48924523.svg?
   :alt: Zendo
   :target: https://zenodo.org/badge/latestdoi/48924523


.. contents::

What is Fluids?
---------------

Fluids is open-source software for engineers and technicians working in the
fields of chemical, mechanical, or civil engineering. It includes modules
for piping, fittings, pumps, tanks, compressible flow, open-channel flow,
atmospheric properties, solar properties, particle size distributions,
two phase flow, friction factors, control valves, orifice plates and
other flow meters, ejectors, relief valves, and more.

The fluids library is designed to be a low-overhead, lightweight repository
of engineering knowledge and utilities that relate to fluid dynamics.

Fluids was originally tightly integrated with SciPy and NumPy; today they
are optional components used for only a small amount of functionality
which do not have pure-Python numerical methods implemented.
Fluids targets Python 2.7 and up as well as PyPy2 and PyPy3. Additionally,
fluids has been tested by the author to load in IronPython, Jython,
and micropython.

While the routines in Fluids are normally quite fast and as efficiently
coded as possible, depending on the application there can still be a need
for further speed. PyPy provides a substantial speed boost of 6-12 times
for most methods. Fluids also
supports Numba, a powerful accelerator that works well with NumPy.
The Numba interface to fluids also makes it easy to multithread
execution as well, avoiding Python GIL issue.

Fluids runs on all operating systems which support Python, is quick to
install, and is free of charge. Fluids is designed to
be easy to use while still providing powerful functionality.
If you need to perform some fluid dynamics calculations, give
fluids a try.

Installation
------------

Get the latest version of fluids from
https://pypi.python.org/pypi/fluids/

If you have an installation of Python with pip, simple install it with:

    $ pip install fluids

Alternatively, if you are using `conda <https://conda.io/en/latest/>`_ as your package management, you can simply
install fluids in your environment from `conda-forge <https://conda-forge.org/>`_ channel with:

    $ conda install -c conda-forge fluids 

To get the git version, run:

    $ git clone git://github.com/CalebBell/fluids.git

Documentation
-------------

fluids's documentation is available on the web:

    http://fluids.readthedocs.io/

Latest source code
------------------

The latest development version of fluids's sources can be obtained at

    https://github.com/CalebBell/fluids


Bug reports
-----------

To report bugs, please use the fluids's Bug Tracker at:

    https://github.com/CalebBell/fluids/issues

If you have further questions about the usage of the library, feel free
to contact the author at Caleb.Andrew.Bell@gmail.com.


License information
-------------------

Fluids is MIT licensed. See ``LICENSE.txt`` for full information
on the terms & conditions for usage of this software, and a
DISCLAIMER OF ALL WARRANTIES.

Although not required by the fluids license, if it is convenient for you,
please cite fluids if used in your work. Please also consider contributing
any changes you make back, such that they may be incorporated into the
main library and all of us will benefit from them.


Citation
--------

To cite fluids in publications use::

    Caleb Bell (2016-2021). fluids: Fluid dynamics component of Chemical Engineering Design Library (ChEDL)
    https://github.com/CalebBell/fluids.
