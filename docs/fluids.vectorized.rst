Support for NumPy arrays (fluids.vectorized)
============================================

Basic submodule that wraps all fluids functions with numpy's :obj:`vectorize <numpy.vectorize>`.
All other object - dicts, classes, etc - are not wrapped. 
Where speed is a concern, the newer :obj:`fluids.numba` numpy interface may be used to obtain
C/C++/Fortran-level performance on array calculations.

>>> import numpy as np
>>> import fluids
>>> fluids.vectorized.Reynolds(V=2.5, D=np.array([0.25, 0.5, 0.75, 1.0]), rho=1.1613, mu=1.9E-5)
array([ 38200.65789474,  76401.31578947, 114601.97368421, 152802.63157895])

Vectorization follows the numpy broadcast rules.

>>> fluids.vectorized.Reynolds(V=np.array([1.5, 2.5, 3.5, 5.0]), D=np.array([0.25, 0.5, 0.75, 1.0]), rho=1.1613, mu=1.9E-5)
array([ 22920.39473684,  76401.31578947, 160442.76315789, 305605.26315789])

Inputs do not need to be numpy arrays; they can be any iterable:

>>> fluids.vectorized.friction_factor(Re=[100, 1000, 10000], eD=0)
array([0.64      , 0.064     , 0.03088295])

If you just want to use numpy arrays as inputs to all fluids functions, you can use Python's import aliasing feature to replace fluids with the vectorized version. Note that there are no submodules in `fluids.vectorized`.

>>> import fluids.vectorized as fluids

.. warning::
    This module does not replace the functions in the `fluids` module; it
    copies all the functions into the `fluids.vectorized` module and makes
    them vectorized there.

    For example by importing `fluids.vectorized`,
    `fluids.friction.friction_factor` won't become vectorized, 
    but `fluids.vectorized.friction_factor` will become available and is vectorized.

.. warning:: :obj:`np.vectorize <numpy.vectorize>` does not use NumPy to accelerate any computations;
   it is a convenience wrapper. If you are working on a problem large enough for
   speed to be an issue and Numba is compatible with your version of Python,
   an interface to that library is available at :obj:`fluids.numba` which does
   accelerate NumPy array computations and is normally faster than using numpy
   directly.
