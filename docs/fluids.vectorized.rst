Support for numpy arrays (fluids.vectorized)
============================================


Basic module which wraps all fluids functions with numpy's vectorize.
All other object - dicts, classes, etc - are not wrapped. Supports star 
imports; so the same objects exported when importing from the main library
will be imported from here. 

>>> from fluids.vectorized import *

Inputs do not need to be numpy arrays; they can be any iterable:

>>> import fluids
>>> fluids.vectorized.friction_factor(Re=[100, 1000, 10000], eD=0)
array([0.64      , 0.064     , 0.03088295])

Note that because this needs to import fluids itself, fluids.vectorized
needs to be imported separately; the following will cause an error:
    
>>> import fluids
>>> fluids.vectorized # doctest: +SKIP
The correct syntax is as follows:

>>> import fluids.vectorized # Necessary
>>> from fluids.vectorized import * # May be used without first importing fluids
