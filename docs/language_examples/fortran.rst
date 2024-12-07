fortran Integration
===================

This example demonstrates how to use fluids from fortran.

Source Code
-----------

.. literalinclude:: fortran/fluids_test.f90
   :language: fortran
   :linenos:

Requirements
------------

- Python with fluids installed

Usage Notes
-----------

- The example is incomplete. `iso_c_binding` is used to interface without actually writing C code. 
- The speed is comparable to pybind11, 2 microseconds per friction factor call.
- The excess amount of code required to interface the two languages is very significant