nim Integration
===============

This example demonstrates how to use fluids from nim.

Source Code
-----------

.. literalinclude:: nim/check_fluids_can_be_called_from_nim.nim
   :language: nim
   :linenos:

Requirements
------------

- Python with fluids installed
- Nimpy: https://github.com/yglukhov/nimpy

Usage Notes
-----------

- Nim has a great package manager and is easy to read/write. This example took 10 minutes to develop.
- 1.5 microsecond friction factor, 9 microsecond tank creation observed by author; very comparable to pybind11