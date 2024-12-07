C++ Integration
===============

This example demonstrates how to use fluids from cpp.

Source Code
-----------

.. literalinclude:: cpp/main.cpp
   :language: cpp
   :linenos:

Requirements
------------

- Python with fluids installed
- pybind11

Usage Notes
-----------

- The example demonstrates basic integration with fluids
- Essentially zero overhead - 1.7 microseconds for friction factor, 8 microseconds for tank creation; only C using the has been observed to have a faster interface, able to save < 0.5 microseconds/call
