go Integration
==============

This example demonstrates how to use fluids from go.

Source Code
-----------

.. literalinclude:: go/main.go
   :language: go
   :linenos:

Requirements
------------

- Python with fluids installed
- cgo

Usage Notes
-----------

- cgo and low-level python primitives are used to interface the two languages
- 3.5 microsecond friction factor, 25 microsecond tank creation observed by author
- There is some memory issue in the example that causes it to crash at the end