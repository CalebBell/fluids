Calling Fluids/Python from Other Languages
==========================================

Fluids can be called from many different programming languages through various Python bindings and interfaces. 
These examples demonstrate some of the ways to communicate with or embed Python into a program written in another language.

The examples focus on showing how to call functions (including optional arguments), classes and their various methods, and the performance of the language binding.

The author has developed these examples with AI assistance and does not claim to be proficient in all of these languages.
The difficulty of a language integration varies from trivial in Julia to manually managing reference counts and dealing with the C API in Fortran. 
The intent of these examples is to show possibilities, not to provide complete wrappers for all functionality.

These examples should also show how the `ht <https://github.com/CalebBell/ht>`_, 
`chemicals <https://github.com/CalebBell/chemicals>`_, and `thermo <https://github.com/CalebBell/thermo>`_ libraries can be used from other languages.


Languages
---------

.. toctree::
   :maxdepth: 1

   julia.rst
   perl.rst
   ruby.rst
   r.rst
   octave.rst
   lisp.rst
   cpp.rst
   c.rst
   fortran.rst
   go.rst
   haskell.rst
   nim.rst
   d.rst

Common Considerations
---------------------

When using fluids from another programming language, keep in mind:

* Performance will be lower due to the overhead of cross-language communication. 
  Compiled languages like C++, C, D, Fortran, Go, haskell, and Nim tend to have a constant 
  overhead of a couple microseconds per call, and any serialization/data copying costs
  which depending on the speed of the fluids function can be 2-3x the fluids call itself.
* If you use your compiler's FASTMATH setting or floating point settings (things like FLDCW/FSTCW/MXCSR) those settings will impact Python's own floating point handling
* For compiled languages error handling can be difficult - checking variable ranges in your own language may be wise

Notes for Package Maintainers
-----------------------------

If you're maintaining a package that wraps fluids for your language, feel free to let the author know (especially if it's open source!).

Contributing
------------

If you'd like to improve these examples or add support for another language:

1. Fork the fluids repository
2. Add your example including documentation
3. Submit a pull request
