Developer's Guide and Roadmap
=============================

The `fluids` project has grown to be:

* Efficient
    * Functions do only the work required.
    * Caching various values, precomputing others.
    * Using various macros and automated expressions to run code with Numba at its optimal speed.
    * Not using Numpy/SciPy most of the time, allowing PyPy and Numba to speed code up.
* Capable of vectorized computation
    * Wrapped with Numba's ufunc machinery.
    * Wrapped with numpy's np.vectorize.
    * Numba's nogil mechanism is set and used on all methods.
* Comprehensive
    * Most correlations taught at the undergrad level are included.
    * Most ancillary calculations such as atmospheric properties and tank geometry are included.
* Capable of handling units
    * Pint interface.
    * All docstrings/code in base SI units.

There is no official road map, no full time developers, and no commercial support for this library - this is a hobby project developed primarily by Caleb Bell. Contributors are welcome! Fluid dynamics is really big field and one author can't do everything.

Scope and Future Features
-------------------------

The following is a list of things that have crossed the author's mind as that would be cool in `fluids` and might make it there at some point. The author has no immediate plans to do any of this. Work by the author happens when inspiration - and free time - happens. Please feel free to work on any of these items:

* More pipe schedules. The author has been asked for EN 10255 in https://github.com/CalebBell/fluids/issues/16 and ASTM A269/A270 in an email.
* More fitting types.
* More tee/wye/junction correlations.
* More models for compressible flow, such as: https://www.mathworks.com/help/aerotbx/gas-dynamics-1.html
* Multiphase fitting pressure drop. Literature data is sparse.
* Multiphase flow meter pressure drop. One correlation has been added but there are more available.
* More multiphase flow maps.
* Pump viscosity correction from Hydraulic Institute.
* Tool to download historical weather data to calculate average historical temperatures and weather to use for design. This is partly complete in a non-exposed module `design_climate.py`.
* Models for mixing efficiency.
* Additional fluid-particle interaction correlations.

Contributing
------------

`fluids` has a lot of infrastructure that makes it attractive to add code to the project. Adding new functionality is possible without compromising load speed, RAM usage or maintainability. If you have a fluids engineering correlation, please feel free to open a PR and we can make any changes needed. There is no template - just do your best.

In an ideal world, new contributions would come with unit tests, docstrings, an addition to the tutorial if relevant.

Running Tests
-------------
From the root directory of the project you downloaded with `git clone https://github.com/CalebBell/fluids.git`, run the following command:

python3 -m pytest .

This will run all of the tests including the doctests.

The test suite can take some time to run; tests are marked with various markers to allow a fast subset of tests to run.

python3 -m pytest -m "not slow" .

This should only take a few seconds, and show red output if a test is broken. To keep the main test suite fast, pytest allows a flag which shows how long each test takes.

python3 -m pytest -m "not slow" --durations=100

If a test you added appears in this list, consider splitting it into a fast portion and a slow portion and decorating the slow portion with "@pytest.mark.slow".

Docstrings
----------
The docstrings follow Pep8, most of the numpydoc standard,
More information about numpydoc can be found `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_

In addition to being documentation, the docstrings in `fluids` serve the following purposes:

* Contain LaTeX math formulas for implemented formulas. This makes it easy for the reader and authors to follow code. This is especially important when the code can be optimized by hand significantly, and end up not looking like the math formulas.
* Contain doctests for every public method. These examples often make debugging really easy since they can just be copy-pasted into Jupyter or an IDE/debugger.
* Contain type information for each variable, which is automatically parsed by the unit handling framework around `pint`.
* Contain the units of each argument, which is used by the unit handling framework around `pint`.
* Contain docstrings for every argument - these are checked by the unit tests programmatically to avoid forgetting to add a description, which the author did often before the checker was added.

Doctest
-------
As anyone who has used doctest before knows, floating-point calculations have trivially different results across platforms. An example cause of this is that most compilers have different sin/cos implementations which are not identical. However, docstrings are checked bit-for-bit, so consistent output is important. Python is better than most languages at maintaining the same results between versions but it is still an issue.

Thanks to a fairly new pytest feature, numbers in doctests can be checked against the number of digits given, not against the real result. It is suggested to put numbers in doctests with around 13 digits, instead of the full repr() string for a number. It is convenient to round the number instead of just removing decimals.

Type Hints
----------
The python ecosystem is gradually adding support for type information, which may allow static analyzers to help find bugs in code even before it is ran. The author has not found these helpful in Python yet - the tools are too slow, missing features, and most libraries do not contain type information. However, type hints might still be useful for your program that uses `fluids`!

For that reason `fluids` includes a set of type hints as stub files (.pyi extension). These are not generated by hand - they use the cool `MonkeyType <https://github.com/Instagram/MonkeyType/>`_ library.
An included script `make_test_stubs` interfaces with this library, which runs the test suite and at the end generates the type hints including the types of every argument to every function seen in the test suite. This is another reason comprehensive test suite coverage is required.

Monkeytype on the `fluids` test suite takes ~15 minutes to run, and generates a ~1 GB database file which is deleted at the end of the run. Some manipulation of the result by hand may be required in the future, or MonkeyType may be replaced by making the type hints by hand. It is planned to incorporate the type stubs into the main file at some point in the future when the tooling is better, and Python 2 support has been dropped completely.

Supported Python Versions
-------------------------
There is no plan to drop Python 2.7 support at present, but you are advised to use Python 3.5 or later. Both NumPy and SciPy have dropped support for Python 2.7 and eventually a backwards-incompatible change may be needed.

Fluids targets Python 2.7 and up as well as PyPy2 and PyPy3. Additionally, fluids has been tested by the author to load in IronPython, Jython, and Micropython.

Unfortunately there is no CI infrastructure for these other Python implementations. For IronPython, Jython, and Micropython there is no NumPy/SciPy which means there is no hope of passing the whole test suite on them either; indeed pytest won't load on any of them.
If you have a need for a specific feature to work in an implementation, don't hesitate to reach out to the author to discuss it.

It is intended for IronPython to support everything except functionality which has a hard dependency on NumPy or SciPy. IronPython is currently Python 2 only; with the Python 3 variant being most of the way there. This may lead to integration with other programs in the future as IronPython is often used as a scripting language.

Micropython is designed to run on limited RAM, and fluids is too large for most microprocessors. You will likely have to copy/paste the specific parts of `fluids` you want to use for this application.

Jython is not very popular, but please reach out of you are using `fluids` with it.

Packaging
---------
The most up to date fluids can be obtained on GitHub, and new releases are pushed to PyPi whenever a new release is made.
Fluids is available on Conda thanks to Diego Volpatto and on Debian and thus Ubuntu thanks to Kurt Kremitzki. Conda updates more or less automatically but takes hours to build.

Code Formatting
---------------
Pep8 is loosely followed. Do your best to follow it if possible, otherwise don't worry about it. Please don't submit a PR for just style changes. Some arguments like `Method` or classes like TANK are unfortunately not pep8 for historical reasons.


Documentation
-------------
Sphinx is used with readthedocs. Readthedocs is configured to build whatever is on the release branch. From the root of the `fluids` project, the documentation can be built with the following command, which will output html files into a "_build" folder:

sphinx-build -b html docs _build

Sample Notebooks
----------------
The `nbval <https://pypi.org/project/nbval/>`_ pytest plugin can be used to check the results of running the notebooks against the stored notebooks.

On UNIX/Mac OS/WSL, the notebook results can be regenerated with the following shell command, from the directory with the notebooks:

.. code-block:: bash

   for i in *.ipynb ; do python3 -m nbconvert --to notebook --inplace --execute "$i" ; done

Continuous Integration
----------------------
Github Actions, Travis and Appveyor are presently used. They test only code in the `release` branch. Some tests, like those that download data from the internet, are not ran by design on their platforms. The same goes for testing `numba` online - getting an up to date version of numba is challenging.

Load Speed
----------
On CPython, `fluids` will load Numpy on load if it is available and `SciPy` when it is needed. Numpy loads in ~150 ms. Fluids alone loads in ~10 ms. It is intended for this to increase only slowly.

RAM Usage
---------
Loading fluids alone takes ~4 MB of RAM. About 2 MB of this is actually docstrings. About 200 KB of pipe schedules, 100 KB of pump information, and 200 KB of Sieve data is also included. Using fluids should not increase RAM except by the size of objects you initiate; the only things cached are very small. The -OO flag can be used in Python to cut RAM usage significantly, which will not load any docstrings.

Adding new data and methods is well worth the price of increasing RAM, but it is intended to keep RAM consumption small via lazy-loading any large data sets. Examples of this can be found in atmosphere.py - spa.py and nrlmsie00.py are lazy-loaded.

It is intended for RAM consumption of the library to increase only slowly.

Notes on Pint Integration Implementation
----------------------------------------
Units in square brackets in the docstrings are parsed for all function inputs and outputs. They are parsed by Pint directly.

In some cases, a function has a variable output unit, as in the case of solvers which can solve for different variables. In that case, the variable unit shouldn't put anything in square brackets. Instead, in `units.py`, the variable `variable_output_unit_funcs` needs to have an entry for the new function. The return unit will be based on which variables are not provided as inputs to the function. True represents a present variable, and False represents a variable left as None. The number of variables the dispatch happens on can be less than the number of function arguments, and should be specified after the units signature.

Notes on PyPy
-------------
PyPy is really awesome!

It does have some drawbacks today:

* The C-API which is used by NumPy, SciPy get 2-3 times slower in PyPy. This is originally why `fluids` started implementing its own numerical methods sometimes, although now it is for custom features and increased speed. There is a project to solve this issue: https://github.com/pyhandle/hpy
* If running code only a few times, PyPy won't be able to accelerate it as it is a Just In Time Compiler.
* Sometimes something gets speed up by PyPy some of the time, but not all of the time.
* Uses more memory, typically 1.5x.
* Not as good as Numba at generating vectorized, SIMD instructions for the CPU. PyPy also isn't as good at inlining small functions.

The main pros of PyPy are:

* Really, really fast. Some functions literally save 98% of their time by running in PyPy, although 85% is more typical.
* Accelerates ALL of your code, not a little like numba. 
* For scalar functions PyPy is typically quite a bit faster than numba.
* Doesn't need special handling, does everything CPython can do.
* Doesn't need a special coding style!

A few compromises in the library to make PyPy more performant were made:

* Use the `sqrt` operator to compute powers as much as possible. `sqrt` and a few multiplies is much cheaper than a power operator. This is not really noticeable on CPython, but you can tell in PyPy. CPUs have special hardware to make this computation very cheap.


Notes on Numba
--------------

The main pros of Numba are:

* Works with CPython.
* Pretty good at generating SIMD instructions.
* Fast. Gets all the benefits that LLVM gets. This means if you include a line of code that does nothing in your function, it probably won't run once compiled with numba.
* When a complete set of code is wrapped by Numba, it can be multithreaded easily.

The main cons of Numba are:

* It doesn't come close to supporting all of Python. This really hurts on things like dictionary lookups or functions that return dictionaries.
* It is not available on many platforms, used to require Anaconda.
* Some code can be really, really slow to compile today. Compiling `fluids` with numba takes ~3 minutes today, after some optimizations. Caching of functions that take functions as arguments is not yet supported, nor are jitclasses.
* Can be a pain to work with.

Quite a few compromises in the library were made to add Numba compatibility and in cases to make Numba even more performant:

* A series of `numba` pragmas were invented and are interpreted by a loader that recompiles the transformed source code of `fluids`.
* Functions that accept functions as arguments or use scipy.special functions are not compatible with Numba's caching implementation at this time. To avoid having complaints about that, they are added to a list in numba.py.
* Numba does not support raising exceptions with dynamically created messages. Where possible, this means using a constant message. 
* Sometimes the only way to do something is by changing the code directly. Append "# numba: delete" at the end of a line in a function to delete the line. Add a new commented out line, and append "# numba: uncomment" to it. Then put the name of that function in the variable `to_change` in numba.py, and the changes will be made when using the Numba interface.
* 1D arrays should be initialized like [0.0]*4, [my_thing]*my_count; and they put the function in the same `to_change` variable. This will transform them into the right type of array for Numba.
* Numba uses efficient cbrts while CPython and PyPy do not; any case of x\*\*(1/3) will turn into a cbrt. x\*\*(2/3) will not, but can be done by hand.

It is hoped many of these trade offs can be removed/resolved by future features added to Numba.

Things to Keep In Mind While Coding
-----------------------------------

1. Python is often ran with the -O or -OO flag. This reduces its memory use and increases performance a little. One of those optimizations is that any `assert` statements in python code are skipped. This means they should not be used to control a program's flow. This is normally the equivalent of using a Release build vs. a Debug build in C++.
2. Numpy arrays and functions should be used with care. They will make that portion of the code not run on some implementations, and will add a dependency on NumPy for that function. If it is a vectorization issue, consider letting Numba or PyPy accelerate it for you. If it about using some fancy functionality like a fourier transform, then NumPy is the right choice!
