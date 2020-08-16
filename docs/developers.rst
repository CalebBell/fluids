Developer's Guide and Roadmap
=============================


The `fluids` project has grown to be:

* Efficient
    * Functions do only the work required.
    * Caching various values where memory is not important.
    * Using various macros and automated expressions to run code with Numba at its optimal speed.
    * Not using Numpy/SciPy most of the time, allowing PyPy to speed code up.
* Vectorized functions
    * Wrapped with numpy's np.vectorize
    * Wrapped with numba's ufunc machinery


Docstrings
----------
The docstrings follow Pep8, most of the numpydoc standard,
More information about numpydoc can be found `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_

In addition to being documentation, the docstrings in `fluids` serve the following purposes:

* Contain LaTeX math formulas for implemented formulas. This makes it easy for the reader and authors to follow code. This is especially important when the formulas can be optimized by hand significantly, and end up not looking like the math formulas.
* Contain doctests for every public method. These examples often make debugging really easy since they can just be copy-pasted into Jupyter or an IDE/debugger.
* Contain type information for each variable, which is automatically parsed by the unit handling framework around `pint`.
* Contain the units of each argument, which is used by the unit handling framework around `pint`.
* * Contain docstrings for every argument - these are checked by the unit tests programatically to avoid forgetting to add a description, which the author did often before the checker was added.

No automated style tool is ran on the docstrings at present, but the following command
was used once to format the docstrings with the tool `docformatter <https://github.com/myint/docformatter>`_



python3 -m docformatter --wrap-summaries=80 --wrap-descriptions=80 --in-place --recursive .