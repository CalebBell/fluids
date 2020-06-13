# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
#from fluids import *
from fluids import isothermal_gas
class TimeCompressibleSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        pass

    def time_compressible(self):
        isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, m=145.48475726, D=None)
