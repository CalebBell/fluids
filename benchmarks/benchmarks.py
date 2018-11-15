# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
#from fluids import *
import numpy as np

from fluids.friction import *

def time_Clamond():
    Clamond(1E5, 1E-4)
    
def time_ft_Crane():
    ft_Crane(.1)

def time_friction_factor_curved():
    friction_factor_curved(Re=1E5, Di=0.02, Dc=0.5)

def time_friction_factor():
    friction_factor(Re=1E5, eD=1E-4)
