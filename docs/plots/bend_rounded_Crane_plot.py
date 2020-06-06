import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from fluids.optional.pychebfun import chebfun
from fluids.fittings import bend_rounded_Crane_ratios, bend_rounded_Crane_fds, bend_rounded_Crane_coeffs

ratios = np.linspace(1, 20, 1000)

bend_rounded_Crane_obj = UnivariateSpline(bend_rounded_Crane_ratios, bend_rounded_Crane_fds, s=0)
fun = chebfun(f=bend_rounded_Crane_obj, domain=[1,20], N=10)


plt.plot(bend_rounded_Crane_ratios, bend_rounded_Crane_fds, 'x', label='Crane data')
plt.plot(ratios, bend_rounded_Crane_obj(ratios), label='Cubic spline')
plt.plot(ratios, fun(ratios), label='Chebyshev approximation')

plt.legend()
plt.title("Interpolation of Crane ft multipliers for pipe bend losses")
plt.xlabel('Bend radius/pipe diameter ratio')
plt.ylabel('Friction factor multiplier')
#plt.show()
