import matplotlib.pyplot as plt
import numpy as np
from math import log10
from scipy.interpolate import UnivariateSpline
from fluids.friction import Crane_fts_Ds, Crane_fts, ft_Crane, roughness_Farshad, friction_factor

roughness = .05E-3

plt.plot(Crane_fts_Ds, Crane_fts, 'o', label='Crane data')

spline_obj = UnivariateSpline(Crane_fts_Ds, Crane_fts, k=3, s=5e-6)
Ds_interp = np.linspace(Crane_fts_Ds[0], Crane_fts_Ds[-1], 500)
plt.plot(Ds_interp, spline_obj(Ds_interp), label='Cubic spline')

ft_crane_correlation = [.25/log10(roughness/(Di)/3.7)**2 for Di in Crane_fts_Ds]

plt.plot(Crane_fts_Ds, ft_crane_correlation,
         label='Crane formula')

plt.plot(Crane_fts_Ds, [round(i, 3) for i in ft_crane_correlation], '.',
         label='Crane formula (rounded)')

eDs_Farshad = [roughness_Farshad(ID='Carbon steel, bare', D=D)/D for D in Crane_fts_Ds]

fts_good = [friction_factor(Re=7.5E6*Di, eD=ed) for ed, Di in zip(eDs_Farshad, Crane_fts_Ds)]

plt.plot(Crane_fts_Ds, fts_good, label='Colebrook')

plt.plot(Crane_fts_Ds, [round(i, 3) for i in fts_good], 'x', label='Colebrook (rounded)')


plt.legend()
plt.title("Comparison of implementation options")
plt.xlabel('Pipe actual diameter, [m]')
plt.ylabel('Darcy friction factor, [-]')
#plt.show()
