import matplotlib.pyplot as plt
import numpy as np
from fluids.fittings import contraction_round, contraction_round_methods
styles = ['--', '-.', '-', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

D_ratios = np.linspace(1-1e-9, .01, 1000)
radius_ratios = [.1, .06, .04, .02, 0] #np.linspace(0, 0.2, 3)
for radius_ratio in radius_ratios:
    for method, style in zip(contraction_round_methods, styles):
        Ks = [contraction_round(Di1=1, Di2=Di, rc=Di*radius_ratio, method=method) for Di in D_ratios]
        Ds2 = D_ratios**2
        plt.plot(Ds2, Ks, style, label=method + ', ratio = ' + str(radius_ratio))
plt.legend()
plt.title("Comparison of available methods for rounded pipe contractions")
plt.xlabel('Area ratio')
plt.ylabel('K')
#plt.show()
