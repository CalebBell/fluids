import matplotlib.pyplot as plt
import numpy as np
from fluids.fittings import entrance_beveled_methods
from fluids.fittings import entrance_beveled

angles = np.linspace(0, 90, 200) # 90 or 180? Plotted in Rennels only to 90.
styles = ['--', '-.', '-', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

l_ratios = [0.025, 0.05, 0.1, 0.25, .5, .75, 1]
for l, style in zip(l_ratios, styles):
    for method in entrance_beveled_methods:
        Ks = [entrance_beveled(Di=1.0, l=l, angle=angle, method=method) for angle in angles]
        plt.plot(angles, Ks, style, label=method + ', l/Di=%g' %l)
plt.legend()
plt.title("Comparison of available methods for beveled entrances")
plt.xlabel('angle')
plt.ylabel('K')
#plt.show()
