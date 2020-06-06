import matplotlib.pyplot as plt
import numpy as np
from fluids.fittings import entrance_distance_methods
from fluids.fittings import entrance_distance

ts = np.linspace(0,0.3, 1000)
for method in entrance_distance_methods:
    Ks = [entrance_distance(Di=1.0, t=t, method=method) for t in ts]
    plt.plot(ts, Ks, label=method)
plt.legend()
plt.title("Comparison of available methods for re-entrant entrances")
plt.xlabel('t/Di')
plt.ylabel('K')
#plt.show()
