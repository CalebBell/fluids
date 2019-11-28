import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fluids.fittings import entrance_rounded_methods
from fluids.fittings import entrance_rounded

rcs = np.linspace(0,0.4, 1000)
for method in entrance_rounded_methods:
    Ks = [entrance_rounded(Di=1.0, rc=rc, method=method) for rc in rcs]
    plt.plot(rcs, Ks, label=method)
plt.legend()
plt.title('Comparison of available methods for rounded flush entrances to pipes')
plt.xlabel('rc/Di')
plt.ylabel('K')
#plt.show()
