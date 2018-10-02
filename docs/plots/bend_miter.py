import matplotlib.pyplot as plt
import numpy as np
from fluids.fittings import bend_miter, bend_miter_methods
styles = ['--', '-.', '-', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

angles = np.linspace(0.0, 120.0, 100)

Res = np.array([[1E4, 4E4, 8E4],
            [2E5, 5E5, 1E6],
            [5E6, 1E7, 1E8]])


f, axarr = plt.subplots(3, 3)

for Re, axes in zip(Res.ravel(), axarr.ravel()):
    for method, style in zip(bend_miter_methods, styles):
        Di = 0.05 # Makes Crane go up or down
        Ks = [bend_miter(angle=angle, Di=Di, Re=Re, roughness=.05E-3, L_unimpeded=Di*20, method=method) for angle in angles]
        axes.plot(angles, Ks, label=method) # + ', angle = ' + str(angle)
        
        axes.set_title(r'Re = %g' %Re)
        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
             axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(6.5)
            
        ttl = axes.title.set_position([.5, .98])
    
plt.subplots_adjust(wspace=.35, hspace=.35)

f.suptitle('Comparison of available methods for mitre bend losses\n Angle (x) vs. Loss coefficient (y)')
plt.legend(loc='upper center', bbox_to_anchor=(1.5, 2.4))
plt.subplots_adjust(right=0.82, top=.85, bottom=.05)
#plt.show()



