import matplotlib.pyplot as plt
import numpy as np

from fluids.fittings import contraction_conical, contraction_conical_methods

styles = ['--', '-.', '-', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

D_ratios = np.linspace(1-1e-9, .01, 100)
angles = np.array([[2, 4, 8, 10],
            [15, 20, 25, 30],
            [45, 60, 90, 120],
            [135, 150, 165, 180]])


f, axarr = plt.subplots(4, 4)

for angle, axes in zip(angles.ravel(), axarr.ravel()):
    for method, style in zip(contraction_conical_methods, styles):

        Ks = [contraction_conical(Di1=1, Di2=Di, Re=1E6, angle=angle, method=method) for Di in D_ratios]
        Ds2 = D_ratios**2
        axes.plot(Ds2, Ks, label=method) # + ', angle = ' + str(angle)

        #axes.legend()
        axes.set_title(r'$%g^\circ$ Angle' %angle)
        #axes.set_xlabel('Area ratio')
        #axes.set_ylabel('K')
        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
             axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(6.5)

        ttl = axes.title.set_position([.5, .93])

plt.subplots_adjust(wspace=.35, hspace=.35)

f.suptitle('Comparison of available methods for conical pipe contractions\n Area ratio (x) vs. Loss coefficient (y)')
plt.legend(loc='upper center', bbox_to_anchor=(1.65, 4.7))
plt.subplots_adjust(right=0.82)
#plt.show()



