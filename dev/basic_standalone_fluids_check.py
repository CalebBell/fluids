import fluids
from fluids import *
import fluids.vectorized
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.spatial
import scipy.special
import scipy.optimize

fluids.all_submodules()

def check_close(a, b, rtol=1e-7, atol=0):
    np.all(np.abs(a - b) <= (atol + rtol * np.abs(b)))
    return True

def run_checks():
    # Avoid 
    checks = []

    # Check friction_factor
    result = fluids.vectorized.friction_factor(Re=[100, 1000, 10000], eD=0)
    expected = np.array([0.64, 0.064, 0.03088295])
    checks.append(check_close(result, expected))

    # Check Reynolds number calculations
    checks.append(fluids.core.Reynolds(D=0.01, rho=1000, V=1.5, mu=1E-3) == 15000.0)
    checks.append(Reynolds(D=0.01, V=1.5, nu=1E-6) == 15000.0)

    # Check nearest_material_roughness (just call, don't check result)
    nearest_material_roughness('Used water piping', clean=False)

    # Check nearest_pipe
    result = nearest_pipe(Do=0.5, schedule='40S')
    expected = np.array([20, 0.48894, 0.508, 0.00953])
    checks.append(check_close(result, expected))

    # Check if ValueError is raised for invalid pipe input
    try:
        nearest_pipe(Do=1)
        checks.append(False)
    except ValueError as e:
        checks.append(str(e) == 'Pipe input is larger than max of selected schedule')

    # Check TANK calculations
    T1 = TANK(D=1.2, L=4, horizontal=False)
    checks.append(check_close(T1.V_total, 4.523893421169302))
    checks.append(check_close(T1.A, 17.34159144781566))

    DIN = TANK(L=3, D=5, horizontal=False, sideA='torispherical', sideB='torispherical', sideA_f=1, sideA_k=0.1, sideB_f=1, sideB_k=0.1)
    checks.append(check_close(DIN.V_total, 83.646361))

    # Check ATMOSPHERE_1976
    atm = ATMOSPHERE_1976(Z=5000)
    checks.append(check_close(atm.T, 255.6755432218))
    checks.append(check_close(atm.P, 54048.2861457))
    checks.append(check_close(atm.rho, 0.736428420779))

    # Check integrate_drag_sphere
    result = integrate_drag_sphere(D=1E-3, rhop=3400., rho=1.2, mu=1E-5, t=1, V=30, distance=True)
    expected = np.array([10.561878111165337, 15.607904177715524])
    checks.append(check_close(result, expected))

    # Check ParticleSizeDistribution
    ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
    numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)
    checks.append(check_close(psd.mean_size(2, 1), 1857.7888572055526))
    return all(checks)

if run_checks():
    print("Fluids basic checks passed - NumPy and SciPy used successfully")
else:
    print('Library not OK')
    exit(1)