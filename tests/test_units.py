# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from __future__ import division
#from fluids import *
from numpy.testing import assert_allclose
import pytest
from fluids.units import *
import numpy as np


def test_convert_input():
    from fluids.units import convert_input
    
    ans = convert_input(5, 'm', u, False)
    assert ans == 5
    with pytest.raises(Exception):
        convert_input(5, 'm', u, True)


def test_sample_cases():
    Re = Reynolds(V=3.5*u.m/u.s, D=2*u.m, rho=997.1*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
    assert_allclose(Re.to_base_units().magnitude, 6979700.0)
    assert dict(Re.dimensionality) == {}
    

#    vs = hwm93(5E5*u.m, 45*u.degrees, 50*u.degrees, 365*u.day)
#    vs_known = [-73.00312042236328, 0.1485661268234253]
#    for v_known, v_calc in zip(vs_known, vs):
#        assert_allclose(v_known, v_calc.to_base_units().magnitude)
#        assert dict(v_calc.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}

    A = API520_A_g(m=24270*u.kg/u.hour, T=348.*u.K, Z=0.90, MW=51.*u.g/u.mol, k=1.11, P1=670*u.kPa, Kb=1, Kc=1)
    assert_allclose(A.to_base_units().magnitude, 0.00369904606468)
    assert dict(A.dimensionality) == {u'[length]': 2.0}
    
    T = T_critical_flow(473*u.K, 1.289)
    assert_allclose(T.to_base_units().magnitude, 413.280908694)
    assert dict(T.dimensionality) == {u'[temperature]': 1.0}
    
    T2 = T_critical_flow(473*u.K, 1.289*u.dimensionless)
    
    assert T == T2
    
    with pytest.raises(Exception):
        T_critical_flow(473, 1.289)

    with pytest.raises(Exception):
        T_critical_flow(473*u.m, 1.289)
        
    A = size_control_valve_g(T=433.*u.K, MW=44.01*u.g/u.mol, mu=1.4665E-4*u.Pa*u.s, gamma=1.30,
    Z=0.988, P1=680*u.kPa, P2=310*u.kPa, Q=38/36.*u.m**3/u.s, D1=0.08*u.m, D2=0.1*u.m, d=0.05*u.m,
    FL=0.85, Fd=0.42, xT=0.60)
    assert_allclose(A.to_base_units().magnitude, 0.0201629570705307)
    assert dict(A.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}
    
    A = API520_round_size(A=1E-4*u.m**2)
    assert_allclose(A.to_base_units().magnitude, 0.00012645136)
    assert dict(A.dimensionality) == {u'[length]': 2.0}
    
    SS = specific_speed(0.0402*u.m**3/u.s, 100*u.m, 3550*u.rpm)
    assert_allclose(SS.to_base_units().magnitude, 2.3570565251512066)
    assert dict(SS.dimensionality) == {u'[length]': 0.75, u'[time]': -1.5}
    
    v = Geldart_Ling(1.*u.kg/u.s, 1.2*u.kg/u.m**3, 0.1*u.m, 2E-5*u.Pa*u.s)
    assert_allclose(v.to_base_units().magnitude, 7.467495862402707)
    assert dict(v.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}
    
    s = speed_synchronous(50*u.Hz, poles=12)
    assert_allclose(s.to_base_units().magnitude, 157.07963267948966)
    assert dict(s.dimensionality) == {u'[time]': -1.0}
    
    t = t_from_gauge(.2, False, 'AWG')
    assert_allclose(t.to_base_units().magnitude, 0.5165)
    assert dict(t.dimensionality) == {u'[length]': 1.0}
    
    dP = Robbins(G=2.03*u.kg/u.m**2/u.s, rhol=1000*u.kg/u.m**3, Fpd=24/u.ft, L=12.2*u.kg/u.m**2/u.s, rhog=1.1853*u.kg/u.m**3, mul=0.001*u.Pa*u.s, H=2*u.m)
    assert_allclose(dP.to_base_units().magnitude, 619.662459344 )
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}
    
    dP = dP_packed_bed(dp=8E-4*u.m, voidage=0.4, vs=1E-3*u.m/u.s, rho=1E3*u.kg/u.m**3, mu=1E-3*u.Pa*u.s)
    assert_allclose(dP.to_base_units().magnitude, 1438.28269588 )
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}
    
    dP = dP_packed_bed(dp=8E-4*u.m, voidage=0.4*u.dimensionless, vs=1E-3*u.m/u.s, rho=1E3*u.kg/u.m**3, mu=1E-3*u.Pa*u.s, Dt=0.01*u.m)
    assert_allclose(dP.to_base_units().magnitude, 1255.16256625)
    assert dict(dP.dimensionality) == {u'[length]': -1.0, u'[mass]': 1.0, u'[time]': -2.0}

    n = C_Chezy_to_n_Manning(26.15*u.m**0.5/u.s, Rh=5*u.m)
    assert_allclose(n.to_base_units().magnitude, 0.05000613713238358)
    assert dict(n.dimensionality) == {u'[length]': -0.3333333333333333, u'[time]': 1.0}

    Q = Q_weir_rectangular_SIA(0.2*u.m, 0.5*u.m, 1*u.m, 2*u.m)
    assert_allclose(Q.to_base_units().magnitude, 1.0408858453811165)
    assert dict(Q.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}
    
    t = agitator_time_homogeneous(D=36*.0254*u.m, N=56/60.*u.revolutions/u.second, P=957.*u.W, T=1.83*u.m, H=1.83*u.m, mu=0.018*u.Pa*u.s, rho=1020*u.kg/u.m**3, homogeneity=.995)
    assert_allclose(t.to_base_units().magnitude, 15.143198226374668)
    assert dict(t.dimensionality) == {u'[time]': 1.0}
    
    K = K_separator_Watkins(0.88*u.dimensionless, 985.4*u.kg/u.m**3, 1.3*u.kg/u.m**3, horizontal=True)
    assert_allclose(K.to_base_units().magnitude, 0.07944704064029771)
    assert dict(K.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}

    A = current_ideal(V=120*u.V, P=1E4*u.W, PF=1, phase=1)
    assert_allclose(A.to_base_units().magnitude, 83.33333333333333)
    assert dict(A.dimensionality) == {u'[current]': 1.0}
    
    fd = friction_factor(Re=1E5, eD=1E-4)
    assert_allclose(fd.to_base_units().magnitude, 0.01851386607747165)
    assert dict(fd.dimensionality) == {}
    
    K = Cv_to_K(2.712*u.gallon/u.minute, .015*u.m)
    assert_allclose(K.to_base_units().magnitude, 14.719595348352552)
    assert dict(K.dimensionality) == {}

    Cv = K_to_Cv(16, .015*u.m)
    assert_allclose(Cv.to_base_units().magnitude, 0.0001641116865931214)
    assert dict(Cv.dimensionality) == {u'[length]': 3.0, u'[time]': -1.0}
    
    Cd = drag_sphere(200)
    assert_allclose(Cd.to_base_units().magnitude, 0.7682237950389874)
    assert dict(Cd.dimensionality) == {}

    V, D = integrate_drag_sphere(D=0.001*u.m, rhop=2200.*u.kg/u.m**3, rho=1.2*u.kg/u.m**3, mu=1.78E-5*u.Pa*u.s, t=0.5*u.s, V=30*u.m/u.s, distance=True)
    assert_allclose(V.to_base_units().magnitude, 9.686465044063436)
    assert dict(V.dimensionality) == {u'[length]': 1.0, u'[time]': -1.0}
    assert_allclose(D.to_base_units().magnitude, 7.829454643649386)
    assert dict(D.dimensionality) == {u'[length]': 1.0}
    
    Bo = Bond(1000*u.kg/u.m**3, 1.2*u.kg/u.m**3, .0589*u.N/u.m, 2*u.m)
    assert_allclose(Bo.to_base_units().magnitude, 665187.2339558573)
    assert dict(Bo.dimensionality) == {}
    
    head = head_from_P(P=98066.5*u.Pa, rho=1000*u.kg/u.m**3)
    assert_allclose(head.to_base_units().magnitude, 10.000000000000002)
    assert dict(head.dimensionality) == {u'[length]': 1.0}
    
    roughness = roughness_Farshad('Cr13, bare', 0.05*u.m)
    assert_allclose(roughness.to_base_units().magnitude, 5.3141677781137006e-05)
    assert dict(roughness.dimensionality) == {u'[length]': 1.0}

    A = A_multiple_hole_cylinder(0.01*u.m, 0.1*u.m, [(0.005*u.m, 1)])
    assert_allclose(A.to_base_units().magnitude, 0.004830198704894308)
    assert dict(A.dimensionality) == {u'[length]': 2.0}
    
    V = V_multiple_hole_cylinder(0.01*u.m, 0.1*u.m, [(0.005*u.m, 1)])
    assert_allclose(V.to_base_units().magnitude, 5.890486225480862e-06)
    assert dict(V.dimensionality) == {u'[length]': 3.0}
