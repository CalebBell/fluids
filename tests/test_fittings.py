# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division
import os
from fluids import *
import numpy as np
from math import pi, log10, log
from random import uniform
from numpy.testing import assert_allclose
from scipy.constants import *
from scipy.optimize import *
from scipy.interpolate import *
from fluids import fluids_data_dir
from fluids.core import Engauge_2d_parser
from fluids.optional.pychebfun import *

import pytest


def log_uniform(low, high):
    return 10**uniform(log10(low), log10(high))


def test_fittings():
    
    K = entrance_beveled_orifice(Di=0.1, do=.07, l=0.003, angle=45)
    assert_allclose(K, 1.2987552913818574)

    ### Exits
    assert_allclose(exit_normal(), 1.0)

    K_helix = helix(Di=0.01, rs=0.1, pitch=.03, N=10, fd=.0185)
    assert_allclose(K_helix, 14.525134924495514)

    K_spiral = spiral(Di=0.01, rmax=.1, rmin=.02, pitch=.01, fd=0.0185)
    assert_allclose(K_spiral, 7.950918552775473)

    ### Contractions
    K_sharp = contraction_sharp(Di1=1, Di2=0.4)
    assert_allclose(K_sharp, 0.5301269161591805)


    K_beveled = contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120)
    assert_allclose(K_beveled, 0.40946469413070485)

    ### Expansions (diffusers)
    K_sharp = diffuser_sharp(Di1=.5, Di2=1)
    assert_allclose(K_sharp, 0.5625)


    K = diffuser_curved(Di1=.25**0.5, Di2=1., l=2.)
    assert_allclose(K, 0.2299781250000002)

    K = diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07)
    assert_allclose(K, 0.06873244301714816)
    
    K = diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07, fd2=.08)
    assert_allclose(K, 0.06952256647393829)

    # Misc
    K1 = Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
    K2 = Darby3K(NPS=12., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
    K3 = Darby3K(NPS=12., Re=10000., K1=950,  Ki=0.25,  Kd=4)
    Ks = [1.1572523963562353, 0.819510280626355, 0.819510280626355]
    assert_allclose([K1, K2, K3], Ks)

    with pytest.raises(Exception):
        Darby3K(NPS=12., Re=10000)
    with pytest.raises(Exception):
        Darby3K(NPS=12., Re=10000, name='fail')

    tot = sum([Darby3K(NPS=2., Re=1000, name=i) for i in Darby.keys()])
    assert_allclose(tot, 67.96442287975898)

    K1 = Hooper2K(Di=2., Re=10000., name='Valve, Globe, Standard')
    K2 = Hooper2K(Di=2., Re=10000., K1=900, Kinfty=4)
    assert_allclose([K1, K2], [6.15, 6.09])
    tot = sum([Hooper2K(Di=2., Re=10000., name=i) for i in Hooper.keys()])
    assert_allclose(tot, 46.18)

    with pytest.raises(Exception):
        Hooper2K(Di=2, Re=10000)
    with pytest.raises(Exception):
        Hooper2K(Di=2., Re=10000, name='fail')
        
    K2 = change_K_basis(K1=32.68875692997804, D1=.01, D2=.02)
    assert_allclose(K2, 523.0201108796487)
        

### Entrances


def test_entrance_distance_45_Miller():
    from fluids.fittings import entrance_distance_45_Miller
    K = entrance_distance_45_Miller(Di=0.1, Di0=0.14)
    assert_allclose(K, 0.24407641818143339)



def test_entrance_distance():
    K1 = entrance_distance(0.1, t=0.0005)
    assert_allclose(K1, 1.0154100000000004)
    
    assert_allclose(entrance_distance(Di=0.1, t=0.05), 0.57)

    K = entrance_distance(Di=0.1, t=0.0005, method='Miller')
    assert_allclose(K, 1.0280427936730414)
    
    K = entrance_distance(Di=0.1, t=0.0005, method='Idelchik')
    assert_allclose(K, 0.9249999999999999)
    K = entrance_distance(Di=0.1, t=0.0005, l=.02, method='Idelchik')
    assert_allclose(K, 0.8475000000000001)
    
    K = entrance_distance(Di=0.1, t=0.0005, method='Harris')
    assert_allclose(K, 0.8705806231290558, 3e-3)
    
    K = entrance_distance(Di=0.1, method='Crane')
    assert_allclose(K, 0.78)
    
    with pytest.raises(Exception):
        entrance_distance(Di=0.1, t=0.01, method='BADMETHOD')


def test_entrance_rounded():
    K =  entrance_rounded(Di=0.1, rc=0.0235)
    assert_allclose(K, 0.09839534618360923)
    assert_allclose(entrance_rounded(Di=0.1, rc=0.2), 0.03)

    K = entrance_rounded(Di=0.1, rc=0.0235, method='Miller')
    assert_allclose(K, 0.057734448458542094)
    
    K = entrance_rounded(Di=0.1, rc=0.0235, method='Swamee')
    assert_allclose(K, 0.06818838227156554)
    
    K = entrance_rounded(Di=0.1, rc=0.01, method='Crane')
    assert_allclose(K, .09)
    
    K = entrance_rounded(Di=0.1, rc=0.01, method='Harris')
    assert_allclose(K, 0.04864878230217168)
    
    # Limiting condition
    K = entrance_rounded(Di=0.1, rc=0.0235, method='Harris')
    assert_allclose(K, 0.0)
    
    K = entrance_rounded(Di=0.1, rc=0.01, method='Idelchik')
    assert_allclose(K, 0.11328005177738182)
    
    # Limiting condition
    K = entrance_rounded(Di=0.1, rc=0.0235, method='Idelchik')
    assert_allclose(K, 0.03)
    
    with pytest.raises(Exception):
        entrance_rounded(Di=0.1, rc=0.01, method='BADMETHOD')

def test_entrance_beveled():
    K = entrance_beveled(Di=0.1, l=0.003, angle=45)
    assert_allclose(K, 0.45086864221916984)

    K = entrance_beveled(Di=0.1, l=0.003, angle=45, method='Idelchik')
    assert_allclose(K, 0.3995000000000001)


def test_entrance_sharp():
    assert_allclose(entrance_sharp(), 0.57)
    
    with pytest.raises(Exception):
        entrance_sharp(method='BADMETHOD')

    for method in ['Swamee', 'Blevins', 'Idelchik', 'Crane']:
        assert_allclose(0.5, entrance_sharp(method=method))
        
    entrance_sharp(method='Miller') # Don't bother checking a value for the Miller method


def test_entrance_angled():
    K_30_Idelchik = 0.9798076211353316
    assert_allclose(entrance_angled(30), K_30_Idelchik)
    assert_allclose(entrance_angled(30, method='Idelchik'), K_30_Idelchik)

    with pytest.raises(Exception):
        entrance_angled(30, method='BADMETHOD')


### Bends


def test_bend_rounded_Crane():
    K = bend_rounded_Crane(Di=.4020, rc=.4*5, angle=30)
    assert_allclose(K, 0.09321910015613409)
    
    K_max = bend_rounded_Crane(Di=.400, rc=.4*25, angle=30)
    K_limit = bend_rounded_Crane(Di=.400, rc=.4*20, angle=30)
    assert_allclose(K_max, K_limit)


def test_bend_rounded_Miller():
    # Miller examples - 9.12
    D = .6
    Re = Reynolds(V=4, D=D, nu=1.14E-6)
    kwargs = dict(Di=D, bend_diameters=2, angle=90,  Re=Re, roughness=.02E-3)
    
    K = bend_rounded_Miller(L_unimpeded=30*D, **kwargs)
    assert_allclose(K, 0.1513266131915296, rtol=1e-4)# 0.150 in Miller- 1% difference due to fd
    K = bend_rounded_Miller(L_unimpeded=0*D, **kwargs)
    assert_allclose(K, 0.1414607344374372, rtol=1e-4) # 0.135 in Miller - Difference mainly from Co interpolation method, OK with that
    K = bend_rounded_Miller(L_unimpeded=2*D, **kwargs)
    assert_allclose(K, 0.09343184457353562, rtol=1e-4) # 0.093 in miller

def test_bend_rounded():
    ### Bends
    K_5_rc = [bend_rounded(Di=4.020, rc=4.0*5, angle=i, fd=0.0163) for i in [15, 30, 45, 60, 75, 90]]
    K_5_rc_values = [0.07038212630028828, 0.10680196344492195, 0.13858204974134541, 0.16977191374717754, 0.20114941557508642, 0.23248382866658507]
    assert_allclose(K_5_rc, K_5_rc_values)

    K_10_rc = [bend_rounded(Di=34.500, rc=36*10, angle=i, fd=0.0106) for i in [15, 30, 45, 60, 75, 90]]
    K_10_rc_values =  [0.061075866683922314, 0.10162621862720357, 0.14158887563243763, 0.18225270014527103, 0.22309967045081655, 0.26343782210280947]
    assert_allclose(K_10_rc, K_10_rc_values)

    K = bend_rounded(Di=4.020, bend_diameters=5, angle=30, fd=0.0163)
    assert_allclose(K, 0.106920213333191)
    
    K = bend_rounded(Di=4.020, bend_diameters=5, angle=30, Re=1E5)
    assert_allclose(K, 0.11532121658742862)
    
    K = bend_rounded(Di=4.020, bend_diameters=5, angle=30, Re=1E5, method='Miller')
    assert_allclose(K, 0.10276501180879682)
    
    K = bend_rounded(Di=.5, bend_diameters=5, angle=30, Re=1E5, method='Crane')
    assert_allclose(K, 0.08959057097762159)
    
    K = bend_rounded(Di=.5, bend_diameters=5, angle=30, Re=1E5, method='Ito')
    assert_allclose(K, 0.10457946464978755)
    
    K = bend_rounded(Di=.5, bend_diameters=5, angle=30, Re=1E5, method='Swamee')
    assert_allclose(K, 0.055429466248839564)



def test_bend_miter():
    K_miters =  [bend_miter(i) for i in [150, 120, 90, 75, 60, 45, 30, 15]]
    K_miter_values = [2.7128147734758103, 2.0264994448555864, 1.2020815280171306, 0.8332188430731828, 0.5299999999999998, 0.30419633092708653, 0.15308822558050816, 0.06051389308126326]
    assert_allclose(K_miters, K_miter_values)
    
    K = bend_miter(Di=.6, angle=45, Re=1e6, roughness=1e-5, L_unimpeded=20, method='Miller')
    assert_allclose(K, 0.2944060416245167)
    
    K = bend_miter(Di=.05, angle=45, Re=1e6, roughness=1e-5, method='Crane')
    assert_allclose(K, 0.28597953150073047)
    
    K = bend_miter(angle=45, Re=1e6, method='Rennels')
    assert_allclose(K, 0.30419633092708653)
    
    with pytest.raises(Exception):
        bend_miter(angle=45, Re=1e6, method='BADMETHOD')



def test_bend_miter_Miller():
    K = bend_miter_Miller(Di=.6, angle=45, Re=1e6, roughness=1e-5, L_unimpeded=20)
    assert_allclose(K, 0.2944060416245167)
    K_default_L_unimpeded = bend_miter_Miller(Di=.6, angle=45, Re=1e6, roughness=1e-5)
    assert_allclose(K, K_default_L_unimpeded)
    
    
    K_high_angle = bend_miter_Miller(Di=.6, angle=120, Re=1e6, roughness=1e-5, L_unimpeded=20)
    K_higher_angle = bend_miter_Miller(Di=.6, angle=150, Re=1e6, roughness=1e-5, L_unimpeded=20)
    assert_allclose(K_high_angle, K_higher_angle)
    
    
@pytest.mark.slow
@pytest.mark.fuzz
def test_bend_rounded_Miller_fuzz():
    # Tested for quite a while without problems
    answers = []
    for i in range(500):
        Di = log_uniform(1e-5, 100)
        rc = uniform(0, 100)
        angle = uniform(0, 180) 
        Re = log_uniform(1e-5, 1E15)
        roughness = uniform(1e-10, Di*.95)
        L_unimpeded = log_uniform(1e-10, Di*1000)
        ans = bend_rounded_Miller(Di=Di, rc=rc, angle=angle, Re=Re, roughness=roughness, L_unimpeded=L_unimpeded)
        if np.isnan(ans) or np.isinf(ans):
            raise Exception
        answers.append(ans)
        
    assert min(answers) >= 0
    assert max(answers) < 1E10


@pytest.mark.slow
@pytest.mark.fuzz
def test_bend_miter_Miller_fuzz():
    # Tested for quite a while without problems
    answers = []
    for i in range(10**3):
        Di = log_uniform(1e-5, 100)
        angle = uniform(0, 120) 
        Re = log_uniform(1e-5, 1E15)
        roughness = uniform(1e-10, Di*.95)
        L_unimpeded = log_uniform(1e-10, Di*1000)
        ans = bend_miter_Miller(Di=Di, angle=angle, Re=Re, roughness=roughness, L_unimpeded=L_unimpeded)
        if np.isnan(ans) or np.isinf(ans):
            raise Exception
        answers.append(ans)
    assert min(answers) >= 0
    assert max(answers) < 1E10


### Diffusers


def test_diffuser_conical():
    K1 = diffuser_conical(Di1=.1**0.5, Di2=1, angle=10., fd=0.020)
    K2 = diffuser_conical(Di1=1/3., Di2=1, angle=50, fd=0.03) # 2
    K3 = diffuser_conical(Di1=2/3., Di2=1, angle=40, fd=0.03) # 3
    K4 = diffuser_conical(Di1=1/3., Di2=1, angle=120, fd=0.0185) # #4
    K5 = diffuser_conical(Di1=2/3., Di2=1, angle=120, fd=0.0185) # Last
    K6 = diffuser_conical(Di1=.1**0.5, Di2=1, l=3.908, fd=0.020)
    Ks = [0.12301652230915454, 0.8081340270019336, 0.32533470783539786, 0.812308728765127, 0.3282650135070033, 0.12300865396254032]
    assert_allclose([K1, K2, K3, K4, K5, K6], Ks)
    with pytest.raises(Exception):
        diffuser_conical(Di1=.1, Di2=0.1, angle=1800., fd=0.020)
    with pytest.raises(Exception):
        diffuser_conical(Di1=.1, Di2=0.1, fd=0.020)

    K1 = diffuser_conical_staged(Di1=1., Di2=10., DEs=[2,3,4,5,6,7,8,9], ls=[1,1,1,1,1,1,1,1,1], fd=0.01)
    K2 = diffuser_conical(Di1=1., Di2=10.,l=9, fd=0.01)
    Ks = [1.7681854713484308, 0.973137914861591]
    assert_allclose([K1, K2], Ks)
    
    # Idelchilk
    Ks_Idelchik = [diffuser_conical(Di1=.1**0.5, Di2=1, l=l,  method='Idelchik') for l in [.1, .5, 1, 2, 3, 4, 5, 20]]
    Ks_Idelchik_expect = [0.8617385829640242, 0.9283647028367953, 0.7082429168951839, 0.291016580744589, 0.18504484868875992, 0.147705693811332, 0.12911637682462676, 0.17]
    assert_allclose(Ks_Idelchik, Ks_Idelchik_expect, rtol=1e-2)




### Contractions

def test_contraction_conical_Crane():
    K2 = contraction_conical_Crane(Di1=0.0779, Di2=0.0525, l=0)
    assert_allclose(K2, 0.2729017979998056)
        


def test_contraction_round():
    K_round = contraction_round(Di1=1, Di2=0.4, rc=0.04)
    assert_allclose(K_round, 0.1783332490866574)

    K = contraction_round(Di1=1, Di2=0.4, rc=0.04, method='Miller')
    assert_allclose(K, 0.085659530512986387)

    K = contraction_round(Di1=1, Di2=0.4, rc=0.04, method='Idelchik')
    assert_allclose(K, 0.1008)
    
    with pytest.raises(Exception):
        contraction_round(Di1=1, Di2=0.4, rc=0.04, method='BADMETHOD')

def test_contraction_round_Miller():
    K = contraction_round_Miller(Di1=1, Di2=0.4, rc=0.04)
    assert_allclose(K, 0.085659530512986387)


def test_contraction_conical():
    K_conical1 = contraction_conical(Di1=0.1, Di2=0.04, l=0.04, fd=0.0185)
    K_conical2 = contraction_conical(Di1=0.1, Di2=0.04, angle=73.74, fd=0.0185)
    assert_allclose([K_conical1, K_conical2], [0.15779041548350314, 0.15779101784158286])
    
    with pytest.raises(Exception):
        contraction_conical(Di1=0.1, Di2=0.04, fd=0.0185)

    K = contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='Rennels')
    assert_allclose(K, 0.47462419839494946)
    
    K = contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='Idelchik')
    assert_allclose(K, 0.391723)
    
    K = contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='Crane')
    assert_allclose(K, 0.41815380146594)
    
    K = contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='Swamee')
    assert_allclose(K, 0.4479863925376303)
    
    K = contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='Blevins')
    assert_allclose(K, 0.365)
    
    K = contraction_conical(Di1=0.1, Di2=0.04, l=0.04, Re=1E6, method='Miller')
    assert_allclose(K, 0.0918289683812792)

    # high l ratio rounding    
    K = contraction_conical(Di1=0.1, Di2=0.06, l=0.04, Re=1E6, method='Miller')
    assert_allclose(K, 0.08651515699621345)

    # low a ratio rounding
    K = contraction_conical(Di1=0.1, Di2=0.099, l=0.04, Re=1E6, method='Miller')
    assert_allclose(K, 0.03065262382984957)
    
    # low l ratio
    K = contraction_conical(Di1=0.1, Di2=0.04, l=0.001, Re=1E6, method='Miller')
    assert_allclose(K, 0.5)
    
    # high l ratio rounding
    K = contraction_conical(Di1=0.1, Di2=0.05, l=1, Re=1E6, method='Miller')
    assert_allclose(K, 0.04497085709551787)
    
    with pytest.raises(Exception):
        contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='BADMETHOD')

### Valves

def test_valve_coefficients():
    Cv = Kv_to_Cv(2)
    assert_allclose(Cv, 2.3121984567073133)
    Kv = Cv_to_Kv(2.312)
    assert_allclose(Kv, 1.9998283393826013)
    K = Kv_to_K(2.312, .015)
    assert_allclose(K, 15.15337460039990)
    Kv = K_to_Kv(15.15337460039990, .015)
    assert_allclose(Kv, 2.312)
    
    # Two way conversions
    K = Cv_to_K(2.712, .015)
    assert_allclose(K, 14.719595348352552)
    assert_allclose(K, Kv_to_K(Cv_to_Kv(2.712), 0.015))
    
    Cv = K_to_Cv(14.719595348352552, .015)
    assert_allclose(Cv, 2.712)
    assert_allclose(Cv, Kv_to_Cv(K_to_Kv(14.719595348352552, 0.015)))
    
    # Code to generate the Kv Cv conversion factor
    # Round 1 trip; randomly assume Kv = 12, rho = 900; they can be anything
    # an tit still works
    dP = 1E5
    rho = 900.
    Kv = 12.
    Q = Kv/3600.
    D = .01
    V = Q/(pi/4*D**2)
    K = dP/(.5*rho*V*V)
    good_K = K

    def to_solve(x):
        from scipy.constants import gallon, minute, hour, psi
        conversion = gallon/minute*hour # from gpm to m^3/hr
        dP = 1*psi
        Cv = Kv*x*conversion
        Q = Cv/3600
        D = .01
        V = Q/(pi/4*D**2)
        K = dP/(.5*rho*V*V)
        return K - good_K
    
    from scipy.optimize import newton
    
    ans = newton(to_solve, 1.2)
    assert_allclose(ans, 1.1560992283536566)


def test_K_gate_valve_Crane():
    K = K_gate_valve_Crane(D1=.01, D2=.02, angle=45, fd=.015)
    assert_allclose(K, 14.548553268047963)
    
    K = K_gate_valve_Crane(D1=.1, D2=.1, angle=0, fd=.015)
    assert_allclose(K, 0.12)
    
    # non-smooth transition test
    K = K_gate_valve_Crane(D1=.1, D2=.146, angle=45, fd=.015)
    assert_allclose(K, 2.5577948931946746)
    K = K_gate_valve_Crane(D1=.1, D2=.146, angle=45.01, fd=.015)
    assert_allclose(K, 2.5719286772143595)
    
    K = K_gate_valve_Crane(D1=.1, D2=.146, angle=13.115)
    assert_allclose(K, 1.1466029421844073, rtol=1e-4)

def test_K_globe_valve_Crane():
    K =  K_globe_valve_Crane(.01, .02, fd=.015)
    assert_allclose(K, 87.1)
    
    assert_allclose(K_globe_valve_Crane(.01, .01, fd=.015), .015*340)
    
    K = K_globe_valve_Crane(.01, .02)
    assert_allclose(K, 135.9200548324305)
    
    
def test_K_angle_valve_Crane():
    K =  K_angle_valve_Crane(.01, .02, fd=.016)
    assert_allclose(K, 19.58)
    
    K = K_angle_valve_Crane(.01, .02, fd=.016, style=1)
    assert_allclose(K, 43.9)
    
    K = K_angle_valve_Crane(.01, .01, fd=.016, style=1)
    assert_allclose(K, 2.4)
    
    with pytest.raises(Exception):
        K_angle_valve_Crane(.01, .02, fd=.016, style=-1)
        
    K = K_angle_valve_Crane(.01, .02)
    assert_allclose(K, 26.597361811128465)
    
    
def test_K_swing_check_valve_Crane():
    K = K_swing_check_valve_Crane(D=.1, fd=.016)
    assert_allclose(K, 1.6)
    K = K_swing_check_valve_Crane(D=.1, fd=.016, angled=False)
    assert_allclose(K, 0.8)
    
    K = K_swing_check_valve_Crane(D=.02)
    assert_allclose(K, 2.3974274785373257)
    
    
def test_K_lift_check_valve_Crane():
    K = K_lift_check_valve_Crane(.01, .02, fd=.016)
    assert_allclose(K, 21.58)
    
    K = K_lift_check_valve_Crane(.01, .01, fd=.016)
    assert_allclose(K, 0.88)
    
    K = K_lift_check_valve_Crane(.01, .01, fd=.016, angled=False)
    assert_allclose(K, 9.6)
    
    K = K_lift_check_valve_Crane(.01, .02, fd=.016, angled=False)
    assert_allclose(K, 161.1)
    
    K = K_lift_check_valve_Crane(.01, .02)
    assert_allclose(K, 28.597361811128465)
    
    
def test_K_tilting_disk_check_valve_Crane():
    K = K_tilting_disk_check_valve_Crane(.01, 5, fd=.016)
    assert_allclose(K, 0.64)
    
    K = K_tilting_disk_check_valve_Crane(.25, 5, fd=.016)
    assert_allclose(K, .48)
    
    K = K_tilting_disk_check_valve_Crane(.9, 5, fd=.016)
    assert_allclose(K, 0.32)
    
    K = K_tilting_disk_check_valve_Crane(.01, 15, fd=.016)
    assert_allclose(K, 1.92)

    K = K_tilting_disk_check_valve_Crane(.25, 15, fd=.016)
    assert_allclose(K, 1.44)

    K = K_tilting_disk_check_valve_Crane(.9, 15, fd=.016)
    assert_allclose(K, 0.96)
    
    K =  K_tilting_disk_check_valve_Crane(.01, 5)
    assert_allclose(K, 1.1626516551826345)
    

def test_K_globe_stop_check_valve_Crane():
    K = K_globe_stop_check_valve_Crane(.1, .02, .0165)
    assert_allclose(K, 4.5225599999999995)
    
    K = K_globe_stop_check_valve_Crane(.1, .02, .0165, style=1)
    assert_allclose(K, 4.51992)
    
    K = K_globe_stop_check_valve_Crane(.1, .02, .0165, style=2)
    assert_allclose(K, 4.513452)
    
    with pytest.raises(Exception):
        K_globe_stop_check_valve_Crane(.1, .02, .0165, style=-1)
        
    K = K_globe_stop_check_valve_Crane(.1, .1, .0165)
    assert_allclose(K, 6.6)
    
    K = K_globe_stop_check_valve_Crane(.1, .02, style=1)
    assert_allclose(K, 4.5235076518969795)
        
        
def test_K_angle_stop_check_valve_Crane():
    K = K_angle_stop_check_valve_Crane(.1, .02, .0165)
    assert_allclose(K, 4.51728)
    
    K = K_angle_stop_check_valve_Crane(.1, .02, .0165, style=1)
    assert_allclose(K, 4.52124)
    
    K = K_angle_stop_check_valve_Crane(.1, .02, .0165, style=2)
    assert_allclose(K, 4.513452)

    with pytest.raises(Exception):
        K_angle_stop_check_valve_Crane(.1, .02, .0165, style=-1)
    
    K = K_angle_stop_check_valve_Crane(.1, .1, .0165)
    assert_allclose(K, 3.3)
    
    K =  K_angle_stop_check_valve_Crane(.1, .02, style=1)
    assert_allclose(K, 4.525425593879809)


def test_K_ball_valve_Crane():
    K = K_ball_valve_Crane(.01, .02, 50, .025)
    assert_allclose(K, 14.100545785228675)
    
    K = K_ball_valve_Crane(.01, .02, 40, .025)
    assert_allclose(K, 12.48666472974707)
    
    K = K_ball_valve_Crane(.01, .01, 0, .025)
    assert_allclose(K, 0.07500000000000001)
    
    K = K_ball_valve_Crane(.01, .02, 50)
    assert_allclose(K, 14.051310974926592)
    
    
def test_K_diaphragm_valve_Crane():
    K = K_diaphragm_valve_Crane(fd=0.015, style=0)
    assert_allclose(2.235, K)
    
    K = K_diaphragm_valve_Crane(fd=0.015, style=1)
    assert_allclose(K, 0.585)
    
    with pytest.raises(Exception):
        K_diaphragm_valve_Crane(fd=0.015, style=-1)
    
    K = K_diaphragm_valve_Crane(D=.1, style=0)
    assert_allclose(K, 2.4269804835982565)
    
    
def test_K_foot_valve_Crane():
    K = K_foot_valve_Crane(fd=0.015, style=0)
    assert_allclose(K, 6.3)
    
    K = K_foot_valve_Crane(fd=0.015, style=1)
    assert_allclose(K, 1.125)
    
    with pytest.raises(Exception):
        K_foot_valve_Crane(fd=0.015, style=-1)
        
    K = K_foot_valve_Crane(D=0.2, style=0)
    assert_allclose(K, 5.912221498436275)
        
        
def test_K_butterfly_valve_Crane():
    K = K_butterfly_valve_Crane(.1, 0.0165)
    assert_allclose(K, 0.7425)
    
    K = K_butterfly_valve_Crane(.3, 0.0165, style=1)
    assert_allclose(K, 0.8580000000000001)
    
    K = K_butterfly_valve_Crane(.6, 0.0165, style=2)
    assert_allclose(K, 0.9075000000000001)
    
    with pytest.raises(Exception):
        K_butterfly_valve_Crane(.6, 0.0165, style=-1)
        
    K = K_butterfly_valve_Crane(D=.1, style=2)
    assert_allclose(K, 3.5508841974793284)
        
        
def test_K_plug_valve_Crane():
    K = K_plug_valve_Crane(.01, .02, 50, .025)
    assert_allclose(K, 20.100545785228675)
    
    K = K_plug_valve_Crane(.01, .02, 50, .025, style=1)
    assert_allclose(K, 24.900545785228676)
    
    K = K_plug_valve_Crane(.01, .02, 50, .025, style=2)
    assert_allclose(K, 48.90054578522867)
    
    K = K_plug_valve_Crane(.01, .01, 50, .025, style=2)
    assert_allclose(K, 2.25)
    
    
    with pytest.raises(Exception):
        K_plug_valve_Crane(.01, .01, 50, .025, style=-1)
        
    K = K_plug_valve_Crane(D1=.01, D2=.02, angle=50)
    assert_allclose(K, 19.80513692341617)


def test_v_lift_valve_Crane():
    v = v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight')
    assert_allclose(v, 1.0252301935349286)
    
    v = v_lift_valve_Crane(rho=998.2, style='swing check angled')
    assert_allclose(v, 1.4243074011010037)
    

### Tees

def test_K_branch_converging_Crane():
    K = K_branch_converging_Crane(0.1023, 0.1023, 1135*liter/minute, 380*liter/minute, angle=90)
    assert_allclose(K, -0.04026, atol=.0001)
    
    K = K_branch_converging_Crane(0.1023, 0.05, 1135*liter/minute, 380*liter/minute, angle=90)
    assert_allclose(K, 0.9799379575823042)
    
    K = K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.0133)
    assert_allclose(K, 0.2644824555594152)
    
    K = K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.0133, angle=45)
    assert_allclose(K, 0.13231793346761025)
    
    
def test_K_run_converging_Crane():
    K = K_run_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633)
    assert_allclose(K, 0.32575847854551254)
    
    K =   K_run_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633, angle=30)
    assert_allclose(K, 0.32920396892611553)
    
    K = K_run_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633, angle=60)
    assert_allclose(K, 0.3757218131135227)
    
    
def test_K_branch_diverging_Crane():
    K = K_branch_diverging_Crane(0.146, 0.146, 1515*liter/minute, 950*liter/minute, angle=45)
    assert_allclose(K, 0.4640, atol=0.0001)
    
    K = K_branch_diverging_Crane(0.146, 0.146, 0.02525, 0.01583, angle=90)
    assert_allclose(K, 1.0910792393446236)
    
    K = K_branch_diverging_Crane(0.146, 0.07, 0.02525, 0.01583, angle=45)
    assert_allclose(K, 1.1950718299625727)
    
    K = K_branch_diverging_Crane(0.146, 0.07, 0.01425, 0.02283, angle=45)
    assert_allclose(K, 3.7281052908078762)
    
    K = K_branch_diverging_Crane(0.146, 0.146, 0.02525, 0.01983, angle=90)
    assert_allclose(K, 1.1194688533508077)
    
    # New test cases post-errata
    K = K_branch_diverging_Crane(0.146, 0.146, 0.02525, 0.04183, angle=45)
    assert_allclose(K, 0.30418565498014477)
    
    K = K_branch_diverging_Crane(0.146, 0.116, 0.02525, 0.01983, angle=90)
    assert_allclose(K, 1.1456727552755597)
    
    
def test_K_run_diverging_Crane():
    K = K_run_diverging_Crane(0.146, 0.146, 1515*liter/minute, 950*liter/minute, angle=45)
    assert_allclose(K, -0.06809, atol=.00001)
    
    K =  K_run_diverging_Crane(0.146, 0.146, 0.01025, 0.01983, angle=45)
    assert_allclose(K, 0.041523953539921235)
    
    K = K_run_diverging_Crane(0.146, 0.08, 0.02525, 0.01583, angle=90)
    assert_allclose(K, 0.0593965132275684)
    

