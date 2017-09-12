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
from fluids import *
from math import pi
from numpy.testing import assert_allclose
from scipy.constants import *
import pytest

def test_fittings():
    assert_allclose(entrance_sharp(), 0.57)

    K1 = entrance_distance(0.1, t=0.0005)
    assert_allclose(K1, 1.0154100000000004)
    
    assert_allclose(entrance_distance(Di=0.1, t=0.05), 0.57)
    
    assert_allclose(entrance_angled(30), 0.9798076211353316)

    K =  entrance_rounded(Di=0.1, rc=0.0235)
    assert_allclose(K, 0.09839534618360923)
    assert_allclose(entrance_rounded(Di=0.1, rc=0.2), 0.03)

    K = entrance_beveled(Di=0.1, l=0.003, angle=45)
    assert_allclose(K, 0.45086864221916984)
    
    K = entrance_beveled_orifice(Di=0.1, do=.07, l=0.003, angle=45)
    assert_allclose(K, 1.2987552913818574)

    ### Exits
    assert_allclose(exit_normal(), 1.0)

    ### Bends
    K_5_rc = [bend_rounded(Di=4.020, rc=4.0*5, angle=i, fd=0.0163) for i in [15, 30, 45, 60, 75, 90]]
    K_5_rc_values = [0.07038212630028828, 0.10680196344492195, 0.13858204974134541, 0.16977191374717754, 0.20114941557508642, 0.23248382866658507]
    assert_allclose(K_5_rc, K_5_rc_values)

    K_10_rc = [bend_rounded(Di=34.500, rc=36*10, angle=i, fd=0.0106) for i in [15, 30, 45, 60, 75, 90]]
    K_10_rc_values =  [0.061075866683922314, 0.10162621862720357, 0.14158887563243763, 0.18225270014527103, 0.22309967045081655, 0.26343782210280947]
    assert_allclose(K_10_rc, K_10_rc_values)

    K = bend_rounded(Di=4.020, bend_diameters=5, angle=30, fd=0.0163)
    assert_allclose(K, 0.106920213333191)

    K_miters =  [bend_miter(i) for i in [150, 120, 90, 75, 60, 45, 30, 15]]
    K_miter_values = [2.7128147734758103, 2.0264994448555864, 1.2020815280171306, 0.8332188430731828, 0.5299999999999998, 0.30419633092708653, 0.15308822558050816, 0.06051389308126326]
    assert_allclose(K_miters, K_miter_values)

    K_helix = helix(Di=0.01, rs=0.1, pitch=.03, N=10, fd=.0185)
    assert_allclose(K_helix, 14.525134924495514)

    K_spiral = spiral(Di=0.01, rmax=.1, rmin=.02, pitch=.01, fd=0.0185)
    assert_allclose(K_spiral, 7.950918552775473)

    ### Contractions
    K_sharp = contraction_sharp(Di1=1, Di2=0.4)
    assert_allclose(K_sharp, 0.5301269161591805)

    K_round = contraction_round(Di1=1, Di2=0.4, rc=0.04)
    assert_allclose(K_round, 0.1783332490866574)

    K_conical1 = contraction_conical(Di1=0.1, Di2=0.04, l=0.04, fd=0.0185)
    K_conical2 = contraction_conical(Di1=0.1, Di2=0.04, angle=73.74, fd=0.0185)
    assert_allclose([K_conical1, K_conical2], [0.15779041548350314, 0.15779101784158286])
    with pytest.raises(Exception):
        contraction_conical(Di1=0.1, Di2=0.04, fd=0.0185)

    K_beveled = contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120)
    assert_allclose(K_beveled, 0.40946469413070485)

    ### Expansions (diffusers)
    K_sharp = diffuser_sharp(Di1=.5, Di2=1)
    assert_allclose(K_sharp, 0.5625)

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


def test_K_globe_valve_Crane():
    K =  K_globe_valve_Crane(.01, .02, fd=.015)
    assert_allclose(K, 87.1)
    
    assert_allclose(K_globe_valve_Crane(.01, .01, fd=.015), .015*340)
    
    
def test_K_angle_valve_Crane():
    K =  K_angle_valve_Crane(.01, .02, fd=.016)
    assert_allclose(K, 19.58)
    
    K = K_angle_valve_Crane(.01, .02, fd=.016, style=1)
    assert_allclose(K, 43.9)
    
    K = K_angle_valve_Crane(.01, .01, fd=.016, style=1)
    assert_allclose(K, 2.4)
    
    with pytest.raises(Exception):
        K_angle_valve_Crane(.01, .02, fd=.016, style=-1)
    
    
def test_K_swing_check_valve_Crane():
    K = K_swing_check_valve_Crane(fd=.016)
    assert_allclose(K, 1.6)
    K = K_swing_check_valve_Crane(fd=.016, angled=False)
    assert_allclose(K, 0.8)
    
    
def test_K_lift_check_valve_Crane():
    K = K_lift_check_valve_Crane(.01, .02, fd=.016)
    assert_allclose(K, 21.58)
    
    K = K_lift_check_valve_Crane(.01, .01, fd=.016)
    assert_allclose(K, 0.88)
    
    K = K_lift_check_valve_Crane(.01, .01, fd=.016, angled=False)
    assert_allclose(K, 9.6)
    
    K = K_lift_check_valve_Crane(.01, .02, fd=.016, angled=False)
    assert_allclose(K, 161.1)
    
    
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


def test_K_ball_valve_Crane():
    K = K_ball_valve_Crane(.01, .02, 50, .025)
    assert_allclose(K, 14.100545785228675)
    
    K = K_ball_valve_Crane(.01, .02, 40, .025)
    assert_allclose(K, 12.48666472974707)
    
    K = K_ball_valve_Crane(.01, .01, 0, .025)
    assert_allclose(K, 0.07500000000000001)
    
    
def test_K_diaphragm_valve_Crane():
    K = K_diaphragm_valve_Crane(0.015, style=0)
    assert_allclose(2.235, K)
    
    K = K_diaphragm_valve_Crane(0.015, style=1)
    assert_allclose(K, 0.585)
    
    with pytest.raises(Exception):
        K_diaphragm_valve_Crane(0.015, style=-1)
    
    
def test_K_foot_valve_Crane():
    K = K_foot_valve_Crane(0.015, style=0)
    assert_allclose(K, 6.3)
    
    K = K_foot_valve_Crane(0.015, style=1)
    assert_allclose(K, 1.125)
    
    with pytest.raises(Exception):
        K_foot_valve_Crane(0.015, style=-1)
        
        
def test_K_butterfly_valve_Crane():
    K = K_butterfly_valve_Crane(.1, 0.0165)
    assert_allclose(K, 0.7425)
    
    K = K_butterfly_valve_Crane(.3, 0.0165, style=1)
    assert_allclose(K, 0.8580000000000001)
    
    K = K_butterfly_valve_Crane(.6, 0.0165, style=2)
    assert_allclose(K, 0.9075000000000001)
    
    with pytest.raises(Exception):
        K_butterfly_valve_Crane(.6, 0.0165, style=-1)
        
        
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
    

def test_v_lift_valve_Crane():
    v = v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight')
    assert_allclose(v, 1.0252301935349286)
    
    v = v_lift_valve_Crane(rho=998.2, style='swing check angled')
    assert_allclose(v, 1.4243074011010037)