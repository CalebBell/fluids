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
from math import pi, log10
from random import uniform
from numpy.testing import assert_allclose
from scipy.constants import *
from scipy.optimize import *
from scipy.interpolate import *
from fluids import fluids_data_dir
from fluids.core import Engauge_2d_parser
from fluids.optional.pychebfun import *

import pytest

def test_contraction_conical_Miller_coefficients():
    from fluids.fittings import tck_diffuser_conical_Miller
    path = os.path.join(fluids_data_dir, 'Miller 2E 1990 conical contraction Kd.csv')
    Kds, l_ratios, A_ratios = Engauge_2d_parser(open(path).readlines())
    # Fixup stupidity
    A_ratios = [[i+1.0 for i in j] for j in A_ratios]
#    for K, ls, As in zip(Kds, l_ratios, A_ratios):
#        plt.loglog(ls, np.array(As)-1)
#    plt.show()

    interp_objs = []
    for K, ls, As in zip(Kds, l_ratios, A_ratios):
        univar = UnivariateSpline(np.log10(ls), np.log10(As), s=4e-5)
        interp_objs.append(univar)
    
    # Extrapolation to the left and right looks bad
    # Extrapolation upwards looks bad too
    ls_full = np.logspace(np.log10(0.1), np.log10(20))
    ls_stored = []
    As_stored = []
    for i, (K, ls, As) in enumerate(zip(Kds, l_ratios, A_ratios)):
#        plt.loglog(ls, As)
        univar = interp_objs[i]
        As_full = 10**univar(np.log10(ls_full))
    #     plt.loglog(ls_full, As_full)
    #     print(len(univar.get_coeffs()), len(univar.get_knots()))
        ls_smoothed = np.logspace(np.log10(ls[0]), np.log10(ls[-1]), 100)
        As_smoothed = 10**univar(np.log10(ls_smoothed))
    #     plt.loglog(ls_smoothed, As_smoothed)
        ls_stored.append(ls_smoothed)
        As_stored.append(As_smoothed)
    
    # plt.show()
    all_zs = []
    all_xs = []
    all_ys = []
    for z, xs, ys in zip(Kds, ls_stored, As_stored):
        for x, y in zip(xs, ys):
            all_zs.append(z)
            all_xs.append(x)
            all_ys.append(y)
    
    tck_recalc = bisplrep(np.log(all_xs), np.log(all_ys), all_zs, s=.002)
    [assert_allclose(i, j) for i, j in zip(tck_diffuser_conical_Miller, tck_recalc)]

    # Plotting code to re-create the graph through solving for points
#    print([len(i) for i in tck[0:3]])
#    
#    for K, ls in zip(Kds, ls_stored):
#        def get_right_y(l, K_goal):
#            try:
#                def err(y_guess):
#                    if y_guess <= 1.1:
#                        y_guess = 1.1
#                    if y_guess > 4:
#                        y_guess = 4
#                    return bisplev(log(l), log(y_guess), tck) - K_goal
#    #             ans = newton(err, 1.3)
#                ans = bisect(err, 1.1, 4)
#                
#    #             if abs(err(ans)) > .1:
#    #                 ans = None
#                return ans
#            except:
#                return None
#        As_needed = [get_right_y(l, K) for l in ls]
#        plt.loglog(ls, As_needed, 'x')
#    plt.show()
    
def test_contraction_conical_Crane():
    K2 = contraction_conical_Crane(Di1=0.0779, Di2=0.0525, l=0)
    assert_allclose(K2, 0.2729017979998056)
        
def test_entrance_distance_Miller_coeffs():
    from fluids.fittings import entrance_distance_Miller_coeffs
    t_ds = [0.006304, 0.007586, 0.009296, 0.011292, 0.013288, 0.015284, 0.019565, 0.022135, 0.024991, 0.02842, 0.032136, 0.036426, 0.040145, 0.043149, 0.048446, 0.054745, 0.061332, 0.067919, 0.075081, 0.081957, 0.089121, 0.096284, 0.099722, 0.106886, 0.110897, 0.118061, 0.125224, 0.132101, 0.139264, 0.147, 0.153877, 0.16104, 0.167917, 0.175081, 0.181957, 0.189121, 0.196284, 0.199723, 0.206886, 0.214049, 0.221213, 0.228376, 0.235539, 0.242416, 0.249579, 0.250726, 0.257889, 0.264766, 0.271929, 0.279093, 0.286256, 0.293419, 0.300009]
    Ks = [1.00003, 0.97655, 0.94239, 0.90824, 0.87408, 0.83993, 0.78301, 0.75028, 0.71756, 0.68626, 0.65638, 0.62793, 0.6066, 0.59166, 0.57532, 0.56111, 0.54833, 0.5384, 0.53416, 0.53135, 0.53138, 0.53142, 0.53143, 0.53147, 0.53149, 0.53152, 0.53156, 0.53159, 0.53162, 0.53023, 0.53027, 0.5303, 0.53033, 0.53179, 0.5304, 0.53186, 0.53189, 0.53191, 0.53194, 0.53198, 0.53201, 0.53347, 0.53208, 0.53353, 0.53215, 0.53215, 0.53218, 0.53364, 0.53367, 0.53371, 0.53374, 0.53378, 0.5331]
    # plt.plot(t_ds, Ks)
    t_ds2 = np.linspace(t_ds[0], t_ds[-1], 1000)
#    Ks_Rennels = [entrance_distance(Di=1, t=t) for t in t_ds2]
    # plt.plot(t_ds2, Ks_Rennels)
    # plt.show()
    
    obj = UnivariateSpline(t_ds, Ks, s=3e-5)
    # print(len(obj.get_coeffs()), len(obj.get_knots()))
    # plt.plot(t_ds2, obj(t_ds2))
    
    fun = chebfun(f=obj, domain=[0,.3], N=15)
    coeffs = chebfun_to_poly(fun, text=False)
    assert_allclose(coeffs, entrance_distance_Miller_coeffs)

def test_entrance_distance_45_Miller():
    from fluids.fittings import entrance_distance_45_Miller
    K = entrance_distance_45_Miller(Di=0.1, Di0=0.14)
    assert_allclose(K, 0.24407641818143339)

def test_entrance_distance_45_Miller_coeffs():
    from fluids.fittings import entrance_distance_45_Miller_coeffs
    t_ds_re_entrant_45 = [0.006375, 0.007586, 0.009296, 0.011292, 0.013288, 0.015284, 0.019565, 0.022135, 0.024991, 0.02842, 0.032136, 0.036426, 0.040109, 0.043328, 0.046868, 0.048443, 0.053379, 0.053594, 0.059318, 0.059855, 0.065044, 0.068836, 0.070768, 0.07678, 0.082793, 0.088805, 0.089663, 0.095963, 0.104267, 0.110566, 0.116866, 0.123451, 0.129751, 0.136337, 0.142637, 0.146933, 0.153807, 0.160394, 0.167268, 0.174143, 0.181018, 0.187893, 0.194769, 0.199927, 0.20709, 0.213966, 0.221129, 0.228292, 0.235455, 0.242332, 0.249495, 0.250641, 0.257804, 0.264967, 0.27213, 0.279006, 0.286169, 0.293333, 0.299815]
    Ks_re_entrant_45 = [1.0, 0.97655, 0.94239, 0.90824, 0.87408, 0.83993, 0.78301, 0.75028, 0.71756, 0.68626, 0.65638, 0.62793, 0.60642, 0.59113, 0.57033, 0.56535, 0.54225, 0.54403, 0.52128, 0.52003, 0.5028, 0.48752, 0.48147, 0.463, 0.44737, 0.42889, 0.4232, 0.41184, 0.39053, 0.3749, 0.3607, 0.34507, 0.33086, 0.31666, 0.30388, 0.29678, 0.28685, 0.27549, 0.26699, 0.25848, 0.25282, 0.24715, 0.24434, 0.24437, 0.24298, 0.24158, 0.2402, 0.24023, 0.23884, 0.23745, 0.23606, 0.23606, 0.2361, 0.23329, 0.23332, 0.23193, 0.23054, 0.23057, 0.22989]
#    plt.plot(t_ds_re_entrant_45, Ks_re_entrant_45)
    
    obj = UnivariateSpline(t_ds_re_entrant_45, Ks_re_entrant_45, s=1e-4)
    t_ds_re_entrant_45_long = np.linspace(0, 0.3, 1000)
#    plt.plot(t_ds_re_entrant_45_long, obj(t_ds_re_entrant_45_long))
    
    fun = chebfun(f=obj, domain=[0,.3], N=15)
    
#    plt.plot(t_ds_re_entrant_45_long, fun(t_ds_re_entrant_45_long), '--')
#    plt.show()
    
    coeffs = chebfun_to_poly(fun)
    assert_allclose(coeffs, entrance_distance_45_Miller_coeffs)


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
    
    K = entrance_rounded(Di=0.1, rc=0.01, method='Idelchik')
    assert_allclose(K, 0.11328005177738182)
    
    with pytest.raises(Exception):
        entrance_rounded(Di=0.1, rc=0.01, method='BADMETHOD')

def test_entrance_beveled():
    K = entrance_beveled(Di=0.1, l=0.003, angle=45)
    assert_allclose(K, 0.45086864221916984)

    K = entrance_beveled(Di=0.1, l=0.003, angle=45, method='Idelchik')
    assert_allclose(K, 0.3995000000000001)

def test_entrance_rounded_Miller_coeffs():
    from fluids.fittings import entrance_rounded_Miller_coeffs
    path = os.path.join(fluids_data_dir, 'Miller 2E 1990 entrances rounded beveled K.csv')
    lines = open(path).readlines()
    _, ratios, Ks = Engauge_2d_parser(lines)
    ratios_45, ratios_30, ratios_round = ratios
    Ks_45, Ks_30, Ks_round = Ks
    
#    plt.plot(ratios_round, Ks_round)
    t_ds2 = np.linspace(ratios_round[0], ratios_round[1], 1000)
#    Ks_Rennels = [entrance_rounded(Di=1, rc=t) for t in t_ds2]
#    plt.plot(t_ds2, Ks_Rennels)
    obj = UnivariateSpline(ratios_round, Ks_round, s=6e-5)
#    plt.plot(t_ds2, obj(t_ds2))
    fun = chebfun(f=obj, domain=[0,.3], N=8)    
#    plt.plot(t_ds2, fun(t_ds2), '--')
#    plt.show()
    coeffs = chebfun_to_poly(fun)
    assert_allclose(coeffs, entrance_rounded_Miller_coeffs)





def test_fittings():
    assert_allclose(entrance_sharp(), 0.57)
    
    assert_allclose(entrance_angled(30), 0.9798076211353316)

    
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
        

def test_bend_rounded_Crane():
    K = bend_rounded_Crane(Di=.4020, rc=.4*5, angle=30)
    assert_allclose(K, 0.09321910015613409)
    
    K_max = bend_rounded_Crane(Di=.400, rc=.4*25, angle=30)
    K_limit = bend_rounded_Crane(Di=.400, rc=.4*20, angle=30)
    assert_allclose(K_max, K_limit)

def test_bend_rounded_Crane_coefficients(): 
    from fluids.fittings import bend_rounded_Crane_ratios, bend_rounded_Crane_fds, bend_rounded_Crane_coeffs
    bend_rounded_Crane_obj = UnivariateSpline(bend_rounded_Crane_ratios, bend_rounded_Crane_fds, s=0)

    fun = chebfun(f=bend_rounded_Crane_obj, domain=[1,20], N=10)
    coeffs = chebfun_to_poly(fun)
    assert_allclose(coeffs, bend_rounded_Crane_coeffs)
    
    xs = np.linspace(1, 20, 2000)
    diffs = (abs(fun(xs)-bend_rounded_Crane_obj(xs))/bend_rounded_Crane_obj(xs))
    assert np.max(diffs) < .02
    assert np.mean(diffs) < .002


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




def test_bend_rounded_Miller_K():
    from fluids import fluids_data_dir
    from fluids.core import Engauge_2d_parser
    from fluids.fittings import tck_bend_rounded_Miller
    Kb_curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 smooth bends Kb.csv')
    lines = open(Kb_curve_path).readlines()
    all_zs, all_xs, all_ys = Engauge_2d_parser(lines, flat=True)
    
    tck_recalc = bisplrep(all_xs, all_ys, all_zs, kx=3, ky=3, s=.001)
    [assert_allclose(i, j) for i, j in zip(tck_bend_rounded_Miller, tck_recalc)]
    
    
def test_bend_rounded_Miller_Re_correction():
    from fluids import fluids_data_dir
    from fluids.core import Engauge_2d_parser
    from fluids.fittings import tck_bend_rounded_Miller_C_Re
    Re_curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 smooth bends Re correction.csv')
    text = open(Re_curve_path).readlines()
    rds, Re_lists, C_lists = Engauge_2d_parser(text)
    
    inter_objs = []
    for rd, Res, Cs in zip(rds, Re_lists, C_lists):
        univar = UnivariateSpline(np.log10(Res), Cs) # Default smoothing is great!
        inter_objs.append(univar)
        
    for i, (rd, Res, Cs) in enumerate(zip(rds, Re_lists, C_lists)):
    #     plt.semilogx(Res, Cs)
        univar = inter_objs[i]
        Cs_smoothed = univar(np.log10(Res))
    #     plt.semilogx(Res, Cs_smoothed)
    #     print(univar.get_coeffs(), univar.get_knots())
    # plt.show()
    
    # make a rectangular grid
    Res = np.logspace(np.log10(1E4), np.log10(1E8), 100)
    Cs_stored = []
    for obj in inter_objs:
        Cs_smoothed = obj(np.log10(Res))
#        plt.semilogx(Res, Cs_smoothed)
        Cs_stored.append(Cs_smoothed)
#    plt.show()

    # Flatten the data to the form used in creating the spline
    all_zs = []
    all_xs = []
    all_ys = []
    for z, x, ys in zip(rds, Res, Cs_stored):
        for x, y in zip(Res, ys):
            all_zs.append(z)
            all_xs.append(x)
            all_ys.append(y)

    tck_recalc = bisplrep(np.log10(all_xs), all_zs, all_ys)
    [assert_allclose(i, j) for i, j in zip(tck_bend_rounded_Miller_C_Re, tck_recalc)]
    
    spline_obj = lambda Re, r_D : bisplev(np.log10(Re), r_D, tck_recalc)
    Res = np.logspace(np.log10(1E4), np.log10(1E8), 100)
    for obj, r_d in zip(inter_objs, rds):
        Cs_smoothed = obj(np.log10(Res))
#        plt.semilogx(Res, Cs_smoothed)
    #     Cs_spline = spline_obj(Res, r_d)
    #     plt.semilogx(Res, Cs_spline, '--')
    for r in np.linspace(1, 2, 10):
        Cs_spline = spline_obj(Res, r)
#        plt.semilogx(Res, Cs_spline, '-')
#    plt.show()

    from fluids.fittings import bend_rounded_Miller_C_Re_limit_1
    from fluids.fittings import bend_rounded_Miller_C_Re
    ps = np.linspace(1, 2)
    qs = [newton(lambda x: bend_rounded_Miller_C_Re(x, i)-1, 2e5) for i in ps]
    rs = np.polyfit(ps, qs, 4).tolist()
    assert_allclose(rs, bend_rounded_Miller_C_Re_limit_1)



def test_bend_rounded_Miller_outlet_tangent_correction():
    from fluids.fittings import tck_bend_rounded_Miller_C_Re
    Re_curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 smooth bends outlet tangent length correction.csv')
    text = open(Re_curve_path).readlines()
    
    Kbs, length_ratio_lists, Co_lists = Engauge_2d_parser(text)
    
    def BioScience_GeneralizedSubstrateDepletion_model(x_in):
        '''Fit created using zunzun.com, comparing the non-linear,
        non-logarithmic plot values with pixel positions on the graph.
        
        0	0.00
        1	311
        2	493
        4	721
        6	872
        10	1074
        20	1365
        30	1641
        40	1661
        '''
        temp = 0.0
        a = 1.0796070184265327E+03
        b = 2.7557612059844967E+00
        c = -2.1529870432577212E+01
        d = 4.1229208061974096E-03
        temp = (a * x_in) / (b + x_in) - (c * x_in) - d
        return temp
    
    def fix(y):
        # Reverse the plot
        # Convert input "y" to between 0 and 1661
        y = y/30 # 0-1 linear
        y *= 1641 # to max
        err = lambda x: BioScience_GeneralizedSubstrateDepletion_model(x) - y
        return float(fsolve(err, 1))

    for values in length_ratio_lists:
        for i in range(len(values)):
            x = min(values[i], 30) # Do not allow values over 30
            values[i] = fix(x)

    
#     Plotting code
#    inter_objs = []
#    for Kb, lrs, Cos in zip(Kbs, length_ratio_lists, Co_lists):
#        univar = UnivariateSpline(lrs, Cos, s=4e-4) # Default smoothing is great!
#        inter_objs.append(univar)
#    for i, (Kb, lrs, Cos) in enumerate(zip(Kbs, length_ratio_lists, Co_lists)):
#        plt.semilogx(lrs, Cos, 'x')
#        univar = inter_objs[i]
#        Cs_smoothed = univar(lrs)
#        plt.semilogx(lrs, Cs_smoothed)
    # plt.ylim([0.3, 3])
    # plt.xlim([0.1, 30])
    # plt.show()

#   Code to literally write the code
    min_vals = []
    tcks = []
    for Kb, lrs, Cos in zip(Kbs, length_ratio_lists, Co_lists):
        univar = splrep(lrs, Cos, s=4e-4) # Default smoothing is great!
        s = ('tck_bend_rounded_Miller_C_o_%s = ' %str(Kb).replace('.', '_'))
        template = 'np.array(%s),\n'
        t1 = template%str(univar[0].tolist())
        t2 = template%str(univar[1].tolist())
        s = s + '[%s%s3]' %(t1, t2)
#        print(s)
        min_vals.append(float(splev(0.01, univar)))
        tcks.append(univar)
    
    # Check the fixed constants above the function
    from fluids.fittings import tck_bend_rounded_Miller_C_os
    for tck, tck_recalc in zip(tck_bend_rounded_Miller_C_os, tcks):
        [assert_allclose(i, j) for i, j in zip(tck, tck_recalc)]


    from fluids.fittings import bend_rounded_Miller_C_o_limit_0_01
    assert_allclose(min_vals, bend_rounded_Miller_C_o_limit_0_01)
    
    from fluids.fittings import bend_rounded_Miller_C_o_limits
    max_ratios = [i[-1] for i in length_ratio_lists]
    assert_allclose(max_ratios, bend_rounded_Miller_C_o_limits)



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



def log_uniform(low, high):
    return 10**uniform(log10(low), log10(high))


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


def test_bend_miter_Miller_coefficients():
    from fluids.optional.pychebfun import chebfun, chebfun_to_poly
    curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 Kb mitre bend.csv')
    text = open(curve_path).readlines()
    zs, x_lists, y_lists = Engauge_2d_parser(text)
    x_raw, y_raw = x_lists[0], y_lists[0]
    univar = UnivariateSpline(x_raw, y_raw, s=1e-4)
    fun = chebfun(f=univar, domain=[0,120], N=15) # 15 max for many coeffs
    
    recalc_coeffs = chebfun_to_poly(fun)
    from fluids.fittings import bend_miter_Miller_coeffs
    assert_allclose(bend_miter_Miller_coeffs, recalc_coeffs)



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
    
    with pytest.raises(Exception):
        contraction_conical(Di1=0.1, Di2=.04, l=.004, Re=1E6, method='BADMETHOD')
        
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
    
    
def test_contraction_abrupt_Miller_data():
    from fluids.fittings import tck_contraction_abrupt_Miller
    curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 abrupt contraction K.csv')
    text = open(curve_path).readlines()
    
    zs, x_lists, y_lists = Engauge_2d_parser(text)
    for xs, values in zip(x_lists, y_lists):
        values[-1] = 0
        low = 1e-8
        for i in range(2):
            low = low/10
            values.insert(-1, low)
            xs.insert(-1, 1-low)
        xs[-1] = 1

    inter_objs = []
    for rd, As, Ks in zip(zs, x_lists, y_lists):
        univar = UnivariateSpline(As, Ks, s=1e-5)
        inter_objs.append(univar)
    
    # make a rectangular grid
    As = np.linspace(0, 1, 1000)
    Ks_stored = []
    for obj in inter_objs:
        Ks_smoothed = obj(As)
        Ks_smoothed[Ks_smoothed < 0] = 0 # Avoid zeros
        Ks_stored.append(Ks_smoothed)
    
    # Flatten the data to the form used in creating the spline
    all_zs = []
    all_xs = []
    all_ys = []
    for z, x, ys in zip(zs, As, Ks_stored):
        for x, y in zip(As, ys):
            all_zs.append(z)
            all_xs.append(x)
            all_ys.append(y)
    tck_recalc = bisplrep(all_xs, all_zs, all_ys, s=5e-4)
    [assert_allclose(i, j) for i, j in zip(tck_contraction_abrupt_Miller, tck_recalc)]
    
#   Plotting code
#     print([i.tolist() for i in tck[:3]])
#    for i, (rd, As, Ks) in enumerate(zip(zs, x_lists, y_lists)):
#        plt.plot(As, Ks, '.')
#        univar = inter_objs[i]
#        As2 = np.linspace(0, 1, 1000)
#        Ks_smoothed = univar(As2)
#        plt.plot(As2, Ks_smoothed)
#        # Compute with the spline
#        Ks_new = bisplev(As2, rd, tck)
#        plt.plot(As2, Ks_new)
#    for rd in np.linspace(.1, 0, 100):
#        As2 = np.linspace(0, 1, 1000)
#        Ks_new = bisplev(As2, rd, tck)
#        plt.plot(As2, Ks_new)
#    plt.show()
