# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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


### Contractions

def test_contraction_conical_Miller_coefficients():
    from fluids.fittings import contraction_conical_Miller_tck
    path = os.path.join(fluids_data_dir, 'Miller 2E 1990 conical contractions K part 1.csv')
    Kds, l_ratios, A_ratios = Engauge_2d_parser(open(path).readlines())
    path = os.path.join(fluids_data_dir, 'Miller 2E 1990 conical contractions K part 2.csv')
    Kd2, l_ratio2, A_ratio2 = Engauge_2d_parser(open(path).readlines())
    Kds.extend(Kd2)
    l_ratios.extend(l_ratio2)
    A_ratios.extend(A_ratio2)
    A_ratios = [[i+1.0 for i in j] for j in A_ratios]

    #    # The second set of data obviously looks terirble when plotted
    #    # Normally the data should be smoothed, but, well, the smoothing
    #    # function also requires smooth functions.
    #    for K, ls, As in zip(Kds, l_ratios, A_ratios):
    #        plt.loglog(ls, np.array(As)-1, label=str(K))
    #    plt.legend()
    #    plt.show()

    all_zs = []
    all_xs = []
    all_ys = []
    for z, xs, ys in zip(Kds, l_ratios, A_ratios):
        for x, y in zip(xs, ys):
            all_zs.append(z)
            all_xs.append(x)
            all_ys.append(y)

    tck = bisplrep(np.log(all_xs), np.log(all_ys), all_zs, kx=3, ky=1, s=.0001)
    [assert_allclose(i, j) for i, j in zip(contraction_conical_Miller_tck, tck)]

#    err = 0.0
#    n = 0
#    for z, xs, ys in zip(Kds, l_ratios, A_ratios):
#        for x, y in zip(xs, ys):
#            predict = bisplev(log(x), log(y), tck)
#            err += abs(predict - z)/z
#            n += 1
    # 5% relative error seems like the sweetspot
#    print(err/n, n, err)

#    import matplotlib.pyplot as plt
#    ax = plt.gca()
#    ax.set_xscale("log")
#    ax.set_yscale("log")
#    x = np.logspace(np.log10(.1), np.log10(10), 200)
#    y = np.logspace(np.log10(1.1), np.log10(4), 200)
#    X, Y = np.meshgrid(x, y, indexing='ij')
#    func = np.vectorize(lambda x, y: max(min(bisplev(log(x), log(y), tck), .5), 0))
#
#    Z = func(X.ravel(), Y.ravel())
#    Z = [[func(xi, yi) for yi in y.tolist()] for xi in x]
#
#    levels = [.001, .01, .03, .04, .05, .1, .2, .3, .4]
#    plt.contourf(X, Y-1, Z, levels=levels, cmap='RdGy')
#    plt.colorbar()
#    plt.show()

@pytest.mark.slow
def test_contraction_abrupt_Miller_coefficients():
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
    [assert_allclose(i, j, rtol=1e-2) for i, j in zip(tck_contraction_abrupt_Miller, tck_recalc)]

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

### Diffusers

@pytest.mark.slow
def test_diffuser_conical_Miller_coefficients():
    from fluids.fittings import tck_diffuser_conical_Miller
    path = os.path.join(fluids_data_dir, 'Miller 2E 1990 conical diffuser Kd.csv')
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
    ls_full = np.logspace(log10(0.1), log10(20.0))
    ls_stored = []
    As_stored = []
    for i, (K, ls, As) in enumerate(zip(Kds, l_ratios, A_ratios)):
#        plt.loglog(ls, As)
        univar = interp_objs[i]
        As_full = 10**univar(np.log10(ls_full))
    #     plt.loglog(ls_full, As_full)
    #     print(len(univar.get_coeffs()), len(univar.get_knots()))
        ls_smoothed = np.logspace(log10(ls[0]), log10(ls[-1]), 100)
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
    [assert_allclose(i, j, rtol=1e-2) for i, j in zip(tck_diffuser_conical_Miller, tck_recalc)]

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

### Entrances

def test_entrance_distance_Miller_coefficients():
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

def test_entrance_distance_45_Miller_coefficients():
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

def test_entrance_rounded_Miller_coefficients():
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


### Bends

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

@pytest.mark.slow
def test_bend_rounded_Miller_K_coefficients():
    from fluids import fluids_data_dir
    from fluids.core import Engauge_2d_parser
    from fluids.fittings import tck_bend_rounded_Miller
    Kb_curve_path = os.path.join(fluids_data_dir, 'Miller 2E 1990 smooth bends Kb.csv')
    lines = open(Kb_curve_path).readlines()
    all_zs, all_xs, all_ys = Engauge_2d_parser(lines, flat=True)

    tck_recalc = bisplrep(all_xs, all_ys, all_zs, kx=3, ky=3, s=.001)
    [assert_allclose(i, j) for i, j in zip(tck_bend_rounded_Miller, tck_recalc)]


@pytest.mark.slow
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


@pytest.mark.slow
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



def test_diffuser_conical_Idelchik_coefficients():
    from fluids.fittings import diffuser_conical_Idelchik_tck, diffuser_conical_Idelchik_angles, diffuser_conical_Idelchik_A_ratios, diffuser_conical_Idelchik_data

    diffuser_conical_Idelchik_obj = RectBivariateSpline(np.array(diffuser_conical_Idelchik_A_ratios),
                                                    np.array(diffuser_conical_Idelchik_angles),
                                                    np.array(diffuser_conical_Idelchik_data),
                                                    kx=3, ky=1)


    [assert_allclose(i, j) for i, j in zip(diffuser_conical_Idelchik_obj.tck, diffuser_conical_Idelchik_tck)]


def test_entrance_rounded_Idelchik_coeffs():
    from fluids.fittings import entrance_rounded_ratios_Idelchik, entrance_rounded_Ks_Idelchik, entrance_rounded_Idelchik_tck

    tck_refit = splrep(entrance_rounded_ratios_Idelchik, entrance_rounded_Ks_Idelchik, s=0, k=2)
    [assert_allclose(i, j, rtol=1e-3) for i, j in zip(tck_refit, entrance_rounded_Idelchik_tck)]
    #entrance_rounded_Idelchik = UnivariateSpline(entrance_rounded_ratios_Idelchik,
#                                             entrance_rounded_Ks_Idelchik,
#                                             s=0, k=2, ext=3)
#
def test_entrance_rounded_Harris_coeffs():
    from fluids.fittings import entrance_rounded_ratios_Harris, entrance_rounded_Ks_Harris, entrance_rounded_Harris_tck

    tck_refit = splrep(entrance_rounded_ratios_Harris, entrance_rounded_Ks_Harris, s=0, k=2)
    [assert_allclose(i, j, rtol=1e-3) for i, j in zip(tck_refit, entrance_rounded_Harris_tck)]


#entrance_rounded_Harris = UnivariateSpline(entrance_rounded_ratios_Harris,
#                                           entrance_rounded_Ks_Harris,
#                                           s=0, k=2, ext=3)

def test_entrance_distance_Harris_coeffs():
    from fluids.fittings import( entrance_distance_Harris_t_Di,
                                entrance_distance_Harris_Ks,
                                entrance_distance_Harris_tck)

    tck_refit = splrep(entrance_distance_Harris_t_Di, entrance_distance_Harris_Ks, s=0, k=3)
    [assert_allclose(i, j, rtol=1e-3) for i, j in zip(tck_refit, entrance_distance_Harris_tck)]
#entrance_distance_Harris_obj = UnivariateSpline(entrance_distance_Harris_t_Di,
#                                                entrance_distance_Harris_Ks,
#                                                s=0, k=3)
