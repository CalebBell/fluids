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
from fluids import *
from fluids.numerics import linspace, assert_close, assert_close1d
import pytest


def test_liquid_jet_pump_ancillary():
    # This equation has been checked from theory 2018-05-08 - it is
    # confirmed to be correct more than the large one!!!
    rhop=998.
    rhos=1098.
    Ks=0.11
    Kp=.04

    solution_vars = {'P1': 426256.1597041593,
     'P2': 133600,
     'Qp': 0.01,
     'Qs': 0.01,
     'd_mixing': 0.045,
     'd_nozzle': 0.022382858811037732}

    for key, value in solution_vars.items():
        kwargs = dict(solution_vars)
        del kwargs[key]
        new_value = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, **kwargs)
        assert_close(new_value, value)


@pytest.mark.slow
@pytest.mark.fuzz
def test_liquid_jet_pump_ancillary_rhos_Ks_Ps():
    for rhop in [998., 1050, 1150, 1250, 800]:
        for rhos in [1098., 1100, 1200, 1600, 4000, 100]:
            for Ks in [1E-9, 1E-3, 0.11, .5, 1, 5, 10, 100, 1000]:
                for Kp in [1E-9, 1E-3, 0.11, .5, 1, 5, 10, 100, 1000]:
                    for P_mult in [0.1, 0.5, 1, 2, 10]:
                        solution_vars = {'P1': 426256.1597041593,
                         'P2': 133600,
                         'Qp': 0.01,
                         'd_mixing': 0.045,
                         'd_nozzle': 0.022382858811037732}
                        solution_vars['P1'] *= P_mult
                        if solution_vars['P1'] < solution_vars['P2']:
                            continue

                        # Finish calculating good known values
                        solution_vars['Qs'] = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, **solution_vars)
                        if solution_vars['Qs'].imag:
                            # Do not keep testing if obtained an imaginary flow rate
                            continue
                        # Try each variable with the solver
                        for key, value in solution_vars.items():
                            kwargs = dict(solution_vars)
                            del kwargs[key]
                            new_value = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, **kwargs)
                            assert_close(new_value, value)

@pytest.mark.slow
@pytest.mark.fuzz
def test_liquid_jet_pump_ancillary_d_mixing():
    rhop=998.
    rhos=1098.
    Ks=0.11
    Kp=.04


    for rhos in [1098., 1100, 1200, 1600, 4000, 100]:
        for Ks in [1E-9, 1E-3, 0.11, .5, 1, 5, 10, 100, 1000]:
            for D_mult in linspace(0.1, 10, 100):
                solution_vars = {'P1': 426256.1597041593,
                 'P2': 133600,
                 'Qp': 0.01,
                 'd_mixing': 0.045,
                 'd_nozzle': 0.022382858811037732}
                solution_vars['d_mixing'] *= D_mult
                if solution_vars['d_mixing'] < solution_vars['d_nozzle']*1.43:
                    continue

                # Finish calculating good known values
                solution_vars['Qs'] = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, **solution_vars)
                if solution_vars['Qs'].imag:
                    # Do not keep testing if obtained an imaginary flow rate
                    continue
                # Try each variable with the solver
                for key, value in solution_vars.items():
                    kwargs = dict(solution_vars)
                    del kwargs[key]
            #         print(solution_vars, key)

                    new_value = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, **kwargs)
                    assert_close(new_value, value)


def validate_liquid_jet_pump(rhop, rhos, Ks, Kp, Km, Kd, nozzle_retracted,
                             solution_vars, d_diffuser=None, full=False):
    '''Helper function for testing `liquid_jet_pump`.
    Returns the number of solutions where the return values are the same as
    those given in `solution_vars`, and the number of cases where it is not.

    There is nothing wrong with getting a different answer; there are multiple
    solutions in the case of many variable sets.

    Raises an exception if a solution cannot be found.
    '''
    if full:
        all_solution_vars = dict(solution_vars)
        solution_vars = dict(solution_vars)
        del solution_vars['M']
        del solution_vars['N']
        del solution_vars['R']
        del solution_vars['alpha']
        del solution_vars['d_diffuser']
        del solution_vars['efficiency']


    same, different = 0, 0
    done = {}
    for i in solution_vars.keys():
        for j in solution_vars.keys():
            if i == j:
                continue
            elif frozenset([i, j]) in done:
                continue
            # Skip tests with alreardy tested variables; and where the two variables are the same

            kwargs = dict(solution_vars)
            del kwargs[i]
            del kwargs[j]
#             print('SOLVING FOR', i, j, kwargs) #
            ans = liquid_jet_pump(rhop=rhop, rhos=rhos, Ks=Ks, Kp=Kp, Km=Km, d_diffuser=d_diffuser, Kd=Kd, max_variations=10000, nozzle_retracted=nozzle_retracted, **kwargs)
#             print(i, j, ans[i], ans[j])
#             print('SOLVED, STARTING NEXT')
            try:
                for key, value in solution_vars.items():
                    assert_close(value, abs(ans[key]))
                same += 1
                # Since it matched, check the other parameters as well
                if full:
                    for key, value in all_solution_vars.items():
                        assert_close(value, abs(ans[key]))
            except:
                for key, value in ans.items():
                    # Had some issues with under zero values
                    assert value > 0

                different += 1
            done[frozenset([i, j])] = True
    return same, different


@pytest.mark.slow
def test_liquid_jet_pump_examples_round_robin():

    # Example one and two variants
    solution_vars = {'P1': 426256.1597041593,
     'P2': 133600,
     'P5': 200000.0,
     'Qp': 0.01,
     'Qs': 0.01,
     'd_mixing': 0.045,
     'd_nozzle': 0.022382858811037732}
    validate_liquid_jet_pump(rhop=998., rhos=1098., Ks=0.11, Kp=0.04, Km=.186, Kd=0.12, nozzle_retracted=False, solution_vars=solution_vars)

    solution_vars = {
     'P1': 468726.56966322445,
     'P2': 133600,
     'P5': 200000.0,
     'Qp': 0.01,
     'Qs': 0.001,
     'd_mixing': 0.0665377148831667,
     'd_nozzle': 0.022382858811037732}
    validate_liquid_jet_pump(rhop=998., rhos=1098., Ks=0.11, Kp=0.04, Km=.186, Kd=0.12, nozzle_retracted=False, solution_vars=solution_vars)
    solution_vars = {
     'P1': 426256.1597041593,
     'P2': 133600,
     'P5': 200000.0,
     'Qp': 0.1,
     'Qs': 0.0201,
     'd_mixing': 0.19926717348339726,
     'd_nozzle': 0.07320212423451278}
    validate_liquid_jet_pump(rhop=998., rhos=1098., Ks=0.11, Kp=0.04, Km=.186, Kd=0.12, nozzle_retracted=False, solution_vars=solution_vars)


    # Example 2
    solution_vars = {'P1': 550000.0,
     'P2': 170000.0,
     'P5': 192362.72123108635,
     'Qp': 0.0005588580085548165,
     'Qs': 0.0018975332068311196,
     'd_mixing': 0.024,
     'd_nozzle': 0.0048}

    validate_liquid_jet_pump(rhop=790.5, rhos=790.5, Km=.1, Kd=0.1, Ks=0.1, Kp=0.03, nozzle_retracted=False, solution_vars=solution_vars)


#@pytest.mark.slow
def test_liquid_jet_pump_examples_round_robin_Ex3():
    # Not worth testing - requires further work
    # Example 3
    rhop=765.0
    rhos=765.0
    Km=.15
    Kd=0.12
    Ks=0.38
    Kp=0.05
    nozzle_retracted=True
    d_diffuser=0.0318

    # point 5
    solution_vars = {
     'P1': 1000000.0,
     'P2': 47500.0,
     'P5': 109500.0,
     'Qp': 0.0005587193619566122,
     'Qs': 0.001400084261324908,
    'd_mixing': 0.017,
     'd_nozzle': 0.0038}
    same, different = validate_liquid_jet_pump(nozzle_retracted=nozzle_retracted, d_diffuser=d_diffuser,rhop=rhop, rhos=rhos, Km=Km, Kd=Kd, Ks=Ks, Kp=Kp, solution_vars=solution_vars)
    assert same > 10
    # Point 4
    solution_vars = {
     'P1': 800000.0,
     'P2': 46020.0,
     'P5': 95000.0,
     'Qp': 0.0004971366273938245,
     'Qs': 0.0012500084707104235,
     'd_mixing': 0.017,
     'd_nozzle': 0.0038}
    same, different = validate_liquid_jet_pump(nozzle_retracted=nozzle_retracted, d_diffuser=d_diffuser,rhop=rhop, rhos=rhos, Km=Km, Kd=Kd, Ks=Ks, Kp=Kp, solution_vars=solution_vars)
    assert same > 10

    # Custom point with full validation
    expected = {'M': 2.633280822186772,
                 'N': 0.06818823529411765,
                 'P1': 500000.0,
                 'P2': 46020.0,
                 'P5': 75000.0,
                 'Qp': 0.0003864114714478578,
                 'Qs': 0.0010175299172366153,
                 'R': 0.05097107567130409,
                 'alpha': 0.28014905021125414,
                 'd_diffuser': 0.0318,
                 'd_mixing': 0.016831456429424897,
                 'd_nozzle': 0.0038,
                 'efficiency': 0.1795587722987592}
    same, different = validate_liquid_jet_pump(nozzle_retracted=nozzle_retracted, d_diffuser=d_diffuser,rhop=rhop, rhos=rhos, Km=Km, Kd=Kd, Ks=Ks, Kp=Kp, solution_vars=expected, full=True)
    assert same > 15
del test_liquid_jet_pump_examples_round_robin_Ex3