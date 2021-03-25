# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import log10, log, exp, isnan, pi
from fluids.numerics import secant, logspace, linspace, assert_close, assert_close1d, assert_close2d, assert_close3d
from fluids import *
import pytest
from fluids.friction import _roughness, _Farshad_roughness

try:
    import fuzzywuzzy
    has_fuzzywuzzy = True
except:
    has_fuzzywuzzy = False

try:
    import mpmath
    has_mpmath = True
except:
    has_mpmath = False

def test_friction_basic():
    assert_close(Moody(1E5, 1E-4), 0.01809185666808665)
    assert_close(Alshul_1952(1E5, 1E-4), 0.018382997825686878)
    assert_close(Wood_1966(1E5, 1E-4), 0.021587570560090762)
    assert_close(Churchill_1973(1E5, 1E-4), 0.01846708694482294)
    assert_close(Eck_1973(1E5, 1E-4), 0.01775666973488564)
    assert_close(Jain_1976(1E5, 1E-4), 0.018436560312693327)
    assert_close(Swamee_Jain_1976(1E5, 1E-4), 0.018452424431901808)
    assert_close(Churchill_1977(1E5, 1E-4), 0.018462624566280075)
    assert_close(Chen_1979(1E5, 1E-4), 0.018552817507472126)
    assert_close(Round_1980(1E5, 1E-4), 0.01831475391244354)
    assert_close(Shacham_1980(1E5, 1E-4), 0.01860641215097828)
    assert_close(Barr_1981(1E5, 1E-4), 0.01849836032779929)
    assert_close(Zigrang_Sylvester_1(1E5, 1E-4), 0.018646892425980794)
    assert_close(Zigrang_Sylvester_2(1E5, 1E-4), 0.01850021312358548)
    assert_close(Haaland(1E5, 1E-4), 0.018265053014793857)
    assert_close(Serghides_1(1E5, 1E-4), 0.01851358983180063)
    assert_close(Serghides_2(1E5, 1E-4), 0.018486377560664482)
    assert_close(Tsal_1989(1E5, 1E-4), 0.018382997825686878)
    assert_close(Tsal_1989(1E8, 1E-4), 0.012165854627780102)
    assert_close(Manadilli_1997(1E5, 1E-4), 0.01856964649724108)
    assert_close(Romeo_2002(1E5, 1E-4), 0.018530291219676177)
    assert_close(Sonnad_Goudar_2006(1E5, 1E-4), 0.0185971269898162)
    assert_close(Rao_Kumar_2007(1E5, 1E-4), 0.01197759334600925)
    assert_close(Buzzelli_2008(1E5, 1E-4), 0.018513948401365277)
    assert_close(Avci_Karagoz_2009(1E5, 1E-4), 0.01857058061066499)
    assert_close(Papaevangelo_2010(1E5, 1E-4), 0.015685600818488177)
    assert_close(Brkic_2011_1(1E5, 1E-4), 0.01812455874141297)
    assert_close(Brkic_2011_2(1E5, 1E-4), 0.018619745410688716)
    assert_close(Fang_2011(1E5, 1E-4), 0.018481390682985432)
    assert_close(Clamond(1E5, 1E-4), 0.01851386607747165)
    assert_close(Clamond(1E5, 1E-4, fast=True), 0.01851486771096876)

    assert_close(friction_laminar(128), 0.5)

    assert_close(Blasius(10000.0), 0.03164)

    fd = ft_Crane(.1)
    assert_close(fd, 0.01628845962146481)
    assert_close(ft_Crane(1e-5), 604.8402578042682)

def test_friction():
    assert_close(sum(_roughness.values()), 0.01504508)


    assert_close(friction_factor(Re=1E5, eD=1E-4), 0.01851386607747165)
    methods_1 = friction_factor_methods(Re=1E5, eD=1E-4)
    methods_1.sort()

    methods_2 = ['Clamond', 'Colebrook', 'Manadilli_1997', 'Haaland', 'Alshul_1952', 'Avci_Karagoz_2009', 'Rao_Kumar_2007', 'Zigrang_Sylvester_2', 'Eck_1973', 'Buzzelli_2008', 'Tsal_1989', 'Papaevangelo_2010', 'Barr_1981', 'Jain_1976', 'Moody', 'Brkic_2011_2', 'Brkic_2011_1', 'Swamee_Jain_1976', 'Wood_1966', 'Shacham_1980', 'Romeo_2002', 'Chen_1979', 'Fang_2011', 'Round_1980', 'Sonnad_Goudar_2006', 'Churchill_1973', 'Churchill_1977', 'Serghides_2', 'Serghides_1', 'Zigrang_Sylvester_1']
    methods_2.sort()
    assert methods_1 == methods_2

    assert_close(friction_factor(Re=1E5, eD=1E-4, Darcy=False), 0.01851386607747165/4)
    assert_close(friction_factor(Re=128), 0.5)

    assert_close(friction_factor(Re=1E5, eD=0, Method=None), 0.01798977308427384)
    assert_close(friction_factor(20000, eD=0.0, Method='laminar'), 0.0032)

    with pytest.raises(ValueError):
        friction_factor(Re=1E5, eD=0, Method='BADMETHOD')

    assert ['laminar'] == friction_factor_methods(200, 0, True)
    assert 31 == len(friction_factor_methods(200, 0, False))

    for m in friction_factor_methods(200, 0, False):
        friction_factor(Re=1E5, eD=1e-6, Method=m)

    fd = ft_Crane(.1)
    Di = 0.1
    fd_act = Clamond(7.5E6*Di, eD=roughness_Farshad(ID='Carbon steel, bare', D=Di)/Di)
    assert_close(fd, fd_act, rtol=5e-6)

def test_friction_Colebrook():
    assert_close(Colebrook(1E5, 1E-4), 0.018513866077471648)

    # Test the colebrook is the clamond when tol=-1
    assert Colebrook(1E5, 1E-4, -1) == Clamond(1E5, 1E-4)
    # Test the colebrook is the analytical solution when Re < 10
    # even when the clamond solution is specified
    assert Colebrook(1, 1E-4, -1) == Colebrook(1, 1e-4)


@pytest.mark.slow
@pytest.mark.mpmath
@pytest.mark.skipif(not has_mpmath, reason='mpmath is not installed')
def test_Colebrook_numerical_mpmath():
    # tested at n=500 for both Re and eD
    Res = logspace(log10(1e-12), log10(1E12), 30) # 1E12 is too large for sympy - it slows down too much
    eDs = logspace(log10(1e-20), log10(.1), 21) # 1-1e-9
    for Re in Res:
        for eD in eDs:
            fd_exact = Colebrook(Re, eD, tol=0)
            fd_numerical = Colebrook(Re, eD, tol=1e-12)
            assert_close(fd_exact, fd_numerical, rtol=1e-5)

@pytest.mark.slow
@pytest.mark.mpmath
@pytest.mark.skipif(not has_mpmath, reason='mpmath is not installed')
def test_Colebrook_scipy_mpmath():
    # Faily grueling test - check the lambertw implementations are matching mostly
    # NOTE the test is to Re = 1E7; at higher Res the numerical solver is almost
    # always used
    Res = logspace(log10(1e-12), log10(1e7), 20) # 1E12 is too large for sympy
    eDs = logspace(log10(1e-20), log10(.1), 19) # 1-1e-9

    for Re in Res:
        for eD in eDs:
            Re = float(Re)
            eD = float(eD)
            fd_exact = Colebrook(Re, eD, tol=0)
            fd_scipy = Colebrook(Re, eD)
            assert_close(fd_exact, fd_scipy, rtol=1e-9)


@pytest.mark.slow
def test_Colebrook_vs_Clamond():
    Res = logspace(log10(10), log10(1E50), 40)
    eDs = logspace(log10(1e-20), log10(1), 40)
    for Re in Res:
        for eD in eDs:
            fd_exact = Colebrook(Re, eD)
            fd_clamond = Clamond(Re, eD)
            # Interestingly, matches to rtol=1e-9 vs. numerical solver
            # But does not have such accuracy compared to mpmath
            if isnan(fd_exact) or isnan(fd_clamond):
                continue # older scipy on 3.4 returns a nan sometimes
            assert_close(fd_exact, fd_clamond, rtol=1e-9)
            # If rtol is moved to 1E-7, eD can be increased to 1




@pytest.mark.mpmath
def test_Colebrook_hard_regimes():
    fd_inf_regime = Colebrook(104800000000, 2.55e-08)
    assert_close(fd_inf_regime, 0.0037751087365339906, rtol=1e-10)


def test_one_phase_dP():
    dP = one_phase_dP(10.0, 1000., 1E-5, .1, L=1.000)
    assert_close(dP, 63.43447321097365)

def test_one_phase_dP_gravitational():
    dP = one_phase_dP_gravitational(angle=90., rho=2.6)
    assert_close(dP, 25.49729)

    dP = one_phase_dP_gravitational(angle=90, rho=2.6, L=2.)
    assert_close(dP, 25.49729*2)


def test_one_phase_dP_dz_acceleration():
    dP = one_phase_dP_dz_acceleration(m=1., D=0.1, rho=827.1, dv_dP=-1.1E-5, dP_dL=5E5, dA_dL=0.0001)
    assert_close(dP, 89162.89116373913)


@pytest.mark.slow
@pytest.mark.thermo
@pytest.mark.skip
def test_one_phase_dP_dz_acceleration_example():
    # This requires thermo!
    from thermo import Stream, Vm_to_rho
    from fluids import one_phase_dP, one_phase_dP_acceleration
    import numpy as np
    from scipy.integrate import odeint
    from fluids.numerics import assert_close

    P0 = 1E5
    s = Stream(['nitrogen', 'methane'], T=300, P=P0, zs=[0.5, 0.5], m=1)
    rho0 = s.rho
    D = 0.1
    def dP_dz(P, L, acc=False):
        s.flash(P=float(P), Hm=s.Hm)
        dPf = one_phase_dP(m=s.m, rho=s.rhog, mu=s.rhog, D=D, roughness=0, L=1.0)

        if acc:
            G = 4.0*s.m/(pi*D*D)
            der = s.VolumeGasMixture.property_derivative_P(P=s.P, T=s.T, zs=s.zs, ws=s.ws)
            der = 1/Vm_to_rho(der, s.MW)
            factor = G*G*der
            dP = dPf/(1.0 + factor)
            return -dP
        return -dPf

    ls = linspace(0, .01)

    dP_noacc = odeint(dP_dz, s.P, ls, args=(False,))[-1]
    s.flash(P=float(P0), Hm=s.Hm) # Reset the stream object
    profile = odeint(dP_dz, s.P, ls, args=(True,))

    dP_acc = profile[-1]

    s.flash(P=dP_acc, Hm=s.Hm)
    rho1 = s.rho

    dP_acc_numerical = dP_noacc - dP_acc
    dP_acc_basic = one_phase_dP_acceleration(m=s.m, D=D, rho_o=rho1, rho_i=rho0)

    assert_close(dP_acc_basic, dP_acc_numerical, rtol=1E-4)
del test_one_phase_dP_dz_acceleration_example

def test_transmission_factor():
    assert_close(transmission_factor(fd=0.0185), 14.704292441876154)
    assert_close(transmission_factor(F=14.704292441876154), 0.0185)
    assert_close(transmission_factor(0.0185), 14.704292441876154)

    # Example in [1]_, lists answer as 12.65
    assert_close(transmission_factor(fd=0.025), 12.649110640673516)

    with pytest.raises(Exception):
        transmission_factor()


def test_roughness_Farshad():

    e = roughness_Farshad('Cr13, bare', 0.05)
    assert_close(e, 5.3141677781137006e-05)

    e = roughness_Farshad('Cr13, bare')
    assert_close(e, 5.5e-05)

    e = roughness_Farshad(coeffs=(0.0021, -1.0055), D=0.05)
    assert_close(e, 5.3141677781137006e-05)

    tot = sum([abs(j) for i in _Farshad_roughness.values() for j in i])
    assert_close(tot, 7.0729095)

    with pytest.raises(Exception):
        roughness_Farshad('BADID', 0.05)

@pytest.mark.skipif(not has_fuzzywuzzy, reason='missing fuzzywuzzy')
def test_nearest_material_roughness():
    hit1 = nearest_material_roughness('condensate pipes', clean=False)
    assert hit1 == 'Seamless steel tubes, Condensate pipes in open systems or periodically operated steam pipelines'

    hit2 = nearest_material_roughness('Plastic', clean=True)
    assert hit2 == 'Plastic coated'


@pytest.mark.skipif(not has_fuzzywuzzy, reason='missing fuzzywuzzy')
def test_material_roughness():
    e1 = material_roughness('Plastic coated')
    assert_close(e1, 5e-06)

    e2 = material_roughness('Plastic coated', D=1E-3)
    assert_close(e2, 5.243618447826409e-06)

    e3 = material_roughness('Brass')
    assert_close(e3, 1.52e-06)

    e4 = material_roughness('condensate pipes')
    assert_close(e4, 0.0005)

    ID = 'Old, poor fitting and manufacture; with an overgrown surface'
    e5 = [material_roughness(ID, optimism=i) for i in (True, False)]
    assert_close1d(e5, [0.001, 0.004])


def test_von_Karman():
    f = von_Karman(1E-4)
    f_precalc = 0.01197365149564789
    assert_close(f, f_precalc)


def Prandtl_von_Karman_Nikuradse_numeric(Re):
    rat = 2.51/Re
    def to_solve(f):
        # Good to 1E75, down to 1E-17
        v = f**-0.5
        return v + 2.0*log10(rat*v)
    return secant(to_solve, 0.000001)


def test_Prandtl_von_Karman_Nikuradse():
    Re = 200
    assert_close(Prandtl_von_Karman_Nikuradse_numeric(Re),  Prandtl_von_Karman_Nikuradse(Re))


def test_Prandtl_von_Karman_Nikuradse_full():
    # Tested to a very high number of points
    fds = []
    fds_numeric = []
    for Re in logspace(1E-15, 30, 40):
        fds.append(Prandtl_von_Karman_Nikuradse_numeric(Re))
        fds_numeric.append(Prandtl_von_Karman_Nikuradse(Re))
    assert_close1d(fds, fds_numeric)


def test_helical_laminar_fd_White():
    fd = helical_laminar_fd_White(250., .02, .1)
    assert_close(fd, 0.4063281817830202)
    assert_close(helical_laminar_fd_White(250, .02, 100), 0.256)


def test_helical_laminar_fd_Mori_Nakayama():
    fd = helical_laminar_fd_Mori_Nakayama(250., .02, .1)
    assert_close(fd, 0.4222458285779544)
    assert_close(4.4969472, helical_laminar_fd_Mori_Nakayama(20, .02, .1))


def test_helical_laminar_fd_Schmidt():
    fd = helical_laminar_fd_Schmidt(250., .02, .1)
    assert_close(fd, 0.47460725672835236)
    # Test convergence at low curvature
    assert_close(helical_laminar_fd_Schmidt(250., 1, 1E10), friction_laminar(250))


def test_helical_turbulent_fd_Srinivasan():
    fd = helical_turbulent_fd_Srinivasan(1E4, 0.01, .02)
    assert_close(fd, 0.0570745212117107)

def test_helical_turbulent_fd_Schmidt():
    fd = helical_turbulent_fd_Schmidt(1E4, 0.01, .02)
    assert_close(fd, 0.08875550767040916)
    fd = helical_turbulent_fd_Schmidt(1E4, 0.01, .2)
    assert_close(fd, 0.04476560991345504)
    assert_close(friction_factor(1E4), helical_turbulent_fd_Schmidt(1E4, 0.01, 1E11))

    fd = helical_turbulent_fd_Schmidt(1E6, 0.01, .02)
    assert_close(fd, 0.04312877383550924)


def test_helical_turbulent_fd_Mori_Nakayama():
    # Formula in [1]_ is hard to read, but the powers have been confirmed in
    # two sources to be 1/5. [3]_ butchers the formula's brackets/power raising,
    # but is otherwise correct.
    fd = helical_turbulent_fd_Mori_Nakayama(1E4, 0.01, .2)
    assert_close(fd, 0.037311802071379796)


def test_helical_turbulent_fd_Prasad():
    # Checks out, formula in [2]_ is the same as in [1]_!
    fd = helical_turbulent_fd_Prasad(1E4, 0.01, .2)
    assert_close(fd, 0.043313098093994626)
    assert_close(helical_turbulent_fd_Prasad(1E4, 0.01, 1E20), friction_factor(1E4))


def test_helical_turbulent_fd_Czop():
    fd = helical_turbulent_fd_Czop(1E4, 0.01, .2)
    assert_close(fd, 0.02979575250574106)


def test_helical_turbulent_fd_Guo():
    fd = helical_turbulent_fd_Guo(2E5, 0.01, .2)
    assert_close(fd, 0.022189161013253147)


def test_helical_turbulent_fd_Ju():
    fd = helical_turbulent_fd_Ju(1E4, 0.01, .2)
    assert_close(fd, 0.04945959480770937)
    assert_close(helical_turbulent_fd_Ju(1E4, 0.01, 1E80),  friction_factor(1E4))


def test_helical_turbulent_fd_Mandal_Nigam():
    fd = helical_turbulent_fd_Mandal_Nigam(1E4, 0.01, .2)
    assert_close(fd, 0.03831658117115902)
    assert_close(helical_turbulent_fd_Mandal_Nigam(1E4, 0.01, 1E80),  friction_factor(1E4))


def test_helical_transition_Re_Seth_Stahel():
    # Read the original
    assert_close(helical_transition_Re_Seth_Stahel(1, 7.), 7645.0599897402535)
    assert_close(helical_transition_Re_Seth_Stahel(1, 1E20), 1900)


def test_helical_transition_Re_Ito():
    assert_close(helical_transition_Re_Ito(1, 7.), 10729.972844697186)


def test_helical_transition_Re_Kubair_Kuloor():
    assert_close(helical_transition_Re_Kubair_Kuloor(1, 7), 8625.986927588123)


def test_helical_transition_Re_Kutateladze_Borishanskii():
    assert_close(helical_transition_Re_Kutateladze_Borishanskii(1, 7.),  7121.143774574058)
    assert_close(helical_transition_Re_Kutateladze_Borishanskii(1, 1E20), 2300)


def test_helical_transition_Re_Schmidt():
    assert_close(helical_transition_Re_Schmidt(1, 7.), 10540.094061770815)
    assert_close(helical_transition_Re_Schmidt(1, 1E20), 2300)


def test_helical_transition_Re_Srinivasan():
    assert_close(helical_transition_Re_Srinivasan(1, 7.),  11624.704719832524,)
    assert_close(helical_transition_Re_Srinivasan(1, 1E20),  2100)


def test_friction_factor_curved():
    fd = friction_factor_curved(2E4, 0.01, .02)
    assert_close(fd, 0.050134646621603024)
    fd = friction_factor_curved(250, .02, .1)
    assert_close(fd, 0.47460725672835236)

    fd_transition = [friction_factor_curved(i, 0.01, .02) for i in [16779, 16780]]
    assert_close1d(fd_transition, [0.03323676794260526, 0.057221855744623344])

    with pytest.raises(Exception):
        friction_factor_curved(16779, 0.01, .02, Method='BADMETHOD')
    with pytest.raises(Exception):
        friction_factor_curved(16779, 0.01, .02, Rec_method='BADMETHOD')

    fd_rough_false = friction_factor_curved(20000, 0.01, .02, roughness=.0001, turbulent_method='Guo')
    assert_close(fd_rough_false, 0.1014240343662085)

    methods = friction_factor_curved_methods(20000, 0.01, .02, check_ranges=True)
    assert sorted(methods) == sorted(['Guo','Ju','Schmidt turbulent','Prasad','Mandel Nigam','Mori Nakayama turbulent','Czop', 'Srinivasan turbulent'])
    methods = friction_factor_curved_methods(2000, 0.01, .02, check_ranges=True)
    assert sorted(methods) == sorted(['White', 'Schmidt laminar', 'Mori Nakayama laminar'])
    assert 'Schmidt turbulent' in friction_factor_curved_methods(Re=1E5, Di=0.02, Dc=0.5)
    assert 11 == len(friction_factor_curved_methods(Re=1E5, Di=0.02, Dc=0.5, check_ranges=False))

    for m in friction_factor_curved_methods(Re=1E5, Di=0.02, Dc=0.5, check_ranges=False):
        friction_factor_curved(2000, 0.01, .02, Method=m)

    # Test the Fanning case
    fd = friction_factor_curved(2E4, 0.01, .02, Darcy=False)
    assert_close(fd, 0.012533661655400756)

    for m in ['Seth Stahel', 'Ito', 'Kubair Kuloor', 'Kutateladze Borishanskii', 'Schmidt', 'Srinivasan']:
        helical_Re_crit(Di=0.02, Dc=0.5, Method=m)

def test_friction_plate():
    fd = friction_plate_Martin_1999(Re=20000., plate_enlargement_factor=1.15)
    assert_close(fd, 2.284018089834134)

    fd = friction_plate_Martin_1999(Re=1999., plate_enlargement_factor=1.15)
    assert_close(fd, 2.749383588479863)

    fd = friction_plate_Martin_VDI(Re=20000., plate_enlargement_factor=1.15)
    assert_close(fd, 2.702534119024076)

    fd = friction_plate_Martin_VDI(Re=1999., plate_enlargement_factor=1.15)
    assert_close(fd, 3.294294334690556)

    fd = friction_plate_Muley_Manglik(Re=2000., chevron_angle=45., plate_enlargement_factor=1.2)
    assert_close(fd, 1.0880870804075413)


def test_friction_Kumar():
    from fluids.friction import Kumar_beta_list, Kumar_fd_Res
    fd = friction_plate_Kumar(2000, 30)
    assert_close(fd, 2.9760669055634517)

    all_ans_expect = [[[22.22222222222222, 18.900854099814858, 5.181226661414687, 5.139730745446174],
  [20.88888888888889, 17.09090909090909, 3.656954441625244, 3.609575756782771]],
 [[13.428571428571427, 12.000171923243482, 1.7788367041690634, 1.7788497785371564],
  [9.714285714285714, 8.5, 1.2332865464612235, 1.2320492987599356]],
 [[7.157894736842104, 6.590102034105372, 1.2332865464612235, 1.2320492987599356],
  [5.052631578947368, 4.571428571428571, 0.9576862861589914, 0.9547729646969146]],
 [[2.4615384615384617, 2.374448634025773, 0.8393834232628009, 0.8379103279437352],
  [2.4615384615384617, 2.3414634146341466, 0.7519331759748705, 0.7502394735017442]],
 [[1.9591836734693877, 1.9015330284979595, 0.6797898512309091, 0.6799788644298855],
  [1.9591836734693877, 1.9015330284979595, 0.6797898512309091, 0.6799788644298855]]]

    all_ans = []
    for i, beta_main in enumerate(Kumar_beta_list):
        beta_ans = []
        for beta in (beta_main-1, beta_main+1):
            Re_ans = []
            for Re_main in Kumar_fd_Res[i]:
                for Re in [Re_main-1, Re_main+1]:
                    ans = friction_plate_Kumar(Re, beta)
                    Re_ans.append(ans)
            beta_ans.append(Re_ans)
        all_ans.append(beta_ans)

    assert_close3d(all_ans, all_ans_expect)



