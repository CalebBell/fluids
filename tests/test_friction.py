# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import log10
from scipy.optimize import newton
import numpy as np

from fluids import *
from numpy.testing import assert_allclose
import pytest

def test_friction():
    assert_allclose(Moody(1E5, 1E-4), 0.01809185666808665)
    assert_allclose(Alshul_1952(1E5, 1E-4), 0.018382997825686878)
    assert_allclose(Wood_1966(1E5, 1E-4), 0.021587570560090762)
    assert_allclose(Churchill_1973(1E5, 1E-4), 0.01846708694482294)
    assert_allclose(Eck_1973(1E5, 1E-4), 0.01775666973488564)
    assert_allclose(Jain_1976(1E5, 1E-4), 0.018436560312693327)
    assert_allclose(Swamee_Jain_1976(1E5, 1E-4), 0.018452424431901808)
    assert_allclose(Churchill_1977(1E5, 1E-4), 0.018462624566280075)
    assert_allclose(Chen_1979(1E5, 1E-4), 0.018552817507472126)
    assert_allclose(Round_1980(1E5, 1E-4), 0.01831475391244354)
    assert_allclose(Shacham_1980(1E5, 1E-4), 0.01860641215097828)
    assert_allclose(Barr_1981(1E5, 1E-4), 0.01849836032779929)
    assert_allclose(Zigrang_Sylvester_1(1E5, 1E-4), 0.018646892425980794)
    assert_allclose(Zigrang_Sylvester_2(1E5, 1E-4), 0.01850021312358548)
    assert_allclose(Haaland(1E5, 1E-4), 0.018265053014793857)
    assert_allclose(Serghides_1(1E5, 1E-4), 0.01851358983180063)
    assert_allclose(Serghides_2(1E5, 1E-4), 0.018486377560664482)
    assert_allclose(Tsal_1989(1E5, 1E-4), 0.018382997825686878)
    assert_allclose(Tsal_1989(1E8, 1E-4), 0.012165854627780102)
    assert_allclose(Manadilli_1997(1E5, 1E-4), 0.01856964649724108)
    assert_allclose(Romeo_2002(1E5, 1E-4), 0.018530291219676177)
    assert_allclose(Sonnad_Goudar_2006(1E5, 1E-4), 0.0185971269898162)
    assert_allclose(Rao_Kumar_2007(1E5, 1E-4), 0.01197759334600925)
    assert_allclose(Buzzelli_2008(1E5, 1E-4), 0.018513948401365277)
    assert_allclose(Avci_Karagoz_2009(1E5, 1E-4), 0.01857058061066499)
    assert_allclose(Papaevangelo_2010(1E5, 1E-4), 0.015685600818488177)
    assert_allclose(Brkic_2011_1(1E5, 1E-4), 0.01812455874141297)
    assert_allclose(Brkic_2011_2(1E5, 1E-4), 0.018619745410688716)
    assert_allclose(Fang_2011(1E5, 1E-4), 0.018481390682985432)
    assert_allclose(Clamond(1E5, 1E-4), 0.01851386607747165)
    assert_allclose(Colebrook(1E5, 1E-4), 0.018513866077471648)

    assert_allclose(friction_laminar(128), 0.5)

    assert_allclose(Blasius(10000), 0.03164)

    assert_allclose(sum(_roughness.values()), 0.01504508)

    assert_allclose(friction_factor(Re=1E5, eD=1E-4), 0.01851386607747165)
    methods_1 = friction_factor(Re=1E5, eD=1E-4, AvailableMethods=True)
    methods_1.sort()

    methods_2 = ['Clamond', 'Colebrook', 'Manadilli_1997', 'Haaland', 'Alshul_1952', 'Avci_Karagoz_2009', 'Rao_Kumar_2007', 'Zigrang_Sylvester_2', 'Eck_1973', 'Buzzelli_2008', 'Tsal_1989', 'Papaevangelo_2010', 'Barr_1981', 'Jain_1976', 'Moody', 'Brkic_2011_2', 'Brkic_2011_1', 'Swamee_Jain_1976', 'Wood_1966', 'Shacham_1980', 'Romeo_2002', 'Chen_1979', 'Fang_2011', 'Round_1980', 'Sonnad_Goudar_2006', 'Churchill_1973', 'Churchill_1977', 'Serghides_2', 'Serghides_1', 'Zigrang_Sylvester_1']
    methods_2.sort()
    assert methods_1 == methods_2

    assert_allclose(friction_factor(Re=1E5, eD=1E-4, Darcy=False), 0.01851386607747165*4)
    assert_allclose(friction_factor(Re=128), 0.5)
    
    assert_allclose(friction_factor(Re=1E5, eD=0, Method=None), 0.01798977308427384)
    


def test_transmission_factor():
    assert_allclose(transmission_factor(fd=0.0185), 14.704292441876154)
    assert_allclose(transmission_factor(F=14.704292441876154), 0.0185)
    assert_allclose(transmission_factor(0.0185), 14.704292441876154)

    # Example in [1]_, lists answer as 12.65
    assert_allclose(transmission_factor(fd=0.025), 12.649110640673516)

    with pytest.raises(Exception):
        transmission_factor()


def test_roughness_Farshad():

    e = roughness_Farshad('Cr13, bare', 0.05)
    assert_allclose(e, 5.3141677781137006e-05)

    e = roughness_Farshad('Cr13, bare')
    assert_allclose(e, 5.5e-05)

    e = roughness_Farshad(coeffs=(0.0021, -1.0055), D=0.05)
    assert_allclose(e, 5.3141677781137006e-05)

    tot = sum([abs(j) for i in _Farshad_roughness.values() for j in i])
    assert_allclose(tot, 7.0729095)

    with pytest.raises(Exception):
        roughness_Farshad('BADID', 0.05)

def test_nearest_material_roughness():
    hit1 = nearest_material_roughness('condensate pipes', clean=False)
    assert hit1 == 'Seamless steel tubes, Condensate pipes in open systems or periodically operated steam pipelines'
    
    hit2 = nearest_material_roughness('Plastic', clean=True)
    assert hit2 == 'Plastic coated'


def test_material_roughness():
    e1 = material_roughness('Plastic coated')
    assert_allclose(e1, 5e-06)
    
    e2 = material_roughness('Plastic coated', D=1E-3)
    assert_allclose(e2, 5.243618447826409e-06)

    e3 = material_roughness('Brass')
    assert_allclose(e3, 1.52e-06)

    e4 = material_roughness('condensate pipes')
    assert_allclose(e4, 0.0005)

    ID = 'Old, poor fitting and manufacture; with an overgrown surface'
    e5 = [material_roughness(ID, optimism=i) for i in (True, False)]
    assert_allclose(e5, [0.001, 0.004])


def test_von_Karman():
    f = von_Karman(1E-4)
    f_precalc = 0.01197365149564789
    assert_allclose(f, f_precalc)


def test_Prandtl_von_Karman_Nikuradse():
    def Prandtl_von_Karman_Nikuradse_numeric(Re):
        def to_solve(f):
            # Good to 1E75, down to 1E-17
            return 1./f**0.5 + 2*log10(2.51/Re/f**0.5)
        return newton(to_solve, 0.000001)

    for Re in np.logspace(1E-15,30,200):
        assert_allclose(Prandtl_von_Karman_Nikuradse_numeric(Re), Prandtl_von_Karman_Nikuradse(Re))


def test_helical_laminar_fd_White():
    fd = helical_laminar_fd_White(250, .02, .1)
    assert_allclose(fd, 0.4063281817830202)
    assert_allclose(helical_laminar_fd_White(250, .02, 100), 0.256)
    
    
def test_helical_laminar_fd_Mori_Nakayama():
    fd = helical_laminar_fd_Mori_Nakayama(250, .02, .1)
    assert_allclose(fd, 0.4222458285779544)
    assert_allclose(4.4969472, helical_laminar_fd_Mori_Nakayama(20, .02, .1))
    
    
def test_helical_laminar_fd_Schmidt():
    fd = helical_laminar_fd_Schmidt(250, .02, .1)
    assert_allclose(fd, 0.47460725672835236)
    # Test convergence at low curvature 
    assert_allclose(helical_laminar_fd_Schmidt(250, 1, 1E10), friction_laminar(250))
    
    
def test_helical_turbulent_fd_Schmidt():
    fd = helical_turbulent_fd_Schmidt(1E4, 0.01, .02)
    assert_allclose(fd, 0.08875550767040916)
    fd = helical_turbulent_fd_Schmidt(1E4, 0.01, .2)
    assert_allclose(fd, 0.04476560991345504)
    assert_allclose(friction_factor(1E4), helical_turbulent_fd_Schmidt(1E4, 0.01, 1E11))
    
    fd = helical_turbulent_fd_Schmidt(1E6, 0.01, .02)
    assert_allclose(fd, 0.04312877383550924)
        
    
def test_helical_turbulent_fd_Mori_Nakayama():
    # Formula in [1]_ is hard to read, but the powers have been confirmed in
    # two sources to be 1/5. [3]_ butchers the formula's brackets/power raising,
    # but is otherwise correct.
    fd = helical_turbulent_fd_Mori_Nakayama(1E4, 0.01, .2)
    assert_allclose(fd, 0.037311802071379796)


def test_helical_turbulent_fd_Prasad():
    # Checks out, formula in [2]_ is the same as in [1]_!
    fd = helical_turbulent_fd_Prasad(1E4, 0.01, .2)
    assert_allclose(fd, 0.043313098093994626)
    assert_allclose(helical_turbulent_fd_Prasad(1E4, 0.01, 1E20), friction_factor(1E4))
    
    
def test_helical_turbulent_fd_Czop():
    fd = helical_turbulent_fd_Czop(1E4, 0.01, .2)
    assert_allclose(fd, 0.02979575250574106)
    
    
def test_helical_turbulent_fd_Guo():
    fd = helical_turbulent_fd_Guo(2E5, 0.01, .2)
    assert_allclose(fd, 0.022189161013253147)
    
    
def test_helical_turbulent_fd_Ju():
    fd = helical_turbulent_fd_Ju(1E4, 0.01, .2)
    assert_allclose(fd, 0.04945959480770937)
    assert_allclose(helical_turbulent_fd_Ju(1E4, 0.01, 1E80),  friction_factor(1E4))
    
    
def test_helical_turbulent_fd_Mandal_Nigam():
    fd = helical_turbulent_fd_Mandal_Nigam(1E4, 0.01, .2)
    assert_allclose(fd, 0.03831658117115902)
    assert_allclose(helical_turbulent_fd_Mandal_Nigam(1E4, 0.01, 1E80),  friction_factor(1E4))


def test_helical_transition_Re_Seth_Stahel():
    # Read the original
    assert_allclose(helical_transition_Re_Seth_Stahel(1, 7.), 7645.0599897402535)
    assert_allclose(helical_transition_Re_Seth_Stahel(1, 1E20), 1900)
    

def test_helical_transition_Re_Ito():
    assert_allclose(helical_transition_Re_Ito(1, 7.), 10729.972844697186)
    
    
def test_helical_transition_Re_Kubair_Kuloor():
    assert_allclose(helical_transition_Re_Kubair_Kuloor(1, 7), 8625.986927588123)
    
    
def test_helical_transition_Re_Kutateladze_Borishanskii():
    assert_allclose(helical_transition_Re_Kutateladze_Borishanskii(1, 7.),  7121.143774574058)
    assert_allclose(helical_transition_Re_Kutateladze_Borishanskii(1, 1E20), 2300)
    
    
def test_helical_transition_Re_Schmidt():
    assert_allclose(helical_transition_Re_Schmidt(1, 7.), 10540.094061770815)
    assert_allclose(helical_transition_Re_Schmidt(1, 1E20), 2300)
    
    
def test_helical_transition_Re_Srinivasan():
    assert_allclose(helical_transition_Re_Srinivasan(1, 7.),  11624.704719832524,)
    assert_allclose(helical_transition_Re_Srinivasan(1, 1E20),  2100)
    
    
def test_friction_factor_curved():
    fd = friction_factor_curved(2E4, 0.01, .02)
    assert_allclose(fd, 0.050134646621603024)
    fd = friction_factor_curved(250, .02, .1)
    assert_allclose(fd, 0.47460725672835236)
    
    fd_transition = [friction_factor_curved(i, 0.01, .02) for i in [16779, 16780]]
    assert_allclose(fd_transition, [0.03323676794260526, 0.057221855744623344])
    
    with pytest.raises(Exception):
        friction_factor_curved(16779, 0.01, .02, Method='BADMETHOD')
    with pytest.raises(Exception):
        friction_factor_curved(16779, 0.01, .02, Rec_method='BADMETHOD')
        
    fd_rough_false = friction_factor_curved(20000, 0.01, .02, roughness=.0001, turbulent_method='Guo')
    assert_allclose(fd_rough_false, 0.1014240343662085)
    
    methods = friction_factor_curved(20000, 0.01, .02, AvailableMethods=True)
    assert sorted(methods) == sorted(['Guo','Ju','Schmidt turbulent','Prasad','Mandel Nigam','Mori Nakayama turbulent','Czop'])
    methods = friction_factor_curved(2000, 0.01, .02, AvailableMethods=True)
    assert sorted(methods) == sorted(['White', 'Schmidt laminar', 'Mori Nakayama laminar'])
    
    # Test the Fanning case
    fd = friction_factor_curved(2E4, 0.01, .02, Darcy=False)
    assert_allclose(fd, 0.2005385864864121)