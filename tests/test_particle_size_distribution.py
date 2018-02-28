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

from fluids import *
import numpy as np
from numpy.testing import assert_allclose
import pytest
from fluids.particle_size_distribution import *
import scipy.stats
from random import uniform
from scipy.integrate import quad


def test_ParticleSizeDistribution_basic():
    ds = [240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532]
    counts = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
    
    # this is calculated from (Ds, counts)
    count_fractions = [0.010640039286298903, 0.01947945653953184, 0.03797675560648224, 0.06711409395973154, 0.102962841708954, 0.13897528237027337, 0.16205598297593715, 0.160582746767065, 0.13504665247994763, 0.09477819610410869, 0.048616794892781146, 0.01816991324275659, 0.0034375511540350305, 0.0001636929120969062]
    fractions = [4.8560356399310335e-05, 0.00021291794698947167, 0.0008107432330218852, 0.0027975134942445257, 0.00836789808490677, 0.02201901107895143, 0.05010399231412809, 0.0968727835386488, 0.15899879607747244, 0.2178784903712532, 0.21825921197532888, 0.159302671180342, 0.05885464261922434, 0.0054727677290887945]
    
    opts = [{'count_fractions': count_fractions},  {'counts': counts}, {'fractions': fractions}]
    for opt in opts:
        asme_e799 = ParticleSizeDistribution(ds=ds, **opt)
        
        d10 = asme_e799.mean_size(1, 0)
        assert_allclose(d10, 1459.3725650679328)
        
        d21 = asme_e799.mean_size(2, 1)
        assert_allclose(d21, 1857.7888572055529)
        d20 = asme_e799.mean_size(2, 0)
        assert_allclose(d20, 1646.5740462835831)
        
        d32 = asme_e799.mean_size(3, 2)
        assert_allclose(d32, 2269.3210317450453)
        # This one is rounded to 2280 in ASME - weird
        
        d31 = asme_e799.mean_size(3, 1)
        assert_allclose(d31, 2053.2703977309357)
        # This one is rounded to 2060 in ASME - weird
        
        d30 = asme_e799.mean_size(3, 0)
        assert_allclose(d30, 1832.39665294744)
        
        d43 = asme_e799.mean_size(4, 3)
        assert_allclose(d43, 2670.751954612969)
        # The others are, rounded to the nearest 10, correct.
        # There's something weird about the end points of some intermediate values of
        #  D3 and D4. Likely just rounding issues.
        
        vol_percents_exp = [0.005, 0.021, 0.081, 0.280, 0.837, 2.202, 5.010, 9.687, 15.900, 21.788, 21.826, 15.930, 5.885, 0.547]
        assert vol_percents_exp == [round(i*100, 3) for i in asme_e799.fractions]
        
        assert_allclose(asme_e799.fractions, fractions)
        assert_allclose(asme_e799.count_fractions, count_fractions)
        
        # i, i distributions
        d00 = asme_e799.mean_size(0, 0)
        assert_allclose(d00, 1278.7057976023061)
        
        d11 = asme_e799.mean_size(1, 1)
        assert_allclose(d11, 1654.6665309027303)
        
        d22 = asme_e799.mean_size(2, 2)
        assert_allclose(d22, 2054.3809583432208)
        
        d33 = asme_e799.mean_size(3, 3)
        assert_allclose(d33, 2450.886241250387)
        
        d44 = asme_e799.mean_size(4, 4)
        assert_allclose(d44, 2826.0471682278476)


def test_pdf_lognormal():
    pdf = pdf_lognormal(d=1E-4, d_characteristic=1E-5, s=1.1)
    assert_allclose(pdf, 405.5420921156425, rtol=1E-12)
    
    pdf_sp = scipy.stats.lognorm.pdf(x=1E-4/1E-5, s=1.1)/1E-5
    assert_allclose(pdf_sp, pdf)
    
    assert 0.0 == pdf_lognormal(d=0, d_characteristic=1E-5, s=1.1)
    
    # Check we can get down almost to zero
    pdf = pdf_lognormal(d=3.7E-24, d_characteristic=1E-5, s=1.1)
    assert_allclose(pdf, 4.842842147909424e-301)
    
    
def test_cdf_lognormal():
    cdf = cdf_lognormal(d=1E-4, d_characteristic=1E-5, s=1.1)
    assert_allclose(cdf, 0.98183698757981763)
    
    cdf_sp = scipy.stats.lognorm.cdf(x=1E-4/1E-5, s=1.1)
    assert_allclose(cdf, cdf_sp)
    
    assert cdf_lognormal(d=1e300, d_characteristic=1E-5, s=1.1) == 1.0
    assert cdf_lognormal(d=0, d_characteristic=1E-5, s=1.1) == 0.0
    
    
def test_pdf_lognormal_basis_integral():
    ans = pdf_lognormal_basis_integral(d=1E-4, d_characteristic=1E-5, s=1.1, n=-2)
    assert_allclose(ans, 56228306549.263626)
    
    
@pytest.mark.slow
def test_ParticleSizeDistributionLognormal_mean_sizes_numerical():
    '''Takes like 10 seconds.
    '''
    # ISO standard example, done numerically
    a = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    ds = a.ds_discrete(dmax=1, pts=1E5)
    fractions = a.fractions_discrete(ds)
    
    disc = ParticleSizeDistribution(ds=ds, fractions=fractions)
    d20 = disc.mean_size(2, 0)
    assert_allclose(d20, 3.033E-6, rtol=0, atol=1E-9)
    
    d10 = disc.mean_size(1, 0)
    assert_allclose(d10, 2.676E-6, rtol=0, atol=1E-9)
    
    d21 = disc.mean_size(2, 1)
    assert_allclose(d21, 3.436E-6, rtol=0, atol=1E-9)
    
    d32 = disc.mean_size(3, 2)
    # Does match, need 6E6 pts to get the last digit to round right
    assert_allclose(d32, 4.412E-6, rtol=0, atol=1E-9)
    
    d43 = disc.mean_size(4, 3)
    assert_allclose(d43, 5.666E-6, rtol=0, atol=1E-9)
    
    d33 = disc.mean_size(3.0, 3.0)
    assert_allclose(d33, 5.000E-6, rtol=0, atol=1E-9)
    
    d00 = disc.mean_size(0.0, 0.0)
    assert_allclose(d00, 2.362E-6, rtol=0, atol=1E-9)


def test_ParticleSizeDistributionLognormal_mean_sizes_analytical():
    disc = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    
    d20 = disc.mean_size(2, 0)
    assert_allclose(d20, 3.033E-6, rtol=0, atol=1E-9)
    assert_allclose(d20, 3.0326532985631672e-06, rtol=1E-12)
    assert_allclose(d20, disc.mean_size_ISO(2, 0), rtol=1E-12)


    d10 = disc.mean_size(1, 0)
    assert_allclose(d10, 2.676E-6, rtol=0, atol=1E-9)
    assert_allclose(d10, 2.6763071425949508e-06, rtol=1E-12)
    assert_allclose(d10, disc.mean_size_ISO(1, 0), rtol=1E-12)

    d21 = disc.mean_size(2, 1)
    assert_allclose(d21, 3.436E-6, rtol=0, atol=1E-9)
    assert_allclose(d21, 3.4364463939548618e-06, rtol=1E-12)
    assert_allclose(d21, disc.mean_size_ISO(1, 1), rtol=1E-12)


    d32 = disc.mean_size(3, 2)
    assert_allclose(d32, 4.412E-6, rtol=0, atol=1E-9)
    assert_allclose(d32, 4.4124845129229773e-06, rtol=1E-12)
    assert_allclose(d32, disc.mean_size_ISO(1, 2), rtol=1E-12)

    d43 = disc.mean_size(4, 3)
    assert_allclose(d43, 5.666E-6, rtol=0, atol=1E-9)
    assert_allclose(d43, 5.6657422653341318e-06, rtol=1E-12)
    assert_allclose(d43, disc.mean_size_ISO(1, 3), rtol=1E-12)

    # There guys - need more work
#    d33 = disc.mean_size(3.0, 3.0)
#    assert_allclose(d33, 5.000E-6, rtol=0, atol=1E-9)
#    
#    d00 = disc.mean_size(0.0, 0.0)
#    assert_allclose(d00, 2.362E-6, rtol=0, atol=1E-9)

def test_ParticleSizeDistributionLognormal_dn():
    disc = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    
    # Test input of 1
    ans = disc.dn(1)
    # The answer can vary quite a lot near the end, so it is safest just to 
    # compare with the reverse, plugging it back to cdf
    assert_allclose(disc.cdf(ans), 1, rtol=1E-12)
#    assert_allclose(ans, 0.0002964902595794474)
    
    # Test zero input
    assert_allclose(disc.dn(0), 0)
    
    # Test 50% input
    ans = disc.dn(.5)
    assert_allclose(ans,  5.0e-06, rtol=1E-6)
    
    with pytest.raises(Exception):
        disc.dn(1.5)
    with pytest.raises(Exception):
        disc.dn(-.5)
        
    # Other orders of n - there is no comparison data for this yet!!
    assert_allclose(disc.pdf(1E-5), disc.pdf(1E-5, 3))
    assert_allclose(disc.pdf(1E-5, 2), 13468.122877854335)
    assert_allclose(disc.pdf(1E-5, 1), 4628.2482296943508)
    assert_allclose(disc.pdf(1E-5, 0), 1238.6613794833427)
        

def test_ParticleSizeDistributionLognormal_dn_order_0_1_2():
    '''Simple point to test where the order of n should be 0
    
    Yes, the integrals need this many points (which makes them slow) to get
    the right accuracy. They've been tested and reduced already quite a bit.
    '''
    # test 2, 0 -> 2, 0
    disc = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    to_int = lambda d: d**2*disc.pdf(d=d, n=0)
    points  = [5E-6*i for i in np.logspace(np.log10(.1), np.log10(50), 8)]
    
    ans_numerical = (quad(to_int, 1E-7, 5E-3, points=points)[0])**0.5
    ans_analytical = 3.0326532985631672e-06
    # The integral is able to give over to decimals!
    assert_allclose(ans_numerical, ans_analytical, rtol=1E-10)
       
    # test 2, 1 -> 1, 1 integrated pdf
    
    to_int = lambda d: d*disc.pdf(d=d, n=1)    
    ans_numerical = (quad(to_int, 1E-7, 5E-3, points=points)[0])**1
    ans_analytical = 3.4364463939548618e-06
    assert_allclose(ans_numerical, ans_analytical, rtol=1E-10)
    
    # test 3, 2 -> 1, 2 integrated pdf
    
    to_int = lambda d: d*disc.pdf(d=d, n=2)    
    ans_numerical = (quad(to_int, 1E-7, 5E-3, points=points)[0])**1
    ans_analytical = 4.4124845129229773e-06
    assert_allclose(ans_numerical, ans_analytical, rtol=1E-8)
    
    
def test_ParticleSizeDistributionLognormal_cdf_orders():
    # Test cdf of different orders a bunch
    disc = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    # 16 x 4 = 64 points
    ds = [1E-7, 2E-6, 3E-6, 4E-6, 5E-6, 6E-6, 7E-6, 1E-5, 2E-5, 3E-5, 5E-5, 7E-5, 1E-4, 2E-4, 4E-4, 1E-3]
    ans_expect = [[1.2740109519816568e-10, 0.36972511868508079, 0.68379899882263917, 0.85399280886562479, 0.9331927987311418, 0.96888427729983861, 0.98510775165387232, 0.9980509630571377, 0.99999033916822688, 0.99999981474719124, 0.99999999948654394, 0.99999999999391209, 0.99999999999996592, 1.0, 1.0, 1.0], 
                  [4.4255886816609946e-12, 0.20254040832522924, 0.49136307673913149, 0.71011232639847854, 0.84134474606854293, 0.91381737643345484, 0.95283088619207579, 0.99149043874391107, 0.99991921875653167, 0.99999771392273817, 0.99999998959747816, 0.99999999982864851, 0.99999999999863987, 1.0, 1.0, 1.0000000000000002], 
                  [1.2032972848417788e-13, 0.091334595732478152, 0.30095658738958558, 0.52141804648990697, 0.69146246127401301, 0.80638264936531312, 0.87959096325267283, 0.9703723506333426, 0.99946716289796089, 0.99997782059383111, 0.99999983475152954, 0.99999999622288371, 0.999999999957497, 0.99999999999999833, 1.0, 1.0], 
                  [2.55351295663786e-15, 0.033432418408916864, 0.15347299656473007, 0.3276949357115424, 0.5, 0.64231108623683952, 0.74950869138681098, 0.91717148099830148, 0.99721938213769046, 0.99983050191355338, 0.99999793935660408, 0.99999993474010451, 0.99999999896020164, 0.99999999999991951, 1.0, 1.0]]
    calc = []
    for n in range(0, 4):
        calc.append([disc.cdf(i, n=n) for i in ds])
    
    assert_allclose(ans_expect, calc, rtol=1E-9)    
    
    
def test_ParticleSizeDistributionLognormal_cdf_vs_pdf():
    
    # test PDF against CDF
    
    disc = ParticleSizeDistributionLognormal(s=0.5, d_characteristic=5E-6)
    ans_calc = []
    ans_expect = []
    for i in range(5):
        # Pick a random start
        start = uniform(0, 1)
        end = uniform(start, 1)
        d_start = disc.dn(start)
        d_end = disc.dn(end)
    
        delta = disc.cdf(d_end) - disc.cdf(d_start)
        delta_numerical = quad(disc.pdf, d_start, d_end)[0]
        ans_calc.append(delta_numerical)
        ans_expect.append(delta)
    assert_allclose(ans_calc, ans_expect)
    
