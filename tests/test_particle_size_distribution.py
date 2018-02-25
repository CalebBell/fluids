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
