'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.
'''

from math import exp, log

from fluids.numerics import assert_close, assert_close1d, derivative
from fluids.numerics import numpy as np
from fluids.numerics.polynomial_evaluation import (
    exp_horner_stable,
    exp_horner_stable_and_der,
    exp_horner_stable_and_der2,
    exp_horner_stable_and_der3,
    exp_horner_stable_ln_tau,
    exp_horner_stable_ln_tau_and_der,
    exp_horner_stable_ln_tau_and_der2,
    horner_domain,
    horner_log,
    horner_stable,
    horner_stable_and_der,
    horner_stable_and_der2,
    horner_stable_and_der3,
    horner_stable_and_der4,
    horner_stable_ln_tau,
    horner_stable_ln_tau_and_der,
    horner_stable_ln_tau_and_der2,
    horner_stable_ln_tau_and_der3,
    horner_stable_log,
)
from fluids.numerics.polynomial_utils import polynomial_offset_scale


def test_horner():
    from fluids.numerics.polynomial_evaluation import horner, horner_and_der2, horner_and_der3, horner_and_der4, horner_backwards
    assert_close(horner([1.0, 3.0], 2.0), 5.0, rtol=1e-15)
    assert_close(horner_backwards(2.0, [1.0, 3.0]), 5.0, rtol=1e-15)
    assert_close(horner([3.0], 2.0), 3.0, rtol=1e-15)

    poly = [1.12, 432.32, 325.5342, .235532, 32.235]
    assert_close1d(horner_and_der2(poly, 3.0), (14726.109396, 13747.040732, 8553.7884), rtol=1e-15)
    assert_close1d(horner_and_der3(poly, 3.0), (14726.109396, 13747.040732, 8553.7884, 2674.56), rtol=1e-15)

    poly = [1.12, 432.32, 325.5342, .235532, 32.235, 1.01]
    assert_close1d(horner_and_der4(poly, 3.0), (44179.338188, 55967.231592, 53155.446664, 33685.04519999999, 10778.880000000001), rtol=1e-15)
    assert_close1d(horner_and_der4(poly, 3.0), [np.polyval(np.polyder(poly,o), 3) for o in range(5)])


def test_exp_horner_backwards():
    from fluids.numerics.polynomial_evaluation import (
        exp_horner_backwards,
        exp_horner_backwards_and_der,
        exp_horner_backwards_and_der2,
        exp_horner_backwards_and_der3,
    )
    assert_close((exp_horner_backwards(2.0, [1.0, 3.0])), exp(5.0))

    # Test the derivatives
    coeffs = [1,.2,.03,.0004,.00005]
    x = 1.1
    val = exp_horner_backwards(x, coeffs)
    assert_close(val, 5.853794011493425)

    der_num = derivative(lambda x: exp_horner_backwards(x, coeffs), x, dx=x*8e-7, order=7)
    der_ana = exp_horner_backwards_and_der(x, coeffs)[1]
    assert_close(der_ana, 35.804145691898384, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-10)

    der_num = derivative(lambda x: exp_horner_backwards_and_der(x, coeffs)[1], x, dx=x*8e-7, order=7)
    der_ana = exp_horner_backwards_and_der2(x, coeffs)[2]
    assert_close(der_ana, 312.0678014926728, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-10)

    der_num = derivative(lambda x: exp_horner_backwards_and_der2(x, coeffs)[2], x, dx=x*8e-7, order=7)
    der_ana = exp_horner_backwards_and_der3(x, coeffs)[3]
    assert_close(der_ana, 3208.8680487693714, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-10)



def test_horner_backwards_ln_tau():
    from fluids.numerics.polynomial_evaluation import (
        horner_backwards_ln_tau,
        horner_backwards_ln_tau_and_der,
        horner_backwards_ln_tau_and_der2,
        horner_backwards_ln_tau_and_der3,
    )
    coeffs = [9.661381155485653, 224.16316385569456, 2195.419519751738, 11801.26111760343, 37883.05110910901, 74020.46380982929, 87244.40329893673, 69254.45831263301, 61780.155823216155]
    Tc = 591.75
    val = horner_backwards_ln_tau(500.0, Tc, coeffs)
    assert_close(val, 24168.867169087476)

    T = 300.0
    val = horner_backwards_ln_tau(T, Tc, coeffs)
    assert_close(val, 37900.38881665646)

    der_num = derivative(lambda T: horner_backwards_ln_tau(T, Tc, coeffs), T, dx=T*8e-7, order=7)
    der_ana = horner_backwards_ln_tau_and_der(T, Tc, coeffs)[1]
    assert_close(der_ana, -54.63227984184944, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-10)

    der_num = derivative(lambda T: horner_backwards_ln_tau_and_der(T, Tc, coeffs)[1], T, dx=T*8e-7, order=7)
    der_ana = horner_backwards_ln_tau_and_der2(T, Tc, coeffs)[2]
    assert_close(der_ana, 0.037847046150971016, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-8)

    der_num = derivative(lambda T: horner_backwards_ln_tau_and_der2(T, Tc, coeffs)[2], T, dx=T*160e-7, order=7)
    der_ana = horner_backwards_ln_tau_and_der3(T, Tc, coeffs)[3]
    assert_close(der_ana, -0.001920502581912092, rtol=1e-10)
    assert_close(der_num,der_ana, rtol=1e-10)

    assert 0 == horner_backwards_ln_tau(600.0, Tc, coeffs)
    assert_close1d(horner_backwards_ln_tau_and_der(600.0, Tc, coeffs), (0.0, 0.0))
    assert_close1d(horner_backwards_ln_tau_and_der2(600.0, Tc, coeffs), (0.0, 0.0, 0.0))
    assert_close1d(horner_backwards_ln_tau_and_der3(600.0, Tc, coeffs), (0.0, 0.0, 0.0, 0.0))


def test_exp_horner_backwards_ln_tau():
    from fluids.numerics.polynomial_evaluation import exp_horner_backwards_ln_tau, exp_horner_backwards_ln_tau_and_der, exp_horner_backwards_ln_tau_and_der2
    # Coefficients for water from REFPROP, fit
    cs=[-1.2616237655927602e-05, -0.0004061873638525952, -0.005563382112542401, -0.04240531802937599, -0.19805733513004808, -0.5905741856310869, -1.1388001144550794, -0.1477584393673108, -2.401287527958821]
    Tc = 647.096
    T = 300.0
    expect = 0.07175344047522199
    val = exp_horner_backwards_ln_tau(T, Tc, cs)
    assert_close(val, expect, rtol=1e-9)

    assert 0 == exp_horner_backwards_ln_tau(Tc, Tc, cs)

    expect_der = -0.000154224581713238
    val, der = exp_horner_backwards_ln_tau_and_der(T, Tc, cs)
    assert_close(der, expect_der, rtol=1e-13)
    assert_close(val, expect, rtol=1e-9)


    val, der, der2 = exp_horner_backwards_ln_tau_and_der2(T, Tc, cs)
    assert_close(der, expect_der, rtol=1e-13)
    assert_close(val, expect, rtol=1e-9)
    expect_der2 = -5.959538970287795e-07
    assert_close(der2, expect_der2, rtol=1e-13)

    assert 0 == exp_horner_backwards_ln_tau(Tc+1, Tc, cs)
    assert_close1d(exp_horner_backwards_ln_tau_and_der(1000.0, Tc, cs), (0.0, 0.0))
    assert_close1d(exp_horner_backwards_ln_tau_and_der2(1000.0, Tc, cs), (0.0, 0.0, 0.0))

def test_horner_domain():
    test_stable_coeffs = [28.0163226043884, 24.92038363551981, -7.469247118451516, 16.400149851861975, 67.52558234042988, 176.7837155284216]
    xmin, xmax = (162.0, 570.0)
    x = 300
    expect = 157.0804912518053
    calc = horner_domain(x, test_stable_coeffs, xmin, xmax)
    assert_close(calc, expect, rtol=1e-14)


def test_horner_stable():
    x = 300

    test_stable_coeffs = [28.0163226043884, 24.92038363551981, -7.469247118451516, 16.400149851861975, 67.52558234042988, 176.7837155284216]
    xmin, xmax = (162.0, 570.0)
    expect = 157.0804912518053
    offset, scale = polynomial_offset_scale(xmin, xmax)
    calc = horner_stable(x, test_stable_coeffs, offset, scale)
    assert_close(calc, expect, rtol=1e-14)

    der_num = derivative(horner_stable, x, args=(test_stable_coeffs, offset, scale), dx=x*1e-7)
    der_analytical = horner_stable_and_der(x, test_stable_coeffs, offset, scale)[1]
    assert_close(der_analytical, 0.25846754626830115, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)


    der_num = derivative(lambda *args: horner_stable_and_der(*args)[1], x,
                         args=(test_stable_coeffs, offset, scale), dx=x*1e-7)
    der_analytical = horner_stable_and_der2(x, test_stable_coeffs, offset, scale)[2]
    assert_close(der_analytical, 0.0014327609525395784, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)


    der_num = derivative(lambda *args: horner_stable_and_der2(*args)[-1], x,
                         args=(test_stable_coeffs, offset, scale), dx=x*1e-7)
    der_analytical = horner_stable_and_der3(x, test_stable_coeffs, offset, scale)[-1]
    assert_close(der_analytical, -7.345952769973301e-06, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)


    der_num = derivative(lambda *args: horner_stable_and_der3(*args)[-1], x,
                         args=(test_stable_coeffs, offset, scale), dx=x*1e-7)
    der_analytical = horner_stable_and_der4(x, test_stable_coeffs, offset, scale)[-1]
    assert_close(der_analytical, -2.8269861583547557e-07, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)

    five_vals = horner_stable_and_der4(x, test_stable_coeffs, offset, scale)
    assert_close1d(five_vals, (157.0804912518053, 0.25846754626830115, 0.0014327609525395784, -7.345952769973301e-06, -2.8269861583547557e-07), rtol=1e-14)

    four_vals = horner_stable_and_der3(x, test_stable_coeffs, offset, scale)
    assert_close1d(four_vals, (157.0804912518053, 0.25846754626830115, 0.0014327609525395784, -7.345952769973301e-06), rtol=1e-14)

    three_vals = horner_stable_and_der2(x, test_stable_coeffs, offset, scale)
    assert_close1d(three_vals, (157.0804912518053, 0.25846754626830115, 0.0014327609525395784), rtol=1e-14)

    two_vals = horner_stable_and_der(x, test_stable_coeffs, offset, scale)
    assert_close1d(two_vals, (157.0804912518053, 0.25846754626830115), rtol=1e-14)

def test_stablepoly_ln_tau():
    Tmin, Tmax, Tc = 178.18, 591.74, 591.75
    coeffs = [-0.00854738149791956, 0.05600738152861595, -0.30758192972280085, 1.6848304651211947, -8.432931053161155, 37.83695791102946, -150.87603890354512, 526.4891248463246, -1574.7593541151946, 3925.149223414621, -7826.869059381197, 11705.265444382389, -11670.331914006258, 5817.751307862842]

    expect = 24498.131947494512
    expect_d, expect_d2, expect_d3 = -100.77476796035525, -0.6838185833621794, -0.012093191888904656
    xmin, xmax = log(1-Tmin/Tc), log(1-Tmax/Tc)

    offset, scale = polynomial_offset_scale(xmin, xmax)
    T = 500.0

    calc = horner_stable_ln_tau(T, Tc, coeffs, offset, scale)
    assert_close(expect, calc)
    assert_close(horner_stable_ln_tau(700.0, Tc, coeffs, offset, scale), 0.0)

    calc2 = horner_stable_ln_tau_and_der(T, Tc, coeffs, offset, scale)
    assert_close(expect, calc2[0])
    assert_close(expect_d, calc2[1])

    assert_close1d(horner_stable_ln_tau_and_der(700.0, Tc, coeffs, offset, scale), (0.0, 0.0))



    calc3 = horner_stable_ln_tau_and_der2(T, Tc, coeffs, offset, scale)
    assert_close(expect, calc3[0])
    assert_close(expect_d, calc3[1])
    assert_close(expect_d2, calc3[2])
    assert_close1d(horner_stable_ln_tau_and_der2(700.0, Tc, coeffs, offset, scale), (0.0, 0.0, 0.0))


    calc4 = horner_stable_ln_tau_and_der3(T, Tc, coeffs, offset, scale)
    assert_close(expect, calc4[0])
    assert_close(expect_d, calc4[1])
    assert_close(expect_d2, calc4[2])
    assert_close(expect_d3, calc4[3])
    assert_close1d(horner_stable_ln_tau_and_der3(700.0, Tc, coeffs, offset, scale), (0.0, 0.0, 0.0, 0.0))

def test_exp_stablepoly_fit():
    xmin, xmax = 309.0, 591.72
    coeffs = [0.008603558174828078, 0.007358688688856427, -0.016890323025782954, -0.005289197721114957, -0.0028824712174469625, 0.05130960832946553, -0.12709896610233662, 0.37774977659528036, -0.9595325030688526, 2.7931528759840174, 13.10149649770156]
    x = 400
    offset, scale = polynomial_offset_scale(xmin, xmax)
    expect = 157191.01706242564
    calc = exp_horner_stable(x, coeffs, offset, scale)
    assert_close(calc, expect, rtol=1e-14)

    der_num = derivative(exp_horner_stable, x, args=(coeffs, offset, scale), dx=x*1e-7)
    der_analytical = exp_horner_stable_and_der(x, coeffs, offset, scale)[1]
    assert_close(der_num, der_analytical, rtol=1e-7)
    assert_close(der_analytical, 4056.436943642117, rtol=1e-14)

    der_num = derivative(lambda *args: exp_horner_stable_and_der(*args)[1], x,
                         args=(coeffs, offset, scale), dx=x*1e-7)
    der_analytical = exp_horner_stable_and_der2(x, coeffs, offset, scale)[-1]
    assert_close(der_analytical, 81.32645570045084, rtol=1e-14)
    assert_close(der_num, der_analytical, rtol=1e-7)


    der_num = derivative(lambda *args: exp_horner_stable_and_der2(*args)[-1], x,
                         args=(coeffs, offset, scale), dx=x*1e-7)
    der_analytical = exp_horner_stable_and_der3(x, coeffs, offset, scale)[-1]
    assert_close(der_num, der_analytical, rtol=1e-7)
    assert_close(der_analytical, 1.103603650822488, rtol=1e-14)

    vals = exp_horner_stable_and_der3(x, coeffs, offset, scale)
    assert_close1d(vals, (157191.01706242564, 4056.436943642117, 81.32645570045084, 1.103603650822488), rtol=1e-14)

    vals = exp_horner_stable_and_der2(x, coeffs, offset, scale)
    assert_close1d(vals, (157191.01706242564, 4056.436943642117, 81.32645570045084), rtol=1e-14)
    vals = exp_horner_stable_and_der(x, coeffs, offset, scale)
    assert_close1d(vals, (157191.01706242564, 4056.436943642117), rtol=1e-14)

def test_exp_stablepoly_fit_ln_tau():
    coeffs = [0.011399360373616219, -0.014916568994522095, -0.06881296308711171, 0.0900153056718409, 0.19066633691545576, -0.24937350547406822, -0.3148389292182401, 0.41171834646956995, 0.3440581845934503, -0.44989947455906076, -0.2590532901358529, 0.33869134876113094, 0.1391329435696207, -0.18195230788023764, -0.050437145563137165, 0.06583166394466389, 0.01685157036382634, -0.022266583863000733, 0.003539388708205138, -0.005171064606571463, 0.012264455189935575, -0.018085676249990357, 0.026950795197264732, -0.04077120220662778, 0.05786417011592615, -0.07222889554773304, 0.07433570330647113, -0.05829288696590232, -3.7182636506596722, -5.844828481765601]
    Tmin, Tmax, Tc = 233.22, 646.15, 647.096
    xmin, xmax = log(1-Tmin/Tc), log(1-Tmax/Tc)
    offset, scale = polynomial_offset_scale(xmin, xmax)

    T = 500
    expect = 0.03126447402046822
    expect_d, expect_d2 = -0.0002337992205182661, -1.0453011134030858e-07
    calc = exp_horner_stable_ln_tau(T, Tc, coeffs, offset, scale)
    assert_close(expect, calc)
    assert 0 == exp_horner_stable_ln_tau(700, Tc, coeffs, offset, scale)

    calc2 = exp_horner_stable_ln_tau_and_der(T, Tc, coeffs, offset, scale)
    assert (0,0) == exp_horner_stable_ln_tau_and_der(700, Tc, coeffs, offset, scale)
    assert_close(expect, calc2[0])
    assert_close(expect_d, calc2[1])


    calc3 = exp_horner_stable_ln_tau_and_der2(T, Tc, coeffs, offset, scale)
    assert (0,0, 0) == exp_horner_stable_ln_tau_and_der2(700, Tc, coeffs, offset, scale)
    assert_close(expect, calc3[0])
    assert_close(expect_d, calc3[1])
    assert_close(expect_d2, calc3[2])




def test_horner_log():
    coeffs = [1.0, 2.0, 3.0]
    calc = horner_log(coeffs, 5.3, 4.5)
    expect = 40.22161020291425
    assert_close(calc, expect, rtol=1e-13)


def test_horner_stable_log():
    # Have yet to be able to get coeffs for horner_stable_log directly
    # Would work with mpmath for sure though
    int_T_stable_coeffs = [3.2254691548856482, 6.923549962582991, 1.5349425170308109, -1.2154278928596742, 0.2691917024300867, 0.08207134574884214, -0.00891350763588927, -0.14463666181618448, 0.20857726049059955, -0.16593159587411, 0.034375567510854466, 0.0906937625089119, -0.0049590621229368415, -0.11605604886860361, -0.23898450392809112, 0.5368870089626024, 0.1796614633975993, -0.7638508609027873, -0.07854882554664272, 0.7411591719079632, -0.06004401012118543, -0.45423261676213483, 0.13111045246315745, 0.11263751330617959, -0.04919816763192376]

    int_T_log_coeff = 33.94307866530405
    offset = -1.2872369320147412
    scale= 0.0011436184660073706

    calc = horner_stable_log(300, int_T_stable_coeffs, offset, scale, int_T_log_coeff)
    assert_close(calc, 193.51466126959178)
