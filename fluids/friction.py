# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains correlations for single-phase friction factor
in a range of geometries.  It also contains several tables of reported material
roughnesses, and some basic functionality showing how to calculate
single-phase pressure drop.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Friction Factor Interfaces
--------------------------
.. autofunction:: friction_factor
.. autofunction:: friction_factor_methods
.. autofunction:: friction_factor_curved
.. autofunction:: friction_factor_curved_methods
.. autofunction:: helical_Re_crit

Pipe Friction Factor Correlations
---------------------------------
.. autofunction:: ft_Crane
.. autofunction:: Colebrook
.. autofunction:: Clamond
.. autofunction:: friction_laminar
.. autofunction:: Moody
.. autofunction:: Blasius
.. autofunction:: von_Karman
.. autofunction:: Prandtl_von_Karman_Nikuradse
.. autofunction:: Alshul_1952
.. autofunction:: Wood_1966
.. autofunction:: Churchill_1973
.. autofunction:: Eck_1973
.. autofunction:: Jain_1976
.. autofunction:: Swamee_Jain_1976
.. autofunction:: Churchill_1977
.. autofunction:: Chen_1979
.. autofunction:: Round_1980
.. autofunction:: Shacham_1980
.. autofunction:: Barr_1981
.. autofunction:: Zigrang_Sylvester_1
.. autofunction:: Zigrang_Sylvester_2
.. autofunction:: Haaland
.. autofunction:: Serghides_1
.. autofunction:: Serghides_2
.. autofunction:: Tsal_1989
.. autofunction:: Manadilli_1997
.. autofunction:: Romeo_2002
.. autofunction:: Sonnad_Goudar_2006
.. autofunction:: Rao_Kumar_2007
.. autofunction:: Buzzelli_2008
.. autofunction:: Avci_Karagoz_2009
.. autofunction:: Papaevangelo_2010
.. autofunction:: Brkic_2011_1
.. autofunction:: Brkic_2011_2
.. autofunction:: Fang_2011
.. autodata:: LAMINAR_TRANSITION_PIPE

Curved Pipe Friction Factor Correlations
----------------------------------------
.. autofunction:: helical_laminar_fd_White
.. autofunction:: helical_laminar_fd_Mori_Nakayama
.. autofunction:: helical_laminar_fd_Schmidt
.. autofunction:: helical_turbulent_fd_Schmidt
.. autofunction:: helical_turbulent_fd_Mori_Nakayama
.. autofunction:: helical_turbulent_fd_Prasad
.. autofunction:: helical_turbulent_fd_Czop
.. autofunction:: helical_turbulent_fd_Guo
.. autofunction:: helical_turbulent_fd_Ju
.. autofunction:: helical_turbulent_fd_Srinivasan
.. autofunction:: helical_turbulent_fd_Mandal_Nigam
.. autofunction:: helical_transition_Re_Seth_Stahel
.. autofunction:: helical_transition_Re_Ito
.. autofunction:: helical_transition_Re_Kubair_Kuloor
.. autofunction:: helical_transition_Re_Kutateladze_Borishanskii
.. autofunction:: helical_transition_Re_Schmidt
.. autofunction:: helical_transition_Re_Srinivasan

Other Geometry Friction Factor Correlations
-------------------------------------------
.. autofunction:: friction_plate_Martin_1999
.. autofunction:: friction_plate_Martin_VDI
.. autofunction:: friction_plate_Kumar
.. autofunction:: friction_plate_Muley_Manglik

Experimental Friction Data
--------------------------
.. autodata:: oregon_smooth_data

Roughness
---------
.. autofunction:: material_roughness
.. autofunction:: nearest_material_roughness
.. autofunction:: roughness_Farshad
.. autodata:: HHR_roughness

Pressure Drop Calculation
-------------------------
.. autofunction:: one_phase_dP
.. autofunction:: one_phase_dP_gravitational
.. autofunction:: one_phase_dP_dz_acceleration
.. autofunction:: one_phase_dP_acceleration

Utilities
---------
.. autofunction:: transmission_factor

"""

from __future__ import division
from math import sqrt, log, log10, exp, cos, sin, tan, pi, radians, isinf
from fluids.constants import inch, g
from fluids.numerics import secant, lambertw
from fluids.core import Dean, Reynolds


__all__ = ['friction_factor', 'friction_factor_methods',
           'friction_factor_curved', 'helical_Re_crit',
           'friction_factor_curved_methods', 'Colebrook',
           'Clamond',
           'friction_laminar', 'one_phase_dP', 'one_phase_dP_gravitational',
           'one_phase_dP_dz_acceleration', 'one_phase_dP_acceleration',
           'transmission_factor', 'material_roughness',
           'nearest_material_roughness', 'roughness_Farshad',
           '_Farshad_roughness', '_roughness', 'HHR_roughness',
           'Moody', 'Alshul_1952', 'Wood_1966', 'Churchill_1973',
'Eck_1973', 'Jain_1976', 'Swamee_Jain_1976', 'Churchill_1977', 'Chen_1979',
'Round_1980', 'Shacham_1980', 'Barr_1981', 'Zigrang_Sylvester_1',
'Zigrang_Sylvester_2', 'Haaland', 'Serghides_1', 'Serghides_2', 'Tsal_1989',
'Manadilli_1997', 'Romeo_2002', 'Sonnad_Goudar_2006', 'Rao_Kumar_2007',
'Buzzelli_2008', 'Avci_Karagoz_2009', 'Papaevangelo_2010', 'Brkic_2011_1',
'Brkic_2011_2', 'Fang_2011', 'Blasius', 'von_Karman',
'Prandtl_von_Karman_Nikuradse', 'ft_Crane', 'helical_laminar_fd_White',
'helical_laminar_fd_Mori_Nakayama', 'helical_laminar_fd_Schmidt',
'helical_turbulent_fd_Schmidt', 'helical_turbulent_fd_Mori_Nakayama',
'helical_turbulent_fd_Prasad', 'helical_turbulent_fd_Czop',
'helical_turbulent_fd_Guo', 'helical_turbulent_fd_Ju',
'helical_turbulent_fd_Srinivasan',
'helical_turbulent_fd_Mandal_Nigam', 'helical_transition_Re_Seth_Stahel',
'helical_transition_Re_Ito', 'helical_transition_Re_Kubair_Kuloor',
'helical_transition_Re_Kutateladze_Borishanskii',
'helical_transition_Re_Schmidt', 'helical_transition_Re_Srinivasan',
'LAMINAR_TRANSITION_PIPE', 'oregon_smooth_data',
'friction_plate_Martin_1999', 'friction_plate_Martin_VDI',
'friction_plate_Kumar', 'friction_plate_Muley_Manglik']


fuzzy_match_fun = None
def fuzzy_match(name, strings):
    global fuzzy_match_fun
    if fuzzy_match_fun is not None:
        return fuzzy_match_fun(name, strings)

    try:
        from fuzzywuzzy import process, fuzz
        fuzzy_match_fun = lambda name, strings: process.extract(name, strings, limit=10)[0][0]
        # extractOne is faster but less reliable
        #fuzzy_match_fun = lambda name, strings: process.extractOne(name, strings, scorer=fuzz.partial_ratio)[0]
    except ImportError: # pragma: no cover
        import difflib
        fuzzy_match_fun = lambda name, strings: difflib.get_close_matches(name, strings, n=1, cutoff=0)[0]
    return fuzzy_match_fun(name, strings)

LAMINAR_TRANSITION_PIPE = 2040.
'''Believed to be the most accurate result to date. Accurate to +/- 10.
Avila, Kerstin, David Moxey, Alberto de Lozar, Marc Avila, Dwight Barkley, and
Björn Hof. "The Onset of Turbulence in Pipe Flow." Science 333, no. 6039
(July 8, 2011): 192-196. doi:10.1126/science.1203223.
'''

oregon_Res = [11.21, 20.22, 29.28, 43.19, 57.73, 64.58, 86.05, 113.3, 135.3,
              157.5, 179.4, 206.4, 228.0, 270.9, 315.2, 358.9, 402.9, 450.2,
              522.5, 583.1, 671.8, 789.8, 891.0, 1013.0, 1197.0, 1300.0,
              1390.0, 1669.0, 1994.0, 2227.0, 2554.0, 2868.0, 2903.0, 2926.0,
              2955.0, 2991.0, 2997.0, 3047.0, 3080.0, 3264.0, 3980.0, 4835.0,
              5959.0, 8162.0, 10900.0, 13650.0, 18990.0, 29430.0, 40850.0,
              59220.0, 84760.0, 120000.0, 176000.0, 237700.0, 298200.0,
              467800.0, 587500.0, 824200.0, 1050000.0]

oregon_fd_smooth = [5.537, 3.492, 2.329, 1.523, 1.173, 0.9863, 0.7826, 0.5709,
                    0.4815, 0.4182, 0.3655, 0.3237, 0.2884, 0.2433, 0.2077,
                    0.1834, 0.1656, 0.1475, 0.1245, 0.1126, 0.09917, 0.08501,
                    0.07722, 0.06707, 0.0588, 0.05328, 0.04815, 0.04304,
                    0.03739, 0.03405, 0.03091, 0.02804, 0.03182, 0.03846,
                    0.03363, 0.04124, 0.035, 0.03875, 0.04285, 0.0426, 0.03995,
                    0.03797, 0.0361, 0.03364, 0.03088, 0.02903, 0.0267,
                    0.02386, 0.02086, 0.02, 0.01805, 0.01686, 0.01594, 0.01511,
                    0.01462, 0.01365, 0.01313, 0.01244, 0.01198]

oregon_smooth_data = (oregon_Res, oregon_fd_smooth)
'''Holds a tuple of experimental results from the smooth pipe flow experiments
presented in McKEON, B. J., C. J. SWANSON, M. V. ZAGAROLA, R. J. DONNELLY, and
A. J. SMITS. "Friction Factors for Smooth Pipe Flow." Journal of Fluid
Mechanics 511 (July 1, 2004): 41-44. doi:10.1017/S0022112004009796.
'''

def friction_laminar(Re):
    r'''Calculates Darcy friction factor for laminar flow, as shown in [1]_ or
    anywhere else.

    .. math::
        f_d = \frac{64}{Re}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    For round pipes, this valid for :math:`Re \approx< 2040`.

    Results in [2]_ show that this theoretical solution calculates too low of
    friction factors from Re = 10 and up, with an average deviation of 4%.

    Examples
    --------
    >>> friction_laminar(128)
    0.5

    References
    ----------
    .. [1] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    .. [2] McKEON, B. J., C. J. SWANSON, M. V. ZAGAROLA, R. J. DONNELLY, and
       A. J. SMITS. "Friction Factors for Smooth Pipe Flow." Journal of Fluid
       Mechanics 511 (July 1, 2004): 41-44. doi:10.1017/S0022112004009796.
    '''
    return 64./Re


def Blasius(Re):
    r'''Calculates Darcy friction factor according to the Blasius formulation,
    originally presented in [1]_ and described more recently in [2]_.

    .. math::
        f_d=\frac{0.3164}{Re^{0.25}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Developed for 3000 < Re < 200000.

    Examples
    --------
    >>> Blasius(10000)
    0.03164

    References
    ----------
    .. [1] Blasius, H."Das Aehnlichkeitsgesetz bei Reibungsvorgängen in
       Flüssigkeiten." In Mitteilungen über Forschungsarbeiten auf dem Gebiete
       des Ingenieurwesens, edited by Verein deutscher Ingenieure, 1-41.
       Berlin, Heidelberg: Springer Berlin Heidelberg, 1913.
       http://rd.springer.com/chapter/10.1007/978-3-662-02239-9_1.
    .. [2] Hager, W. H. "Blasius: A Life in Research and Education." In
       Experiments in Fluids, 566–571, 2003.
    '''
    return 0.3164/sqrt(sqrt(Re))


def Colebrook(Re, eD, tol=None):
    r'''Calculates Darcy friction factor using the Colebrook equation
    originally published in [1]_. Normally, this function uses an exact
    solution to the Colebrook equation, derived with a CAS. A numerical can
    also be used.

    .. math::
        \frac{1}{\sqrt{f}}=-2\log_{10}\left(\frac{\epsilon/D}{3.7}
        +\frac{2.51}{\text{Re}\sqrt{f}}\right)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]
    tol : float, optional
        None for analytical solution (default); user specified value to use the
        numerical solution; 0 to use `mpmath` and provide a bit-correct exact
        solution to the maximum fidelity of the system's `float`;
        -1 to apply the Clamond solution where appropriate for greater speed
        (Re > 10), [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    The solution is as follows:

    .. math::
        f_d = \frac{\ln(10)^2\cdot {3.7}^2\cdot{2.51}^2}
        {\left(\ln(10)\epsilon/D\cdot\text{Re} - 2\cdot 2.51\cdot 3.7\cdot
        \text{lambertW}\left[\ln(\sqrt{10})\sqrt{
        10^{\left(\frac{\epsilon \text{Re}}{2.51\cdot 3.7D}\right)}
        \cdot \text{Re}^2/{2.51}^2}\right]\right)}

    Some effort to optimize this function has been made. The `lambertw`
    function from scipy is used, and is defined to solve the specific function:

    .. math::
        y = x\exp(x)

        \text{lambertW}(y) = x

    This is relatively slow despite its explicit form as it uses the
    mathematical function `lambertw` which is expensive to compute.

    For high relative roughness and Reynolds numbers, an OverflowError can be
    encountered in the solution of this equation. The numerical solution is
    then used.

    The numerical solution provides values which are generally within an
    rtol of 1E-12 to the analytical solution; however, due to the different
    rounding order, it is possible for them to be as different as rtol 1E-5 or
    higher. The 1E-5 accuracy regime has been tested and confirmed numerically
    for hundreds of thousand of points within the region 1E-12 < Re < 1E12
    and 0 < eD < 0.1.

    The numerical solution attempts the secant method using `scipy`'s `newton`
    solver, and in the event of nonconvergence, attempts the `fsolve` solver
    as well. An initial guess is provided via the `Clamond` function.

    The numerical and analytical solution take similar amounts of time; the
    `mpmath` solution used when `tol=0` is approximately 45 times slower. This
    function takes approximately 8 us normally.

    Examples
    --------
    >>> Colebrook(1E5, 1E-4)
    0.018513866077471

    References
    ----------
    .. [1] Colebrook, C F."Turbulent Flow in Pipes, with Particular Reference
       to the Transition Region Between the Smooth and Rough Pipe Laws."
       Journal of the ICE 11, no. 4 (February 1, 1939): 133-156.
       doi:10.1680/ijoti.1939.13150.
    '''
    if tol == -1:
        if Re > 10.0:
            return Clamond(Re, eD, False)
        else:
            tol = None
    elif tol == 0:
#        from sympy import LambertW, Rational, log, sqrt
#        Re = Rational(Re)
#        eD_Re = Rational(eD)*Re
#        sub = 1/Rational('6.3001')*10**(1/Rational('9.287')*eD_Re)*Re*Re
#        lambert_term = LambertW(log(sqrt(10))*sqrt(sub))
#        den = log(10)*eD_Re - 18.574*lambert_term
#        return float(log(10)**2*Rational('3.7')**2*Rational('2.51')**2/(den*den))
        try:
            from mpmath import mpf, log, mp, sqrt as sqrtmp
            from mpmath import lambertw as mp_lambertw
        except:
            raise ImportError('For exact solutions, the `mpmath` library is '
                              'required')
        mp.dps = 50
        Re = mpf(Re)
        eD_Re = mpf(eD)*Re
        sub = 1/mpf('6.3001')*10**(1/mpf('9.287')*eD_Re)*Re*Re
        lambert_term = mp_lambertw(log(sqrtmp(10))*sqrtmp(sub))
        den = log(10)*eD_Re - 18.574*lambert_term
        return float(log(10)**2*mpf('3.7')**2*mpf('2.51')**2/(den*den))
    if tol is None:
        try:
            eD_Re = eD*Re
            # 9.287 = 2.51*3.7; 6.3001 = 2.51**2
            # xn = 1/6.3001 = 0.15872763924382155
            # 1/9.287 = 0.10767739851405189
            sub = 0.15872763924382155*10.0**(0.10767739851405189*eD_Re)*Re*Re
            if isinf(sub):
                #  Can't continue, need numerical approach
                raise OverflowError
            # 1.15129... = log(sqrt(10))
            lambert_term = float(lambertw(1.151292546497022950546806896454654633998870849609375*sqrt(sub)).real)
            # log(10) = 2.302585...; 2*2.51*3.7 = 18.574
            # 457.28... = log(10)**2*3.7**2*2.51**2
            den = 2.30258509299404590109361379290930926799774169921875*eD_Re - 18.574*lambert_term
            return 457.28006463294371997108100913465023040771484375/(den*den)
        except OverflowError:
            pass
    # Either user-specified tolerance, or an error in the analytical solution
    if tol is None:
        tol = 1e-12
    try:
        fd_guess = Clamond(Re, eD)
    except ValueError:
        fd_guess = Blasius(Re)
    def err(x):
        # Convert the newton search domain to always positive
        f_12_inv = 1.0/sqrt(abs(x))
        # 0.27027027027027023 = 1/3.7
        return f_12_inv + 2.0*log10(eD*0.27027027027027023 + 2.51/Re*f_12_inv)
    fd = abs(secant(err, fd_guess, xtol=tol))
    return fd


def Clamond(Re, eD, fast=False):
    r"""Calculates Darcy friction factor using a solution accurate to almost
    machine precision. Recommended very strongly. For details of the algorithm,
    see [1]_.

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]
    fast : bool, optional
        If true, performs only one iteration, which gives roughly half the
        number of decimals of accuracy, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    This is a highly optimized function, 4 times faster than the solution using
    the LambertW function, and faster than many other approximations which are
    much less accurate.

    The code used here is only slightly modified than that in [1]_, for further
    performance improvements.

    For 10 < Re < 1E12, and 0 < eD < 0.01, this equation has been confirmed
    numerically to provide a solution to the Colebrook equation accurate to an
    rtol of 1E-9 or better - the same level of accuracy as the analytical
    solution to the Colebrook equation due to floating point precision.

    Comparing this to the numerical solution of the Colebrook equation,
    identical values are given accurate to an rtol of 1E-9 for 10 < Re < 1E100,
    and 0 < eD < 1 and beyond.

    However, for values of Re under 10, different answers from the `Colebrook`
    equation appear and then quickly a ValueError is raised.

    Examples
    --------
    >>> Clamond(1E5, 1E-4)
    0.01851386607747165

    References
    ----------
    .. [1] Clamond, Didier. "Efficient Resolution of the Colebrook Equation."
       Industrial & Engineering Chemistry Research 48, no. 7 (April 1, 2009):
       3665-71. doi:10.1021/ie801626g.
       http://math.unice.fr/%7Edidierc/DidPublis/ICR_2009.pdf
    """
    X1 = eD*Re*0.1239681863354175460160858261654858382699 # (log(10)/18.574).evalf(40)
    X2 = log(Re) - 0.7793974884556819406441139701653776731705 # log(log(10)/5.02).evalf(40)
    F = X2 - 0.2
    X1F = X1 + F
    X1F1 = 1. + X1F

    E = (log(X1F) - 0.2)/(X1F1)
    F = F - (X1F1 + 0.5*E)*E*(X1F)/(X1F1 + E*(1. + (1.0/3.0)*E))

    if not fast:
        X1F = X1 + F
        X1F1 = 1. + X1F
        E = (log(X1F) + F - X2)/(X1F1)

        b = (X1F1 + E*(1. + 1.0/3.0*E))
        F = b/(b*F -  ((X1F1 + 0.5*E)*E*(X1F)))
        return 1.325474527619599502640416597148504422899*(F*F) # ((0.5*log(10))**2).evalf(40)

    return 1.325474527619599502640416597148504422899/(F*F) # ((0.5*log(10))**2).evalf(40)


def Moody(Re, eD):
    r'''Calculates Darcy friction factor using the method in Moody (1947)
    as shown in [1]_ and originally in [2]_.

    .. math::
        f_f = 1.375\times 10^{-3}\left[1+\left(2\times10^4\frac{\epsilon}{D} +
        \frac{10^6}{Re}\right)^{1/3}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is Re >= 4E3 and Re <= 1E8; eD >= 0 < 0.01.

    Examples
    --------
    >>> Moody(1E5, 1E-4)
    0.01809185666808665

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Moody, L.F.: An approximate formula for pipe friction factors.
       Trans. Am. Soc. Mech. Eng. 69,1005-1006 (1947)
    '''
    return 4*(1.375E-3*(1 + (2E4*eD + 1E6/Re)**(1/3.)))


def Alshul_1952(Re, eD):
    r'''Calculates Darcy friction factor using the method in Alshul (1952)
    as shown in [1]_.

    .. math::
        f_d = 0.11\left( \frac{68}{Re} + \frac{\epsilon}{D}\right)^{0.25}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Alshul_1952(1E5, 1E-4)
    0.018382997825686878

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    '''
    return 0.11*sqrt(sqrt(68/Re + eD))


def Wood_1966(Re, eD):
    r'''Calculates Darcy friction factor using the method in Wood (1966) [2]_
    as shown in [1]_.

    .. math::
        f_d = 0.094(\frac{\epsilon}{D})^{0.225} + 0.53(\frac{\epsilon}{D})
        + 88(\frac{\epsilon}{D})^{0.4}Re^{-A_1}

    .. math::
        A_1 = 1.62(\frac{\epsilon}{D})^{0.134}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 5E7;  1E-5 <= eD <= 4E-2.

    Examples
    --------
    >>> Wood_1966(1E5, 1E-4)
    0.021587570560090762

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Wood, D.J.: An Explicit Friction Factor Relationship, vol. 60.
       Civil Engineering American Society of Civil Engineers (1966)
    '''
    A1 = 1.62*eD**0.134
    return 0.094*eD**0.225 + 0.53*eD +88*eD**0.4*Re**-A1


def Churchill_1973(Re, eD):
    r'''Calculates Darcy friction factor using the method in Churchill (1973)
    [2]_ as shown in [1]_

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log_{10}\left[\frac{\epsilon}{3.7D} +
        (\frac{7}{Re})^{0.9}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Churchill_1973(1E5, 1E-4)
    0.01846708694482294

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Churchill, Stuart W. "Empirical Expressions for the Shear
       Stress in Turbulent Flow in Commercial Pipe." AIChE Journal 19, no. 2
       (March 1, 1973): 375-76. doi:10.1002/aic.690190228.
    '''
    return (-2*log10(eD/3.7 + (7./Re)**0.9))**-2


def Eck_1973(Re, eD):
    r'''Calculates Darcy friction factor using the method in Eck (1973)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log_{10}\left[\frac{\epsilon}{3.715D}
        + \frac{15}{Re}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Eck_1973(1E5, 1E-4)
    0.01775666973488564

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Eck, B.: Technische Stromungslehre. Springer, New York (1973)
    '''
    return (-2*log10(eD/3.715 + 15/Re))**-2


def Jain_1976(Re, eD):
    r'''Calculates Darcy friction factor using the method in Jain (1976)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = 2.28 - 4\log_{10}\left[ \frac{\epsilon}{D} +
        \left(\frac{29.843}{Re}\right)^{0.9}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 5E3 <= Re <= 1E7;  4E-5 <= eD <= 5E-2.

    Examples
    --------
    >>> Jain_1976(1E5, 1E-4)
    0.018436560312693327

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Jain, Akalank K."Accurate Explicit Equation for Friction Factor."
       Journal of the Hydraulics Division 102, no. 5 (May 1976): 674-77.
    '''
    ff = (2.28-4*log10(eD+(29.843/Re)**0.9))**-2
    return 4*ff


def Swamee_Jain_1976(Re, eD):
    r'''Calculates Darcy friction factor using the method in Swamee and
    Jain (1976) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log_{10}\left[\left(\frac{6.97}{Re}\right)^{0.9}
        + (\frac{\epsilon}{3.7D})\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 5E3 <= Re <= 1E8;  1E-6 <= eD <= 5E-2.

    Examples
    --------
    >>> Swamee_Jain_1976(1E5, 1E-4)
    0.018452424431901808

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Swamee, Prabhata K., and Akalank K. Jain."Explicit Equations for
       Pipe-Flow Problems." Journal of the Hydraulics Division 102, no. 5
       (May 1976): 657-664.
    '''
    ff = (-4*log10((6.97/Re)**0.9 + eD/3.7))**-2
    return 4*ff


def Churchill_1977(Re, eD):
    r'''Calculates Darcy friction factor using the method in Churchill and
    (1977) [2]_ as shown in [1]_.

    .. math::
        f_f = 2\left[(\frac{8}{Re})^{12} + (A_2 + A_3)^{-1.5}\right]^{1/12}

    .. math::
        A_2 = \left\{2.457\ln\left[(\frac{7}{Re})^{0.9}
        + 0.27\frac{\epsilon}{D}\right]\right\}^{16}

    .. math::
        A_3 = \left( \frac{37530}{Re}\right)^{16}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Churchill_1977(1E5, 1E-4)
    0.018462624566280075

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2]	Churchill, S.W.: Friction factor equation spans all fluid flow
       regimes. Chem. Eng. J. 91, 91-92 (1977)
    '''
    A3 = (37530/Re)**16
    A2 = (2.457*log((7./Re)**0.9 + 0.27*eD))**16
    ff = 2*((8/Re)**12 + 1/(A2+A3)**1.5)**(1/12.)
    return 4*ff


def Chen_1979(Re, eD):
    r'''Calculates Darcy friction factor using the method in Chen (1979) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log_{10}\left[\frac{\epsilon}{3.7065D}
        -\frac{5.0452}{Re}\log_{10} A_4\right]

    .. math::
        A_4 = \frac{(\epsilon/D)^{1.1098}}{2.8257}
        + \left(\frac{7.149}{Re}\right)^{0.8981}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 4E8;  1E-7 <= eD <= 5E-2.

    Examples
    --------
    >>> Chen_1979(1E5, 1E-4)
    0.018552817507472126

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Chen, Ning Hsing. "An Explicit Equation for Friction Factor in
       Pipe." Industrial & Engineering Chemistry Fundamentals 18, no. 3
       (August 1, 1979): 296-97. doi:10.1021/i160071a019.
    '''
    A4 = eD**1.1098/2.8257 + (7.149/Re)**0.8981
    ff = (-4*log10(eD/3.7065 - 5.0452/Re*log10(A4)))**-2
    return 4*ff


def Round_1980(Re, eD):
    r'''Calculates Darcy friction factor using the method in Round (1980) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -3.6\log_{10}\left[\frac{Re}{0.135Re
        \frac{\epsilon}{D}+6.5}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 4E8;  0 <= eD <= 5E-2.

    Examples
    --------
    >>> Round_1980(1E5, 1E-4)
    0.01831475391244354

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Round, G. F."An Explicit Approximation for the Friction
       Factor-Reynolds Number Relation for Rough and Smooth Pipes." The
       Canadian Journal of Chemical Engineering 58, no. 1 (February 1, 1980):
       122-23. doi:10.1002/cjce.5450580119.
    '''
    ff = (-3.6*log10(Re/(0.135*Re*eD+6.5)))**-2
    return 4*ff


def Shacham_1980(Re, eD):
    r'''Calculates Darcy friction factor using the method in Shacham (1980) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log_{10}\left[\frac{\epsilon}{3.7D} -
        \frac{5.02}{Re} \log_{10}\left(\frac{\epsilon}{3.7D}
        + \frac{14.5}{Re}\right)\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 4E8

    Examples
    --------
    >>> Shacham_1980(1E5, 1E-4)
    0.01860641215097828

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Shacham, M. "Comments on: 'An Explicit Equation for Friction
       Factor in Pipe.'" Industrial & Engineering Chemistry Fundamentals 19,
       no. 2 (May 1, 1980): 228-228. doi:10.1021/i160074a019.
    '''
    ff = (-4*log10(eD/3.7 - 5.02/Re*log10(eD/3.7 + 14.5/Re)))**-2
    return 4*ff


def Barr_1981(Re, eD):
    r'''Calculates Darcy friction factor using the method in Barr (1981) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log_{10}\left\{\frac{\epsilon}{3.7D} +
        \frac{4.518\log_{10}(\frac{Re}{7})}{Re\left[1+\frac{Re^{0.52}}{29}
        \left(\frac{\epsilon}{D}\right)^{0.7}\right]}\right\}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Barr_1981(1E5, 1E-4)
    0.01849836032779929

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2]	Barr, Dih, and Colebrook White."Technical Note. Solutions Of The
       Colebrook-White Function For Resistance To Uniform Turbulent Flow."
       ICE Proceedings 71, no. 2 (January 6, 1981): 529-35.
       doi:10.1680/iicep.1981.1895.
    '''
    fd = (-2*log10(eD/3.7 + 4.518*log10(Re/7.)/(Re*(1+Re**0.52/29*eD**0.7))))**-2
    return fd


def Zigrang_Sylvester_1(Re, eD):
    r'''Calculates Darcy friction factor using the method in
     Zigrang and Sylvester (1982) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log_{10}\left[\frac{\epsilon}{3.7D}
        - \frac{5.02}{Re}\log_{10} A_5\right]

        A_5 = \frac{\epsilon}{3.7D} + \frac{13}{Re}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 1E8;  4E-5 <= eD <= 5E-2.

    Examples
    --------
    >>> Zigrang_Sylvester_1(1E5, 1E-4)
    0.018646892425980794

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Zigrang, D. J., and N. D. Sylvester."Explicit Approximations to the
       Solution of Colebrook's Friction Factor Equation." AIChE Journal 28,
       no. 3 (May 1, 1982): 514-15. doi:10.1002/aic.690280323.
    '''
    A5 = eD/3.7 + 13/Re
    ff = (-4*log10(eD/3.7 - 5.02/Re*log10(A5)))**-2
    return 4*ff


def Zigrang_Sylvester_2(Re, eD):
    r'''Calculates Darcy friction factor using the second method in
     Zigrang and Sylvester (1982) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log_{10}\left[\frac{\epsilon}{3.7D}
        - \frac{5.02}{Re}\log_{10} A_6\right]

    .. math::
        A_6 = \frac{\epsilon}{3.7D} - \frac{5.02}{Re}\log_{10} A_5

    .. math::
        A_5 = \frac{\epsilon}{3.7D} + \frac{13}{Re}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 1E8;  4E-5 <= eD <= 5E-2

    Examples
    --------
    >>> Zigrang_Sylvester_2(1E5, 1E-4)
    0.01850021312358548

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Zigrang, D. J., and N. D. Sylvester."Explicit Approximations to the
       Solution of Colebrook's Friction Factor Equation." AIChE Journal 28,
       no. 3 (May 1, 1982): 514-15. doi:10.1002/aic.690280323.
    '''
    A5 = eD/3.7 + 13/Re
    A6 = eD/3.7 - 5.02/Re*log10(A5)
    ff = (-4*log10(eD/3.7 - 5.02/Re*log10(A6)))**-2
    return 4*ff


def Haaland(Re, eD):
    r'''Calculates Darcy friction factor using the method in
     Haaland (1983) [2]_ as shown in [1]_.

    .. math::
        f_f = \left(-1.8\log_{10}\left[\left(\frac{\epsilon/D}{3.7}
        \right)^{1.11} + \frac{6.9}{Re}\right]\right)^{-2}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 1E8;  1E-6 <= eD <= 5E-2

    Examples
    --------
    >>> Haaland(1E5, 1E-4)
    0.018265053014793857

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Haaland, S. E."Simple and Explicit Formulas for the Friction Factor
       in Turbulent Pipe Flow." Journal of Fluids Engineering 105, no. 1
       (March 1, 1983): 89-90. doi:10.1115/1.3240948.
    '''
    ff = (-3.6*log10(6.9/Re +(eD/3.7)**1.11))**-2
    return 4*ff


def Serghides_1(Re, eD):
    r'''Calculates Darcy friction factor using the method in Serghides (1984)
    [2]_ as shown in [1]_.

    .. math::
        f=\left[A-\frac{(B-A)^2}{C-2B+A}\right]^{-2}

    .. math::
        A=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{12}{Re}\right]

    .. math::
        B=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{2.51A}{Re}\right]

    .. math::
        C=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{2.51B}{Re}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Serghides_1(1E5, 1E-4)
    0.01851358983180063

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Serghides T.K (1984)."Estimate friction factor accurately"
       Chemical Engineering, Vol. 91(5), pp. 63-64.
    '''
    A = -2*log10(eD/3.7 + 12/Re)
    B = -2*log10(eD/3.7 + 2.51*A/Re)
    C = -2*log10(eD/3.7 + 2.51*B/Re)
    return (A - (B-A)**2/(C-2*B + A))**-2


def Serghides_2(Re, eD):
    r'''Calculates Darcy friction factor using the method in Serghides (1984)
    [2]_ as shown in [1]_.

    .. math::
        f_d = \left[ 4.781 - \frac{(A - 4.781)^2}
        {B-2A+4.781}\right]^{-2}

    .. math::
        A=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{12}{Re}\right]

    .. math::
        B=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{2.51A}{Re}\right]


    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Serghides_2(1E5, 1E-4)
    0.018486377560664482

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2]	Serghides T.K (1984)."Estimate friction factor accurately"
       Chemical Engineering, Vol. 91(5), pp. 63-64.
    '''
    A = -2*log10(eD/3.7 + 12/Re)
    B = -2*log10(eD/3.7 + 2.51*A/Re)
    return (4.781 - (A - 4.781)**2/(B - 2*A + 4.781))**-2


def Tsal_1989(Re, eD):
    r'''Calculates Darcy friction factor using the method in Tsal (1989)
    [2]_ as shown in [1]_.

    .. math::
        A = 0.11(\frac{68}{Re} + \frac{\epsilon}{D})^{0.25}

    if :math:`A >= 0.018` then `fd = A`;

    if :math:`A < 0.018` then :math:`fd = 0.0028 + 0.85 A`.

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 1E8;  0 <= eD <= 5E-2

    Examples
    --------
    >>> Tsal_1989(1E5, 1E-4)
    0.018382997825686878

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Tsal, R.J.: Altshul-Tsal friction factor equation.
       Heat-Piping-Air Cond. 8, 30-45 (1989)
    '''
    A = 0.11*sqrt(sqrt(68/Re + eD))
    if A >= 0.018:
        return A
    else:
        return 0.0028 + 0.85*A


def Manadilli_1997(Re, eD):
    r'''Calculates Darcy friction factor using the method in Manadilli (1997)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log_{10}\left[\frac{\epsilon}{3.7D} +
        \frac{95}{Re^{0.983}} - \frac{96.82}{Re}\right]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 5.245E3 <= Re <= 1E8;  0 <= eD <= 5E-2

    Examples
    --------
    >>> Manadilli_1997(1E5, 1E-4)
    0.01856964649724108

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Manadilli, G.: Replace implicit equations with signomial functions.
       Chem. Eng. 104, 129 (1997)
    '''
    return (-2*log10(eD/3.7 + 95/Re**0.983 - 96.82/Re))**-2


def Romeo_2002(Re, eD):
    r'''Calculates Darcy friction factor using the method in Romeo (2002)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log_{10}\left\{\frac{\epsilon}{3.7065D}\times
        \frac{5.0272}{Re}\times\log_{10}\left[\frac{\epsilon}{3.827D} -
        \frac{4.567}{Re}\times\log_{10}\left(\frac{\epsilon}{7.7918D}^{0.9924} +
        \left(\frac{5.3326}{208.815+Re}\right)^{0.9345}\right)\right]\right\}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 3E3 <= Re <= 1.5E8;  0 <= eD <= 5E-2

    Examples
    --------
    >>> Romeo_2002(1E5, 1E-4)
    0.018530291219676177

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Romeo, Eva, Carlos Royo, and Antonio Monzon."Improved Explicit
       Equations for Estimation of the Friction Factor in Rough and Smooth
       Pipes." Chemical Engineering Journal 86, no. 3 (April 28, 2002): 369-74.
       doi:10.1016/S1385-8947(01)00254-6.
    '''
    fd = (-2*log10(eD/3.7065-5.0272/Re*log10(eD/3.827-4.567/Re*log10((eD/7.7918)**0.9924+(5.3326/(208.815+Re))**0.9345))))**-2
    return fd


def Sonnad_Goudar_2006(Re, eD):
    r'''Calculates Darcy friction factor using the method in Sonnad and Goudar
    (2006) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = 0.8686\ln\left(\frac{0.4587Re}{S^{S/(S+1)}}\right)

    .. math::
        S = 0.1240\times\frac{\epsilon}{D}\times Re + \ln(0.4587Re)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 4E3 <= Re <= 1E8;  1E-6 <= eD <= 5E-2

    Examples
    --------
    >>> Sonnad_Goudar_2006(1E5, 1E-4)
    0.0185971269898162

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Travis, Quentin B., and Larry W. Mays."Relationship between
       Hazen-William and Colebrook-White Roughness Values." Journal of
       Hydraulic Engineering 133, no. 11 (November 2007): 1270-73.
       doi:10.1061/(ASCE)0733-9429(2007)133:11(1270).
    '''
    S = 0.124*eD*Re + log(0.4587*Re)
    return (.8686*log(.4587*Re/S**(S/(S+1))))**-2


def Rao_Kumar_2007(Re, eD):
    r'''Calculates Darcy friction factor using the method in Rao and Kumar
    (2007) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = 2\log_{10}\left(\frac{(2\frac{\epsilon}{D})^{-1}}
        {\left(\frac{0.444 + 0.135Re}{Re}\right)\beta}\right)

    .. math::
        \beta = 1 - 0.55\exp(-0.33\ln\left[\frac{Re}{6.5}\right]^2)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.
    This equation is fit to original experimental friction factor data.
    Accordingly, this equation should not be used unless appropriate
    consideration is given.

    Examples
    --------
    >>> Rao_Kumar_2007(1E5, 1E-4)
    0.01197759334600925

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] Rao, A.R., Kumar, B.: Friction factor for turbulent pipe flow.
       Division of Mechanical Sciences, Civil Engineering Indian Institute of
       Science Bangalore, India ID Code 9587 (2007)
    '''
    beta = 1 - 0.55*exp(-0.33*(log(Re/6.5))**2)
    return (2*log10((2*eD)**-1/beta/((0.444+0.135*Re)/Re)))**-2


def Buzzelli_2008(Re, eD):
    r'''Calculates Darcy friction factor using the method in Buzzelli (2008)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = B_1 - \left[\frac{B_1 +2\log_{10}(\frac{B_2}{Re})}
        {1 + \frac{2.18}{B_2}}\right]

    .. math::
        B_1 = \frac{0.774\ln(Re)-1.41}{1+1.32\sqrt{\frac{\epsilon}{D}}}

    .. math::
        B_2 = \frac{\epsilon}{3.7D}Re+2.51\times B_1

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Buzzelli_2008(1E5, 1E-4)
    0.018513948401365277

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Buzzelli, D.: Calculating friction in one step.
       Mach. Des. 80, 54-55 (2008)
    '''
    B1 = (.774*log(Re)-1.41)/(1.0 + 1.32*sqrt(eD))
    B2 = eD/3.7*Re + 2.51*B1
    return (B1- (B1+2*log10(B2/Re))/(1+2.18/B2))**-2


def Avci_Karagoz_2009(Re, eD):
    r'''Calculates Darcy friction factor using the method in Avci and Karagoz
    (2009) [2]_ as shown in [1]_.

    .. math::
        f_D = \frac{6.4} {\left\{\ln(Re) - \ln\left[
        1 + 0.01Re\frac{\epsilon}{D}\left(1 + 10(\frac{\epsilon}{D})^{0.5}
        \right)\right]\right\}^{2.4}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Avci_Karagoz_2009(1E5, 1E-4)
    0.01857058061066499

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2]	Avci, Atakan, and Irfan Karagoz."A Novel Explicit Equation for
       Friction Factor in Smooth and Rough Pipes." Journal of Fluids
       Engineering 131, no. 6 (2009): 061203. doi:10.1115/1.3129132.
    '''
    return 6.4*(log(Re) - log(1 + 0.01*Re*eD*(1+10*sqrt(eD))))**-2.4


def Papaevangelo_2010(Re, eD):
    r'''Calculates Darcy friction factor using the method in Papaevangelo
    (2010) [2]_ as shown in [1]_.

    .. math::
        f_D = \frac{0.2479 - 0.0000947(7-\ln Re)^4}{\left[\log_{10}\left
        (\frac{\epsilon}{3.615D} + \frac{7.366}{Re^{0.9142}}\right)\right]^2}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 1E4 <= Re <= 1E7;  1E-5 <= eD <= 1E-3

    Examples
    --------
    >>> Papaevangelo_2010(1E5, 1E-4)
    0.015685600818488177

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Papaevangelou, G., Evangelides, C., Tzimopoulos, C.: A New Explicit
       Relation for the Friction Factor Coefficient in the Darcy-Weisbach
       Equation, pp. 166-172. Protection and Restoration of the Environment
       Corfu, Greece: University of Ioannina Greece and Stevens Institute of
       Technology New Jersey (2010)
    '''
    return (0.2479-0.0000947*(7-log(Re))**4)/(log10(eD/3.615 + 7.366/Re**0.9142))**2


def Brkic_2011_1(Re, eD):
    r'''Calculates Darcy friction factor using the method in Brkic
    (2011) [2]_ as shown in [1]_.

    .. math::
        f_d = [-2\log_{10}(10^{-0.4343\beta} + \frac{\epsilon}{3.71D})]^{-2}

    .. math::
        \beta = \ln \frac{Re}{1.816\ln\left(\frac{1.1Re}{\ln(1+1.1Re)}\right)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Brkic_2011_1(1E5, 1E-4)
    0.01812455874141297

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Brkic, Dejan."Review of Explicit Approximations to the Colebrook
       Relation for Flow Friction." Journal of Petroleum Science and
       Engineering 77, no. 1 (April 2011): 34-48.
       doi:10.1016/j.petrol.2011.02.006.
    '''
    beta = log(Re/(1.816*log(1.1*Re/log(1+1.1*Re))))
    return (-2*log10(10**(-0.4343*beta)+eD/3.71))**-2


def Brkic_2011_2(Re, eD):
    r'''Calculates Darcy friction factor using the method in Brkic
    (2011) [2]_ as shown in [1]_.

    .. math::
        f_d = [-2\log_{10}(\frac{2.18\beta}{Re}+ \frac{\epsilon}{3.71D})]^{-2}

    .. math::
        \beta = \ln \frac{Re}{1.816\ln\left(\frac{1.1Re}{\ln(1+1.1Re)}\right)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    No range of validity specified for this equation.

    Examples
    --------
    >>> Brkic_2011_2(1E5, 1E-4)
    0.018619745410688716

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Brkic, Dejan."Review of Explicit Approximations to the Colebrook
       Relation for Flow Friction." Journal of Petroleum Science and
       Engineering 77, no. 1 (April 2011): 34-48.
       doi:10.1016/j.petrol.2011.02.006.
    '''
    beta = log(Re/(1.816*log(1.1*Re/log(1+1.1*Re))))
    return (-2*log10(2.18*beta/Re + eD/3.71))**-2


def Fang_2011(Re, eD):
    r'''Calculates Darcy friction factor using the method in Fang
    (2011) [2]_ as shown in [1]_.

    .. math::
        f_D = 1.613\left\{\ln\left[0.234\frac{\epsilon}{D}^{1.1007}
        - \frac{60.525}{Re^{1.1105}}
        + \frac{56.291}{Re^{1.0712}}\right]\right\}^{-2}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Range is 3E3 <= Re <= 1E8;  0 <= eD <= 5E-2

    Examples
    --------
    >>> Fang_2011(1E5, 1E-4)
    0.018481390682985432

    References
    ----------
    .. [1] Winning, H. and T. Coole. "Explicit Friction Factor Accuracy and
       Computational Efficiency for Turbulent Flow in Pipes." Flow, Turbulence
       and Combustion 90, no. 1 (January 1, 2013): 1-27.
       doi:10.1007/s10494-012-9419-7
    .. [2] 	Fang, Xiande, Yu Xu, and Zhanru Zhou."New Correlations of
       Single-Phase Friction Factor for Turbulent Pipe Flow and Evaluation of
       Existing Single-Phase Friction Factor Correlations." Nuclear Engineering
       and Design, The International Conference on Structural Mechanics in
       Reactor Technology (SMiRT19) Special Section, 241, no. 3 (March 2011):
       897-902. doi:10.1016/j.nucengdes.2010.12.019.
    '''
    return log(0.234*eD**1.1007 - 60.525/Re**1.1105 + 56.291/Re**1.0712)**-2*1.613


def von_Karman(eD):
    r'''Calculates Darcy friction factor for rough pipes at infinite Reynolds
    number from the von Karman equation (as given in [1]_ and [2]_:

    .. math::
        \frac{1}{\sqrt{f_d}} = -2 \log_{10} \left(\frac{\epsilon/D}{3.7}\right)

    Parameters
    ----------
    eD : float
        Relative roughness, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    This case does not actually occur; Reynolds number is always finite.
    It is normally applied as a "limiting" value when a pipe's roughness is so
    high it has a friction factor curve effectively independent of Reynods
    number.

    Examples
    --------
    >>> von_Karman(1E-4)
    0.01197365149564789

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] McGovern, Jim. "Technical Note: Friction Factor Diagrams for Pipe
       Flow." Paper, October 3, 2011. http://arrow.dit.ie/engschmecart/28.
    '''
    x = log10(eD/3.71)
    return 0.25/(x*x)


def Prandtl_von_Karman_Nikuradse(Re):
    r'''Calculates Darcy friction factor for smooth pipes as a function of
    Reynolds number from the Prandtl-von Karman Nikuradse equation as given
    in [1]_ and [2]_:

    .. math::
        \frac{1}{\sqrt{f}} = -2\log_{10}\left(\frac{2.51}{Re\sqrt{f}}\right)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    This equation is often stated as follows; the correct constant is not 0.8,
    but 2log10(2.51) or approximately 0.7993474:

    .. math::
        \frac{1}{\sqrt{f}}\approx 2\log_{10}(\text{Re}\sqrt{f})-0.8

    This function is calculable for all Reynolds numbers between 1E151 and
    1E-151. It is solved with the LambertW function from SciPy. The solution is:

    .. math::
        f_d = \frac{\frac{1}{4}\log_{10}^2}{\left(\text{lambertW}\left(\frac{
        \lb(10)Re}{2(2.51)}\right)\right)^2}

    Examples
    --------
    >>> Prandtl_von_Karman_Nikuradse(1E7)
    0.008102669430

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] McGovern, Jim. "Technical Note: Friction Factor Diagrams for Pipe
       Flow." Paper, October 3, 2011. http://arrow.dit.ie/engschmecart/28.
    '''
    # Good 1E150 to 1E-150
    c1 = 1.151292546497022842008995727342182103801 # log(10)/2
    c2 = 1.325474527619599502640416597148504422899 # log(10)**2/4
    return c2/float(lambertw((c1*Re)/2.51).real)**2


# Values still in table at least to 2013
Crane_fts_nominal_Ds = [.015, .02, .025, .032, .04, .05, .065, .08, .1, .125,
                        .15, .2, .25, .35, .4, .55, .6, .9]

Crane_fts_Ds = [0.01576, 0.02096, 0.02664, 0.03508, 0.04094, 0.05248, 0.06268,
                0.07792, 0.10226, 0.1282, 0.154, 0.20274, 0.25446, 0.33334,
                0.381, 0.53994, 0.57504, 0.8759]

Crane_fts = [.026, .024, .022, .021, .02, .019, .018, .017, .016, .015, .015,
             .014, .013, .013, .012, .012, .011, .011]


def ft_Crane(D):
    r'''Calculates the Crane fully turbulent Darcy friction factor for flow in
    commercial pipe, as used in the Crane formulas for loss coefficients in
    various fittings. Note that this is **not generally applicable to loss
    due to friction in pipes**, as it does not take into account the roughness
    of various pipe materials. But for fittings in any type of pipe, this is
    the friction factor to use in the Crane [1]_ method to get their loss
    coefficients.

    Parameters
    ----------
    D : float
        Pipe inner diameter, [m]

    Returns
    -------
    fd : float
        Darcy Crane friction factor for fully turbulent flow, [-]

    Notes
    -----
    There is confusion and uncertainty regarding the friction factor table
    given in Crane TP 410M [1]_. This function does not help: it implements a
    new way to obtain Crane friction factors, so that it can better be based in
    theory and give more precision (not accuracy) and trend better with
    diameters not tabulated in [1]_.

    The data in [1]_ was digitized, and nominal pipe diameters were converted
    to actual pipe diameters. An objective function was sought which would
    produce the exact same values as in [1]_ when rounded to the same decimal
    place. One was found fairly easily by using the standard `Colebrook`
    friction factor formula, and using the diameter-dependent roughness values
    calculated with the `roughness_Farshad` method for bare Carbon steel. A
    diameter-dependent Reynolds number was required to match the values;
    the :math:`\rho V/\mu` term is set to 7.5E6.

    The formula given in [1]_ is:

    .. math::
        f_T = \frac{0.25}{\left[\log_{10}\left(\frac{\epsilon/D}{3.7}\right)
        \right]^2}

    However, this function does not match the rounded values in [1]_ well and
    it is not very clear which roughness to use. Using both the value for new
    commercial steel (.05 mm) or a diameter-dependent value
    (`roughness_Farshad`), values were found to be too high and too low
    respectively. That function is based in theory - the limit of the
    `Colebrook` equation when `Re` goes to infinity - but in the end real pipe
    flow is not infinity, and so a large error occurs from that use.

    The following plot shows all these options, and that the method implemented
    here matches perfectly the rounded values in [1]_.

    .. plot:: plots/ft_Crane_plot.py

    Examples
    --------
    >>> ft_Crane(.1)
    0.01628845962146481

    Explicitly spelling out the function (note the exact same answer is not
    returned; it is accurate to 5-8 decimals however, for increased speed):

    >>> Di = 0.1
    >>> Colebrook(7.5E6*Di, eD=roughness_Farshad(ID='Carbon steel, bare', D=Di)/Di)
    0.0162884254312

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    fast = True
    if D < 1E-2:
        fast = False
    return Clamond(7.5E6*D, 3.4126825352925e-5*D**-1.0112, fast)


fmethods = {'Moody': (4000.0, 100000000.0, 0.0, 1.0),
 'Alshul_1952': (None, None, None, None),
 'Wood_1966': (4000.0, 50000000.0, 1e-05, 0.04),
 'Churchill_1973': (None, None, None, None),
 'Eck_1973': (None, None, None, None),
 'Jain_1976': (5000.0, 10000000.0, 4e-05, 0.05),
 'Swamee_Jain_1976': (5000.0, 100000000.0, 1e-06, 0.05),
 'Churchill_1977': (None, None, None, None),
 'Chen_1979': (4000.0, 400000000.0, 1e-07, 0.05),
 'Round_1980': (4000.0, 400000000.0, 0.0, 0.05),
 'Shacham_1980': (4000.0, 400000000.0, None, None),
 'Barr_1981': (None, None, None, None),
 'Zigrang_Sylvester_1': (4000.0, 100000000.0, 4e-05, 0.05),
 'Zigrang_Sylvester_2': (4000.0, 100000000.0, 4e-05, 0.05),
 'Haaland': (4000.0, 100000000.0, 1e-06, 0.05),
 'Serghides_1': (None, None, None, None),
 'Serghides_2': (None, None, None, None),
 'Tsal_1989': (4000.0, 100000000.0, 0.0, 0.05),
 'Manadilli_1997': (5245.0, 100000000.0, 0.0, 0.05),
 'Romeo_2002': (3000.0, 150000000.0, 0.0, 0.05),
 'Sonnad_Goudar_2006': (4000.0, 100000000.0, 1e-06, 0.05),
 'Rao_Kumar_2007': (None, None, None, None),
 'Buzzelli_2008': (None, None, None, None),
 'Avci_Karagoz_2009': (None, None, None, None),
 'Papaevangelo_2010': (10000.0, 10000000.0, 1e-05, 0.001),
 'Brkic_2011_1': (None, None, None, None),
 'Brkic_2011_2': (None, None, None, None),
 'Fang_2011': (3000.0, 100000000.0, 0.0, 0.05),
 'Clamond': (0, None, 0.0, None),
 'Colebrook': (0, None, 0.0, None)}


def friction_factor_methods(Re, eD=0.0, check_ranges=True):
    r'''Returns a list of correlation names for calculating friction factor
    for internal pipe flow.

    Examples
    --------
    >>> len(friction_factor_methods(Re=1E5, eD=1E-4))
    30

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float, optional
        Relative roughness of the wall, [-]
    check_ranges : bool, optional
        Whether to filter the list for correlations which claim to be valid for
        the given values, [-]

    Returns
    -------
    methods : list
        List of methods which claim to be valid for the range of `Re` and `eD`
        given, [-]
    '''
    if check_ranges:
        if Re < LAMINAR_TRANSITION_PIPE:
            return ['laminar']
        methods = []
        for n, (Re_min, Re_max, eD_min, eD_max) in fmethods.items():
            if Re_min is not None and Re < Re_min:
                continue
            if Re_max is not None and Re > Re_max:
                continue
            if eD_min is not None and eD < eD_min:
                continue
            if eD_max is not None and eD > eD_max:
                continue
            methods.append(n)
        return methods
    else:
        return list(fmethods.keys()) + ['laminar']


def friction_factor(Re, eD=0.0, Method='Clamond', Darcy=True):
    r'''Calculates friction factor. Uses a specified method, or automatically
    picks one from the dictionary of available methods. 29 approximations are
    available as well as the direct solution, described in the table below.
    The default is to use the exact solution.

    For Re < 2040, [1]_ the laminar solution is always returned, regardless of
    selected method.

    Examples
    --------
    >>> friction_factor(Re=1E5, eD=1E-4)
    0.01851386607747165
    >>> friction_factor(Re=2.9E5, eD=1E-5, Method='Serghides_2')
    0.0146199041093456

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float, optional
        Relative roughness of the wall, [-]

    Returns
    -------
    f : float
        Friction factor, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use
    Darcy : bool, optional
        If False, will return fanning friction factor, 1/4 of the Darcy value

    See Also
    --------
    Colebrook
    Clamond

    Notes
    -----
    A table of the supposed limits of each correlation is as follows. Note that
    the spaces in the method names are placed by underscores in the actual
    function names and when provided as the `Method` argument. The default
    method is likely to be sufficient.


    +-------------------+------+------+----------------------+----------------------+
    |Nice name          |Re min|Re max|:math:`\epsilon/D` Min|:math:`\epsilon/D` Max|
    +===================+======+======+======================+======================+
    |Clamond            |0     |None  |0                     |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Rao Kumar 2007     |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Eck 1973           |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Jain 1976          |5000  |1.0E+7|4.0E-5                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Avci Karagoz 2009  |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Swamee Jain 1976   |5000  |1.0E+8|1.0E-6                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Churchill 1977     |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Brkic 2011 1       |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Chen 1979          |4000  |4.0E+8|1.0E-7                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Round 1980         |4000  |4.0E+8|0                     |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Papaevangelo 2010  |10000 |1.0E+7|1.0E-5                |0.001                 |
    +-------------------+------+------+----------------------+----------------------+
    |Fang 2011          |3000  |1.0E+8|0                     |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Shacham 1980       |4000  |4.0E+8|None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Barr 1981          |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Churchill 1973     |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Moody              |4000  |1.0E+8|0                     |1                     |
    +-------------------+------+------+----------------------+----------------------+
    |Zigrang Sylvester 1|4000  |1.0E+8|4.0E-5                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Zigrang Sylvester 2|4000  |1.0E+8|4.0E-5                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Buzzelli 2008      |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Haaland            |4000  |1.0E+8|1.0E-6                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Serghides 1        |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Serghides 2        |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Tsal 1989          |4000  |1.0E+8|0                     |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Alshul 1952        |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Wood 1966          |4000  |5.0E+7|1.0E-5                |0.04                  |
    +-------------------+------+------+----------------------+----------------------+
    |Manadilli 1997     |5245  |1.0E+8|0                     |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Brkic 2011 2       |None  |None  |None                  |None                  |
    +-------------------+------+------+----------------------+----------------------+
    |Romeo 2002         |3000  |1.5E+8|0                     |0.05                  |
    +-------------------+------+------+----------------------+----------------------+
    |Sonnad Goudar 2006 |4000  |1.0E+8|1.0E-6                |0.05                  |
    +-------------------+------+------+----------------------+----------------------+

    References
    ----------
    .. [1] Avila, Kerstin, David Moxey, Alberto de Lozar, Marc Avila, Dwight
       Barkley, and Björn Hof. "The Onset of Turbulence in Pipe Flow." Science
       333, no. 6039 (July 8, 2011): 192-96. doi:10.1126/science.1203223.
    '''
    if Method is None:
        Method = 'Clamond'

    if Re < LAMINAR_TRANSITION_PIPE or Method == 'laminar':
        f = friction_laminar(Re)
    elif Method == "Clamond":
        f = Clamond(Re, eD, False)
    elif Method == "Colebrook":
        f = Colebrook(Re, eD)
    elif Method == "Moody":
        f = Moody(Re, eD)
    elif Method == "Alshul_1952":
        f = Alshul_1952(Re, eD)
    elif Method == "Wood_1966":
        f = Wood_1966(Re, eD)
    elif Method == "Churchill_1973":
        f = Churchill_1973(Re, eD)
    elif Method == "Eck_1973":
        f = Eck_1973(Re, eD)
    elif Method == "Jain_1976":
        f = Jain_1976(Re, eD)
    elif Method == "Swamee_Jain_1976":
        f = Swamee_Jain_1976(Re, eD)
    elif Method == "Churchill_1977":
        f = Churchill_1977(Re, eD)
    elif Method == "Chen_1979":
        f = Chen_1979(Re, eD)
    elif Method == "Round_1980":
        f = Round_1980(Re, eD)
    elif Method == "Shacham_1980":
        f = Shacham_1980(Re, eD)
    elif Method == "Barr_1981":
        f = Barr_1981(Re, eD)
    elif Method == "Zigrang_Sylvester_1":
        f = Zigrang_Sylvester_1(Re, eD)
    elif Method == "Zigrang_Sylvester_2":
        f = Zigrang_Sylvester_2(Re, eD)
    elif Method == "Haaland":
        f = Haaland(Re, eD)
    elif Method == "Serghides_1":
        f = Serghides_1(Re, eD)
    elif Method == "Serghides_2":
        f = Serghides_2(Re, eD)
    elif Method == "Tsal_1989":
        f = Tsal_1989(Re, eD)
    elif Method == "Manadilli_1997":
        f = Manadilli_1997(Re, eD)
    elif Method == "Romeo_2002":
        f = Romeo_2002(Re, eD)
    elif Method == "Sonnad_Goudar_2006":
        f = Sonnad_Goudar_2006(Re, eD)
    elif Method == "Rao_Kumar_2007":
        f = Rao_Kumar_2007(Re, eD)
    elif Method == "Buzzelli_2008":
        f = Buzzelli_2008(Re, eD)
    elif Method == "Avci_Karagoz_2009":
        f = Avci_Karagoz_2009(Re, eD)
    elif Method == "Papaevangelo_2010":
        f = Papaevangelo_2010(Re, eD)
    elif Method == "Brkic_2011_1":
        f = Brkic_2011_1(Re, eD)
    elif Method == "Brkic_2011_2":
        f = Brkic_2011_2(Re, eD)
    elif Method == "Fang_2011":
        f = Fang_2011(Re, eD)
    else:
        raise ValueError("Method not recognized")
    if not Darcy:
        f *= 0.25
    return f


def helical_laminar_fd_White(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under laminar conditions, using the method of
    White [1]_ as shown in [2]_.

    .. math::
        f_{curved} = f_{\text{straight,laminar}} \left[1 - \left(1-\left(
        \frac{11.6}{De}\right)^{0.45}\right)^{\frac{1}{0.45}}\right]^{-1}

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    The range of validity of this equation is :math:`11.6< De < 2000`,
    :math:`3.878\times 10^{-4}<D_i/D_c < 0.066`.

    The form of the equation means it yields nonsense results for De < 11.6;
    at De < 11.6, the equation is modified to return the straight pipe value.

    This model is recommended in [3]_, with a slight modification for Dean
    numbers larger than 2000.

    Examples
    --------
    >>> helical_laminar_fd_White(250, .02, .1)
    0.4063281817830202

    References
    ----------
    .. [1] White, C. M. "Streamline Flow through Curved Pipes." Proceedings of
       the Royal Society of London A: Mathematical, Physical and Engineering
       Sciences 123, no. 792 (April 6, 1929): 645-63.
       doi:10.1098/rspa.1929.0089.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    De = Dean(Re=Re, Di=Di, D=Dc)
    fd = friction_laminar(Re)
    if De < 11.6:
        return fd
    return fd/(1. - (1. - (11.6/De)**0.45)**(1./0.45)) # 1/.45 sometimes said to be 2.2


def helical_laminar_fd_Mori_Nakayama(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under laminar conditions, using the method of
    Mori and Nakayama [1]_ as shown in [2]_ and [3]_.

    .. math::
        f_{curved} = f_{\text{straight,laminar}} \left(\frac{0.108\sqrt{De}}
        {1-3.253De^{-0.5}}\right)

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    The range of validity of this equation is :math:`100 < De < 2000`.

    The form of the equation means it yields nonsense results for De < 42.328;
    under that, the equation is modified to return the value at De=42.328,
    which is a multiplier of 1.405296 on the straight pipe friction factor.

    Examples
    --------
    >>> helical_laminar_fd_Mori_Nakayama(250, .02, .1)
    0.42224582857795434

    References
    ----------
    .. [1] Mori, Yasuo, and Wataru Nakayama. "Study on Forced Convective Heat
       Transfer in Curved Pipes : 1st Report, Laminar Region." Transactions of
       the Japan Society of Mechanical Engineers 30, no. 216 (1964): 977-88.
       doi:10.1299/kikai1938.30.977.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Pimenta, T. A., and J. B. L. M. Campos. "Friction Losses of
       Newtonian and Non-Newtonian Fluids Flowing in Laminar Regime in a
       Helical Coil." Experimental Thermal and Fluid Science 36 (January 2012):
       194-204. doi:10.1016/j.expthermflusci.2011.09.013.
    '''
    De = Dean(Re=Re, Di=Di, D=Dc)
    fd = friction_laminar(Re)
    if De < 42.328036:
        return fd*1.405296
    return fd*(0.108*sqrt(De))/(1. - 3.253*1.0/sqrt(De))


def helical_laminar_fd_Schmidt(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under laminar conditions, using the method of
    Schmidt [1]_ as shown in [2]_ and [3]_.

    .. math::
        f_{curved} = f_{\text{straight,laminar}} \left[1 + 0.14\left(\frac{D_i}
        {D_c}\right)^{0.97}Re^{\left[1 - 0.644\left(\frac{D_i}{D_c}
        \right)^{0.312}\right]}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    The range of validity of this equation is specified only for Re,
    :math:`100 < Re < Re_{critical}`.

    The form of the equation is such that as the curvature becomes negligible,
    straight tube result is obtained.

    Examples
    --------
    >>> helical_laminar_fd_Schmidt(250, .02, .1)
    0.47460725672835236

    References
    ----------
    .. [1] Schmidt, Eckehard F. "Wärmeübergang Und Druckverlust in
       Rohrschlangen." Chemie Ingenieur Technik 39, no. 13 (July 10, 1967):
       781-89. doi:10.1002/cite.330391302.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Pimenta, T. A., and J. B. L. M. Campos. "Friction Losses of
       Newtonian and Non-Newtonian Fluids Flowing in Laminar Regime in a
       Helical Coil." Experimental Thermal and Fluid Science 36 (January 2012):
       194-204. doi:10.1016/j.expthermflusci.2011.09.013.
    '''
    fd = friction_laminar(Re)
    D_ratio = Di/Dc
    return fd*(1. + 0.14*D_ratio**0.97*Re**(1. - 0.644*D_ratio**0.312))


def helical_turbulent_fd_Srinivasan(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Srinivasan [1]_, as shown in [2]_ and [3]_.

    .. math::
        f_d = \frac{0.336}{{\left[Re\sqrt{\frac{D_i}{D_c}}\right]^{0.2}}}

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Valid for 0.01 < Di/Dc < 0.15, with no Reynolds number criteria given in
    [2]_ or [3]_.

    [2]_ recommends this method, using the transition criteria of Srinivasan as
    well. [3]_ recommends using either this method or the Ito method. This
    method did not make it into the popular review articles on curved flow.

    Examples
    --------
    >>> helical_turbulent_fd_Srinivasan(1E4, 0.01, .02)
    0.0570745212117107

    References
    ----------
    .. [1] Srinivasan, PS, SS Nandapurkar, and FA Holland. "Friction Factors
       for Coils." TRANSACTIONS OF THE INSTITUTION OF CHEMICAL ENGINEERS AND
       THE CHEMICAL ENGINEER 48, no. 4-6 (1970): T156
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    De = Dean(Re=Re, Di=Di, D=Dc)
    return 0.336*De**-0.2


def helical_turbulent_fd_Schmidt(Re, Di, Dc, roughness=0):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Schmidt [1]_, also shown in [2]_.

    For :math:`Re_{crit} < Re < 2.2\times 10^{4}`:

    .. math::
        f_{curv} = f_{\text{str,turb}} \left[1 + \frac{2.88\times10^{4}}{Re}
        \left(\frac{D_i}{D_c}\right)^{0.62}\right]

    For :math:`2.2\times 10^{4} < Re < 1.5\times10^{5}`:

    .. math::
        f_{curv} = f_{\text{str,turb}} \left[1 + 0.0823\left(1 + \frac{D_i}
        {D_c}\right)\left(\frac{D_i}{D_c}\right)^{0.53} Re^{0.25}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Valid from the transition to turbulent flow up to
    :math:`Re=1.5\times 10^{5}`. At very low curvatures, converges on the
    straight pipe result.

    Examples
    --------
    >>> helical_turbulent_fd_Schmidt(1E4, 0.01, .02)
    0.08875550767040916

    References
    ----------
    .. [1] Schmidt, Eckehard F. "Wärmeübergang Und Druckverlust in
       Rohrschlangen." Chemie Ingenieur Technik 39, no. 13 (July 10, 1967):
       781-89. doi:10.1002/cite.330391302.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    fd = friction_factor(Re=Re, eD=roughness/Di)
    if Re < 2.2E4:
        return fd*(1. + 2.88E4/Re*(Di/Dc)**0.62)
    else:
        return fd*(1. + 0.0823*(1. + Di/Dc)*(Di/Dc)**0.53*sqrt(sqrt(Re)))


def helical_turbulent_fd_Mori_Nakayama(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Mori and Nakayama [1]_, also shown in [2]_ and [3]_.

    .. math::
        f_{curv} = 0.3\left(\frac{D_i}{D_c}\right)^{0.5}
        \left[Re\left(\frac{D_i}{D_c}\right)^2\right]^{-0.2}\left[1
        + 0.112\left[Re\left(\frac{D_i}{D_c}\right)^2\right]^{-0.2}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Valid from the transition to turbulent flow up to
    :math:`Re=6.5\times 10^{5}\sqrt{D_i/D_c}`. Does not use a straight pipe
    correlation, and so will not converge on the
    straight pipe result at very low curvature.

    Examples
    --------
    >>> helical_turbulent_fd_Mori_Nakayama(1E4, 0.01, .2)
    0.037311802071379796

    References
    ----------
    .. [1] Mori, Yasuo, and Wataru Nakayama. "Study of Forced Convective Heat
       Transfer in Curved Pipes (2nd Report, Turbulent Region)." International
       Journal of Heat and Mass Transfer 10, no. 1 (January 1, 1967): 37-59.
       doi:10.1016/0017-9310(67)90182-2.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Ali, Shaukat. "Pressure Drop Correlations for Flow through Regular
       Helical Coil Tubes." Fluid Dynamics Research 28, no. 4 (April 2001):
       295-310. doi:10.1016/S0169-5983(00)00034-4.
    '''
    term = (Re*(Di/Dc)**2)**-0.2
    return 0.3*1.0/sqrt(Dc/Di)*term*(1. + 0.112*term)


def helical_turbulent_fd_Prasad(Re, Di, Dc,roughness=0):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Prasad [1]_, also shown in [2]_.

    .. math::
        f_{curv} = f_{\text{str,turb}}\left[1 + 0.18\left[Re\left(\frac{D_i}
        {D_c}\right)^2\right]^{0.25}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    No range of validity was specified, but the experiments used were with
    coil/tube diameter ratios of 17.24 and 34.9, hot water in the tube, and
    :math:`1780 < Re < 59500`. At very low curvatures, converges on the
    straight pipe result.

    Examples
    --------
    >>> helical_turbulent_fd_Prasad(1E4, 0.01, .2)
    0.043313098093994626

    References
    ----------
    .. [1] Prasad, B. V. S. S. S., D. H. Das, and A. K. Prabhakar. "Pressure
       Drop, Heat Transfer and Performance of a Helically Coiled Tubular
       Exchanger." Heat Recovery Systems and CHP 9, no. 3 (January 1, 1989):
       249-56. doi:10.1016/0890-4332(89)90008-2.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    fd = friction_factor(Re=Re, eD=roughness/Di)
    return fd*(1. + 0.18*sqrt(sqrt(Re*(Di/Dc)**2)))


def helical_turbulent_fd_Czop (Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Czop [1]_, also shown in [2]_.

    .. math::
        f_{curv} = 0.096De^{-0.1517}

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Valid for :math:`2\times10^4 < Re < 1.5\times10^{5}`. Does not use a
    straight pipe correlation, and so will not converge on the
    straight pipe result at very low curvature.

    Examples
    --------
    >>> helical_turbulent_fd_Czop(1E4, 0.01, .2)
    0.02979575250574106

    References
    ----------
    .. [1] Czop, V., D. Barbier, and S. Dong. "Pressure Drop, Void Fraction and
       Shear Stress Measurements in an Adiabatic Two-Phase Flow in a Coiled
       Tube." Nuclear Engineering and Design 149, no. 1 (September 1, 1994):
       323-33. doi:10.1016/0029-5493(94)90298-4.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    De = Dean(Re=Re, Di=Di, D=Dc)
    return 0.096*De**-0.1517


def helical_turbulent_fd_Guo(Re, Di, Dc):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Guo [1]_, also shown in [2]_.

    .. math::
        f_{curv} = 0.638Re^{-0.15}\left(\frac{D_i}{D_c}\right)^{0.51}

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Valid for :math:`2\times10^4 < Re < 1.5\times10^{5}`. Does not use a
    straight pipe correlation, and so will not converge on the
    straight pipe result at very low curvature.

    Examples
    --------
    >>> helical_turbulent_fd_Guo(2E5, 0.01, .2)
    0.022189161013253147

    References
    ----------
    .. [1] Guo, Liejin, Ziping Feng, and Xuejun Chen. "An Experimental
       Investigation of the Frictional Pressure Drop of Steam–water Two-Phase
       Flow in Helical Coils." International Journal of Heat and Mass Transfer
       44, no. 14 (July 2001): 2601-10. doi:10.1016/S0017-9310(00)00312-4.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    return 0.638*Re**-0.15*(Di/Dc)**0.51


def helical_turbulent_fd_Ju(Re, Di, Dc,roughness=0.0):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Ju et al. [1]_, also shown in [2]_.

    .. math::
        f_{curv} = f_{\text{str,turb}}\left[1 +0.11Re^{0.23}\left(\frac{D_i}
        {D_c}\right)^{0.14}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Claimed to be valid for all turbulent conditions with :math:`De>11.6`.
    At very low curvatures, converges on the straight pipe result.

    Examples
    --------
    >>> helical_turbulent_fd_Ju(1E4, 0.01, .2)
    0.04945959480770937

    References
    ----------
    .. [1] Ju, Huaiming, Zhiyong Huang, Yuanhui Xu, Bing Duan, and Yu Yu.
       "Hydraulic Performance of Small Bending Radius Helical Coil-Pipe."
       Journal of Nuclear Science and Technology 38, no. 10 (October 1, 2001):
       826-31. doi:10.1080/18811248.2001.9715102.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    fd = friction_factor(Re=Re, eD=roughness/Di)
    return fd*(1. + 0.11*Re**0.23*(Di/Dc)**0.14)


def helical_turbulent_fd_Mandal_Nigam(Re, Di, Dc, roughness=0):
    r'''Calculates Darcy friction factor for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Mandal and Nigam [1]_, also shown in [2]_.

    .. math::
        f_{curv} = f_{\text{str,turb}} [1 + 0.03{De}^{0.27}]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    fd : float
        Darcy friction factor for a curved pipe [-]

    Notes
    -----
    Claimed to be valid for all turbulent conditions with
    :math:`2500 < De < 15000`. At very low curvatures, converges on the
    straight pipe result.

    Examples
    --------
    >>> helical_turbulent_fd_Mandal_Nigam(1E4, 0.01, .2)
    0.03831658117115902

    References
    ----------
    .. [1] Mandal, Monisha Mridha, and K. D. P. Nigam. "Experimental Study on
       Pressure Drop and Heat Transfer of Turbulent Flow in Tube in Tube
       Helical Heat Exchanger." Industrial & Engineering Chemistry Research 48,
       no. 20 (October 21, 2009): 9318-24. doi:10.1021/ie9002393.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    De = Dean(Re=Re, Di=Di, D=Dc)
    fd = friction_factor(Re=Re, eD=roughness/Di)
    return fd*(1. + 0.03*De**0.27)


def helical_transition_Re_Seth_Stahel(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_.

    .. math::
        Re_{crit} = 1900\left[1 + 8 \sqrt{\frac{D_i}{D_c}}\right]

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 1900.

    Examples
    --------
    >>> helical_transition_Re_Seth_Stahel(1, 7.)
    7645.0599897402535

    References
    ----------
    .. [1] Seth, K. K., and E. P. Stahel. "HEAT TRANSFER FROM HELICAL COILS
       IMMERSED IN AGITATED VESSELS." Industrial & Engineering Chemistry 61,
       no. 6 (June 1, 1969): 39-49. doi:10.1021/ie50714a007.
    '''
    return 1900.*(1. + 8.*sqrt(Di/Dc))


def helical_transition_Re_Ito(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_,
    as shown in [2]_ and in [3]_.

    .. math::
        Re_{crit} = 20000 \left(\frac{D_i}{D_c}\right)^{0.32}

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 0.
    Recommended for :math:`0.00116 < d_i/D_c  < 0.067`

    Examples
    --------
    >>> helical_transition_Re_Ito(1, 7.)
    10729.972844697186

    References
    ----------
    .. [1] H. Ito. "Friction factors for turbulent flow in curved pipes."
       Journal Basic Engineering, Transactions of the ASME, 81 (1959): 123-134.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Mori, Yasuo, and Wataru Nakayama. "Study on Forced Convective Heat
       Transfer in Curved Pipes." International Journal of Heat and Mass
       Transfer 10, no. 5 (May 1, 1967): 681-95.
       doi:10.1016/0017-9310(67)90113-5.
    '''
    return 2E4*(Di/Dc)**0.32


def helical_transition_Re_Kubair_Kuloor(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_,
    as shown in [2]_.

    .. math::
        Re_{crit} = 12730 \left(\frac{D_i}{D_c}\right)^{0.2}

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 0.
    Recommended for :math:`0.0005 < d_i/D_c < 0.103`

    Examples
    --------
    >>> helical_transition_Re_Kubair_Kuloor(1, 7.)
    8625.986927588123

    References
    ----------
    .. [1] Kubair, Venugopala, and N. R. Kuloor. "Heat Transfer to Newtonian
       Fluids in Coiled Pipes in Laminar Flow." International Journal of Heat
       and Mass Transfer 9, no. 1 (January 1, 1966): 63-75.
       doi:10.1016/0017-9310(66)90057-3.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    return 1.273E4*(Di/Dc)**0.2


def helical_transition_Re_Kutateladze_Borishanskii(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_,
    also shown in [2]_.

    .. math::
        Re_{crit} = 2300 + 1.05\times 10^4 \left(\frac{D_i}{D_c}\right)^{0.3}

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 2300.
    Recommended for :math:`0.0417 < d_i/D_c < 0.1667`

    Examples
    --------
    >>> helical_transition_Re_Kutateladze_Borishanskii(1, 7.)
    7121.143774574058

    References
    ----------
    .. [1] Kutateladze, S. S, and V. M Borishanskiĭ. A Concise Encyclopedia of
       Heat Transfer. Oxford; New York: Pergamon Press, 1966.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    '''
    return 2300. + 1.05E4*(Di/Dc)**0.4


def helical_transition_Re_Schmidt(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_,
    also shown in [2]_ and [3]_. Correlation recommended in [3]_.

    .. math::
        Re_{crit} = 2300\left[1 + 8.6\left(\frac{D_i}{D_c}\right)^{0.45}\right]

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 2300.
    Recommended for :math:`d_i/D_c < 0.14`

    Examples
    --------
    >>> helical_transition_Re_Schmidt(1, 7.)
    10540.094061770815

    References
    ----------
    .. [1] Schmidt, Eckehard F. "Wärmeübergang Und Druckverlust in
       Rohrschlangen." Chemie Ingenieur Technik 39, no. 13 (July 10, 1967):
       781-89. doi:10.1002/cite.330391302.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    return 2300.*(1. + 8.6*(Di/Dc)**0.45)


def helical_transition_Re_Srinivasan(Di, Dc):
    r'''Calculates the transition Reynolds number for flow inside a curved or
    helical coil between laminar and turbulent flow, using the method of [1]_,
    also shown in [2]_ and [3]_. Correlation recommended in [3]_.

    .. math::
        Re_{crit} = 2100\left[1 + 12\left(\frac{D_i}{D_c}\right)^{0.5}\right]

    Parameters
    ----------
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Re_crit : float
        Transition Reynolds number between laminar and turbulent [-]

    Notes
    -----
    At very low curvatures, converges to Re = 2100.
    Recommended for :math:`0.004 < d_i/D_c < 0.1`.

    Examples
    --------
    >>> helical_transition_Re_Srinivasan(1, 7.)
    11624.704719832524

    References
    ----------
    .. [1] Srinivasan, P. S., Nandapurkar, S. S., and Holland, F. A., "Pressure
       Drop and Heat Transfer in Coils", Chemical Engineering, 218, CE131-119,
       (1968).
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    return 2100.*(1. + 12.*sqrt(Di/Dc))


curved_friction_laminar_methods = {'White': helical_laminar_fd_White,
                           'Mori Nakayama laminar': helical_laminar_fd_Mori_Nakayama,
                           'Schmidt laminar': helical_laminar_fd_Schmidt}

# Format: 'key': (correlation, supports_roughness)
curved_friction_turbulent_methods = {'Schmidt turbulent': (helical_turbulent_fd_Schmidt, True),
                                     'Mori Nakayama turbulent': (helical_turbulent_fd_Mori_Nakayama, False),
                                     'Prasad': (helical_turbulent_fd_Prasad, True),
                                     'Czop': (helical_turbulent_fd_Czop, False),
                                     'Guo': (helical_turbulent_fd_Guo, False),
                                     'Ju': (helical_turbulent_fd_Ju, True),
                                     'Mandel Nigam': (helical_turbulent_fd_Mandal_Nigam, True),
                                     'Srinivasan turbulent': (helical_turbulent_fd_Srinivasan, False)}

curved_friction_transition_methods = {'Seth Stahel': helical_transition_Re_Seth_Stahel,
                                      'Ito': helical_transition_Re_Ito,
                                      'Kubair Kuloor': helical_transition_Re_Kubair_Kuloor,
                                      'Kutateladze Borishanskii': helical_transition_Re_Kutateladze_Borishanskii,
                                      'Schmidt': helical_transition_Re_Schmidt,
                                      'Srinivasan': helical_transition_Re_Srinivasan}

_bad_curved_transition_method = '''Invalid method specified for transition Reynolds number;
valid methods are %s''' % list(curved_friction_transition_methods.keys())

curved_friction_turbulent_methods_list = ['Schmidt turbulent', 'Mori Nakayama turbulent', 'Prasad', 'Czop', 'Guo', 'Ju', 'Mandel Nigam', 'Srinivasan turbulent']
curved_friction_laminar_methods_list = ['White', 'Mori Nakayama laminar', 'Schmidt laminar']

def helical_Re_crit(Di, Dc, Method='Schmidt'):
    r'''Calculates the transition Reynolds number for fluid flowing in a
    curved pipe or helical coil. Selects the appropriate regime by default.
    Optionally, a specific correlation can be specified with the `Method`
    keyword.

    The default correlations are those recommended in [1]_, and are believed to
    be the best publicly available.

    Examples
    --------
    >>> helical_Re_crit(Di=0.02, Dc=0.5)
    6946.792538856203

    Parameters
    ----------
    Di : float
        Inner diameter of the tube making up the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    Method : str, optional
        Critical Reynolds number transition criteria correlation; one of ['Seth Stahel',
        'Ito', 'Kubair Kuloor', 'Kutateladze Borishanskii', 'Schmidt',
        'Srinivasan']; the default is 'Schmidt'.

    Returns
    -------
    Re_crit : float
        Reynolds number for critical transition between laminar and turbulent
        flow, [-]

    See Also
    --------
    fluids.geometry.HelicalCoil
    helical_transition_Re_Schmidt
    helical_transition_Re_Srinivasan
    helical_transition_Re_Kutateladze_Borishanskii
    helical_transition_Re_Kubair_Kuloor
    helical_transition_Re_Ito
    helical_transition_Re_Seth_Stahel

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    if Method == 'Schmidt':
        Re_crit = helical_transition_Re_Schmidt(Di, Dc)
    elif Method == 'Seth Stahel':
        Re_crit = helical_transition_Re_Seth_Stahel(Di, Dc)
    elif Method == 'Ito':
        Re_crit = helical_transition_Re_Ito(Di, Dc)
    elif Method == 'Kubair Kuloor':
        Re_crit = helical_transition_Re_Kubair_Kuloor(Di, Dc)
    elif Method == 'Kutateladze Borishanskii':
        Re_crit = helical_transition_Re_Kutateladze_Borishanskii(Di, Dc)
    elif Method == 'Srinivasan':
        Re_crit = helical_transition_Re_Srinivasan(Di, Dc)
    else:
        raise ValueError(_bad_curved_transition_method)
    return Re_crit


def friction_factor_curved_methods(Re, Di, Dc, roughness=0.0,
                                   check_ranges=True):
    r'''Returns a list of correlation names for calculating friction factor
    of fluid flowing in a curved pipe or helical coil, supporting both laminar
    and turbulent regimes.

    Examples
    --------
    >>> friction_factor_curved_methods(Re=1E5, Di=0.02, Dc=0.5)[0]
    'Schmidt turbulent'

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the tube making up the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list
        List of methods in the regime the specified `Re` is in at the given
        `Di` and `Dc`.
    '''
    Re_crit = helical_Re_crit(Di=Di, Dc=Dc, Method='Schmidt')
    turbulent = False if Re < Re_crit else True
    if check_ranges:
        if turbulent:
            return list(curved_friction_turbulent_methods_list)
        else:
            return list(curved_friction_laminar_methods_list)
    else:
        return curved_friction_turbulent_methods_list + curved_friction_laminar_methods_list


def friction_factor_curved(Re, Di, Dc, roughness=0.0, Method=None,
                           Rec_method='Schmidt',
                           laminar_method='Schmidt laminar',
                           turbulent_method='Schmidt turbulent', Darcy=True):
    r'''Calculates friction factor fluid flowing in a curved pipe or helical
    coil, supporting both laminar and turbulent regimes. Selects the
    appropriate regime by default, and has default correlation choices.
    Optionally, a specific correlation can be specified with the `Method`
    keyword.

    The default correlations are those recommended in [1]_, and are believed to
    be the best publicly available.

    Examples
    --------
    >>> friction_factor_curved(Re=1E5, Di=0.02, Dc=0.5)
    0.022961996738387523

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Di : float
        Inner diameter of the tube making up the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]
    roughness : float, optional
        Roughness of pipe wall [m]

    Returns
    -------
    f : float
        Friction factor, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, overriding the default turbulent/
        laminar selection.
    Rec_method : str, optional
        Critical Reynolds number transition criteria; one of ['Seth Stahel',
        'Ito', 'Kubair Kuloor', 'Kutateladze Borishanskii', 'Schmidt',
        'Srinivasan']; the default is 'Schmidt'.
    laminar_method : str, optional
        Friction factor correlation for the laminar regime; one of
        ['White', 'Mori Nakayama laminar', 'Schmidt laminar']; the default is
        'Schmidt laminar'.
    turbulent_method : str, optional
        Friction factor correlation for the turbulent regime; one of
        ['Guo', 'Ju', 'Schmidt turbulent', 'Prasad', 'Mandel Nigam',
        'Mori Nakayama turbulent', 'Czop']; the default is 'Schmidt turbulent'.
    Darcy : bool, optional
        If False, will return fanning friction factor, 1/4 of the Darcy value

    See Also
    --------
    fluids.geometry.HelicalCoil
    helical_turbulent_fd_Schmidt
    helical_turbulent_fd_Srinivasan
    helical_turbulent_fd_Mandal_Nigam
    helical_turbulent_fd_Ju
    helical_turbulent_fd_Guo
    helical_turbulent_fd_Czop
    helical_turbulent_fd_Prasad
    helical_turbulent_fd_Mori_Nakayama
    helical_laminar_fd_Schmidt
    helical_laminar_fd_Mori_Nakayama
    helical_laminar_fd_White
    helical_transition_Re_Schmidt
    helical_transition_Re_Srinivasan
    helical_transition_Re_Kutateladze_Borishanskii
    helical_transition_Re_Kubair_Kuloor
    helical_transition_Re_Ito
    helical_transition_Re_Seth_Stahel

    Notes
    -----
    The range of accuracy of these correlations is much than that in a
    straight pipe.

    References
    ----------
    .. [1] Schlunder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1983.
    '''
    Re_crit = helical_Re_crit(Di=Di, Dc=Dc, Method=Rec_method)
    turbulent = False if Re < Re_crit else True

    if Method is None:
        Method2 = turbulent_method if turbulent else laminar_method
    else:
        Method2 = Method # Use second variable to keep numba types happy
    # Laminar
    if Method2 == 'Schmidt laminar':
        f = helical_laminar_fd_Schmidt(Re, Di, Dc)
    elif Method2 == 'White':
        f = helical_laminar_fd_White(Re, Di, Dc)
    elif Method2 == 'Mori Nakayama laminar':
        f = helical_laminar_fd_Mori_Nakayama(Re, Di, Dc)
    # Turbulent with roughness support
    elif Method2 == 'Schmidt turbulent':
        f = helical_turbulent_fd_Schmidt(Re, Di, Dc, roughness)
    elif Method2 == 'Prasad':
        f = helical_turbulent_fd_Prasad(Re, Di, Dc, roughness)
    elif Method2 == 'Ju':
        f = helical_turbulent_fd_Ju(Re, Di, Dc, roughness)
    elif Method2 == 'Mandel Nigam':
        f = helical_turbulent_fd_Mandal_Nigam(Re, Di, Dc, roughness)
    # Turbulent without roughness support
    elif Method2 == 'Mori Nakayama turbulent':
        f = helical_turbulent_fd_Mori_Nakayama(Re, Di, Dc)
    elif Method2 == 'Czop':
        f = helical_turbulent_fd_Czop(Re, Di, Dc)
    elif Method2 == 'Guo':
        f = helical_turbulent_fd_Guo(Re, Di, Dc)
    elif Method2 == 'Srinivasan turbulent':
        f = helical_turbulent_fd_Srinivasan(Re, Di, Dc)
    else:
        raise ValueError('Invalid method for friction factor calculation')
    if not Darcy:
        f *= 0.25
    return f

### Plate heat exchanger single phase

def friction_plate_Martin_1999(Re, plate_enlargement_factor):
    r'''Calculates Darcy friction factor for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = \frac{\cos \phi}{\sqrt{0.045\tan\phi
        + 0.09\sin\phi + f_0/\cos(\phi)}} + \frac{1-\cos\phi}{\sqrt{3.8f_1}}

    .. math::
        f_0 = 16/Re \text{ for } Re < 2000

    .. math::
        f_0 = (1.56\ln Re - 3)^{-2} \text{ for } Re \ge 2000

    .. math::
        f_1 = \frac{149}{Re} + 0.9625 \text{ for } Re < 2000

    .. math::
        f_1 = \frac{9.75}{Re^{0.289}} \text{ for } Re \ge 2000

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Based on experimental data from Re from 200 - 10000 and enhancement
    factors calculated with chevron angles of 0 to 80 degrees. See
    `PlateExchanger` for further clarification on the definitions.

    The length the friction factor gets multiplied by is not the flow path
    length, but rather the straight path length from port to port as if there
    were no chevrons.

    Note there is a discontinuity at Re = 2000 for the transition from
    laminar to turbulent flow, although the literature suggests the transition
    is actually smooth.

    This was first developed in [2]_ and only minor modifications by the
    original author were made before its republication in [1]_.
    This formula is also suggested in [3]_

    Examples
    --------
    >>> friction_plate_Martin_1999(Re=20000, plate_enlargement_factor=1.15)
    2.284018089834135

    References
    ----------
    .. [1] Martin, Holger. "Economic optimization of compact heat exchangers."
       EF-Conference on Compact Heat Exchangers and Enhancement Technology for
       the Process Industries, Banff, Canada, July 18-23, 1999, 1999.
       https://publikationen.bibliothek.kit.edu/1000034866.
    .. [2] Martin, Holger. "A Theoretical Approach to Predict the Performance
       of Chevron-Type Plate Heat Exchangers." Chemical Engineering and
       Processing: Process Intensification 35, no. 4 (January 1, 1996): 301-10.
       https://doi.org/10.1016/0255-2701(95)04129-X.
    .. [3] Shah, Ramesh K., and Dusan P. Sekulic. Fundamentals of Heat
       Exchanger Design. 1st edition. Hoboken, NJ: Wiley, 2002.
    '''
    phi = plate_enlargement_factor

    if Re < 2000.:
        f0 = 16./Re
        f1 = 149./Re + 0.9625
    else:
        f0 = (1.56*log(Re) - 3.0)**-2
        f1 = 9.75*Re**-0.289

    rhs = cos(phi)*1.0/sqrt(0.045*tan(phi) + 0.09*sin(phi) + f0/cos(phi))
    rhs += (1. - cos(phi))*1.0/sqrt(3.8*f1)
    ff = rhs**-2.
    return ff*4.0


def friction_plate_Martin_VDI(Re, plate_enlargement_factor):
    r'''Calculates Darcy friction factor for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = \frac{\cos \phi}{\sqrt{0.28\tan\phi
        + 0.36\sin\phi + f_0/\cos(\phi)}} + \frac{1-\cos\phi}{\sqrt{3.8f_1}}

    .. math::
        f_0 = 64/Re \text{ for } Re < 2000

    .. math::
        f_0 = (1.56\ln Re - 3)^{-2} \text{ for } Re \ge 2000

    .. math::
        f_1 = \frac{597}{Re} + 3.85 \text{ for } Re < 2000

    .. math::
        f_1 = \frac{39}{Re^{0.289}} \text{ for } Re \ge 2000

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Based on experimental data from Re from 200 - 10000 and enhancement
    factors calculated with chevron angles of 0 to 80 degrees. See
    `PlateExchanger` for further clarification on the definitions.

    The length the friction factor gets multiplied by is not the flow path
    length, but rather the straight path length from port to port as if there
    were no chevrons.

    Note there is a discontinuity at Re = 2000 for the transition from
    laminar to turbulent flow, although the literature suggests the transition
    is actually smooth.

    This is a revision of the Martin's earlier model, adjusted to predidct
    higher friction factors.

    There are three parameters in this model, a, b and c; it is posisble
    to adjust them to better fit a know exchanger's pressure drop.

    See Also
    --------
    friction_plate_Martin_1999

    Examples
    --------
    >>> friction_plate_Martin_VDI(Re=20000, plate_enlargement_factor=1.15)
    2.702534119024076

    References
    ----------
    .. [1] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    phi = plate_enlargement_factor

    if Re < 2000.:
        f0 = 64./Re
        f1 = 597./Re + 3.85
    else:
        f0 = (1.8*log10(Re) - 1.5)**-2
        f1 = 39.*Re**-0.289

    a, b, c = 3.8, 0.28, 0.36

    rhs = cos(phi)*1.0/sqrt(b*tan(phi) + c*sin(phi) + f0/cos(phi))
    rhs += (1. - cos(phi))*1.0/sqrt(a*f1)
    return rhs**-2.0

Kumar_beta_list = [30.0, 45.0, 50.0, 60.0, 65.0]

Kumar_fd_Res = [[10.0, 100.0],
      [15.0, 300.0],
      [20.0, 300.0],
      [40.0, 400.0],
      [50.0, 500.0]]

Kumar_C2s = [[50.0, 19.40, 2.990],
       [47.0, 18.29, 1.441],
       [34.0, 11.25, 0.772],
       [24.0, 3.24, 0.760],
       [24.0, 2.80, 0.639]]

# Is the second in the first row 0.589 (paper) or 0.598 (PHEWorks)
# Believed to be the values from the paper, where this graph was
# curve fit as the original did not contain and coefficients only a plot
Kumar_Ps = [[1.0, 0.589, 0.183],
      [1.0, 0.652, 0.206],
      [1.0, 0.631, 0.161],
      [1.0, 0.457, 0.215],
      [1.0, 0.451, 0.213]]


def friction_plate_Kumar(Re, chevron_angle):
    r'''Calculates Darcy friction factor for single-phase flow in a
    **well-designed** Chevron-style plate heat exchanger according to [1]_.
    The data is believed to have been developed by APV International Limited,
    since acquired by SPX Corporation. This uses a curve fit of that data
    published in [2]_.

    .. math::
        f_f = \frac{C_2}{Re^p}

    C2 and p are coefficients looked up in a table, with varying ranges
    of Re validity and chevron angle validity. See the source for their
    exact values.

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Data on graph from Re=0.1 to Re=10000, with chevron angles 30 to 65 degrees.
    See `PlateExchanger` for further clarification on the definitions.

    It is believed the constants used in this correlation were curve-fit to
    the actual graph in [1]_ by the author of [2]_ as there is no

    The length the friction factor gets multiplied by is not the flow path
    length, but rather the straight path length from port to port as if there
    were no chevrons.

    As the coefficients change, there are numerous small discontinuities,
    although the data on the graphs is continuous with sharp transitions
    of the slope.

    The author of [1]_ states clearly this correlation is "applicable only to
    well designed Chevron PHEs".

    Examples
    --------
    >>> friction_plate_Kumar(Re=2000, chevron_angle=30)
    2.9760669055634517

    References
    ----------
    .. [1] Kumar, H. "The plate heat exchanger: construction and design." In
       First U.K. National Conference on Heat Transfer: Held at the University
       of Leeds, 3-5 July 1984, Institute of Chemical Engineering Symposium
       Series, vol. 86, pp. 1275-1288. 1984.
    .. [2] Ayub, Zahid H. "Plate Heat Exchanger Literature Survey and New Heat
       Transfer and Pressure Drop Correlations for Refrigerant Evaporators."
       Heat Transfer Engineering 24, no. 5 (September 1, 2003): 3-16.
       doi:10.1080/01457630304056.
    '''
    beta_list_len = len(Kumar_beta_list)

    for i in range(beta_list_len):
        if chevron_angle <= Kumar_beta_list[i]:
            C2_options, p_options, Re_ranges = Kumar_C2s[i], Kumar_Ps[i], Kumar_fd_Res[i]
            break
        elif i == beta_list_len-1:
            C2_options, p_options, Re_ranges = Kumar_C2s[-1], Kumar_Ps[-1], Kumar_fd_Res[-1]

    Re_len = len(Re_ranges)

    for j in range(Re_len):
        if Re <= Re_ranges[j]:
            C2, p = C2_options[j], p_options[j]
            break
        elif j == Re_len-1:
            C2, p = C2_options[-1], p_options[-1]

    # Originally in Fanning friction factor basis
    return 4.0*C2*Re**-p


def friction_plate_Muley_Manglik(Re, chevron_angle, plate_enlargement_factor):
    r'''Calculates Darcy friction factor for single-phase flow in a
    Chevron-style plate heat exchanger according to [1]_, also shown and
    recommended in [2]_.

    .. math::
        f_f = [2.917 - 0.1277\beta + 2.016\times10^{-3} \beta^2]
        \times[20.78 - 19.02\phi + 18.93\phi^2 - 5.341\phi^3]
        \times Re^{-[0.2 + 0.0577\sin[(\pi \beta/45)+2.1]]}

    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]
    plate_enlargement_factor : float
        The extra surface area multiplier as compared to a flat plate
        caused the corrugations, [-]

    Returns
    -------
    fd : float
        Darcy friction factor [-]

    Notes
    -----
    Based on experimental data of plate enacement factors up to 1.5, and valid
    for Re > 1000 and chevron angles from 30 to 60 degrees with sinusoidal
    shape. See `PlateExchanger` for further clarification on the definitions.

    The length the friction factor gets multiplied by is not the flow path
    length, but rather the straight path length from port to port as if there
    were no chevrons.

    This is a continuous model with no discontinuities.

    Examples
    --------
    >>> friction_plate_Muley_Manglik(Re=2000, chevron_angle=45, plate_enlargement_factor=1.2)
    1.0880870804075413

    References
    ----------
    .. [1] Muley, A., and R. M. Manglik. "Experimental Study of Turbulent Flow
       Heat Transfer and Pressure Drop in a Plate Heat Exchanger With Chevron
       Plates." Journal of Heat Transfer 121, no. 1 (February 1, 1999): 110-17.
       doi:10.1115/1.2825923.
    .. [2] Ayub, Zahid H. "Plate Heat Exchanger Literature Survey and New Heat
       Transfer and Pressure Drop Correlations for Refrigerant Evaporators."
       Heat Transfer Engineering 24, no. 5 (September 1, 2003): 3-16.
       doi:10.1080/01457630304056.
    '''
    beta, phi = chevron_angle, plate_enlargement_factor
    # Beta is indeed chevron angle; with respect to angle of mvoement
    # Still might be worth another check
    t1 = (2.917 - 0.1277*beta + 2.016E-3*beta**2)
    t2 = (5.474 - 19.02*phi + 18.93*phi**2 - 5.341*phi**3)
    t3 = -(0.2 + 0.0577*sin(pi*beta/45. + 2.1))
    # Equation returns fanning friction factor
    return 4*t1*t2*Re**t3


# Data from the Handbook of Hydraulic Resistance, 4E, in format (min, max, avg)
#  roughness in m; may have one, two, or three of the values.
seamless_other_metals = {'Commercially smooth': (1.5E-6, 1.0E-5, None)}

seamless_steel = {'New and unused': (2.0E-5, 1.0E-4, None),
    'Cleaned, following years of use': (None, 4.0E-5, None),
    'Bituminized': (None, 4.0E-5, None),
    'Heating systems piping; either superheated steam pipes, or just water pipes of systems with deaerators and chemical treatment':
    (None, None, 1.0E-4),
    'Following one year as a gas pipeline': (None, None, 1.2E-4),
    'Following multiple year as a gas pipeline': (4.0E-5, 2.0E-4, None),
    'Casings in gas wells, different conditions, several years of use':
    (6.0E-5, 2.2E-4, None),
    'Heating systems, saturated steam ducts or water pipes (with minor water leakage < 0.5%, and balance water deaerated)':
    (None, None, 2.0E-4),
    'Water heating system pipelines, any source': (None, None, 2.0E-4),
    'Oil pipelines, intermediate operating conditions ': (None, None, 2.0E-4),
    'Corroded, moderately ': (None, None, 4.0E-4),
    'Scale, small depositions only ': (None, None, 4.0E-4),
    'Condensate pipes in open systems or periodically operated steam pipelines':
    (None, None, 5.0E-4),
    'Compressed air piping': (None, None, 8.0E-4),
    'Following multiple years of operation, generally corroded or with small amounts of scale':
    (1.5E-4, 1.0E-3, None),
    'Water heating piping without deaeration but with chemical treatment of water; leakage up to 3%; or condensate piping operated periodically':
    (None, None, 1.0E-3),
    'Used water piping': (1.2E-3, 1.5E-3, None),
    'Poor condition': (5.0E-3, None, None)}

welded_steel = {'Good condition': (4.0E-5, 1.0E-4, None),
    'New and covered with bitumen': (None, None, 5.0E-5),
    'Used and covered with partially dissolved bitumen; corroded':
    (None, None, 1.0E-4),
    'Used, suffering general corrosion': (None, None, 1.5E-4),
    'Surface looks like new, 10 mm lacquer inside, even joints':
    (3.0E-4, 4.0E-4, None),
    'Used Gas mains': (None, None, 5.0E-4),
    'Double or simple transverse riveted joints; with or without lacquer; without corrosion':
    (6.0E-4, 7.0E-4, None),
    'Lacquered inside but rusted': (9.5E-4, 1.0E-3, None),
    'Gas mains, many years of use, with layered deposits': (None, None, 1.1E-3),
    'Non-corroded and with double transverse riveted joints':
    (1.2E-3, 1.5E-3, None),
    'Small deposits': (None, None, 1.5E-3),
    'Heavily corroded and with  double transverse riveted joints':
    (None, None, 2.0E-3),
    'Appreciable deposits': (2.0E-3, 4.0E-3, None),
    'Gas mains, many years of use, deposits of resin/naphthalene':
        (None, None, 2.4E-3),
    'Poor condition': (5.0E-3, None, None)}

riveted_steel = {
    'Riveted laterally and longitudinally with one line; lacquered on the inside':
    (3.0E-4, 4.0E-4, None),
    'Riveted laterally and longitudinally with two lines; with or without lacquer on the inside and without corrosion':
    (6.0E-4, 7.0E-4, None),
    'Riveted laterally with one line and longitudinally with two lines; thickly lacquered or torred on the inside':
    (1.2E-3, 1.4E-3, None),
    'Riveted longitudinally with six lines, after extensive use':
    (None, None, 2.0E-3),
    'Riveted laterally with four line and longitudinally with six lines; overlapping joints inside':
    (None, None, 4.0E-3),
    'Extremely poor surface; overlapping and uneven joints':
    (5.0E-3, None, None)}

roofing_metal = {'Oiled': (1.5E-4, 1.1E-3, None),
                 'Not Oiled': (2.0E-5, 4.0E-5, None)}

galvanized_steel_tube = {'Bright galvanization; new': (7.0E-5, 1.0E-4, None),
                         'Ordinary galvanization': (1.0E-4, 1.5E-4, None)}

galvanized_steel_sheet = {'New': (None, None, 1.5E-4),
                          'Used previously for water': (None, None, 1.8E-4)}

steel = {'Glass enamel coat': (1.0E-6, 1.0E-5, None),
         'New': (2.5E-4, 1.0E-3, None)}

cast_iron = {'New, bituminized': (1.0E-4, 1.5E-4, None),
             'Coated with asphalt': (1.2E-4, 3.0E-4, None),
             'Used water pipelines': (None, None, 1.4E-3),
             'Used and corroded': (1.0E-3, 1.5E-3, None),
             'Deposits visible': (1.0E-3, 1.5E-3, None),
             'Substantial deposits': (2.0E-3, 4.0E-3, None),
             'Cleaned after extensive use': (3.0E-4, 1.5E-3, None),
             'Severely corroded': (None, 3.0E-3, None)}

water_conduit_steel = {
    'New, clean, seamless (without joints), well fitted':
    (1.5E-5, 4.0E-5, None),
    'New, clean, welded lengthwise and well fitted': (1.2E-5, 3.0E-5, None),
    'New, clean, welded lengthwise and well fitted, with transverse welded joints':
    (8.0E-5, 1.7E-4, None),
    'New, clean, coated, bituminized when manufactured': (1.4E-5, 1.8E-5, None),
    'New, clean, coated, bituminized when manufactured, with transverse welded joints':
    (2.0E-4, 6.0E-4, None),
    'New, clean, coated, galvanized': (1.0E-4, 2.0E-4, None),
    'New, clean, coated, roughly galvanized': (4.0E-4, 7.0E-4, None),
    'New, clean, coated, bituminized, curved': (1.0E-4, 1.4E-3, None),
    'Used, clean, slight corrosion': (1.0E-4, 3.0E-4, None),
    'Used, clean, moderate corrosion or slight deposits':
    (3.0E-4, 7.0E-4, None),
    'Used, clean, severe corrosion': (8.0E-4, 1.5E-3, None),
    'Used, clean, previously cleaned of either deposits or rust':
        (1.5E-4, 2.0E-4, None)}

water_conduit_steel_used = {
    'Used, all welded, <2 years use, no deposits': (1.2E-4, 2.4E-4, None),
    'Used, all welded, <20 years use, no deposits': (6.0E-4, 5.0E-3, None),
    'Used, iron-bacterial corrosion': (3.0E-3, 4.0E-3, None),
    'Used, heavy corrosion, or with incrustation (deposit 1.5 - 9 mm deep)':
    (3.0E-3, 5.0E-3, None),
    'Used, heavy corrosion, or with incrustation (deposit 3 - 25 mm deep)':
    (6.0E-3, 6.5E-3, None),
    'Used, inside coating, bituminized, < 2 years use': (1.0E-4, 3.5E-4, None)}

steels = {'Seamless tubes made from brass, copper, lead, aluminum':
          seamless_other_metals,
          'Seamless steel tubes': seamless_steel,
          'Welded steel tubes': welded_steel,
          'Riveted steel tubes': riveted_steel,
          'Roofing steel sheets': roofing_metal,
          'Galzanized steel tubes': galvanized_steel_tube,
          'Galzanized sheet steel': galvanized_steel_sheet,
          'Steel tubes': steel,
          'Cast-iron tubes': cast_iron,
          'Steel water conduits in generating stations': water_conduit_steel,
          'Used steel water conduits in generating stations':
          water_conduit_steel_used}


concrete_water_conduits = {
    'New and finished with plater; excellent manufacture (joints aligned, prime coated and smoothed)':
    (5.0E-5, 1.5E-4, None),
    'Used and corroded; with a wavy surface and wood framework':
    (1.0E-3, 4.0E-3, None),
    'Old, poor fitting and manufacture; with an overgrown surface and deposits of sand and gravel':
    (1.0E-3, 4.0E-3, None),
    'Very old; damaged surface, very overgrown': (5.0E-3, None, None),
    'Water conduit, finished with smoothed plaster': (5.0E-3, None, None),
    'New, very well manufactured, hand smoothed, prime-coated joints':
    (1.0E-4, 2.0E-4, None),
    'Hand-smoothed cement finish and smoothed joints': (1.5E-4, 3.5E-4, None),
    'Used, no deposits, moderately smooth, steel or wooden casing, joints prime coated but not smoothed':
    (3.0E-4, 6.0E-4, None),
    'Used, prefabricated monoliths, cement plaster (wood floated), rough joints':
    (5.0E-4, 1.0E-3, None),
    'Conduits for water, sprayed surface of concrete': (5.0E-4, 1.0E-3, None),
    'Smoothed air-placed, either sprayed concrete or concrete on more concrete':
    (None, None, 5.0E-4),
    'Brushed air-placed, either sprayed concrete or concrete on more concrete':
    (None, None, 2.3E-3),
    'Non-smoothed air-placed, either sprayed concrete or concrete on more concrete':
    (3.0E-3, 6.0E-3, None),
    'Smoothed air-placed, either sprayed concrete or concrete on more concrete':
    (6.0E-3, 1.7E-2, None)}

concrete_reinforced_tubes = {'New': (2.5E-4, 3.4E-4, None),
                             'Nonprocessed': (2.5E-3, None, None)}

asbestos_cement = {'New': (5.0E-5, 1.0E-4, None),
                   'Average': (6.0E-4, None, None)}

cement_tubes = {'Smoothed': (3.0E-4, 8.0E-4, None),
                'Non processed': (1.0E-3, 2.0E-3, None),
                'Joints, non smoothed': (1.9E-3, 6.4E-3, None)}

cement_mortar_channels = {
    'Plaster, cement, smoothed joints and protrusions, and a casing':
    (5.0E-5, 2.2E-4, None),
    'Steel trowled': (None, None, 5.0E-4)}

cement_other = {'Plaster over a screen': (1.0E-2, 1.5E-2, None),
                'Salt-glazed ceramic': (None, None, 1.4E-3),
                'Slag-concrete': (None, None, 1.5E-3),
                'Slag and alabaster-filling': (1.0E-3, 1.5E-3, None)}

concretes = {'Concrete water conduits, no finish': concrete_water_conduits,
             'Reinforced concrete tubes': concrete_reinforced_tubes,
             'Asbestos cement tubes': asbestos_cement,
             'Cement tubes': cement_tubes,
             'Cement-mortar plaster channels': cement_mortar_channels,
             'Other': cement_other}


wood_tube = {'Boards, thoroughly dressed': (None, None, 1.5E-4),
             'Boards, well dressed': (None, None, 3.0E-4),
             'Boards, undressed but fitted': (None, None, 7.0E-4),
             'Boards, undressed': (None, None, 1.0E-3),
             'Staved': (None, None, 6.0E-4)}

plywood_tube = {'Birch plywood, transverse grain, good quality':
                (None, None, 1.2E-4),
                'Birch plywood, longitudal grain, good quality':
                (3.0E-5, 5.0E-5, None)}

glass_tube = {'Glass': (1.5E-6, 1.0E-5, None)}

wood_plywood_glass = {'Wood tubes': wood_tube,
                      'Plywood tubes': plywood_tube,
                      'Glass tubes': glass_tube}


rock_channels = {'Blast-hewed, little jointing': (1.0E-1, 1.4E-1, None),
                 'Blast-hewed, substantial jointing': (1.3E-1, 5.0E-1, None),
                 'Roughly cut or very uneven surface': (5.0E-1, 1.5E+0, None)}

unlined_tunnels = {'Rocks, gneiss, diameter 3-13.5 m': (3.0E-1, 7.0E-1, None),
                   'Rocks, granite, diameter 3-9 m': (2.0E-1, 7.0E-1, None),
                   'Shale, diameter, diameter 9-12 m': (2.5E-1, 6.5E-1, None),
                   'Shale, quartz, quartzile, diameter 7-10 m':
                   (2.0E-1, 6.0E-1, None),
                   'Shale, sedimentary, diameter 4-7 m': (None, None, 4.0E-1),
                   'Shale, nephrite bearing, diameter 3-8 m':
                   (None, None, 2.0E-1)}

tunnels = {'Rough channels in rock': rock_channels,
           'Unlined tunnels': unlined_tunnels}


# Roughness, in m
_roughness = {'Brass': .00000152, 'Lead': .00000152, 'Glass': .00000152,
'Steel': .00000152, 'Asphalted cast iron': .000122, 'Galvanized iron': .000152,
'Cast iron': .000259, 'Wood stave': .000183, 'Rough wood stave': .000914,
'Concrete': .000305, 'Rough concrete': .00305, 'Riveted steel': .000914,
'Rough riveted steel': .00914}


# Create a more friendly data structure

'''Holds a dict of tuples in format (min, max, average) roughness values in
meters from the source
Idelʹchik, I. E, and A. S Ginevskiĭ. Handbook of Hydraulic
Resistance. Redding, CT: Begell House, 2007.
'''
HHR_roughness = {}


HHR_roughness_dicts = [tunnels, wood_plywood_glass, concretes, steels]
HHR_roughness_categories = {}
[HHR_roughness_categories.update(i) for i in HHR_roughness_dicts]
for d in HHR_roughness_dicts:
    for k, v in d.items():
        for name, values in v.items():
            HHR_roughness[str(k)+', ' + name] = values

# For searching only
_all_roughness = HHR_roughness.copy()
_all_roughness.update(_roughness)

# Format : ID: (avg_roughness, coef A (inches), coef B (inches))
_Farshad_roughness = {'Plastic coated': (5E-6, 0.0002, -1.0098),
                      'Carbon steel, honed bare': (12.5E-6, 0.0005, -1.0101),
                      'Cr13, electropolished bare': (30E-6, 0.0012, -1.0086),
                      'Cement lining': (33E-6, 0.0014, -1.0105),
                      'Carbon steel, bare': (36E-6, 0.0014, -1.0112),
                      'Fiberglass lining': (38E-6, 0.0016, -1.0086),
                      'Cr13, bare': (55E-6, 0.0021, -1.0055)  }

try:
    if IS_NUMBA: # type: ignore
        _Farshad_roughness_keys = tuple(_Farshad_roughness.keys())
        _Farshad_roughness_values = tuple(_Farshad_roughness.values())
except:
    pass

def roughness_Farshad(ID=None, D=None, coeffs=None):
    r'''Calculates of retrieves the roughness of a pipe based on the work of
    [1]_. This function will return an average value for pipes of a given
    material, or if diameter is provided, will calculate one specifically for
    the pipe inner diameter according to the following expression with
    constants `A` and `B`:

    .. math::
        \epsilon = A\cdot D^{B+1}

    Please not that `A` has units of inches, and `B` requires `D` to be in
    inches as well.

    The list of supported materials is as follows:

        * 'Plastic coated'
        * 'Carbon steel, honed bare'
        * 'Cr13, electropolished bare'
        * 'Cement lining'
        * 'Carbon steel, bare'
        * 'Fiberglass lining'
        * 'Cr13, bare'

    If `coeffs` and `D` are given, the custom coefficients for the equation as
    given by the user will be used and `ID` is not required.

    Parameters
    ----------
    ID : str, optional
        Name of pipe material from above list
    D : float, optional
        Actual inner diameter of pipe, [m]
    coeffs : tuple, optional
        (A, B) Coefficients to use directly, instead of looking them up;
        they are actually dimensional, in the forms (inch^-B, -) but only
        coefficients with those dimensions are available [-]

    Returns
    -------
    epsilon : float
        Roughness of pipe [m]

    Notes
    -----
    The diameter-dependent form provides lower roughness values for larger
    diameters.

    The measurements were based on DIN 4768/1 (1987), using both a
    "Dektak ST Surface Profiler" and a "Hommel Tester T1000". Both instruments
    were found to be in agreement. A series of flow tests, in which pressure
    drop directly measured, were performed as well, with nitrogen gas as an
    operating fluid. The accuracy of the data from these tests is claimed to be
    within 1%.

    Using those results, the authors back-calculated what relative roughness
    values would be necessary to produce the observed pressure drops. The
    average difference between this back-calculated roughness and the measured
    roughness was 6.75%.

    For microchannels, this model will predict roughness much larger than the
    actual channel diameter.

    Examples
    --------
    >>> roughness_Farshad('Cr13, bare', 0.05)
    5.3141677781137006e-05

    References
    ----------
    .. [1] Farshad, Fred F., and Herman H. Rieke. "Surface Roughness Design
       Values for Modern Pipes." SPE Drilling & Completion 21, no. 3 (September
       1, 2006): 212-215. doi:10.2118/89040-PA.
    '''
    # Case 1, coeffs given; only run if ID is not given.
    if ID is None and coeffs is not None:
        A, B = coeffs
        return A*(D/inch)**(B + 1.0)*inch
    # Case 2, lookup parameters
    if ID in _Farshad_roughness: # numba: delete
        dat = _Farshad_roughness[ID] # numba: delete
#    try: # numba: uncomment
#        dat = _Farshad_roughness_values[_Farshad_roughness_keys.index(ID)] # numba: uncomment
#    except: # numba: uncomment
#        raise KeyError('ID was not in _Farshad_roughness.') # numba: uncomment
    if D is None:
        return dat[0]
    else:
        A, B = dat[1], dat[2]
        return A*(D/inch)**(B+1)*inch


roughness_clean_names = set(_roughness.keys())
roughness_clean_names.update(_Farshad_roughness.keys())


def nearest_material_roughness(name, clean=None):
    r'''Searches through either a dict of clean pipe materials or used pipe
    materials and conditions and returns the ID of the nearest material.
    Search is performed with either the standard library's difflib or with
    the fuzzywuzzy module if available.

    Parameters
    ----------
    name : str
        Search term for matching pipe materials
    clean : bool, optional
        If True, search only clean pipe database; if False, search only the
        dirty database; if None, search both

    Returns
    -------
    ID : str
        String for lookup of roughness of a pipe, in either
        `roughness_clean_names` or `HHR_roughness` depending on if clean is
        True, [-]

    Examples
    --------
    >>> nearest_material_roughness('condensate pipes', clean=False) # doctest: +SKIP
    'Seamless steel tubes, Condensate pipes in open systems or periodically operated steam pipelines'

    References
    ----------
    .. [1] Idelʹchik, I. E, and A. S Ginevskiĭ. Handbook of Hydraulic
       Resistance. Redding, CT: Begell House, 2007.
    '''
    if clean is None:
        d = _all_roughness.keys()
    else:
        if clean:
            d = roughness_clean_names
        else:
            d = HHR_roughness.keys()
    return fuzzy_match(name, d)


def material_roughness(ID, D=None, optimism=None):
    r'''Searches through either a dict of clean pipe materials or used pipe
    materials and conditions and returns the ID of the nearest material.
    Search is performed with either the standard library's difflib or with
    the fuzzywuzzy module if available.

    Parameters
    ----------
    ID : str
        Search terms for matching pipe materials, [-]
    D : float, optional
        Diameter of desired pipe; used only if ID is in [2]_, [m]
    optimism : bool, optional
        For values in [1]_, a minimum, maximum, and average value is normally
        given; if True, returns the minimum roughness; if False, the maximum
        roughness; and if None, returns the average roughness. Most entries do
        not have all three values, so fallback logic to return the closest
        entry is used, [-]

    Returns
    -------
    roughness : float
        Retrieved or calculated roughness, [m]

    Examples
    --------
    >>> material_roughness('condensate pipes') # doctest: +SKIP
    0.0005

    References
    ----------
    .. [1] Idelʹchik, I. E, and A. S Ginevskiĭ. Handbook of Hydraulic
       Resistance. Redding, CT: Begell House, 2007.
    .. [2] Farshad, Fred F., and Herman H. Rieke. "Surface Roughness Design
       Values for Modern Pipes." SPE Drilling & Completion 21, no. 3 (September
       1, 2006): 212-215. doi:10.2118/89040-PA.
    '''
    if ID in _Farshad_roughness:
        return roughness_Farshad(ID, D)
    elif ID in _roughness:
        return _roughness[ID]
    elif ID in HHR_roughness:
        minimum, maximum, avg = HHR_roughness[ID]
        if optimism is None:
            return avg if avg else (maximum if maximum else minimum)
        elif optimism is True:
            return minimum if minimum else (avg if avg else maximum)
        else:
            return maximum if maximum else (avg if avg else minimum)
    else:
        return material_roughness(nearest_material_roughness(ID, clean=False),
                                  D=D, optimism=optimism)

def transmission_factor(fd=None, F=None):
    r'''Calculates either transmission factor from Darcy friction factor,
    or Darcy friction factor from the transmission factor. Raises an exception
    if neither input is given.

    Transmission factor is a term used in compressible gas flow in pipelines.

    .. math::
        F = \frac{2}{\sqrt{f_d}}

    .. math::
        f_d = \frac{4}{F^2}

    Parameters
    ----------
    fd : float, optional
        Darcy friction factor, [-]
    F : float, optional
        Transmission factor, [-]

    Returns
    -------
    fd or F : float
        Darcy friction factor or transmission factor [-]

    Examples
    --------
    >>> transmission_factor(fd=0.0185)
    14.704292441876154

    >>> transmission_factor(F=20)
    0.01

    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton,
       FL: CRC Press, 2005.
    '''
    if fd is not None:
        return 2./sqrt(fd)
    elif F is not None:
        return 4./(F*F)
    else:
        raise ValueError('Either Darcy friction factor or transmission factor is needed')


def one_phase_dP(m, rho, mu, D, roughness=0.0, L=1.0, Method=None):
    r'''Calculates single-phase pressure drop. This is a wrapper
    around other methods.

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    rho : float
        Density of fluid, [kg/m^3]
    mu : float
        Viscosity of fluid, [Pa*s]
    D : float
        Diameter of pipe, [m]
    roughness : float, optional
        Roughness of pipe for use in calculating friction factor, [m]
    L : float, optional
        Length of pipe, [m]
    Method : string, optional
        A string of the function name to use

    Returns
    -------
    dP : float
        Pressure drop of the single-phase flow, [Pa]

    Notes
    -----

    Examples
    --------
    >>> one_phase_dP(10.0, 1000, 1E-5, .1, L=1)
    63.43447321097365

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    D2 = D*D
    V = m/(0.25*pi*D2*rho)
    Re = Reynolds(V=V, rho=rho, mu=mu, D=D)
    fd = friction_factor(Re=Re, eD=roughness/D, Method=Method)
    dP = fd*L/D*(0.5*rho*V*V)
    return dP


def one_phase_dP_acceleration(m, D, rho_o, rho_i):
    r'''This function handles calculation of one-phase fluid pressure drop
    due to acceleration for flow inside channels. This is a discrete
    calculation, providing the total differential in pressure for a given
    length and should be called as part of a segment solver routine.

    .. math::
        - \left(\frac{d P}{dz}\right)_{acc} = G^2 \frac{d}{dz} \left[\frac{
        1}{\rho_o} - \frac{1}{\rho_i} \right]

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    D : float
        Diameter of pipe, [m]
    rho_o : float
        Fluid density out, [kg/m^3]
    rho_i : float
        Fluid density int, [kg/m^3]

    Returns
    -------
    dP : float
        Acceleration component of pressure drop for one-phase flow, [Pa]

    Notes
    -----

    Examples
    --------
    >>> one_phase_dP_acceleration(m=1, D=0.1, rho_o=827.1, rho_i=830)
    0.06848289670840459
    '''
    G = 4.0*m/(pi*D*D)
    return G*G*(1.0/rho_o - 1.0/rho_i)


def one_phase_dP_dz_acceleration(m, D, rho, dv_dP, dP_dL, dA_dL):
    r'''This function handles calculation of one-phase fluid pressure drop
    due to acceleration for flow inside channels. This is a continuous
    calculation, providing the differential in pressure per unit length and
    should be called as part of an integration routine [1]_.

    .. math::
        -\left(\frac{\partial P}{\partial L}\right)_{A} = G^2
        \frac{\partial P}{\partial L}\left[\frac{\partial (1/\rho)}{\partial P}
        \right]- \frac{G^2}{\rho}\frac{1}{A}\frac{\partial A}{\partial L}

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    D : float
        Diameter of pipe, [m]
    rho : float
        Fluid density, [kg/m^3]
    dv_dP : float
        Derivative of mass specific volume of the fluid with respect to
        pressure, [m^3/(kg*Pa)]
    dP_dL : float
        Pressure drop per unit length of pipe, [Pa/m]
    dA_dL : float
        Change in area of pipe per unit length of pipe, [m^2/m]

    Returns
    -------
    dP_dz : float
        Acceleration component of pressure drop for one-phase flow, [Pa/m]

    Notes
    -----
    The value returned here is positive for pressure loss and negative for
    pressure increase.

    As `dP_dL` is not known, this equation is normally used in a more
    complicated way than this function provides; this method can be used to
    check the consistency of that routine.

    Examples
    --------
    >>> one_phase_dP_dz_acceleration(m=1, D=0.1, rho=827.1, dv_dP=-1.1E-5,
    ... dP_dL=5E5, dA_dL=0.0001)
    89162.89116373913

    References
    ----------
    .. [1] Shoham, Ovadia. Mechanistic Modeling of Gas-Liquid Two-Phase Flow in
       Pipes. Pap/Cdr edition. Richardson, TX: Society of Petroleum Engineers,
       2006.
    '''
    A = 0.25*pi*D*D
    G = m/A
    return -G*G*(dP_dL*dv_dP - dA_dL/(rho*A))


def one_phase_dP_gravitational(angle, rho, L=1.0, g=g):
    r'''This function handles calculation of one-phase liquid-gas pressure drop
    due to gravitation for flow inside channels. This is either a differential
    calculation for a segment with an infinitesimal difference in elevation
    `L` = 1 or a discrete calculation.

    .. math::
        -\left(\frac{dP}{dz} \right)_{grav} =  \rho g \sin \theta

    .. math::
        -\left(\Delta P \right)_{grav} =  L \rho g \sin \theta

    Parameters
    ----------
    angle : float
        The angle of the pipe with respect to the horizontal, [degrees]
    rho : float
        Fluid density, [kg/m^3]
    L : float, optional
        Length of pipe, [m]
    g : float, optional
        Acceleration due to gravity, [m/s^2]

    Returns
    -------
    dP : float
        Gravitational component of pressure drop for one-phase flow, [Pa/m] or
        [Pa]

    Notes
    -----

    Examples
    --------
    >>> one_phase_dP_gravitational(angle=90, rho=2.6)
    25.49729
    >>> one_phase_dP_gravitational(angle=90, rho=2.6, L=4)
    101.98916
    '''
    angle = radians(angle)
    return L*g*sin(angle)*rho
