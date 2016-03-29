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
from math import log, log10, exp
__all__ = ['friction_factor', 'Moody', 'Alshul_1952', 'Wood_1966', 'Churchill_1973',
'Eck_1973', 'Jain_1976', 'Swamee_Jain_1976', 'Churchill_1977', 'Chen_1979',
'Round_1980', 'Shacham_1980', 'Barr_1981', 'Zigrang_Sylvester_1',
'Zigrang_Sylvester_2', 'Haaland', 'Serghides_1', 'Serghides_2', 'Tsal_1989',
'Manadilli_1997', 'Romeo_2002', 'Sonnad_Goudar_2006', 'Rao_Kumar_2007',
'Buzzelli_2008', 'Avci_Karagoz_2009', 'Papaevangelo_2010', 'Brkic_2011_1',
'Brkic_2011_2', 'Fang_2011', '_roughness']


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
    ff = 1.375E-3*(1 + (2E4*eD + 1E6/Re)**(1/3.))
    fd = ff*4
    return fd


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
    fd = 0.11*(68/Re + eD)**0.25
    return fd


def Wood_1966(Re, eD):
    r'''Calculates Darcy friction factor using the method in Wood (1966) [2]_
    as shown in [1]_.

    .. math::
        f_d = 0.094(\frac{\epsilon}{D})^{0.225} + 0.53(\frac{\epsilon}{D})
        + 88(\frac{\epsilon}{D})^{0.4}Re^{-A_1}

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
    fd = 0.094*eD**0.225 + 0.53*eD +88*eD**0.4*Re**-A1
    return fd


def Churchill_1973(Re, eD):
    r'''Calculates Darcy friction factor using the method in Churchill (1973)
    [2]_ as shown in [1]_

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log\left[\frac{\epsilon}{3.7D} +
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
    fd = (-2*log10(eD/3.7 + (7./Re)**0.9))**-2
    return fd


def Eck_1973(Re, eD):
    r'''Calculates Darcy friction factor using the method in Eck (1973)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log\left[\frac{\epsilon}{3.715D}
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
    fd = (-2*log10(eD/3.715 + 15/Re))**-2
    return fd


def Jain_1976(Re, eD):
    r'''Calculates Darcy friction factor using the method in Jain (1976)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = 2.28 - 4\log\left[ \frac{\epsilon}{D} +
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
    fd = 4*ff
    return fd


def Swamee_Jain_1976(Re, eD):
    r'''Calculates Darcy friction factor using the method in Swamee and
    Jain (1976) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log\left[\left(\frac{6.97}{Re}\right)^{0.9}
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
    fd = 4*ff
    return fd


def Churchill_1977(Re, eD):
    r'''Calculates Darcy friction factor using the method in Churchill and
    (1977) [2]_ as shown in [1]_.

    .. math::
        f_f = 2\left[(\frac{8}{Re})^{12} + (A_2 + A_3)^{-1.5}\right]^{1/12}

        A_2 = \left\{2.457\ln\left[(\frac{7}{Re})^{0.9}
        + 0.27\frac{\epsilon}{D}\right]\right\}^{16}

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
    fd = 4*ff
    return fd


def Chen_1979(Re, eD):
    r'''Calculates Darcy friction factor using the method in Chen (1979) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log\left[\frac{\epsilon}{3.7065D}
        -\frac{5.0452}{Re}\log A_4\right]

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
    fd = 4*ff
    return fd


def Round_1980(Re, eD):
    r'''Calculates Darcy friction factor using the method in Round (1980) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -3.6\log\left[\frac{Re}{0.135Re
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
    fd = 4*ff
    return fd


def Shacham_1980(Re, eD):
    r'''Calculates Darcy friction factor using the method in Shacham (1980) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log\left[\frac{\epsilon}{3.7D} -
        \frac{5.02}{Re} \log\left(\frac{\epsilon}{3.7D}
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
    fd = 4*ff
    return fd


def Barr_1981(Re, eD):
    r'''Calculates Darcy friction factor using the method in Barr (1981) [2]_
    as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log\left\{\frac{\epsilon}{3.7D} +
        \frac{4.518\log(\frac{Re}{7})}{Re\left[1+\frac{Re^{0.52}}{29}
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
        \frac{1}{\sqrt{f_f}} = -4\log\left[\frac{\epsilon}{3.7D}
        - \frac{5.02}{Re}\log A_5\right]

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
    fd = 4*ff
    return fd


def Zigrang_Sylvester_2(Re, eD):
    r'''Calculates Darcy friction factor using the second method in
     Zigrang and Sylvester (1982) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_f}} = -4\log\left[\frac{\epsilon}{3.7D}
        - \frac{5.02}{Re}\log A_6\right]

        A_6 = \frac{\epsilon}{3.7D} - \frac{5.02}{Re}\log A_5

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
    fd = 4*ff
    return fd


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
    fd = 4*ff
    return fd


def Serghides_1(Re, eD):
    r'''Calculates Darcy friction factor using the method in Serghides (1984)
    [2]_ as shown in [1]_.

    .. math::
        f=\left[A-\frac{(B-A)^2}{C-2B+A}\right]^{-2}

        A=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{12}{Re}\right]

        B=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{2.51A}{Re}\right]

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
    .. [2] 	Serghides T.K (1984)."Estimate friction factor accurately"
       Chemical Engineering, Vol. 91(5), pp. 63-64.
    '''
    A = -2*log10(eD/3.7 + 12/Re)
    B = -2*log10(eD/3.7 + 2.51*A/Re)
    C = -2*log10(eD/3.7 + 2.51*B/Re)
    fd = (A - (B-A)**2/(C-2*B + A))**-2
    return fd


def Serghides_2(Re, eD):
    r'''Calculates Darcy friction factor using the method in Serghides (1984)
    [2]_ as shown in [1]_.

    .. math::
        f_d = \left[ 4.781 - \frac{(A - 4.781)^2}
        {B-2A+4.781}\right]^{-2}

        A=-2\log_{10}\left[\frac{\epsilon/D}{3.7}+\frac{12}{Re}\right]

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
    fd = (4.781 - (A - 4.781)**2/(B - 2*A + 4.781))**-2
    return fd


def Tsal_1989(Re, eD):
    r'''Calculates Darcy friction factor using the method in Tsal (1989)
    [2]_ as shown in [1]_.

    .. math::
        A = 0.11(\frac{68}{Re} + \frac{\epsilon}{D})^{0.25}

    if A >= 0.018 then fd = A
    if A < 0.018 then fd = 0.0028 + 0.85 A

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
    .. [2] 	Tsal, R.J.: Altshul-Tsal friction factor equation.
       Heat-Piping-Air Cond. 8, 30-45 (1989)
    '''
    A = 0.11*(68/Re + eD)**0.25
    if A >= 0.018:
        fd = A
    else:
        fd = 0.0028 + 0.85*A
    return fd


def Manadilli_1997(Re, eD):
    r'''Calculates Darcy friction factor using the method in Manadilli (1997)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log\left[\frac{\epsilon}{3.7D} +
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
    fd = (-2*log10(eD/3.7 + 95/Re**0.983 - 96.82/Re))**-2
    return fd


def Romeo_2002(Re, eD):
    r'''Calculates Darcy friction factor using the method in Romeo (2002)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = -2\log\left\{\frac{\epsilon}{3.7065D}\times
        \frac{5.0272}{Re}\times\log\left[\frac{\epsilon}{3.827D} -
        \frac{4.567}{Re}\times\log\left(\frac{\epsilon}{7.7918D}^{0.9924} +
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
    fd = (.8686*log(.4587*Re/S**(S/(S+1))))**-2
    return fd


def Rao_Kumar_2007(Re, eD):
    r'''Calculates Darcy friction factor using the method in Rao and Kumar
    (2007) [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = 2\log\left(\frac{(2\frac{\epsilon}{D})^{-1}}
        {\left(\frac{0.444 + 0.135Re}{Re}\right)\beta}\right)

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
    fd = (2*log10((2*eD)**-1/beta/((0.444+0.135*Re)/Re)))**-2
    return fd


def Buzzelli_2008(Re, eD):
    r'''Calculates Darcy friction factor using the method in Buzzelli (2008)
    [2]_ as shown in [1]_.

    .. math::
        \frac{1}{\sqrt{f_d}} = B_1 - \left[\frac{B_1 +2\log(\frac{B_2}{Re})}
        {1 + \frac{2.18}{B_2}}\right]

        B_1 = \frac{0.774\ln(Re)-1.41}{1+1.32\sqrt{\frac{\epsilon}{D}}}

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
    B1 = (.774*log(Re)-1.41)/(1+1.32*eD**0.5)
    B2 = eD/3.7*Re + 2.51*B1
    fd = (B1- (B1+2*log10(B2/Re))/(1+2.18/B2))**-2
    return fd


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
    fd = 6.4*(log(Re) - log(1 + 0.01*Re*eD*(1+10*eD**0.5)))**-2.4
    return fd


def Papaevangelo_2010(Re, eD):
    r'''Calculates Darcy friction factor using the method in Papaevangelo
    (2010) [2]_ as shown in [1]_.

    .. math::
        f_D = \frac{0.2479 - 0.0000947(7-\log Re)^4}{\left[\log\left
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
    fd = (0.2479-0.0000947*(7-log(Re))**4)/(log10(eD/3.615 + 7.366/Re**0.9142))**2
    return fd


def Brkic_2011_1(Re, eD):
    r'''Calculates Darcy friction factor using the method in Brkic
    (2011) [2]_ as shown in [1]_.

    .. math::
        f_d = [-2\log(10^{-0.4343\beta} + \frac{\epsilon}{3.71D})]^{-2}

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
    fd = (-2*log10(10**(-0.4343*beta)+eD/3.71))**-2
    return fd


def Brkic_2011_2(Re, eD):
    r'''Calculates Darcy friction factor using the method in Brkic
    (2011) [2]_ as shown in [1]_.

    .. math::
        f_d = [-2\log(\frac{2.18\beta}{Re}+ \frac{\epsilon}{3.71D})]^{-2}

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
    fd = (-2*log10(2.18*beta/Re + eD/3.71))**-2
    return fd


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
    fd = log(0.234*eD**1.1007 - 60.525/Re**1.1105 + 56.291/Re**1.0712)**-2*1.613
    return fd

### Main functions

fmethods = {}
fmethods['Moody'] = {'Nice name': 'Moody', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 1.0, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Alshul_1952'] = {'Nice name': 'Alshul 1952', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Wood_1966'] = {'Nice name': 'Wood 1966', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-05, 'Default': None, 'Max': 0.04, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 50000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Churchill_1973'] = {'Nice name': 'Churchill 1973', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Eck_1973'] = {'Nice name': 'Eck 1973', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Jain_1976'] = {'Nice name': 'Jain 1976', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 4e-05, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 5000.0, 'Default': None, 'Max': 10000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Swamee_Jain_1976'] = {'Nice name': 'Swamee Jain 1976', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-06, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 5000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Churchill_1977'] = {'Nice name': 'Churchill 1977', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Chen_1979'] = {'Nice name': 'Chen 1979', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-07, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 400000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Round_1980'] = {'Nice name': 'Round 1980', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 400000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Shacham_1980'] = {'Nice name': 'Shacham 1980', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 400000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Barr_1981'] = {'Nice name': 'Barr 1981', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Zigrang_Sylvester_1'] = {'Nice name': 'Zigrang Sylvester 1', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 4e-05, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Zigrang_Sylvester_2'] = {'Nice name': 'Zigrang Sylvester 2', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 4e-05, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Haaland'] = {'Nice name': 'Haaland', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-06, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Serghides_1'] = {'Nice name': 'Serghides 1', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Serghides_2'] = {'Nice name': 'Serghides 2', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Tsal_1989'] = {'Nice name': 'Tsal 1989', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Manadilli_1997'] = {'Nice name': 'Manadilli 1997', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 5245.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Romeo_2002'] = {'Nice name': 'Romeo 2002', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 3000.0, 'Default': None, 'Max': 150000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Sonnad_Goudar_2006'] = {'Nice name': 'Sonnad Goudar 2006', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-06, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 4000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Rao_Kumar_2007'] = {'Nice name': 'Rao Kumar 2007', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Buzzelli_2008'] = {'Nice name': 'Buzzelli 2008', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Avci_Karagoz_2009'] = {'Nice name': 'Avci Karagoz 2009', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Papaevangelo_2010'] = {'Nice name': 'Papaevangelo 2010', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 1e-05, 'Default': None, 'Max': 0.001, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 10000.0, 'Default': None, 'Max': 10000000.0, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Brkic_2011_1'] = {'Nice name': 'Brkic 2011 1', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Brkic_2011_2'] = {'Nice name': 'Brkic 2011 2', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': None, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Fang_2011'] = {'Nice name': 'Fang 2011', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': 0.05, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 3000.0, 'Default': None, 'Max': 100000000.0, 'Symbol': '\text{Re}', 'Units': None}}}



def friction_factor(Re=1E5, eD=1E-4, Method=None, Darcy=True, AvailableMethods=False):
    r'''Calculates friction factor. Uses a specified method, or automatically
    picks one from the dictionary of available methods. 28 methods available,
    described in the table below. The default is more than sufficient
    for all applications. Can also be accesed under the name `fd`.

    Examples
    --------
    >>> friction_factor(Re=1E5, eD=1E-4)
    0.018513948401365277

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float
        Relative roughness of the wall, []

    Returns
    -------
    f : float
        Friction factor, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which claim to be valid for the range of `Re` and `eD`
        given

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use
    Darcy : bool, optional
        If False, will return fanning friction factor, 1/4 of the Darcy value
    AvailableMethods : bool, optional
        If True, function will consider which methods claim to be valid for
        the range of `Re` and `eD` given

    Notes
    -----
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Nice name          |Re min|Re max|Re Default|:math:`\epsilon/D` Min|:math:`\epsilon/D` Max|:math:`\epsilon/D` Default|
    +===================+======+======+==========+======================+======================+==========================+
    |Rao Kumar 2007     |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Eck 1973           |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Jain 1976          |5000  |1.0E+7|None      |4.0E-5                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Avci Karagoz 2009  |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Swamee Jain 1976   |5000  |1.0E+8|None      |1.0E-6                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Churchill 1977     |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Brkic 2011 1       |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Chen 1979          |4000  |4.0E+8|None      |1.0E-7                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Round 1980         |4000  |4.0E+8|None      |0                     |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Papaevangelo 2010  |10000 |1.0E+7|None      |1.0E-5                |0.001                 |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Fang 2011          |3000  |1.0E+8|None      |0                     |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Shacham 1980       |4000  |4.0E+8|None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Barr 1981          |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Churchill 1973     |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Moody              |4000  |1.0E+8|None      |0                     |1                     |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Zigrang Sylvester 1|4000  |1.0E+8|None      |4.0E-5                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Zigrang Sylvester 2|4000  |1.0E+8|None      |4.0E-5                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Buzzelli 2008      |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Haaland            |4000  |1.0E+8|None      |1.0E-6                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Serghides 1        |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Serghides 2        |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Tsal 1989          |4000  |1.0E+8|None      |0                     |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Alshul 1952        |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Wood 1966          |4000  |5.0E+7|None      |1.0E-5                |0.04                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Manadilli 1997     |5245  |1.0E+8|None      |0                     |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Brkic 2011 2       |None  |None  |None      |None                  |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Romeo 2002         |3000  |1.5E+8|None      |0                     |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Sonnad Goudar 2006 |4000  |1.0E+8|None      |1.0E-6                |0.05                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    '''
    def list_methods():
        methods = [i for i in fmethods if
        (not fmethods[i]['Arguments']['eD']['Min'] or fmethods[i]['Arguments']['eD']['Min'] <= eD) and
        (not fmethods[i]['Arguments']['eD']['Max'] or eD <= fmethods[i]['Arguments']['eD']['Max']) and
        (not fmethods[i]['Arguments']['Re']['Min'] or Re > fmethods[i]['Arguments']['Re']['Min']) and
        (not fmethods[i]['Arguments']['Re']['Max'] or Re <= fmethods[i]['Arguments']['Re']['Max'])]
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = 'Buzzelli_2008'
    f = globals()[Method](Re=Re, eD=eD)
    if not Darcy:
        f *= 4
    return f


fd = friction_factor # shortcut

#print friction_factor(Re=1E5, eD=1E-4, AvailableMethods=True)

# Roughness, in m
_roughness = {'Brass': .00000152, 'Lead': .00000152, 'Glass': .00000152,
'Steel': .00000152, 'Asphalted cast iron': .000122, 'Galvanized iron': .000152,
'Cast iron': .000259, 'Wood stave': .000183, 'Rough wood stave': .000914,
'Concrete': .000305, 'Rough concrete': .00305, 'Riveted steel': .000914,
'Rough riveted steel': .00914}




#### Code used to create the dictionary
#data = '''Moody	4000	1.0E+8	0	1
#Alshul_1952
#Wood_1966	4000	5.0E+7	1.0E-5	0.04
#Churchill_1973
#Eck_1973
#Jain_1976	5000	1.0E+7	4.0E-5	0.05
#Swamee_Jain_1976	5000	1.0E+8	1.0E-6	0.05
#Churchill_1977
#Chen_1979	4000	4.0E+8	1.0E-7	0.05
#Round_1980	4000	4.0E+8	0	0.05
#Shacham_1980	4000	4.0E+8
#Barr_1981
#Zigrang_Sylvester_1	4000	1.0E+8	4.0E-5	0.05
#Zigrang_Sylvester_2	4000	1.0E+8	4.0E-5	0.05
#Haaland	4000	1.0E+8	1.0E-6	0.05
#Serghides_1
#Serghides_2
#Tsal_1989	4000	1.0E+8	0	0.05
#Manadilli_1997	5245	1.0E+8	0	0.05
#Romeo_2002	3000	1.5E+8	0	0.05
#Sonnad_Goudar_2006	4000	1.0E+8	1.0E-6	0.05
#Rao_Kumar_2007
#Buzzelli_2008
#Avci_Karagoz_2009
#Papaevangelo_2010	10000	1.0E+7	1.0E-5	0.001
#Brkic_2011_1
#Brkic_2011_2
#Fang_2011	3000	1.0E+8	0	0.05'''

#for i in data.split('\n'):
#    j = i.split('\t')
#    if len(j) == 1:
#        fname = j[0]
#        Remin, Remax, eDmin, eDmax = None, None, None, None
#    elif len(j) == 3:
#        fname, Remin, Remax = j
##        Remin = float(Remin)
##        Remax = float(Remax)
#        eDmin, eDmax = None, None
#    elif len(j) == 5:
#        fname, Remin, Remax, eDmin, eDmax = j
##        Remin = float(Remin)
##        Remax = float(Remax)
##        eDmin = float(eDmin)
##        eDmax = float(eDmax)
##
##    args = {}
#    Re_args = (Remin, Remax, None)
#    eD_args = (eDmin, eDmax, None)
#
##    args['eD'] = {'Name': 'Relative roughness', 'Symbol': '\epsilon/D', 'Default': None, 'Min': eDmin, 'Max': eDmax, 'Units': None }
##    args['Re'] = {'Name': 'Reynolds number', 'Symbol': '\text{Re}', 'Default': None, 'Min':  Remin, 'Max': Remax, 'Units': None }
##    print args
#    print '''fmethods[%s] = {'Nice name': '%s', 'Re_details': %s, 'eD_details': %s}'''  %(fname, fname.replace('_', ' '), Re_args, eD_args)
#
#
#fmethods_prototype = {'Re_details': ('Re min', 'Re max', 'Re Default'),
#                      'eD_details': (':math:`\epsilon/D` Min', ':math:`\epsilon/D` Max', ':math:`\epsilon/D` Default')}
#
##print fmethods_prototype
#
#fmethods = {}
#fmethods[Moody] = {'Nice name': 'Moody', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('0', '1', None)}
#fmethods[Alshul_1952] = {'Nice name': 'Alshul 1952', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Wood_1966] = {'Nice name': 'Wood 1966', 'Re_details': ('4000', '5.0E+7', None), 'eD_details': ('1.0E-5', '0.04', None)}
#fmethods[Churchill_1973] = {'Nice name': 'Churchill 1973', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Eck_1973] = {'Nice name': 'Eck 1973', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Jain_1976] = {'Nice name': 'Jain 1976', 'Re_details': ('5000', '1.0E+7', None), 'eD_details': ('4.0E-5', '0.05', None)}
#fmethods[Swamee_Jain_1976] = {'Nice name': 'Swamee Jain 1976', 'Re_details': ('5000', '1.0E+8', None), 'eD_details': ('1.0E-6', '0.05', None)}
#fmethods[Churchill_1977] = {'Nice name': 'Churchill 1977', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Chen_1979] = {'Nice name': 'Chen 1979', 'Re_details': ('4000', '4.0E+8', None), 'eD_details': ('1.0E-7', '0.05', None)}
#fmethods[Round_1980] = {'Nice name': 'Round 1980', 'Re_details': ('4000', '4.0E+8', None), 'eD_details': ('0', '0.05', None)}
#fmethods[Shacham_1980] = {'Nice name': 'Shacham 1980', 'Re_details': ('4000', '4.0E+8', None), 'eD_details': (None, None, None)}
#fmethods[Barr_1981] = {'Nice name': 'Barr 1981', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Zigrang_Sylvester_1] = {'Nice name': 'Zigrang Sylvester 1', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('4.0E-5', '0.05', None)}
#fmethods[Zigrang_Sylvester_2] = {'Nice name': 'Zigrang Sylvester 2', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('4.0E-5', '0.05', None)}
#fmethods[Haaland] = {'Nice name': 'Haaland', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('1.0E-6', '0.05', None)}
#fmethods[Serghides_1] = {'Nice name': 'Serghides 1', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Serghides_2] = {'Nice name': 'Serghides 2', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Tsal_1989] = {'Nice name': 'Tsal 1989', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('0', '0.05', None)}
#fmethods[Manadilli_1997] = {'Nice name': 'Manadilli 1997', 'Re_details': ('5245', '1.0E+8', None), 'eD_details': ('0', '0.05', None)}
#fmethods[Romeo_2002] = {'Nice name': 'Romeo 2002', 'Re_details': ('3000', '1.5E+8', None), 'eD_details': ('0', '0.05', None)}
#fmethods[Sonnad_Goudar_2006] = {'Nice name': 'Sonnad Goudar 2006', 'Re_details': ('4000', '1.0E+8', None), 'eD_details': ('1.0E-6', '0.05', None)}
#fmethods[Rao_Kumar_2007] = {'Nice name': 'Rao Kumar 2007', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Buzzelli_2008] = {'Nice name': 'Buzzelli 2008', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Avci_Karagoz_2009] = {'Nice name': 'Avci Karagoz 2009', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Papaevangelo_2010] = {'Nice name': 'Papaevangelo 2010', 'Re_details': ('10000', '1.0E+7', None), 'eD_details': ('1.0E-5', '0.001', None)}
#fmethods[Brkic_2011_1] = {'Nice name': 'Brkic 2011 1', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Brkic_2011_2] = {'Nice name': 'Brkic 2011 2', 'Re_details': (None, None, None), 'eD_details': (None, None, None)}
#fmethods[Fang_2011] = {'Nice name': 'Fang 2011', 'Re_details': ('3000', '1.0E+8', None), 'eD_details': ('0', '0.05', None)}



#header = ['Nice name']
#rows = [header]
#row_elements = []
#
#for details in fmethods_prototype.values():
#    for detail in details:
#        header += [detail]
#
#
#for f in fmethods.values():
#    row = [f['Nice name']]
#    for detail_type in ['Re_details', 'eD_details']:
#        for detail in f[detail_type]:
#            row.append(str(detail))
#    rows.append(row)
#
#
#lengths = [0 for i in rows[0]]
#for row in rows:
#    for i in range(len(row)):
#        lengths[i] = max(lengths[i], len(row[i]))
#
#def main_line(lengths):
#    new_line = ''
#    for length in lengths:
#        new_line += '+' + '-'*length
#    new_line += '+\n'
#    return new_line
#
#def header_line(lengths):
#    line = ''
#    for length in lengths:
#        line += '+' + '='*length
#    line += '+\n'
#    return line
#
#n_line = main_line(lengths)
#h_line = header_line(lengths)
#
#
#table = n_line
#for k in range(len(rows)):
#    row = rows[k]
#
#    for i in range(len(row)):
#        table += '|' + str(row[i]).ljust(lengths[i])
#    table += '|\n'
#
#    if k == 0:
#        table += h_line
#    else:
#        table += n_line
##table += n_line
#print table

