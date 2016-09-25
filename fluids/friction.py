# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import log, log10, exp
import difflib
from scipy.special import lambertw
from scipy.constants import inch

__all__ = ['friction_factor', 'Colebrook', 'Clamond', 'Moody', 'Alshul_1952', 'Wood_1966', 'Churchill_1973',
'Eck_1973', 'Jain_1976', 'Swamee_Jain_1976', 'Churchill_1977', 'Chen_1979',
'Round_1980', 'Shacham_1980', 'Barr_1981', 'Zigrang_Sylvester_1',
'Zigrang_Sylvester_2', 'Haaland', 'Serghides_1', 'Serghides_2', 'Tsal_1989',
'Manadilli_1997', 'Romeo_2002', 'Sonnad_Goudar_2006', 'Rao_Kumar_2007',
'Buzzelli_2008', 'Avci_Karagoz_2009', 'Papaevangelo_2010', 'Brkic_2011_1',
'Brkic_2011_2', 'Fang_2011', 'friction_laminar', 'Blasius', 'von_Karman', 'Prandtl_von_Karman_Nikuradse', 
'transmission_factor', 'roughness_Farshad', '_Farshad_roughness', 
'_roughness', 'oregon_smooth_data']

oregon_Res = [11.21, 20.22, 29.28, 43.19, 57.73, 64.58, 86.05, 113.3, 135.3, 
              157.5, 179.4, 206.4, 228, 270.9, 315.2, 358.9, 402.9, 450.2, 
              522.5, 583.1, 671.8, 789.8, 891, 1013, 1197, 1300, 1390, 1669, 
              1994, 2227, 2554, 2868, 2903, 2926, 2955, 2991, 2997, 3047, 3080,
              3264, 3980, 4835, 5959, 8162, 10900, 13650, 18990, 29430, 40850, 
              59220, 84760, 120000, 176000, 237700, 298200, 467800, 587500, 
              824200, 1050000]
oregon_fd_smooth = [5.537, 3.492, 2.329, 1.523, 1.173, 0.9863, 0.7826, 0.5709,
                    0.4815, 0.4182, 0.3655, 0.3237, 0.2884, 0.2433, 0.2077, 
                    0.1834, 0.1656, 0.1475, 0.1245, 0.1126, 0.09917, 0.08501, 
                    0.07722, 0.06707, 0.0588, 0.05328, 0.04815, 0.04304, 
                    0.03739, 0.03405, 0.03091, 0.02804, 0.03182, 0.03846, 
                    0.03363, 0.04124, 0.035, 0.03875, 0.04285, 0.0426, 0.03995,
                    0.03797, 0.0361, 0.03364, 0.03088, 0.02903, 0.0267, 
                    0.02386, 0.02086, 0.02, 0.01805, 0.01686, 0.01594, 0.01511,
                    0.01462, 0.01365, 0.01313, 0.01244, 0.01198]
'''Holds a tuple of experimental results from the smooth pipe flow experiments
presented in McKEON, B. J., C. J. SWANSON, M. V. ZAGAROLA, R. J. DONNELLY, and 
A. J. SMITS. "Friction Factors for Smooth Pipe Flow." Journal of Fluid 
Mechanics 511 (July 1, 2004): 41-44. doi:10.1017/S0022112004009796.
'''
oregon_smooth_data = (oregon_Res, oregon_fd_smooth)

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
    For round pipes, this valid for Re < 2320. 
    
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
    return 0.3164*Re**-0.25


def Colebrook(Re, eD):
    r'''Calculates Darcy friction factor using an exact solution to the 
    Colebrook equation, derived with a CAS. Relatively slow despite its
    explicit form. 

    .. math::
        \frac{1}{\sqrt{f}}=-2\log_{10}\left(\frac{\epsilon/D}{3.7}
        +\frac{2.51}{\text{Re}\sqrt{f}}\right)

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
    The solution is as follows:
    
    .. math::
        f_d = \frac{\ln(10)^2\cdot {3.7}^2\cdot{2.51}^2}
        {\left(\log(10)\epsilon/D\cdot\text{Re} - 2\cdot 2.51\cdot 3.7\cdot
        \text{lambertW}\left[\log(\sqrt{10})\sqrt{
        10^{\left(\frac{\epsilon \text{Re}}{2.51\cdot 3.7D}\right)}
        \cdot \text{Re}^2/{2.51}^2}\right]\right)}

    Some effort to optimize this function has been made. The `lambertw`  
    function from scipy is used, and is defined to solve the specific function:
    
    .. math::
        y = x\exp(x)
        
        \text{lambertW}(y) = x

    For high relative roughness and reynolds numbers, an OverflowError is 
    raised in solution of this equation. 

    Examples
    --------
    >>> Colebrook(1E5, 1E-4)
    0.018513866077471648

    References
    ----------
    .. [1] Colebrook, C F."Turbulent Flow in Pipes, with Particular Reference to
       the Transition Region Between the Smooth and Rough Pipe Laws." Journal 
       of the ICE 11, no. 4 (February 1, 1939): 133-156. 
       doi:10.1680/ijoti.1939.13150.
    '''
    # 9.287 = 2.51*3.7; 6.3001 = 2.51**2
    sub = 10**(eD*Re/9.287)*Re**2/6.3001 
    # 1.15129... = log(sqrt(10))
    lambert_term = lambertw(1.151292546497022950546806896454654633998870849609375*sub**0.5).real 
    # log(10) = 2.302585...; 2*2.51*3.7 = 18.574
    # 457.28... = log(10)**2*3.7**2*2.51**2
    return (457.28006463294371997108100913465023040771484375
            /(2.30258509299404590109361379290930926799774169921875*eD*Re - 18.574*lambert_term)**2)


def Clamond(Re, eD):
    r'''Calculates Darcy friction factor using a solution accurate to almost
    machine precision. Recommended very strongly. For details of the algorithm,
    see [1]_. 
    
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
    This is a highly optimized function, 4 times faster than the solution using
    the LambertW function, and faster than many other approximations which are 
    much less accurate.

    The code used here is only slightly modified than that in [1]_, for further
    performance improvements. 
    
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
    '''
    X1 = eD*Re*0.1239681863354175460160858261654858382699 # (log(10)/18.574).evalf(40)
    X2 = log(Re) - 0.7793974884556819406441139701653776731705 # log(log(10)/5.02).evalf(40)
    F = X2 - 0.2
    X1F = X1 + F
    X1F1 = 1. + X1F
    
    E = (log(X1F) - 0.2)/(X1F1)
    F = F - (X1F1 + 0.5*E)*E*(X1F)/ (X1F1 + E*(1. + E/3.))

    X1F = X1 + F
    X1F1 = 1. + X1F
    E = (log(X1F) + F - X2)/(X1F1)
    F = F - (X1F1 + 0.5*E)*E*(X1F)/ (X1F1 + E*(1. + E/3.))

    return 1.325474527619599502640416597148504422899/F/F # ((0.5*log(10))**2).evalf(40)



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
    .. [2] Serghides T.K (1984)."Estimate friction factor accurately"
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
        \log(10)Re}{2(2.51)}\right)\right)^2}

    Examples
    --------
    >>> Prandtl_von_Karman_Nikuradse(1E7)
    0.0081026694308749137

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
    return c2/(lambertw((c1*Re)/2.51).real)**2




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
fmethods['Clamond'] = {'Nice name': 'Clamond 2009', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 0, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}
fmethods['Colebrook'] = {'Nice name': 'Colebrook', 'Notes': '', 'Arguments': {'eD': {'Name': 'Relative roughness', 'Min': 0.0, 'Default': None, 'Max': None, 'Symbol': '\\epsilon/D', 'Units': None}, 'Re': {'Name': 'Reynolds number', 'Min': 0, 'Default': None, 'Max': None, 'Symbol': '\text{Re}', 'Units': None}}}



def friction_factor(Re, eD=0, Method='Clamond', Darcy=True, AvailableMethods=False):
    r'''Calculates friction factor. Uses a specified method, or automatically
    picks one from the dictionary of available methods. 29 approximations are 
    available as well as the direct solution, described in the table below. 
    The default is to use the exact solution. Can also be accesed under the 
    name `fd`.
    
    For Re < 2320, the laminar solution is always returned, regardless of
    selected method.

    Examples
    --------
    >>> friction_factor(Re=1E5, eD=1E-4)
    0.01851386607747165

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    eD : float, optional
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
    
    See Also
    --------
    Colebrook
    Clamond
    
    Notes
    -----
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
    |Nice name          |Re min|Re max|Re Default|:math:`\epsilon/D` Min|:math:`\epsilon/D` Max|:math:`\epsilon/D` Default|
    +===================+======+======+==========+======================+======================+==========================+
    |Clamond            |0     |None  |None      |0                     |None                  |None                      |
    +-------------------+------+------+----------+----------------------+----------------------+--------------------------+
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
    elif not Method:
        Method = 'Clamond'
    if Re < 2320:
        f = friction_laminar(Re)
    else:
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

# Data from the Handbook of Hydraulic Resistance, 4E, in format (min, max, avg) roughness in m
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
    'Corroded, moderately ': (None, None, -4.0E-4),
    'Scale, small depositions only ': (None, None, -4.0E-4),
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
    'New and covered with bitumen': (None, None, -5.0E-5),
    'Used and covered with partially dissolved bitumen; corroded':
    (None, None, -1.0E-4),
    'Used, suffering general corrosion': (None, None, -1.5E-4),
    'Surface looks like new, 10 mm lacquer inside, even joints':
    (3.0E-4, 4.0E-4, None),
    'Used Gas mains': (None, None, -5.0E-4),
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


HHR_roughness_dicts = [tunnels, steels, wood_plywood_glass, concretes]
HHR_roughness_categories, HHR_roughness = {}, {}
[HHR_roughness_categories.update(i) for i in HHR_roughness_dicts]
[[HHR_roughness.update(i) for i in j.values()] for j in HHR_roughness_dicts]


# Format : ID: (avg_roughness, coef A (inches), coef B (inches))
_Farshad_roughness = {'Plastic coated': (5E-6, 0.0002, -1.0098),
                      'Carbon steel, honed bare': (12.5E-6, 0.0005, -1.0101),
                      'Cr13, electropolished bare': (30E-6, 0.0012, -1.0086),
                      'Cement lining': (33E-6, 0.0014, -1.0105),
                      'Carbon steel, bare': (36E-6, 0.0014, -1.0112),
                      'Fiberglass lining': (38E-6, 0.0016, -1.0086),
                      'Cr13, bare': (55E-6, 0.0021, -1.0055)  }


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
        (A, B) Coefficients to use directly, instead of looking them up
        [inch^-B, -]

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
    values would be ncessary to produce the observed pressure drops. The 
    average difference between this back-calculated roughness and the measured
    roughness was 6.75%.

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
    if ID is None and coeffs:
        A, B = coeffs
        return A*(D/inch)**(B+1)*inch
    # Case 2, lookup parameters
    try :
        dat = _Farshad_roughness[ID]
    except:
        raise KeyError('ID was not in _Farshad_roughness.')
    if D is None:
        return dat[0]
    else:
        A, B = dat[1], dat[2]
        return A*(D/inch)**(B+1)*inch


roughness_clean_dict = _roughness.copy()
roughness_clean_dict.update(_Farshad_roughness)


#def nearest_roughness(name, clean=True):
#    if clean:
#        d = roughness_clean_dict
#    else:
#        d = HHR_roughness
#    ID = difflib.get_close_matches(name, d.keys(), n=1, cutoff=0.6)
#    if not ID:
#        ID = difflib.get_close_matches(name, d.keys(), n=1, cutoff=0.3)
#    if not ID:
#        ID = difflib.get_close_matches(name, d.keys(), n=1, cutoff=0)
#    return ID[0]


def transmission_factor(fd=None, F=None):
    r'''Calculates either transmission factor from Darcy friction factor,
    or Darcy friction factor from the transmission factor. Raises an exception
    if neither input is given.
    
    Transmission factor is a term used in compressible gas flow in pipelines.

    .. math::
        F = \frac{2}{\sqrt{f_d}}

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

    References
    ----------
    .. [1] Menon, E. Shashi. Gas Pipeline Hydraulics. 1st edition. Boca Raton, 
       FL: CRC Press, 2005.
    '''
    if fd:
        return 2./fd**0.5
    elif F:
        return 4./(F*F)
    else:
        raise Exception('Either Darcy friction factor or transmission factor is needed')



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

