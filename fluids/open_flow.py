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
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.constants import g
from math import tan, radians

__all__ = ['Q_weir_V_Shen',
'Q_weir_rectangular_Kindsvater_Carter', 'Q_weir_rectangular_SIA',
'Q_weir_rectangular_full_Ackers', 'Q_weir_rectangular_full_SIA',
'Q_weir_rectangular_full_Rehbock', 'Q_weir_rectangular_full_Kindsvater_Carter',
'V_Manning', 'n_Manning_to_C_Chezy', 'C_Chezy_to_n_Manning', 'V_Chezy',
'n_natural', 'n_excavated_dredged', 'n_lined_built', 'n_closed_conduit',
'n_dicts']


nape_types = ['free', 'depressed', 'clinging']
flow_types = ['aerated', 'partially aerated', 'unaerated']
weir_types = ['V-notch', 'rectangular', 'rectangular full-channel',
              'Cipoletti', 'broad-crested', 'Ogee']

angles_Shen = [20, 40, 60, 80, 100]
Cs_Shen = [0.59, 0.58, 0.575, 0.575, 0.58]
k_Shen = [0.0028, 0.0017, 0.0012, 0.001, 0.001]

Cs_Shen_i = interp1d(angles_Shen, Cs_Shen)
k_Shen_i = interp1d(angles_Shen, k_Shen)


### V-Notch Weirs (Triangular weir)

def Q_weir_V_Shen(h1, angle=90):
    r'''Calculates the flow rate across a V-notch (triangular) weir from
    the height of the liquid above the tip of the notch, and with the angle
    of the notch. Most of these type of weir are 90 degrees. Model from [1]_
    as reproduced in [2]_.

    Flow rate is given by:

    .. math::
        Q = C \tan\left(\frac{\theta}{2}\right)\sqrt{g}(h_1 + k)^{2.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the notch [m]
    angle : float, optional
        Angle of the notch [degrees]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    angles = [20, 40, 60, 80, 100]
    Cs = [0.59, 0.58, 0.575, 0.575, 0.58]
    k = [0.0028, 0.0017, 0.0012, 0.001, 0.001]

    The following limits apply to the use of this equation:

    h1 >= 0.05 m
    h2 > 0.45 m
    h1/h2 <= 0.4 m
    b > 0.9 m

    .. math::
        \frac{h_1}{b}\tan\left(\frac{\theta}{2}\right) < 2

    Flows are lower than obtained by the curves at
    http://www.lmnoeng.com/Weirs/vweir.php.

    Examples
    --------
    >>> Q_weir_V_Shen(0.6, angle=45)
    0.21071725775478228

    References
    ----------
    .. [1] Shen, John. "Discharge Characteristics of Triangular-Notch
       Thin-Plate Weirs : Studies of Flow to Water over Weirs and Dams."
       USGS Numbered Series. Water Supply Paper. U.S. Geological Survey :
       U.S. G.P.O., 1981
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    C = Cs_Shen_i(angle)
    k = k_Shen_i(angle)
    Q = C*tan(radians(angle)/2)*g**0.5*(h1 + k)**2.5
    return Q


### Rectangular Weirs


def Q_weir_rectangular_Kindsvater_Carter(h1, h2, b):
    r'''Calculates the flow rate across rectangular weir from
    the height of the liquid above the crest of the notch, the liquid depth
    beneath it, and the width of the notch. Model from [1]_ as reproduced in
    [2]_.

    Flow rate is given by:

    .. math::
        Q = 0.554\left(1 - 0.0035\frac{h_1}{h_2}\right)(b + 0.0025)
        \sqrt{g}(h_1 + 0.0001)^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the rectangular flow section of the weir [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    b/b1 ≤ 0.2
    h1/h2 < 2
    b > 0.15 m
    h1 > 0.03 m
    h2 > 0.1 m

    Examples
    --------
    >>> Q_weir_rectangular_Kindsvater_Carter(0.2, 0.5, 1)
    0.15545928949179422

    References
    ----------
    .. [1] Kindsvater, Carl E., and Rolland W. Carter. "Discharge
       Characteristics of Rectangular Thin-Plate Weirs." Journal of the
       Hydraulics Division 83, no. 6 (December 1957): 1-36.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    Q = 0.554*(1 - 0.0035*h1/h2)*(b + 0.0025)*g**0.5*(h1 + 0.0001)**1.5
    return Q


def Q_weir_rectangular_SIA(h1, h2, b, b1):
    r'''Calculates the flow rate across rectangular weir from
    the height of the liquid above the crest of the notch, the liquid depth
    beneath it, and the width of the notch. Model from [1]_ as reproduced in
    [2]_.

    Flow rate is given by:

    .. math::
        Q = 0.544\left[1 + 0.064\left(\frac{b}{b_1}\right)^2 + \frac{0.00626
        - 0.00519(b/b_1)^2}{h_1 + 0.0016}\right]
        \left[1 + 0.5\left(\frac{b}{b_1}\right)^4\left(\frac{h_1}{h_1+h_2}
        \right)^2\right]b\sqrt{g}h^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the rectangular flow section of the weir [m]
    b1 : float
        Width of the full section of the channel [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    b/b1 ≤ 0.2
    h1/h2 < 2
    b > 0.15 m
    h1 > 0.03 m
    h2 > 0.1 m

    Examples
    --------
    >>> Q_weir_rectangular_SIA(0.2, 0.5, 1, 2)
    1.0408858453811165

    References
    ----------
    .. [1] Normen für Wassermessungen: bei Durchführung von Abnahmeversuchen
       an Wasserkraftmaschinen. SIA, 1924.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    h = h1 + h2
    Q = 0.544*(1 + 0.064*(b/b1)**2 + (0.00626 - 0.00519*(b/b1)**2)/(h1 + 0.0016))\
    *(1 + 0.5*(b/b1)**4*(h1/(h1 + h2))**2)*b*g**0.5*h**1.5
    return Q


### Rectangular Weirs, full channel

def Q_weir_rectangular_full_Ackers(h1, h2, b):
    r'''Calculates the flow rate across a full-channel rectangular weir from
    the height of the liquid above the crest of the weir, the liquid depth
    beneath it, and the width of the channel. Model from [1]_ as reproduced in
    [2]_, confirmed with [3]_.

    Flow rate is given by:

    .. math::
        Q = 0.564\left(1+0.150\frac{h_1}{h_2}\right)b\sqrt{g}(h_1+0.001)^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the channel section [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    h1 > 0.02 m
    h2 > 0.15 m
    h1/h2 ≤ 2.2

    Examples
    --------
    Example as in [3]_, matches. However, example is unlikely in practice.

    >>> Q_weir_rectangular_full_Ackers(h1=0.9, h2=0.6, b=5)
    9.251938159899948

    References
    ----------
    .. [1] Ackers, Peter, W. R. White, J. A. Perkins, and A. J. M. Harrison.
       Weirs and Flumes for Flow Measurement. Chichester ; New York:
       John Wiley & Sons Ltd, 1978.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [3] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    Q = 0.564*(1 + 0.150*h1/h2)*b*g**0.5*(h1 + 0.001)**1.5
    return Q


def Q_weir_rectangular_full_SIA(h1, h2, b):
    r'''Calculates the flow rate across a full-channel rectangular weir from
    the height of the liquid above the crest of the weir, the liquid depth
    beneath it, and the width of the channel. Model from [1]_ as reproduced in
    [2]_.

    Flow rate is given by:

    .. math::
        Q = \frac{2}{3}\sqrt{2}\left(0.615 + \frac{0.000615}{h_1+0.0016}\right)
        b\sqrt{g} h_1 +0.5\left(\frac{h_1}{h_1+h_2}\right)^2b\sqrt{g} h_1^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the channel section [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    0.025 < h < 0.8 m
    b > 0.3 m
    h2 > 0.3 m
    h1/h2 < 1

    Examples
    --------
    Example compares terribly with the Ackers expression - probable error
    in [2]_. DO NOT USE.

    >>> Q_weir_rectangular_full_SIA(h1=0.3, h2=0.4, b=2)
    1.1875825055400384

    References
    ----------
    .. [1] Normen für Wassermessungen: bei Durchführung von Abnahmeversuchen an
       Wasserkraftmaschinen. SIA, 1924.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    Q = 2/3.*2**0.5*(0.615 + 0.000615/(h1 + 0.0016))*b*g**0.5*h1 \
    + 0.5*(h1/(h1+h2))**2*b*g**0.5*h1**1.5
    return Q


def Q_weir_rectangular_full_Rehbock(h1, h2, b):
    r'''Calculates the flow rate across a full-channel rectangular weir from
    the height of the liquid above the crest of the weir, the liquid depth
    beneath it, and the width of the channel. Model from [1]_ as reproduced in
    [2]_.

    Flow rate is given by:

    .. math::
        Q = \frac{2}{3}\sqrt{2}\left(0.602 + 0.0832\frac{h_1}{h_2}\right)
        b\sqrt{g} (h_1 +0.00125)^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the channel section [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    0.03 m < h1 < 0.75 m
    b > 0.3 m
    h2 > 0.3 m
    h1/h2 < 1

    Examples
    --------
    >>> Q_weir_rectangular_full_Rehbock(h1=0.3, h2=0.4, b=2)
    0.6486856330601333

    References
    ----------
    .. [1] King, H. W., Floyd A. Nagler, A. Streiff, R. L. Parshall, W. S.
       Pardoe, R. E. Ballester, Gardner S. Williams, Th Rehbock, Erik G. W.
       Lindquist, and Clemens Herschel. "Discussion of 'Precise Weir
       Measurements.'" Transactions of the American Society of Civil Engineers
       93, no. 1 (January 1929): 1111-78.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    Q = 2/3.*2**0.5*(0.602 + 0.0832*h1/h2)*b*g**0.5*(h1+0.00125)**1.5
    return Q

#print [Q_weir_rectangular_full_Rehbock(h1=0.3, h2=0.4, b=2)]


def Q_weir_rectangular_full_Kindsvater_Carter(h1, h2, b):
    r'''Calculates the flow rate across a full-channel rectangular weir from
    the height of the liquid above the crest of the weir, the liquid depth
    beneath it, and the width of the channel. Model from [1]_ as reproduced in
    [2]_.

    Flow rate is given by:

    .. math::
        Q = \frac{2}{3}\sqrt{2}\left(0.602 + 0.0832\frac{h_1}{h_2}\right)
        b\sqrt{g} (h_1 +0.00125)^{1.5}

    Parameters
    ----------
    h1 : float
        Height of the fluid above the crest of the weir [m]
    h2 : float
        Height of the fluid below the crest of the weir [m]
    b : float
        Width of the channel section [m]

    Returns
    -------
    Q : float
        Volumetric flow rate across the weir [m^3/s]

    Notes
    -----
    The following limits apply to the use of this equation:

    h1 > 0.03 m
    b > 0.15 m
    h2 > 0.1 m
    h1/h2 < 2

    Examples
    --------
    >>> Q_weir_rectangular_full_Kindsvater_Carter(h1=0.3, h2=0.4, b=2)
    0.641560300081563

    References
    ----------
    .. [1] Kindsvater, Carl E., and Rolland W. Carter. "Discharge
       Characteristics of Rectangular Thin-Plate Weirs." Journal of the
       Hydraulics Division 83, no. 6 (December 1957): 1-36.
    .. [2] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    Q = 2/3.*2**0.5*(0.602 + 0.075*h1/h2)*(b - 0.001)*g**0.5*(h1 + 0.001)**1.5
    return Q
#print [Q_weir_rectangular_full_Kindsvater_Carter(h1=0.3, h2=0.4, b=2)]



### Open flow calculations - Manning and Chezy

def V_Manning(Rh, S, n):
    r'''Predicts the average velocity of a flow across an open channel of
    hydraulic radius Rh and slope S, given the Manning roughness coefficient
    n.

    Flow rate is given by:

    .. math::
        V = \frac{1}{n} R_h^{2/3} S^{0.5}

    Parameters
    ----------
    Rh : float
        Hydraulic radius of the channel, Flow Area/Wetted perimeter [m]
    S : float
        Slope of the channel, m/m [-]
    n : float
        Manning roughness coefficient [s/m^(1/3)]

    Returns
    -------
    V : float
        Average velocity of the channel [m/s]

    Notes
    -----
    This is equation is often given in imperial units multiplied by 1.49.

    Examples
    --------
    Example is from [2]_, matches.

    >>> V_Manning(0.2859, 0.005236, 0.03)
    1.0467781958118971

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    '''
    V = Rh**(2/3.)*S**0.5/n
    return V


def n_Manning_to_C_Chezy(n, Rh):
    r'''Converts a Manning roughness coefficient to a Chezy coefficient,
    given the hydraulic radius of the channel.

    .. math::
        C = \frac{1}{n}R_h^{1/6}

    Parameters
    ----------
    n : float
        Manning roughness coefficient [s/m^(1/3)]
    Rh : float
        Hydraulic radius of the channel, Flow Area/Wetted perimeter [m]

    Returns
    -------
    C : float
        Chezy coefficient [m^0.5/s]

    Notes
    -----

    Examples
    --------
    Custom example, checked.

    >>> n_Manning_to_C_Chezy(0.05, Rh=5)
    26.15320972023661

    References
    ----------
    .. [1] Chow, Ven Te. Open-Channel Hydraulics. New York: McGraw-Hill, 1959.
    '''
    C = 1./n*Rh**(1/6.)
    return C


def C_Chezy_to_n_Manning(C, Rh):
    r'''Converts a Chezy coefficient to a Manning roughness coefficient,
    given the hydraulic radius of the channel.

    .. math::
        n = \frac{1}{C}R_h^{1/6}

    Parameters
    ----------
    C : float
        Chezy coefficient [m^0.5/s]
    Rh : float
        Hydraulic radius of the channel, Flow Area/Wetted perimeter [m]

    Returns
    -------
    n : float
        Manning roughness coefficient [s/m^(1/3)]

    Notes
    -----

    Examples
    --------
    Custom example, checked.

    >>> C_Chezy_to_n_Manning(26.15, Rh=5)
    0.05000613713238358

    References
    ----------
    .. [1] Chow, Ven Te. Open-Channel Hydraulics. New York: McGraw-Hill, 1959.
    '''
    n = Rh**(1/6.)/C
    return n


def V_Chezy(Rh, S, C):
    r'''Predicts the average velocity of a flow across an open channel of
    hydraulic radius Rh and slope S, given the Chezy coefficient C.

    Flow rate is given by:

    .. math::
        V = C\sqrt{S R_h}

    Parameters
    ----------
    Rh : float
        Hydraulic radius of the channel, Flow Area/Wetted perimeter [m]
    S : float
        Slope of the channel, m/m [-]
    C : float
        Chezy coefficient [m^0.5/s]

    Returns
    -------
    V : float
        Average velocity of the channel [m/s]

    Notes
    -----

    Examples
    --------
    Custom example, checked.

    >>> V_Chezy(Rh=5, S=0.001, C=26.153)
    1.8492963648371776

    References
    ----------
    .. [1] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [2] Cengel, Yunus, and John Cimbala. Fluid Mechanics: Fundamentals and
       Applications. Boston: McGraw Hill Higher Education, 2006.
    .. [3] Chow, Ven Te. Open-Channel Hydraulics. New York: McGraw-Hill, 1959.
    '''
    V = C*(S*Rh)**0.5
    return V









### Manning coefficients

n_closed_conduit = {
    'Brass': {
        'Smooth': (0.009, 0.01, 0.013),
    },
    'Steel': {
        'Lockbar and welded': (0.01, 0.012, 0.014),
        'Riveted and spiral': (0.013, 0.016, 0.017),
    },
    'Cast Iron': {
        'Coated ': (0.01, 0.013, 0.014),
        'Uncoated': (0.011, 0.014, 0.016),
    },
    'Wrought Iron': {
        'Black ': (0.012, 0.014, 0.015),
        'Galvanized': (0.013, 0.016, 0.017),
    },
    'Corrugated metal': {
        'Subdrain': (0.017, 0.019, 0.021),
        'Storm drain': (0.021, 0.024, 0.03),
    },
    'Acrylic': {
        'Smooth': (0.008, 0.009, 0.01),
    },
    'Glass': {
        'Smooth': (0.009, 0.01, 0.013),
    },
    'Cement': {
        'Neat, surface': (0.01, 0.011, 0.013),
        'Mortar': (0.011, 0.013, 0.015),
    },
    'Concrete': {
        'Culvert, straight and free of debris': (0.01, 0.011, 0.013),
        'Culvert, some bends, connections, and debris': (0.011, 0.013, 0.014),
        'Finished': (0.011, 0.012, 0.014),
        'Sewer with manholes, inlet, straight': (0.013, 0.015, 0.017),
        'Unfinished, steel form': (0.012, 0.013, 0.014),
        'Unfinished, smooth wood form': (0.012, 0.014, 0.016),
        'Unfinished, rough wood form': (0.015, 0.017, 0.02),
    },
    'Wood': {
        'Stave': (0.01, 0.012, 0.014),
        'Laminated, treated': (0.015, 0.017, 0.02),
    },
    'Clay': {
        'Common drainage tile': (0.011, 0.013, 0.017),
        'Vitrified sewer': (0.011, 0.014, 0.017),
        'Vitrified sewer with manholes, inlet, etc.': (0.013, 0.015, 0.017),
        'Vitrified Subdrain with open joint': (0.014, 0.016, 0.018),
    },
    'Brickwork': {
        'Glazed': (0.011, 0.013, 0.015),
        'Lined with cement mortar': (0.012, 0.015, 0.017),
    },
    'Other': {
        'Sanitary sewers coated with sewage slime with bends and connections': (0.012, 0.013, 0.016),
        'Paved invert, sewer, smooth bottom': (0.016, 0.019, 0.02),
        'Rubble masonry, cemented': (0.018, 0.025, 0.03),
    }
}




n_lined_built = {
    'Metal': {
        'Smooth steel, unpainted': (0.011, 0.012, 0.014),
        'Smooth steel, painted': (0.012, 0.013, 0.017),
        'Corrugated': (0.021, 0.025, 0.03),
    },
    'Cement': {
        'Neat, surface': (0.01, 0.011, 0.013),
        'Mortar': (0.011, 0.013, 0.015),
    },
    'Wood': {
        'Planed, untreated': (0.01, 0.012, 0.014),
        'Planed, creosoted': (0.011, 0.012, 0.015),
        'Unplaned': (0.011, 0.013, 0.015),
        'Plank with battens': (0.012, 0.015, 0.018),
        'Lined with Roofing paper': (0.01, 0.014, 0.017),
    },
    'Concrete': {
        'Trowel finish': (0.011, 0.013, 0.015),
        'Float finish': (0.013, 0.015, 0.016),
        'Finished, with gravel on bottom': (0.015, 0.017, 0.02),
        'Unfinished': (0.014, 0.017, 0.02),
        'Gunite, good section': (0.016, 0.019, 0.023),
        'Gunite, wavy section': (0.018, 0.022, 0.025),
        'On good excavated rock': (0.017, 0.02, 0.02),
        'On irregular excavated rock': (0.022, 0.027, 0.027),
    },
    'Concrete bottom float': {
        'Finished with sides of dressed stone in mortar': (0.015, 0.017, 0.02),
        'Finished with sides of random stone in mortar': (0.017, 0.02, 0.024),
        'Finished with sides of cement rubble masonry, plastered': (0.016, 0.02, 0.024),
        'Finished with sides of cement rubble masonry': (0.02, 0.025, 0.03),
        'Finished with sides of dry rubble or riprap': (0.02, 0.03, 0.035),
    },
    'Gravel bottom': {
        'Sides of formed concrete': (0.017, 0.02, 0.025),
        'Sides of random stone in mortar': (0.02, 0.023, 0.026),
        'Sides of dry rubble or riprap': (0.023, 0.033, 0.036),
    },
    'Brick': {
        'Glazed': (0.011, 0.013, 0.015),
        'In-cement mortar': (0.012, 0.015, 0.018),
    },
    'Masonry': {
        'Cemented rubble': (0.017, 0.025, 0.03),
        'Dry rubble': (0.023, 0.032, 0.035),
    },
    'Dressed ashlar': {
        'Stone paving': (0.013, 0.015, 0.017),
    },
    'Asphalt': {
        'Smooth': (0.013, 0.013, 0.013),
        'Rough': (0.016, 0.016, 0.016),
    },
    'Vegatal': {
        'Lined': (0.03, 0.4, 0.5),
    }
}


n_excavated_dredged = {
    'Earth, straight, and uniform': {
        'Clean, recently completed': (0.016, 0.018, 0.02),
        'Clean, after weathering': (0.018, 0.022, 0.025),
        'Gravel, uniform section, clean': (0.022, 0.025, 0.03),
        'With short grass and few weeds': (0.022, 0.027, 0.033),
    },
    'Earth, winding and sluggish': {
        'No vegetation': (0.023, 0.025, 0.03),
        'Grass and some weeds': (0.025, 0.03, 0.033),
        'Dense weeds or aquatic plants, in deep channels': (0.03, 0.035, 0.04),
        'Earth bottom; rubble sides': (0.028, 0.03, 0.035),
        'Stony bottom; weedy banks': (0.025, 0.035, 0.04),
        'Cobble bottom; clean sides': (0.03, 0.04, 0.05),
    },
    'Dragline-excavated or dredged': {
        'No vegetation': (0.025, 0.028, 0.033),
        'Light brush on banks': (0.035, 0.05, 0.06),
    },
    'Rock cuts': {
        'Smooth and Uniform': (0.025, 0.035, 0.04),
        'Jaged and Irregular': (0.035, 0.04, 0.05),
    },
    'Channels not maintained, with weeds and uncut brush': {
        'Dense weeds, as high as the flow depth': (0.05, 0.08, 0.12),
        'Clean bottom, brush on sides': (0.04, 0.05, 0.08),
        'Clean bottom, brush on sides, highest stage of flow': (0.045, 0.07, 0.11),
        'Dense brush, high stage': (0.08, 0.1, 0.14),
    }
}

n_natural = {
    'Major streams': {
        'Irregular, rough': (0.035, 0.07, 0.1),
    },
    'Flood plains': {
        'Pasture, no brush, short grass': (0.025, 0.03, 0.035),
        'Pasture, no brush, high grass': (0.03, 0.035, 0.05),
        'Cultivated areas, no crop': (0.02, 0.03, 0.04),
        'Cultivated areas, mature row crops': (0.025, 0.035, 0.045),
        'Cultivated areas, mature field crops': (0.03, 0.04, 0.05),
        'Brush, scattered brush, heavy weeds': (0.035, 0.05, 0.07),
        'Brush, light brush and trees, in winter': (0.035, 0.05, 0.06),
        'Brush, light brush and trees, in summer': (0.04, 0.06, 0.08),
        'Brush, medium to dense brush, in winter': (0.045, 0.07, 0.11),
        'Brush, medium to dense brush, in summer': (0.07, 0.1, 0.16),
        'Trees, dense willows, summer, straight': (0.11, 0.15, 0.2),
        'Trees, cleared land with tree stumps, no sprouts': (0.03, 0.04, 0.05),
        'Trees, cleared land with tree stumps, heavy growth of sprouts': (0.05, 0.06, 0.08),
        'Trees, heavy stand of timber, a few down trees, little undergrowth, flood stage below branches': (0.08, 0.1, 0.12),
        'Trees, heavy stand of timber, a few down trees, little undergrowth, flood stage reaching branches': (0.1, 0.12, 0.16),
    },
    'Minor streams': {
        'Mountain streams, no vegetation in channel, banks steep, trees and bush on the banks submerged to high stages, with gravel, cobbles and few boulders on bottom': (0.03, 0.04, 0.05),
        'Mountain streams, no vegetation in channel, banks steep, trees and bush on the banks submerged to high stages, with cobbles and large boulders on bottom': (0.04, 0.05, 0.07),
        'Plain streams, clean, straight, full stage, no rifts or deep pools': (0.025, 0.03, 0.033),
        'Plain streams, clean, straight, full stage, no rifts or deep pools, more stones and weeds': (0.03, 0.035, 0.04),
        'Clean, winding, some pools and shoals': (0.033, 0.04, 0.045),
        'Clean, winding, some pools and shoals, some weeds and stones': (0.035, 0.045, 0.05),
        'Clean, winding, some pools and shoals, some weeds and stones, lower stages, less effective slopes and sections': (0.04, 0.048, 0.055),
        'Clean, winding, some pools and shoals, more weeds and stones': (0.045, 0.05, 0.06),
        'Sluggish reaches, weedy, deep pools': (0.05, 0.07, 0.08),
        'Very weedy reaches, deep pools, or floodways with heavy stand of timber and underbrush': (0.075, 0.1, 0.15),
    }
}

n_dicts = [n_natural, n_excavated_dredged, n_lined_built, n_closed_conduit]

#tot = 0
#for thing in n_dicts:
#    for val in thing.values():
#        tot += sum(val.values()[0])

#print tot
#    for i in thing:
#        print i, thing[i].keys()
#        p += len(thing[i].keys())
##print p*3
#import numpy as np
#
##print '2'
#print np.sum(np.array([val.values()[0] for thing in n_dicts for val in thing.values()]))
