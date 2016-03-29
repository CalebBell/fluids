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
from math import cos, sin, tan, atan, pi

__all__ = ['contraction_sharp', 'contraction_round',
'contraction_conical', 'contraction_beveled',  'diffuser_sharp',
'diffuser_conical', 'diffuser_conical_staged', 'diffuser_curved',
'diffuser_pipe_reducer',
'entrance_sharp', 'entrance_distance', 'entrance_angled',
'entrance_rounded', 'entrance_beveled', 'exit_normal', 'bend_rounded',
'bend_miter', 'helix', 'spiral','Darby3K', 'Hooper2K', 'Kv_to_Cv', 'Cv_to_Kv',
'Kv_to_K', 'K_to_Kv', 'Darby', 'Hooper']

### Entrances

def entrance_sharp():
    r'''Returns loss coefficient for a sharp entrance to a pipe
    as shown in [1]_.

    .. math::
        K = 0.57

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Other values used have been 0.5.

    Examples
    --------
    >>> entrance_sharp()
    0.57

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = 0.57
    return K


def entrance_distance(d=None, t=None, l=None):
    r'''Returns loss coefficient for a sharp entrance to a pipe at a distance
    from the wall of a reservoir, as shown in [1]_.

    .. math::
        K = 1.12 - 22\frac{t}{d} + 216\left(\frac{t}{d}\right)^2 +
        80\left(\frac{t}{d}\right)^3

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    t : float
        Thickness of pipe wall, [m]
    l : float, optional
        Length of pipe extending from the wall, [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Requires that l/d be >= 0.5.
    Requires that t/d <= 0.05.
    Will raise an exception if these are not the case.

    Examples
    --------
    >>> entrance_distance(d=0.1, t=0.0005)
    1.0154100000000004

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    if l:
        if l/d < 0.5:
            raise Exception('l/d is under 0.5')
    if t/d > 0.05:
        raise Exception('t/d > 0.05')
    K = 1.12 - 22*t/d + 216*(t/d)**2 + 80*(t/d)**3
    return K


def entrance_angled(angle):
    r'''Returns loss coefficient for a sharp, angled entrance to a pipe
    flush with the wall of a reservoir, as shown in [1]_.

    .. math::
        K = 0.57 + 0.30\cos(\theta) + 0.20\cos(\theta)^2

    Parameters
    ----------
    angle : float
        Angle of inclination, [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Not reliable for angles under 20 degrees.
    Loss coefficient is the same for a upward or downward angle.

    Examples
    --------
    >>> entrance_angled(30)
    0.9798076211353316

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    angle = angle/(180/pi)
    K = 0.57 + 0.30*cos(angle) + 0.20*cos(angle)**2
    return K


def entrance_rounded(Di, rc):
    r'''Returns loss coefficient for a rounded entrance to a pipe
    flush with the wall of a reservoir, as shown in [1]_.

    .. math::
        K = 0.0696\left(1 - 0.569\frac{r}{d}\right)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622\left(1 - 0.30\sqrt{\frac{r}{d}}
        - 0.70\frac{r}{d}\right)^4

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    rd : float
        Radius of curvatuce of the entrance, [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Applies for r/D < 1.
    For generously rounded entrances (r/D ~= 1):  K = 0.03

    Examples
    --------
    >>> entrance_rounded(Di=0.1, rc=0.0235)
    0.09839534618360923

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    lbd = 1 + 0.622*(1 - 0.30*(rc/Di)**0.5 - 0.70*(rc/Di))**4
    K = 0.0696*(1 - 0.569*rc/Di)*lbd**2 + (lbd-1)**2
    return K


def entrance_beveled(Di, l, angle):
    r'''Returns loss coefficient for a beveled entrance to a pipe
    flush with the wall of a reservoir, as shown in [1]_.

    .. math::
        K = 0.0696\left(1 - C_b\frac{l}{d}\right)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622\left[1-1.5C_b\left(\frac{l}{d}
        \right)^{\frac{1-(l/d)^{1/4}}{2}}\right]

        C_b = \left(1 - \frac{\theta}{90}\right)\left(\frac{\theta}{90}
        \right)^{\frac{1}{l+l/d}}

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    l : float
        Length of bevel, [m]
    angle : float
        Angle of bevel, [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    A cheap way of getting a lower pressure drop.
    Little credible data is available.

    Examples
    --------
    >>> entrance_beveled(Di=0.1, l=0.003, angle=45)
    0.45086864221916984

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    Cb = (1-angle/90.)*(angle/90.)**(1./(1 +l/Di ))
    lbd = 1 + 0.622*(1 - 1.5*Cb*(l/Di)**((1-(l/Di)**0.25)/2.))
    K = 0.0696*(1-Cb*l/Di)*lbd**2 + (lbd-1)**2
    return K


### Exits

def exit_normal():
    r'''Returns loss coefficient for any exit to a pipe
    as shown in [1]_ and in other sources.

    .. math::
        K = 1

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    It has been found on occasion that K = 2.0 for laminar flow, and ranges
    from about 1.04 to 1.10 for turbulent flow.

    Examples
    --------
    >>> exit_normal()
    1.0

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = 1.0
    return K

### Bends

def bend_rounded(Di=None, rc=None, angle=None, fd=None, bend_diameters=5):
    r'''Returns loss coefficient for any rounded bend in a pipe
    as shown in [1]_.

    .. math::
        K = f\alpha\frac{r}{d} + (0.10 + 2.4f)\sin(\alpha/2)
        + \frac{6.6f(\sqrt{\sin(\alpha/2)}+\sin(\alpha/2))}
        {(r/d)^{\frac{4\alpha}{\pi}}}

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    rc : float
        Radius of curvatuce of the entrance, optional [m]
    angle : float
        Angle of bend, [degrees]
    fd : float
        Darcy friction factor [-]
    bend_diameters : float
        Number of diameters of pipe making up the bend radius [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    When inputting bend diameters, note that manufacturers often specify
    this as a multiplier of nominal diameter, which is different than actual
    diameter. Those require that rc be specified.

    First term represents surface friction loss; the second, secondary flows;
    and the third, flow separation.
    Encompasses the entire range of elbow and pipe bend configurations.

    Examples
    --------
    >>> bend_rounded(Di=4.020, rc=4.0*5, angle=30, fd=0.0163)
    0.10680196344492195

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    angle = angle/(180/pi)
    if not rc:
        rc = Di*bend_diameters
    K = (fd*angle*rc/Di + (0.10+2.4*fd)*sin(angle/2.)
    + 6.6*fd*(sin(angle/2.)**0.5 + sin(angle/2.))/(rc/Di)**(4.*angle/pi))
    return K


def bend_miter(angle):
    r'''Returns loss coefficient for any single-joint miter bend in a pipe
    as shown in [1]_.

    .. math::
        K = 0.42\sin(\alpha/2) + 2.56\sin^3(\alpha/2)

    Parameters
    ----------
    angle : float
        Angle of bend, [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Applies for bends from 0 to 150 degrees. One joint only.

    Examples
    --------
    >>> bend_miter(150)
    2.7128147734758103

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    angle = angle/(180/pi)
    K = 0.42*sin(angle/2) + 2.56*sin(angle/2)**3
    return K


def helix(Di=None, rs=None, pitch=None, N=None, fd=None):
    r'''Returns loss coefficient for any size constant-pitch helix
    as shown in [1]_. Has applications in immersed coils in tanks.

    .. math::
        K = N \left[f\frac{\sqrt{(2\pi r)^2 + p^2}}{d} + 0.20 + 4.8 f\right]

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    rs : float
        Radius of spiral, [m]
    pitch : float
        Distance between two subsequent coil centers, [m]
    N : float
        Number of coils in the helix [-]
    fd : float
        Darcy friction factor [-]


    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Formulation based on peak secondary flow as in two 180 degree bends per
    coil. Flow separation ignored. No f, Re, geometry limitations.
    Source not compared against others.

    Examples
    --------
    >>> helix(Di=0.01, rs=0.1, pitch=.03, N=10, fd=.0185)
    14.525134924495514

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = N*(fd*( (2*pi*rs)**2 +pitch**2 )**0.5/Di + 0.20 + 4.8*fd)
    return K


def spiral(Di=None, rmax=None, rmin=None, pitch=None, fd=None):
    r'''Returns loss coefficient for any size constant-pitch spiral
    as shown in [1]_. Has applications in immersed coils in tanks.

    .. math::
        K = \frac{r_{max} - r_{min}}{p} \left[ f\pi\left(\frac{r_{max}
        +r_{min}}{d}\right) + 0.20 + 4.8f\right]
        + \frac{13.2f}{(r_{min}/d)^2}

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    rmax : float
        Radius of spiral at extremity, [m]
    rmax : float
        Radius of spiral at end near center, [m]
    pitch : float
        Distance between two subsequent coil centers, [m]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Source not compared against others.

    Examples
    --------
    >>> spiral(Di=0.01, rmax=.1, rmin=.02, pitch=.01, fd=0.0185)
    7.950918552775473

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = (rmax-rmin)/pitch*(fd*pi*(rmax+rmin)/Di + 0.20 + 4.8*fd) + 13.2*fd/(rmin/Di)**2
    return K

### Contractions

def contraction_sharp(Di1, Di2):
    r'''Returns loss coefficient for any sharp edged pipe contraction
    as shown in [1]_.

    .. math::
        K = 0.0696(1-\beta^5)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622(1-0.215\beta^2 -  0.785\beta^5)

        \beta = d_2/d_1

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    A value of 0.506 or simply 0.5 is often used.

    Examples
    --------
    >>> contraction_sharp(Di1=1, Di2=0.4)
    0.5301269161591805

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di2/Di1
    lbd = 1 + 0.622*(1-0.215*beta**2 - 0.785*beta**5)
    K = 0.0696*(1-beta**5)*lbd**2 + (lbd-1)**2
    return K


def contraction_round(Di1, Di2, rc):
    r'''Returns loss coefficient for any round edged pipe contraction
    as shown in [1]_.

    .. math::
        K = 0.0696\left(1 - 0.569\frac{r}{d_2}\right)\left(1-\sqrt{\frac{r}
        {d_2}}\beta\right)(1-\beta^5)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622\left(1 - 0.30\sqrt{\frac{r}{d_2}}
        - 0.70\frac{r}{d_2}\right)^4 (1-0.215\beta^2-0.785\beta^5)

        \beta = d_2/d_1

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]
    rc : float
        Radius of curvatuce of the contraction, [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Rounding radius larger than 0.14Di2 prevents flow separation from the wall.
    Further increase in rounding radius continues to reduce loss coefficient.

    Examples
    --------
    >>> contraction_round(Di1=1, Di2=0.4, rc=0.04)
    0.1783332490866574

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di2/Di1
    lbd = 1 + 0.622*(1 - 0.30*(rc/Di2)**0.5 - 0.70*rc/Di2)**4*(1-0.215*beta**2 - 0.785*beta**5)
    K = 0.0696*(1-0.569*rc/Di2)*(1-(rc/Di2)**0.5*beta)*(1-beta**5)*lbd**2 + (lbd-1)**2
    return K


def contraction_conical(Di1, Di2, l=None, angle=None, fd=None):
    r'''Returns loss coefficient for any conical pipe contraction
    as shown in [1]_.

    .. math::
        K = 0.0696[1+C_B(\sin(\alpha/2)-1)](1-\beta^5)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622(\alpha/180)^{0.8}(1-0.215\beta^2-0.785\beta^5)

        \beta = d_2/d_1

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]
    l : float
        Length of the contraction, optional [m]
    angle : float
        Angle of contraction, optional [degrees]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Cheap and has substantial impact on pressure drop.

    Examples
    --------
    >>> contraction_conical(Di1=0.1, Di2=0.04, l=0.04, fd=0.0185)
    0.15779041548350314

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di2/Di1
    if angle:
        angle = angle/(180/pi)
        l = (Di1 - Di2)/(2*tan(angle/2))
    elif l:
        angle = 2*atan((Di1-Di2)/2/l)
    else:
        raise Exception('Either l or angle is required')

    lbd = 1 + 0.622*(angle/pi)**0.8*(1-0.215*beta**2 - 0.785*beta**5)
    K = fd*(1-beta**4)/(8*sin(angle/2)) + 0.0696*sin(angle/2)*(1-beta**5)*lbd**2 + (lbd-1)**2
    return K


def contraction_beveled(Di1, Di2, l=None, angle=None):
    r'''Returns loss coefficient for any sharp beveled pipe contraction
    as shown in [1]_.

    .. math::
        K = 0.0696[1+C_B(\sin(\alpha/2)-1)](1-\beta^5)\lambda^2 + (\lambda-1)^2

        \lambda = 1 + 0.622\left[1+C_B\left(\left(\frac{\alpha}{180}
        \right)^{0.8}-1\right)\right](1-0.215\beta^2-0.785\beta^5)

        C_B = \frac{l}{d_2}\frac{2\beta\tan(\alpha/2)}{1-\beta}

        \beta = d_2/d_1

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]
    l : float
        Length of the bevel along the pipe axis ,[m]
    angle : float
        Angle of bevel, [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----

    Examples
    --------
    >>> contraction_beveled(Di1=0.5, Di2=0.1, l=.7*.1, angle=120)
    0.40946469413070485

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    angle = angle/(180/pi)
    beta = Di2/Di1
    CB = l/Di2*2*beta*tan(angle/2)/(1-beta)
    lbd  = 1 + 0.622*(1 + CB*((angle/pi)**0.8-1))*(1-0.215*beta**2-0.785*beta**5)
    K = 0.0696*(1 + CB*(sin(angle/2)-1))*(1-beta**5)*lbd**2 + (lbd-1)**2
    return K

### Expansions (diffusers)

def diffuser_sharp(Di1, Di2):
    r'''Returns loss coefficient for any sudded pipe diameter expansion
    as shown in [1]_ and in other sources.

    .. math::
        K_1 = (1-\beta^2)^2

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Highly accurate.

    Examples
    --------
    >>> diffuser_sharp(Di1=.5, Di2=1)
    0.5625

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di1/Di2
    K = (1-beta**2)**2
    return K


def diffuser_conical(Di1, Di2, l=None, angle=None, fd=None):
    r'''Returns loss coefficient for any conical pipe expansion
    as shown in [1]_. Five different formulas are used, depending on
    the angle and the ratio of diameters.

    For 0 to 20 degrees, all aspect ratios:

    .. math::
        K_1 = 8.30[\tan(\alpha/2)]^{1.75}(1-\beta^2)^2 + \frac{f(1-\beta^4)}{8\sin(\alpha/2)}

    For 20 to 60 degrees, beta < 0.5:

    .. math::
        K_1 = \left\{1.366\sin\left[\frac{2\pi(\alpha-15^\circ)}{180}\right]^{0.5}
        - 0.170 - 3.28(0.0625-\beta^4)\sqrt{\frac{\alpha-20^\circ}{40^\circ}}\right\}
        (1-\beta^2)^2 + \frac{f(1-\beta^4)}{8\sin(\alpha/2)}

    For 20 to 60 degrees, beta >= 0.5:

    .. math::
        K_1 = \left\{1.366\sin\left[\frac{2\pi(\alpha-15^\circ)}{180}\right]^{0.5}
        - 0.170 \right\}(1-\beta^2)^2 + \frac{f(1-\beta^4)}{8\sin(\alpha/2)}

    For 60 to 180 degrees, beta < 0.5:

    .. math::
        K_1 = \left[1.205 - 3.28(0.0625-\beta^4)-12.8\beta^6\sqrt{\frac
        {\alpha-60^\circ}{120^\circ}}\right](1-\beta^2)^2

    For 60 to 180 degrees, beta >= 0.5:

    .. math::
        K_1 = \left[1.205 - 0.20\sqrt{\frac{\alpha-60^\circ}{120^\circ}}
        \right](1-\beta^2)^2

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    l : float
        Length of the contraction along the pipe axis, optional[m]
    angle : float
        Angle of contraction, [degrees]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    For angles above 60 degrees, friction factor is not used.

    Examples
    --------
    >>> diffuser_conical(Di1=.1**0.5, Di2=1, angle=10., fd=0.020)
    0.12301652230915454
    >>> diffuser_conical(Di1=1/3., Di2=1, angle=50, fd=0.03) # 2
    0.8081340270019336
    >>> diffuser_conical(Di1=2/3., Di2=1, angle=40, fd=0.03) # 3
    0.32533470783539786
    >>> diffuser_conical(Di1=1/3., Di2=1, angle=120, fd=0.0185) # #4
    0.812308728765127
    >>> diffuser_conical(Di1=2/3., Di2=1, angle=120, fd=0.0185) # Last
    0.3282650135070033

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di1/Di2

    if angle:
        angle_rad = angle/(180/pi)
        l = (Di2 - Di1)/(2*tan(angle_rad/2))
    elif l:
        angle_rad = 2*atan((Di2-Di1)/2/l)
        angle = angle_rad*(180/pi)

    if 0 < angle <= 20:
        K = 8.30*tan(angle_rad/2)**1.75*(1-beta**2)**2 + fd*(1-beta**4)/8./sin(angle_rad/2)
    elif 20 < angle <= 60 and 0 <= beta < 0.5:
        K = (1.366*sin(2*pi*(angle-15)/180.)**0.5-0.170
        - 3.28*(0.0625-beta**4)*((angle-20)/40.)**0.5)*(1-beta**2)**2 + fd*(1-beta**4)/8./sin(angle_rad/2)
    elif 20 < angle <= 60 and beta >= 0.5:
        K = (1.366*sin(2*pi*(angle-15)/180.)**0.5-0.170)*(1-beta**2)**2 + fd*(1-beta**4)/8./sin(angle_rad/2)
    elif 60 < angle <= 180 and 0 <= beta < 0.5:
        K = (1.205 - 3.28*(0.0625-beta**4) - 12.8*beta**6*((angle-60)/120.)**0.5)*(1-beta**2)**2
    elif 60 < angle <= 180 and beta >= 0.5:
        K = (1.205 - 0.20*((angle-60)/120.)**0.5)*(1-beta**2)**2
    else:
        raise Exception('Conical diffuser inputs incorrect')
    return K


def diffuser_conical_staged(Di1, Di2, DEs, ls, fd=None):
    r'''Returns loss coefficient for any series of staged conical pipe expansions
    as shown in [1]_. Five different formulas are used, depending on
    the angle and the ratio of diameters. This function calls diffuser_conical.

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    DEs : array
        Diameters of intermediate sections, [m]
    ls : array
        Lengths of the various sections, [m]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Only lengths of sections currently allowed. This could be changed
    to understand angles also.

    Formula doesn't make much sense, as observed by the example comparing
    a series of conical sections. Use only for small numbers of segments of
    highly differing angles.

    Examples
    --------
    >>> diffuser_conical(Di1=1., Di2=10.,l=9, fd=0.01)
    0.973137914861591

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = 0
    DEs.insert(0, Di1)
    DEs.append(Di2)
    for i in range(len(ls)):
        K += diffuser_conical(Di1=float(DEs[i]), Di2=float(DEs[i+1]), l=float(ls[i]), fd=fd)
    return K


def diffuser_curved(Di1, Di2, l):
    r'''Returns loss coefficient for any curved wall pipe expansion
    as shown in [1]_.

    .. math::
        K_1 = \phi(1.43-1.3\beta^2)(1-\beta^2)^2

        \phi = 1.01 - 0.624\frac{l}{d_1} + 0.30\left(\frac{l}{d_1}\right)^2
        - 0.074\left(\frac{l}{d_1}\right)^3 + 0.0070\left(\frac{l}{d_1}\right)^4

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    l : float
        Length of the curve along the pipe axis, [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Beta^2 should be between 0.1 and 0.9.
    A small mismatch between tabulated values of this function in table 11.3
    is observed with the equation presented.

    Examples
    --------
    >>> diffuser_curved(Di1=.25**0.5, Di2=1., l=2.)
    0.2299781250000002

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    beta = Di1/Di2
    phi = 1.01 - 0.624*l/Di1 + 0.30*(l/Di1)**2 - 0.074*(l/Di1)**3 + 0.0070*(l/Di1)**4
    K = phi*(1.43 - 1.3*beta**2)*(1 - beta**2)**2
    return K


def diffuser_pipe_reducer(Di1, Di2, l, fd1, fd2=None):
    r'''Returns loss coefficient for any pipe reducer pipe expansion
    as shown in [1]. This is an approximate formula.

    .. math::
        K_f = f_1\frac{0.20l}{d_1} + \frac{f_1(1-\beta)}{8\sin(\alpha/2)}
        + f_2\frac{0.20l}{d_2}\beta^4

        \alpha = 2\tan^{-1}\left(\frac{d_1-d_2}{1.20l}\right)

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    l : float
        Length of the pipe reducer along the pipe axis, [m]
    fd1 : float
        Darcy friction factor at inlet diameter [-]
    fd2 : float
        Darcy friction factor at outlet diameter, optional [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Industry lack of standardization prevents better formulas from being
    developed. Add 15% if the reducer is eccentric.
    Friction factor at outlet will be assumed the same as at inlet if not specified.

    Doubt about the validity of this equation is raised.

    Examples
    --------
    >>> diffuser_pipe_reducer(Di1=.5, Di2=.75, l=1.5, fd1=0.07)
    0.06873244301714816

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    if not fd2:
        fd2 = fd1
    beta = Di1/Di2
    angle = -2*atan((Di1-Di2)/1.20/l)
    K = fd1*0.20*l/Di1 + fd1*(1-beta)/8./sin(angle/2) + fd2*0.20*l/Di2*beta**4
    return K

### TODO: Tees

###  3 Darby 3K Method (with valves)
Darby = {}
Darby['Elbow, 90°, threaded, standard, (r/D = 1)'] = {'K1': 800, 'Ki': 0.14, 'Kd': 4}
Darby['Elbow, 90°, threaded, long radius, (r/D = 1.5)'] = {'K1': 800, 'Ki': 0.071, 'Kd': 4.2}
Darby['Elbow, 90°, flanged, welded, bends, (r/D = 1)'] = {'K1': 800, 'Ki': 0.091, 'Kd': 4}
Darby['Elbow, 90°, (r/D = 2)'] = {'K1': 800, 'Ki': 0.056, 'Kd': 3.9}
Darby['Elbow, 90°, (r/D = 4)'] = {'K1': 800, 'Ki': 0.066, 'Kd': 3.9}
Darby['Elbow, 90°, (r/D = 6)'] = {'K1': 800, 'Ki': 0.075, 'Kd': 4.2}
Darby['Elbow, 90°, mitered, 1 weld, (90°)'] = {'K1': 1000, 'Ki': 0.27, 'Kd': 4}
Darby['Elbow, 90°, 2 welds, (45°)'] = {'K1': 800, 'Ki': 0.068, 'Kd': 4.1}
Darby['Elbow, 90°, 3 welds, (30°)'] = {'K1': 800, 'Ki': 0.035, 'Kd': 4.2}
Darby['Elbow, 45°, threaded standard, (r/D = 1)'] = {'K1': 500, 'Ki': 0.071, 'Kd': 4.2}
Darby['Elbow, 45°, long radius, (r/D = 1.5)'] = {'K1': 500, 'Ki': 0.052, 'Kd': 4}
Darby['Elbow, 45°, mitered, 1 weld, (45°)'] = {'K1': 500, 'Ki': 0.086, 'Kd': 4}
Darby['Elbow, 45°, mitered, 2 welds, (22.5°)'] = {'K1': 500, 'Ki': 0.052, 'Kd': 4}
Darby['Elbow, 180°, threaded, close-return bend, (r/D = 1)'] = {'K1': 1000, 'Ki': 0.23, 'Kd': 4}
Darby['Elbow, 180°, flanged, (r/D = 1)'] = {'K1': 1000, 'Ki': 0.12, 'Kd': 4}
Darby['Elbow, 180°, all, (r/D = 1.5)'] = {'K1': 1000, 'Ki': 0.1, 'Kd': 4}
Darby['Tee, Through-branch, (as elbow), threaded, (r/D = 1)'] = {'K1': 500, 'Ki': 0.274, 'Kd': 4}
Darby['Tee, Through-branch,(as elbow), (r/D = 1.5)'] = {'K1': 800, 'Ki': 0.14, 'Kd': 4}
Darby['Tee, Through-branch, (as elbow), flanged, (r/D = 1)'] = {'K1': 800, 'Ki': 0.28, 'Kd': 4}
Darby['Tee, Through-branch, (as elbow), stub-in branch'] = {'K1': 1000, 'Ki': 0.34, 'Kd': 4}
Darby['Tee, Run-through, threaded, (r/D = 1)'] = {'K1': 200, 'Ki': 0.091, 'Kd': 4}
Darby['Tee, Run-through, flanged, (r/D = 1)'] = {'K1': 150, 'Ki': 0.05, 'Kd': 4}
Darby['Tee, Run-through, stub-in branch'] = {'K1': 100, 'Ki': 0, 'Kd': 0}
Darby['Valve, Angle valve, 45°, full line size, β = 1'] = {'K1': 950, 'Ki': 0.25, 'Kd': 4}
Darby['Valve, Angle valve, 90°, full line size, β = 1'] = {'K1': 1000, 'Ki': 0.69, 'Kd': 4}
Darby['Valve, Globe valve, standard, β = 1'] = {'K1': 1500, 'Ki': 1.7, 'Kd': 3.6}
Darby['Valve, Plug valve, branch flow'] = {'K1': 500, 'Ki': 0.41, 'Kd': 4}
Darby['Valve, Plug valve, straight through'] = {'K1': 300, 'Ki': 0.084, 'Kd': 3.9}
Darby['Valve, Plug valve, three-way (flow through)'] = {'K1': 300, 'Ki': 0.14, 'Kd': 4}
Darby['Valve, Gate valve, standard, β = 1'] = {'K1': 300, 'Ki': 0.037, 'Kd': 3.9}
Darby['Valve, Ball valve, standard, β = 1'] = {'K1': 300, 'Ki': 0.017, 'Kd': 3.5}
Darby['Valve, Diaphragm, dam type'] = {'K1': 1000, 'Ki': 0.69, 'Kd': 4.9}
Darby['Valve, Swing check'] = {'K1': 1500, 'Ki': 0.46, 'Kd': 4}
Darby['Valve, Lift check'] = {'K1': 2000, 'Ki': 2.85, 'Kd': 3.8}


def Darby3K(NPS=None, Re=None, name=None, K1=None, Ki=None, Kd=None):
    r'''Returns loss coefficient for any various fittings, depending
    on the name input. Alternatively, the Darby constants K1, Ki and Kd
    may be provided and used instead. Source of data is [1]_.
    Reviews of this model are favorable.

    .. math::
        K_f = \frac{K_1}{Re} + K_i\left(1 + \frac{K_d}{D_{\text{NPS}}^{0.3}}\right)

    Parameters
    ----------
    NPS : float
        Nominal diameter of the pipe, [in]
    Re : float
        Reynolds number, [-]
    name : str
        String from Darby dict representing a fitting
    K1 : float
        K1 parameter of Darby model, optional [-]
    Ki : float
        Ki parameter of Darby model, optional [-]
    Kd : float
        Kd parameter of Darby model, optional [in]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Also described in Albright's Handbook and Ludwig's Applied Process Design.
    Relatively uncommon to see it used.

    The possibility of combining these methods with those above are attractive.

    Examples
    --------
    >>> Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')
    1.1572523963562353
    >>> Darby3K(NPS=12., Re=10000., K1=950,  Ki=0.25,  Kd=4)
    0.819510280626355

    References
    ----------
    .. [1] Silverberg, Peter, and Ron Darby. "Correlate Pressure Drops through
       Fittings: Three Constants Accurately Calculate Flow through Elbows,
       Valves and Tees." Chemical Engineering 106, no. 7 (July 1999): 101.
    .. [2] Silverberg, Peter. "Correlate Pressure Drops Through Fittings."
       Chemical Engineering 108, no. 4 (April 2001): 127,129-130.
    '''
    if name:
        if name in Darby:
            d = Darby[name]
            K1, Ki, Kd = d['K1'], d['Ki'], d['Kd']
        else:
            raise Exception('Name of fitting not in list')
    elif K1 and Ki and Kd:
        pass
    else:
        raise Exception('Name of fitting or constants are required')
    K = K1/Re + Ki*(1. + Kd/NPS**0.3)
    return K


### 2K Hooper Method

Hooper = {}
Hooper['Elbow, 90°, Standard (R/D = 1), Screwed'] = {'K1': 800, 'Kinfty': 0.4}
Hooper['Elbow, 90°, Standard (R/D = 1), Flanged/welded'] = {'K1': 800, 'Kinfty': 0.25}
Hooper['Elbow, 90°, Long-radius (R/D = 1.5), All types'] = {'K1': 800, 'Kinfty': 0.2}
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 1 weld (90° angle)'] = {'K1': 1000, 'Kinfty': 1.15}
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 2 weld (45° angle)'] = {'K1': 800, 'Kinfty': 0.35}
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 3 weld (30° angle)'] = {'K1': 800, 'Kinfty': 0.3}
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 4 weld (22.5° angle)'] = {'K1': 800, 'Kinfty': 0.27}
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 5 weld (18° angle)'] = {'K1': 800, 'Kinfty': 0.25}
Hooper['Elbow, 45°, Standard (R/D = 1), All types'] = {'K1': 500, 'Kinfty': 0.2}
Hooper['Elbow, 45°, Long-radius (R/D 1.5), All types'] = {'K1': 500, 'Kinfty': 0.15}
Hooper['Elbow, 45°, Mitered (R/D=1.5), 1 weld (45° angle)'] = {'K1': 500, 'Kinfty': 0.25}
Hooper['Elbow, 45°, Mitered (R/D=1.5), 2 weld (22.5° angle)'] = {'K1': 500, 'Kinfty': 0.15}
Hooper['Elbow, 45°, Standard (R/D = 1), Screwed'] = {'K1': 1000, 'Kinfty': 0.7}
Hooper['Elbow, 180°, Standard (R/D = 1), Flanged/welded'] = {'K1': 1000, 'Kinfty': 0.35}
Hooper['Elbow, 180°, Long-radius (R/D = 1.5), All types'] = {'K1': 1000, 'Kinfty': 0.3}
Hooper['Elbow, Used as, Standard, Screwed'] = {'K1': 500, 'Kinfty': 0.7}
Hooper['Elbow, Elbow, Long-radius, Screwed'] = {'K1': 800, 'Kinfty': 0.4}
Hooper['Elbow, Elbow, Standard, Flanged/welded'] = {'K1': 800, 'Kinfty': 0.8}
Hooper['Elbow, Elbow, Stub-in type branch'] = {'K1': 1000, 'Kinfty': 1}
Hooper['Tee, Run, Screwed'] = {'K1': 200, 'Kinfty': 0.1}
Hooper['Tee, Through, Flanged or welded'] = {'K1': 150, 'Kinfty': 0.05}
Hooper['Tee, Tee, Stub-in type branch'] = {'K1': 100, 'Kinfty': 0}
Hooper['Valve, Gate, Full line size, Beta = 1'] = {'K1': 300, 'Kinfty': 0.1}
Hooper['Valve, Ball, Reduced trim, Beta = 0.9'] = {'K1': 500, 'Kinfty': 0.15}
Hooper['Valve, Plug, Reduced trim, Beta = 0.8'] = {'K1': 1000, 'Kinfty': 0.25}
Hooper['Valve, Globe, Standard'] = {'K1': 1500, 'Kinfty': 4}
Hooper['Valve, Globe, Angle or Y-type'] = {'K1': 1000, 'Kinfty': 2}
Hooper['Valve, Diaphragm, Dam type'] = {'K1': 1000, 'Kinfty': 2}
Hooper['Valve, Butterfly,'] = {'K1': 800, 'Kinfty': 0.25}
Hooper['Valve, Check, Lift'] = {'K1': 2000, 'Kinfty': 10}
Hooper['Valve, Check, Swing'] = {'K1': 1500, 'Kinfty': 1.5}
Hooper['Valve, Check, Tilting-disc'] = {'K1': 1000, 'Kinfty': 0.5}


def Hooper2K(Di=None, Re=None, name=None, K1=None, Kinfty=None):
    r'''Returns loss coefficient for any various fittings, depending
    on the name input. Alternatively, the Hooper constants K1, Kinfty
    may be provided and used instead. Source of data is [1]_.
    Reviews of this model are favorable less favorable than the Darby method
    but superior to the constant-K method.

    .. math::
        K = \frac{K_1}{Re} + K_\infty\left(1 + \frac{1}{ID_{in}}\right)

    Parameters
    ----------
    Di : float
        Actual inside diameter of the pipe, [in]
    Re : float
        Reynolds number, [-]
    name : str
        String from Hooper dict representing a fitting
    K1 : float
        K1 parameter of Hooper model, optional [-]
    Kinfty : float
        Kinfty parameter of Hooper model, optional [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Also described in Ludwig's Applied Process Design.
    Relatively uncommon to see it used.
    No actual example found.

    Examples
    --------
    >>> Hooper2K(Di=2., Re=10000., name='Valve, Globe, Standard')
    6.15
    >>> Hooper2K(Di=2., Re=10000., K1=900, Kinfty=4)
    6.09

    References
    ----------
    .. [1] Hooper, W. B., "The 2-K Method Predicts Head Losses in Pipe
       Fittings," Chem. Eng., p. 97, Aug. 24 (1981).
    .. [2] Hooper, William B. "Calculate Head Loss Caused by Change in Pipe
       Size." Chemical Engineering 95, no. 16 (November 7, 1988): 89.
    .. [3] Kayode Coker. Ludwig's Applied Process Design for Chemical and
       Petrochemical Plants. 4E. Amsterdam ; Boston: Gulf Professional
       Publishing, 2007.
    '''
    if name:
        if name in Hooper:
            d = Hooper[name]
            K1, Kinfty = d['K1'], d['Kinfty']
        else:
            raise Exception('Name of fitting not in list')
    elif K1 and Kinfty:
        pass
    else:
        raise Exception('Name of fitting or constants are required')
    K = K1/Re + Kinfty*(1. + 1./Di)
    return K


### Valves



def Kv_to_Cv(Kv):
    r'''Convert valve flow coefficient from imperial to common metric units.

    .. math::
        C_v = 1.1560992283540599 K_v

    Parameters
    ----------
    Kv : float
        Valve flow coefficient, [1 m^3 cold water/hour at dP = 1 bar]

    Returns
    -------
    Cv : float
        Valve flow coefficient, [1 gpm water at 1.0 psi dP]

    Notes
    -----
    Kv = 0.865 Cv is in the IEC standard 60534-2-1.
    It has also been said that Cv = 1.17Kv; this is wrong by current standards.

    Examples
    --------
    >>> Kv_to_Cv(2)
    2.3121984567081197

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    Cv = 1.1560992283540599*Kv
    return Cv


def Cv_to_Kv(Cv):
    r'''Convert valve flow coefficient from imperial to common metric units.

    .. math::
        K_v = C_v/1.156

    Parameters
    ----------
    Cv : float
        Valve flow coefficient, [1 gpm water at 1.0 psi dP]

    Returns
    -------
    Kv : float
        Valve flow coefficient, [1 m^3 cold water/hour at dP = 1 bar]

    Notes
    -----
    Kv = 0.865 Cv is in the IEC standard 60534-2-1.
    It has also been said that Cv = 1.17Kv; this is wrong by current standards.

    Examples
    --------
    >>> Cv_to_Kv(2.312)
    1.9998283393819036

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    Kv = Cv/1.1560992283540599
    return Kv


def Kv_to_K(Kv, D):
    r'''Convert valve flow coefficient from common metric units to regular
    loss coefficients.

    .. math::
        K = 0.001604 \frac{D^4}{K_v^2}

    Parameters
    ----------
    Kv : float
        Valve flow coefficient, [1 m^3 cold water/hour at dP = 1 bar]

    Returns
    -------
    K : float
        Loss coefficient, [-]

    Notes
    -----


    Examples
    --------
    >>> Kv_to_K(2.312, .015)
    15.1912580369009

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    K = 0.001604E12*D**4/Kv**2
    return K


def K_to_Kv(K, D):
    r'''Convert regular loss coefficient to valve flow coefficient.

    .. math::
        K_v = \sqrt{0.001604 \frac{D^4}{K}}

    Parameters
    ----------
    K : float
        Loss coefficient, [-]

    Returns
    -------
    Kv : float
        Valve flow coefficient, [1 m^3 cold water/hour at dP = 1 bar]

    Notes
    -----


    Examples
    --------
    >>> K_to_Kv(15.1912580369009, .015)
    2.312

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    Kv = (0.001604E12*D**4/K)**0.5
    return Kv

