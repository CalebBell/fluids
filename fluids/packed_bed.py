# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains correlations for calculating the pressure drop across
a packed bed. Several correlations for void fraction are also included.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Packed Bed Correlation Interfaces
---------------------------------
.. autofunction:: dP_packed_bed
.. autofunction:: dP_packed_bed_methods

Packed Bed Correlations
-----------------------
.. autofunction:: Ergun
.. autofunction:: Kuo_Nydegger
.. autofunction:: Jones_Krier
.. autofunction:: Carman
.. autofunction:: Hicks
.. autofunction:: Brauer
.. autofunction:: KTA
.. autofunction:: Erdim_Akgiray_Demir
.. autofunction:: Fahien_Schriver
.. autofunction:: Tallmadge
.. autofunction:: Idelchik
.. autofunction:: Harrison_Brunner_Hecker
.. autofunction:: Montillet_Akkari_Comiti
.. autofunction:: Guo_Sun

Voidage Correlations
--------------------
.. autofunction:: voidage_Benyahia_Oneil
.. autofunction:: voidage_Benyahia_Oneil_spherical
.. autofunction:: voidage_Benyahia_Oneil_cylindrical

"""

from __future__ import division
from math import exp, pi, sqrt

__all__ = ['dP_packed_bed', 'dP_packed_bed_methods', 'Ergun', 'Kuo_Nydegger', 'Jones_Krier', 'Carman', 'Hicks',
           'Brauer', 'KTA', 'Erdim_Akgiray_Demir', 'Fahien_Schriver',
           'Tallmadge', 'Idelchik',
           'Harrison_Brunner_Hecker', 'Montillet_Akkari_Comiti', 'Guo_Sun',
            'voidage_Benyahia_Oneil',
           'voidage_Benyahia_Oneil_spherical', 'voidage_Benyahia_Oneil_cylindrical']



def Ergun(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_ and [3]_. Eighteenth most
    accurate correlation overall in the review of [2]_.

    Most often presented in the following form:

    .. math::
        \Delta P = \frac{150\mu (1-\epsilon)^2 v_s L}{\epsilon^3 d_p^2}
        + \frac{1.75 (1-\epsilon) \rho v_s^2 L}{\epsilon^3 d_p}

    It is also often presented with a term for sphericity, which is multiplied
    by particle diameter everywhere in the equation. However, this is highly
    empirical and better correlations for beds of differently-shaped particles
    exist. To use sphericity in this model, multiple the input particle
    diameter by the spericity separately.

    In the review of [2]_, it is expressed in terms of a parameter `fp`, shown
    below. This is a convenient means of expressing all forms of pressure drop
    in packed beds correlations in a way that allows for easy comparison.

    .. math::
        f_p = \left(150 + 1.75\left(\frac{Re}{1-\epsilon}\right)\right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    The first term in this equation represents laminar loses, and the second,
    turbulent loses. Developed with data from spheres, sand, and pulverized
    coke. Fluids tested were carbon dioxide, nitrogen, methane, and hydrogen.

    Validity range shown in [3]_ is :math:`1 < Re_{Erg} < 2300`.
    Over predicts pressure drop for :math:`Re_{Erg} > 700`.

    Examples
    --------
    >>> Ergun(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1338.8671874999995

    References
    ----------
    .. [1] Ergun, S. (1952) "Fluid flow through packed columns",
       Chem. Eng. Prog., 48, 89-94.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Jones, D. P., and H. Krier. "Gas Flow Resistance Measurements
       Through Packed Beds at High Reynolds Numbers." Journal of Fluids
       Engineering 105, no. 2 (June 1, 1983): 168-172. doi:10.1115/1.3240959.
    '''
    Re = dp*rho*vs/mu
    fp = (150 + 1.75*(Re/(1-voidage)))*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Kuo_Nydegger(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_ and [3]. Thirty-eighth most
    accurate correlation overall in the review of [2]_.

    .. math::
        f_p = \left(276.23 + 5.05\left(\frac{Re}{1-\epsilon}\right)^{0.87}
        \right)\frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Validity range shown in [2]_ as for a range of
    :math:`460 < Re < 14600`.
    :math:`0.3760 < \epsilon < 0.3901`.
    Developed with data from rough granular ball propellants beds, with air.

    Examples
    --------
    >>> Kuo_Nydegger(dp=8E-1, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    0.025651460973648624

    References
    ----------
    .. [1] Kuo, K. K. and Nydegger, C., "Flow Resistance Measurement and
       Correlation in Packed Beds of WC 870 Ball Propellants," Journal of
       Ballistics , Vol. 2, No. 1, pp. 1-26, 1978.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Jones, D. P., and H. Krier. "Gas Flow Resistance Measurements
       Through Packed Beds at High Reynolds Numbers." Journal of Fluids
       Engineering 105, no. 2 (June 1, 1983): 168-172. doi:10.1115/1.3240959.
    '''
    Re = dp*rho*vs/mu
    fp = (276.23 + 5.05*(Re/(1-voidage))**0.87)*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Tallmadge(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_ and [3].

    .. math::
        f_p = \left(150 + 4.2\left(\frac{Re}{1-\epsilon}\right)^{5/6} \right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    The validity range shown in [2]_ is a range of
    :math:`0.1 < Re < 100000`.

    Examples
    --------
    >>> Tallmadge(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1365.2739144209424

    References
    ----------
    .. [1] Tallmadge, J. A. "Packed Bed Pressure Drop-an Extension to Higher
       Reynolds Numbers." AIChE Journal 16, no. 6 (November 1, 1970): 1092-93.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Montillet, A., E. Akkari, and J. Comiti. "About a Correlating
       Equation for Predicting Pressure Drops through Packed Beds of Spheres
       in a Large Range of Reynolds Numbers." Chemical Engineering and
       Processing: Process Intensification 46, no. 4 (April 2007): 329-33.
       doi:10.1016/j.cep.2006.07.002.
    '''
    Re = dp*rho*vs/mu
    fp = (150.0 + 4.2*(Re/(1-voidage))**(5.0/6.0))*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Jones_Krier(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, also shown in [2]_. Tenth most accurate
    correlation overall in the review of [2]_.

    .. math::
        f_p = \left(150 + 3.89\left(\frac{Re}{1-\epsilon}\right)^{0.87}\right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Validity range shown in [1]_ as for a range of
    :math:`733 < Re < 126,670`.
    :math:`0.3804 < \epsilon < 0.4304`.
    Developed from data of spherical glass beads.

    Examples
    --------
    >>> Jones_Krier(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1362.2719449873746

    References
    ----------
    .. [1] Jones, D. P., and H. Krier. "Gas Flow Resistance Measurements
       Through Packed Beds at High Reynolds Numbers." Journal of Fluids
       Engineering 105, no. 2 (June 1, 1983): 168-172. doi:10.1115/1.3240959.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    '''
    Re = dp*rho*vs/mu
    fp = (150 + 3.89*(Re/(1-voidage))**0.87)*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Carman(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_. Fifth most accurate
    correlation overall in the review of [2]_. Also shown in [3]_.

    .. math::
        f_p = \left(180 + 2.871\left(\frac{Re}{1-\epsilon}\right)^{0.9}\right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Valid in [1]_, [2]_, and [3]_ for a range of
    :math:`300 < Re_{Erg} < 60,000`.

    Examples
    --------
    >>> Carman(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1614.721678121775

    References
    ----------
    .. [1] P.C. Carman, Fluid flow through granular beds, Transactions of the
       London Institute of Chemical Engineers 15 (1937) 150-166.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = dp*rho*vs/mu
    fp = (180 + 2.871*(Re/(1-voidage))**0.9)*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Hicks(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_. Twenty-third most accurate
    correlation overall in the review of [2]_. Also shown in [3]_.

    .. math::
        f_p = 6.8 \frac{(1-\epsilon)^{1.2}}{Re^{0.2}\epsilon^3}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Valid in [1]_, [2]_, and [3]_ for a range of
    :math:`300 < Re_{Erg} < 60,000`.

    Examples
    --------
    >>> Hicks(dp=0.01, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    3.631703956680737

    References
    ----------
    .. [1] Hicks, R. E. "Pressure Drop in Packed Beds of Spheres." Industrial
       Engineering Chemistry Fundamentals 9, no. 3 (August 1, 1970): 500-502.
       doi:10.1021/i160035a032.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = dp*rho*vs/mu
    fp = 6.8*(1-voidage)**1.2/Re**0.2/voidage**3
    return fp*rho*vs**2*L/dp


def Brauer(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_. Seventh most accurate
    correlation overall in the review of [2]_. Also shown in [3]_.

    .. math::
        f_p = \left(160 + 3\left(\frac{Re}{1-\epsilon}\right)^{0.9}\right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    The original article [1]_ has not been reviewed.
    In [2]_, is stated as for a range of :math:`2 < Re_{Erg} < 20,000`.
    In [3]_, is stated as for a range of :math:`0.01 < Re_{Erg} < 40,000`.

    Examples
    --------
    >>> Brauer(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1441.5479196020563

    References
    ----------
    .. [1] H. Brauer, Grundlagen der Einphasen -und Mehrphasenstromungen,
       Sauerlander AG, Aarau, 1971.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    .. [3] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = dp*rho*vs/mu
    fp = (160 + 3.1*(Re/(1-voidage))**0.9)*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def KTA(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_. Third most accurate
    correlation overall in the review of [2]_.

    .. math::
        f_p = \left(160 + 3\left(\frac{Re}{1-\epsilon}\right)^{0.9}\right)
        \frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re= \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Developed for gas flow through pebbles in nuclear reactors.

    In [2]_, stated as for a range of :math:`1 < RE_{Erg} <100,000`.
    In [1]_, a limit on porosity is stated as :math:`0.36 < \epsilon < 0.42`.


    Examples
    --------
    >>> KTA(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1440.409277034248

    References
    ----------
    .. [1] KTA. KTA 3102.3 Reactor Core Design of High-Temperature Gas-Cooled
       Reactors Part 3: Loss of Pressure through Friction in Pebble Bed Cores.
       Germany, 1981.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    '''
    Re = dp*rho*vs/mu
    fp = (160 + 3*(Re/(1-voidage))**0.9)*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Erdim_Akgiray_Demir(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, claiming to be the best model to date.

    .. math::
        f_v = 160 + 2.81Re_{Erg}^{0.904}

    .. math::
        f_v = \frac{\Delta P d_p^2}{\mu v_s L}\frac{\epsilon^3}{(1-\epsilon)^2}

    .. math::
        Re_{Erg} = \frac{\rho v_s  d_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Developed with data in the range of:

    .. math::
        2 < Re_{Erg} <3582\\
        4 < d_t/d_p < 34.1\\
        0.377 < \epsilon <0.470

    Examples
    --------
    >>> Erdim_Akgiray_Demir(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1438.2826958844414

    References
    ----------
    .. [1] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    '''
    Rem = dp*rho*vs/mu/(1-voidage)
    fv = 160 + 2.81*Rem**0.904
    return fv*(mu*vs*L/dp**2)*(1-voidage)**2/voidage**3


def Fahien_Schriver(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, as shown in [2]_. Second most accurate
    correlation overall in the review of [2]_.

    .. math::
        f_p = \left(q\frac{f_{1L}}{Re_{Erg}} + (1-q)\left(f_2 + \frac{f_{1T}}
        {Re_{Erg}}\right)\right)\frac{1-\epsilon}{\epsilon^3}

    .. math::
        q = \exp\left(-\frac{\epsilon^2(1-\epsilon)}{12.6}Re_{Erg}\right)

    .. math::
        f_{1L}=\frac{136}{(1-\epsilon)^{0.38}}

    .. math::
        f_{1T} = \frac{29}{(1-\epsilon)^{1.45}\epsilon^2}

    .. math::
        f_2 = \frac{1.87\epsilon^{0.75}}{(1-\epsilon)^{0.26}}

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re_{Erg} = \frac{\rho v_s  d_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    No range of validity available.

    Examples
    --------
    >>> Fahien_Schriver(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1470.6175541844711

    References
    ----------
    .. [1] R.W. Fahien, C.B. Schriver, Paper presented at the 1961 Denver
       meeting of AIChE, in: R.W. Fahien, Fundamentals of Transport Phenomena,
       McGraw-Hill, New York, 1983.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    '''
    Rem = dp*rho*vs/mu/(1-voidage)
    q = exp(-voidage**2*(1-voidage)/12.6*Rem)
    f1L = 136/(1-voidage)**0.38
    f1T = 29/((1-voidage)**1.45*voidage**2)
    f2 = 1.87*voidage**0.75/(1-voidage)**0.26
    fp = (q*f1L/Rem + (1-q)*(f2 + f1T/Rem))*(1-voidage)/voidage**3
    return fp*rho*vs**2*L/dp


def Idelchik(dp, voidage, vs, rho, mu, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_.

    .. math::
        \frac{\Delta P}{L\rho v_s^2} d_p  = \frac{0.765}{\epsilon^{4.2}}
        \left(\frac{30}{Re_l} + \frac{3}{Re_l^{0.7}} + 0.3\right)

    .. math::
        Re_l = (0.45/\epsilon^{0.5})Re_{Erg}

    .. math::
        Re_{Erg} = \frac{\rho v_s  D_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    :math:`0.001 < Re_{Erg} <1000`
    This equation is valid for void fractions between 0.3 and 0.8. Cited as
    by Bernshtein.

    Examples
    --------
    >>> Idelchik(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1571.909125999067

    References
    ----------
    .. [1] Idelchik, I. E. Flow Resistance: A Design Guide for Engineers.
       Hemisphere Publishing Corporation, New York, 1989.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*dp/mu/(1-voidage)
    Re = (0.45/sqrt(voidage))*Re
    right = 0.765/voidage**4.2*(30./Re + 3./Re**0.7 + 0.3)
    left = dp/L/rho/vs**2
    return right/left


def Harrison_Brunner_Hecker(dp, voidage, vs, rho, mu, L=1, Dt=None):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_, also shown in [2]_. Fourth most accurate
    correlation overall in the review of [2]_.
    Applies a wall correction if diameter of tube is provided.

    .. math::
        f_p = \left(119.8A + 4.63B\left(\frac{Re}{1-\epsilon}\right)^{5/6}
        \right)\frac{(1-\epsilon)^2}{\epsilon^3 Re}

    .. math::
        A = \left(1 + \pi \frac{d_p}{6(1-\epsilon)D_t}\right)^2

    .. math::
        B = 1 - \frac{\pi^2 d_p}{24D_t}\left(1 - \frac{0.5d_p}{D_t}\right)

    .. math::
        f_p = \frac{\Delta P d_p}{\rho v_s^2 L}

    .. math::
        Re = \frac{\rho v_s  d_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]
    Dt : float, optional
        Diameter of the tube, [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Uses data from other sources only. Correlation will underestimate pressure
    drop if tube diameter is not provided. Limits are specified in [1]_ as:

    .. math::
        0.72 < Re < 7700 \\
        8.3 < d_t/d_p < 50 \\
        0.33 < \epsilon < 0.88

    Examples
    --------
    >>> Harrison_Brunner_Hecker(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=1E-2)
    1255.1625662548427

    References
    ----------
    .. [1] KTA. KTA 3102.3 Reactor Core Design of High-Temperature Gas-Cooled
       Reactors Part 3: Loss of Pressure through Friction in Pebble Bed Cores.
       Germany, 1981.
    .. [2] Erdim, Esra, Ömer Akgiray, and İbrahim Demir. "A Revisit of Pressure
       Drop-Flow Rate Correlations for Packed Beds of Spheres." Powder
       Technology 283 (October 2015): 488-504. doi:10.1016/j.powtec.2015.06.017.
    '''
    Re = dp*rho*vs/mu
    if Dt is None:
        A, B = 1.0, 1.0
    else:
        A = (1 + pi*dp/(6*(1-voidage)*Dt))**2
        B = 1 - pi**2*dp/24/Dt*(1 - dp/(2*Dt))
    fp = (119.8*A + 4.63*B*(Re/(1-voidage))**(5/6.))*(1-voidage)**2/(voidage**3*Re)
    return fp*rho*vs**2*L/dp


def Montillet_Akkari_Comiti(dp, voidage, vs, rho, mu, L=1, Dt=None):
    r'''Calculates pressure drop across a packed bed of spheres as in [2]_,
    originally in [1]_. Wall effect adjustment is used if `Dt` is provided.

    .. math::
        \frac{\Delta P}{L\rho V_s^2} D_p \frac{\epsilon^3}{(1-\epsilon)}
        = a\left(\frac{D_c}{D_p}\right)^{0.20}
        \left(\frac{1000}{Re_{p}} + \frac{60}{Re_{p}^{0.5}} + 12 \right)

    .. math::
        Re_{p} = \frac{\rho v_s  D_p}{\mu}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]
    Dt : float, optional
        Diameter of the tube, [m]

    Returns
    -------
    dP : float
        Pressure drop across bed [Pa]

    Notes
    -----
    :math:`10 < REp <2500`
    if Dc/D > 50, set to 2.2.
    a = 0.061 for epsilon < 0.4, 0.050 for > 0.4.

    Examples
    --------
    Custom example:

    >>> Montillet_Akkari_Comiti(dp=0.0008, voidage=0.4, L=0.5, vs=0.00132629120, rho=1000., mu=1.00E-003)
    1148.1905244077548

    References
    ----------
    .. [1] Montillet, A., E. Akkari, and J. Comiti. "About a Correlating
       Equation for Predicting Pressure Drops through Packed Beds of Spheres
       in a Large Range of Reynolds Numbers." Chemical Engineering and
       Processing: Process Intensification 46, no. 4 (April 2007): 329-33.
       doi:10.1016/j.cep.2006.07.002.
    .. [2] Allen, K. G., T. W. von Backstrom, and D. G. Kroger. "Packed Bed
       Pressure Drop Dependence on Particle Shape, Size Distribution, Packing
       Arrangement and Roughness." Powder Technology 246 (September 2013):
       590-600. doi:10.1016/j.powtec.2013.06.022.
    '''
    Re = rho*vs*dp/mu
    if voidage < 0.4:
        a = 0.061
    else:
        a = 0.05
    if Dt is None or Dt/dp > 50:
        Dterm = 2.2
    else:
        Dterm = (Dt/dp)**0.2
    right = a*Dterm*(1000./Re + 60/sqrt(Re) + 12)
    left = dp/L/rho/vs**2*voidage**3/(1-voidage)
    return right/left


def Guo_Sun(dp, voidage, vs, rho, mu, Dt, L=1.0):
    r'''Calculates pressure drop across a packed bed of spheres using a
    correlation developed in [1]_. This is valid for highly-packed particles
    at particle/tube diameter ratios between 2 and 3, where a ring packing
    structure occurs. If a packing ratio is so low, it is important to use this
    model because in some cases its predictions are as low as half those of
    other models!

    .. math::
        f_v = 180 + \left(9.5374\frac{d_p}{D_t} - 2.8054\right)Re_{Erg}^{0.97}

    .. math::
        f_v = \frac{\Delta P d_p^2}{\mu v_s L}\frac{\epsilon^3}{(1-\epsilon)^2}

    .. math::
        Re_{Erg} = \frac{\rho v_s  d_p}{\mu(1-\epsilon)}

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area)[m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    Dt : float
        Diameter of the tube, [m]
    L : float, optional
        Length the fluid flows in the packed bed [m]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Notes
    -----
    Developed with data in the range of:

    .. math::
        100 < Re_{m} <33000\\
        2 < d_t/d_p < 3 1\\
        0.476 < \epsilon <0.492

    Examples
    --------
    >>> Guo_Sun(dp=14.2E-3, voidage=0.492, vs=0.6, rho=1E3, mu=1E-3, Dt=40.9E-3)
    42019.529911473706

    References
    ----------
    .. [1] Guo, Zehua, Zhongning Sun, Nan Zhang, Ming Ding, and Jiaqing Liu.
       "Pressure Drop in Slender Packed Beds with Novel Packing Arrangement."
       Powder Technology 321 (November 2017): 286-92.
       doi:10.1016/j.powtec.2017.08.024.
    '''
    #  2 < D/d < 3, particles in contact with the wall tend to form a highly ordered ring structure.
    Rem = dp*rho*vs/mu/(1-voidage)
    ratio = dp/Dt if Dt is not None else 3.5 # Never ran
    fv = 180 + (9.5374*ratio - 2.8054)*Rem**0.97
    return fv*(mu*vs*L/dp**2)*(1-voidage)**2/voidage**3



# Format: Nice name : (formula, uses_dt)
packed_beds_correlations = {
'Ergun': (Ergun, False),
'Tallmadge': (Tallmadge, False),
'Kuo & Nydegger': (Kuo_Nydegger, False),
'Jones & Krier': (Jones_Krier, False),
'Carman': (Carman, False),
'Hicks': (Hicks, False),
'Brauer': (Brauer, False),
'KTA': (KTA, False),
'Fahien & Schriver': (Fahien_Schriver, False),
'Idelchik': (Idelchik, False),
'Erdim, Akgiray & Demir': (Erdim_Akgiray_Demir, False),

'Harrison, Brunner & Hecker': (Harrison_Brunner_Hecker, True),
'Montillet, Akkari & Comiti': (Montillet_Akkari_Comiti, True),
'Guo, Sun, Zhang, Ding & Liu': (Guo_Sun, True)
}

def dP_packed_bed_methods(dp, voidage, vs, rho, mu, L=1.0, Dt=None, check_ranges=False):
    r'''This function handles determining which pressure drop in a packed bed
    correlation are suitable for the provided inputs.

    Preferred correlations are 'Erdim, Akgiray & Demir' when tube
    diameter is not provided, and 'Harrison, Brunner & Hecker' when tube
    diameter is provided.

    Examples
    --------
    >>> dP_packed_bed_methods(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=1.0)[0]
    'Erdim, Akgiray & Demir'
    >>> dP_packed_bed_methods(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, L=1, Dt=1e-2)[0]
    'Harrison, Brunner & Hecker'

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area) [m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]
    Dt : float, optional
        Diameter of the tube, [m]
    check_ranges : bool, optional
        Added for Future use only

    Returns
    -------
    methods : list
        List of methods which can be used to calculate `dP` with the given inputs
    '''
    methods = []
    if (dp is not None and voidage is not None and vs is not None
        and rho is not None and mu is not None and L is not None):
        if Dt is not None:
            methods = ['Harrison, Brunner & Hecker', 'Montillet, Akkari & Comiti', 'Guo, Sun, Zhang, Ding & Liu']

        methods.extend(['Erdim, Akgiray & Demir', 'Idelchik', 'Fahien & Schriver',
                        'KTA', 'Brauer', 'Hicks', 'Carman', 'Jones & Krier', 'Kuo & Nydegger',
                        'Tallmadge', 'Ergun'])
    return methods


def dP_packed_bed(dp, voidage, vs, rho, mu, L=1, Dt=None, sphericity=None,
                  Method=None):
    r'''This function handles choosing which pressure drop in a packed bed
    correlation is used. Automatically select which correlation
    to use if none is provided. Returns None if insufficient information is
    provided.

    Preferred correlations are 'Erdim, Akgiray & Demir' when tube
    diameter is not provided, and 'Harrison, Brunner & Hecker' when tube
    diameter is provided. If you are using a particles in a narrow tube
    between 2 and 3 particle diameters, expect higher than normal voidages
    (0.4-0.5) and used the method 'Guo, Sun, Zhang, Ding & Liu'.

    Examples
    --------
    >>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3)
    1438.2826958844414
    >>> dP_packed_bed(dp=8E-4, voidage=0.4, vs=1E-3, rho=1E3, mu=1E-3, Dt=0.01)
    1255.1625662548427
    >>> dP_packed_bed(dp=0.05, voidage=0.492, vs=0.1, rho=1E3, mu=1E-3, Dt=0.015, Method='Guo, Sun, Zhang, Ding & Liu')
    18782.499710673364

    Parameters
    ----------
    dp : float
        Particle diameter of spheres [m]
    voidage : float
        Void fraction of bed packing [-]
    vs : float
        Superficial velocity of the fluid (volumetric flow rate/cross-sectional
        area) [m/s]
    rho : float
        Density of the fluid [kg/m^3]
    mu : float
        Viscosity of the fluid, [Pa*s]
    L : float, optional
        Length the fluid flows in the packed bed [m]
    Dt : float, optional
        Diameter of the tube, [m]
    sphericity : float, optional
        Sphericity of the particles [-]

    Returns
    -------
    dP : float
        Pressure drop across the bed [Pa]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        packed_beds_correlations
    '''

    if Method is None:
        Method2 = 'Harrison, Brunner & Hecker' if Dt is not None else 'Erdim, Akgiray & Demir'
    else:
        Method2 = Method

    if dp is not None and sphericity is not None:
        dp = dp*sphericity

    if Method2 == "Ergun":
        return Ergun(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Tallmadge":
        return Tallmadge(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Kuo & Nydegger":
        return Kuo_Nydegger(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Jones & Krier":
        return Jones_Krier(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Carman":
        return Carman(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Hicks":
        return Hicks(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Brauer":
        return Brauer(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "KTA":
        return KTA(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Erdim, Akgiray & Demir":
        return Erdim_Akgiray_Demir(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Fahien & Schriver":
        return Fahien_Schriver(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Idelchik":
        return Idelchik(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L)
    elif Method2 == "Harrison, Brunner & Hecker":
        return Harrison_Brunner_Hecker(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L, Dt=Dt)
    elif Method2 == "Montillet, Akkari & Comiti":
        return Montillet_Akkari_Comiti(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L, Dt=Dt)
    elif Method2 == "Guo, Sun, Zhang, Ding & Liu":
        return Guo_Sun(dp=dp, voidage=voidage, vs=vs, rho=rho, mu=mu, L=L, Dt=Dt)
    else:
        raise ValueError('Unrecognized method')


#import matplotlib.pyplot as plt
#import numpy as np
#
#voidage = 0.4
#rho = 1000.
#mu = 1E-3
#vs = 0.1
#dp = 0.0001
#methods = dP_packed_bed(dp, voidage, vs, rho, mu, L=1, AvailableMethods=True)
#dps = np.logspace(-4, -1, 100)
#
#for method in methods:
#    dPs = [dP_packed_bed(dp, voidage, vs, rho, mu, Method=method) for dp in dps]
#    plt.semilogx(dps, dPs, label=method)
#plt.legend()
#plt.show()



### Voidage correlations

def voidage_Benyahia_Oneil(Dpe, Dt, sphericity):
    r'''Calculates voidage of a bed of arbitrarily shaped uniform particles
    packed into a bed or tube of diameter `Dt`, with equivalent sphere diameter
    `Dp`. Shown in [1]_, and cited by various authors. Correlations exist
    also for spheres, solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.1504 + \frac{0.2024}{\phi} + \frac{1.0814}
        {\left(\frac{d_{t}}{d_{pe}}+0.1226\right)^2}

    Parameters
    ----------
    Dpe : float
        Equivalent spherical particle diameter (diameter of a sphere with the
        same volume), [m]
    Dt : float
        Diameter of the tube, [m]
    sphericity : float
        Sphericity of particles in bed []

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error of 5.2%; valid 1.5 < dtube/dp < 50 and 0.42 < sphericity < 1

    Examples
    --------
    >>> voidage_Benyahia_Oneil(Dpe=1E-3, Dt=1E-2, sphericity=.8)
    0.41395363849210065

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    return 0.1504 + 0.2024/sphericity + 1.0814/(Dt/Dpe + 0.1226)**2


def voidage_Benyahia_Oneil_spherical(Dp, Dt):
    r'''Calculates voidage of a bed of spheres
    packed into a bed or tube of diameter `Dt`, with sphere diameters
    `Dp`. Shown in [1]_, and cited by various authors. Correlations exist
    also for solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.390+\frac{1.740}{\left(\frac{d_{cyl}}{d_p}+1.140\right)^2}

    Parameters
    ----------
    Dp : float
        Spherical particle diameter, [m]
    Dt : float
        Diameter of the tube, [m]

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error 1.5%, 1.5 < ratio < 50.

    Examples
    --------
    >>> voidage_Benyahia_Oneil_spherical(Dp=.001, Dt=.05)
    0.3906653157443224

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    return 0.390 + 1.740/(Dt/Dp + 1.140)**2


def voidage_Benyahia_Oneil_cylindrical(Dpe, Dt, sphericity):
    r'''Calculates voidage of a bed of cylindrical uniform particles
    packed into a bed or tube of diameter `Dt`, with equivalent sphere diameter
    `Dpe`. Shown in [1]_, and cited by various authors. Correlations exist
    also for spheres, solid cylinders, hollow cylinders, and 4-hole cylinders.
    Based on a series of physical measurements.

    .. math::
        \epsilon = 0.373+\frac{1.703}{\left(\frac{d_{cyl}}{d_p}+0.611\right)^2}

    Parameters
    ----------
    Dpe : float
        Equivalent spherical particle diameter (diameter of a sphere with the
        same volume), [m]
    Dt : float
        Diameter of the tube, [m]
    sphericity : float
        Sphericity of particles in bed []

    Returns
    -------
    voidage : float
        Void fraction of bed packing []

    Notes
    -----
    Average error 1.6%; 1.7 < ratio < 26.3.

    Examples
    --------
    >>> voidage_Benyahia_Oneil_cylindrical(Dpe=.01, Dt=.1, sphericity=.6)
    0.38812523109607894

    References
    ----------
    .. [1] Benyahia, F., and K. E. O’Neill. "Enhanced Voidage Correlations for
       Packed Beds of Various Particle Shapes and Sizes." Particulate Science
       and Technology 23, no. 2 (April 1, 2005): 169-77.
       doi:10.1080/02726350590922242.
    '''
    return 0.373 + 1.703/(Dt/Dpe + 0.611)**2
