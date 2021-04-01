# -*- coding: utf-8 -*-
r"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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

This module contains correlations for the loss coefficient of various types
of pipe fittings. Whether you are desining a network or modeling a single
element, the correlations here cover most cases.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Entrances
---------
.. autofunction:: entrance_sharp
.. autofunction:: entrance_distance
.. autofunction:: entrance_angled
.. autofunction:: entrance_rounded
.. autofunction:: entrance_beveled
.. autofunction:: entrance_beveled_orifice
.. autofunction:: entrance_distance_45_Miller

Exits
-----
.. autofunction:: exit_normal

Bends
-----
.. autofunction:: bend_rounded
.. autofunction:: bend_rounded_Miller
.. autofunction:: bend_rounded_Crane
.. autofunction:: bend_miter
.. autofunction:: bend_miter_Miller
.. autofunction:: helix
.. autofunction:: spiral

Contractions
------------
.. autofunction:: contraction_sharp
.. autofunction:: contraction_round
.. autofunction:: contraction_round_Miller
.. autofunction:: contraction_conical
.. autofunction:: contraction_conical_Crane
.. autofunction:: contraction_beveled

Expansions/Diffusers
--------------------
.. autofunction:: diffuser_sharp
.. autofunction:: diffuser_conical
.. autofunction:: diffuser_conical_staged
.. autofunction:: diffuser_curved
.. autofunction:: diffuser_pipe_reducer

Tees
----
.. autofunction:: K_branch_converging_Crane
.. autofunction:: K_run_converging_Crane
.. autofunction:: K_branch_diverging_Crane
.. autofunction:: K_run_diverging_Crane

Valves
------
.. autofunction:: K_gate_valve_Crane
.. autofunction:: K_angle_valve_Crane
.. autofunction:: K_globe_valve_Crane
.. autofunction:: K_swing_check_valve_Crane
.. autofunction:: K_lift_check_valve_Crane
.. autofunction:: K_tilting_disk_check_valve_Crane
.. autofunction:: K_globe_stop_check_valve_Crane
.. autofunction:: K_angle_stop_check_valve_Crane
.. autofunction:: K_ball_valve_Crane
.. autofunction:: K_diaphragm_valve_Crane
.. autofunction:: K_foot_valve_Crane
.. autofunction:: K_butterfly_valve_Crane
.. autofunction:: K_plug_valve_Crane

Hooper 2K fittings
------------------
.. autofunction:: Hooper2K
.. autodata:: Hooper

Darby 3K fittings
------------------
.. autofunction:: Darby3K
.. autodata:: Darby

Loss Coefficient Converters
---------------------------
.. autofunction:: Cv_to_K
.. autofunction:: Kv_to_K
.. autofunction:: K_to_Cv
.. autofunction:: K_to_Kv
.. autofunction:: Cv_to_Kv
.. autofunction:: Kv_to_Cv

Miscellaneous
-------------
.. autofunction:: v_lift_valve_Crane

Sources
-------

The main sources for these correlations are as follows [100]_ [101]_ [102]_ [103]_ [104]_:

.. [100] Crane Co. TP 410 Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
   2009.
.. [101] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
   and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
.. [102] Miller, Donald S. Internal Flow Systems: Design and Performance
   Prediction. Gulf Publishing Company, 1990.
.. [103] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
   Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
   Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
   Treniya). National technical information Service, 1966.
.. [104] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
   Van Nostrand Reinhold Co., 1984.

"""

from __future__ import division
from math import sqrt, cos, sin, tan, atan, pi, radians, degrees, log10, log
from fluids.constants import inch, deg2rad, rad2deg
from fluids.friction import (friction_factor, Clamond,
                             friction_factor_curved, ft_Crane)
from fluids.numerics import (horner, interp, splev, bisplev,
                             implementation_optimize_tck, tck_interp2d_linear)

__all__ = ['contraction_sharp', 'contraction_round',
           'contraction_round_Miller',
'contraction_conical', 'contraction_conical_Crane', 'contraction_beveled',  'diffuser_sharp',
'diffuser_conical', 'diffuser_conical_staged', 'diffuser_curved',
'diffuser_pipe_reducer',
'entrance_sharp', 'entrance_distance', 'entrance_angled',
'entrance_rounded', 'entrance_beveled', 'entrance_beveled_orifice',
'entrance_distance_45_Miller',
'exit_normal', 'bend_rounded', 'bend_rounded_Miller', 'bend_rounded_Crane', 'bend_miter',
'bend_miter_Miller', 'helix', 'spiral','Darby3K', 'Hooper2K', 'Kv_to_Cv', 'Cv_to_Kv',
'Kv_to_K', 'K_to_Kv', 'Cv_to_K', 'K_to_Cv', 'change_K_basis', 'Darby',
'Hooper', 'K_gate_valve_Crane', 'K_angle_valve_Crane', 'K_globe_valve_Crane',
'K_swing_check_valve_Crane', 'K_lift_check_valve_Crane',
'K_tilting_disk_check_valve_Crane', 'K_globe_stop_check_valve_Crane',
'K_angle_stop_check_valve_Crane', 'K_ball_valve_Crane',
'K_diaphragm_valve_Crane', 'K_foot_valve_Crane', 'K_butterfly_valve_Crane',
'K_plug_valve_Crane', 'K_branch_converging_Crane', 'K_run_converging_Crane',
'K_branch_diverging_Crane', 'K_run_diverging_Crane', 'v_lift_valve_Crane']



def change_K_basis(K1, D1, D2):
    r'''Converts a loss coefficient `K1` from the basis of one diameter `D1`
    to another diameter, `D2`. This is necessary when dealing with pipelines
    of changing diameter.

    .. math::
        K_2 = K_1\frac{D_2^4}{D_1^4} = K_1 \frac{A_2^2}{A_1^2}

    Parameters
    ----------
    K1 : float
        Loss coefficient with respect to diameter `D`, [-]
    D1 : float
        Diameter of pipe for which `K1` has been calculated, [m]
    D2 : float
        Diameter of pipe for which `K2` will be calculated, [m]

    Returns
    -------
    K2 : float
        Loss coefficient with respect to the second diameter, [-]

    Notes
    -----
    This expression is shown in [1]_ and can easily be derived:

    .. math::
        \frac{\rho V_{1}^{2}}{2} \cdot K_{1} = \frac{\rho V_{2}^{2} }{2}
        \cdot K_{2}

    Substitute velocities for flow rate divided by area:

    .. math::
        \frac{8 K_{1} Q^{2} \rho}{\pi^{2} D_{1}^{4}} = \frac{8 K_{2} Q^{2}
        \rho}{\pi^{2} D_{2}^{4}}

    From here, simplification and rearrangement is all that is required.

    Examples
    --------
    >>> change_K_basis(K1=32.68875692997804, D1=.01, D2=.02)
    523.020110879

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    r = D2/D1
    r *= r
    return K1*r*r


### Entrances

entrance_sharp_methods = ['Rennels', 'Swamee', 'Blevins', 'Idelchik', 'Crane',
                          'Miller']
entrance_sharp_method_missing = ('Specified method not recognized; methods are %s'
                         %(entrance_sharp_methods))

def entrance_sharp(method='Rennels'):
    r'''Returns loss coefficient for a sharp entrance to a pipe.
    Six sources are available; four of them recommending K = 0.5,
    the most recent 'Rennels', method recommending K = 0.57, and the
    'Miller' method recommending ~0.51 as read from a graph.

    .. figure:: fittings/flush_mounted_sharp_edged_entrance.png
       :scale: 30 %
       :alt: flush mounted sharp edged entrance; after [1]_

    Parameters
    ----------
    method : str, optional
        The method to use; one of 'Rennels', 'Swamee', 'Blevins',
        'Idelchik', 'Crane', or 'Miller, [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    0.5 is the result for  'Swamee', 'Blevins', 'Idelchik', and 'Crane';
    'Miller' returns 0.5093, and 'Rennels' returns 0.57.

    Examples
    --------
    >>> entrance_sharp()
    0.57

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [3] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    .. [4] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [5] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [6] Swamee, Prabhata K., and Ashok K. Sharma. Design of Water Supply
       Pipe Networks. John Wiley & Sons, 2008.
    '''
    if method is None:
        method = 'Rennels'
    if method in ('Swamee', 'Blevins', 'Crane', 'Idelchik'):
        return 0.50
    elif method == 'Miller':
        # From entrance_rounded(Di=0.9, rc=0.0, method='Miller'); Not saying it's right
        return 0.5092676683721356
    elif method == 'Rennels':
        return 0.57
    else:
        raise ValueError(entrance_sharp_method_missing)

entrance_distance_Miller_coeffs = [3.5979871366071166, -2.735407311020481, -14.08678246875138,
                                   10.637236472292983, 21.99568490754116, -16.38501138746954,
                                   -17.62779826803278, 12.945551397987447, 7.715463242992863,
                                   -5.850893341031715, -1.3809402870404826, 1.179637166644488,
                                   0.08781141316107932, -0.09751968111743672, 0.00501792061942849,
                                   0.0026378278251172615, 0.5309019247035696]

entrance_distance_Idelchik_l_Di = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
                                   0.2, 0.3, 0.5, 10.0] # last point infinity
entrance_distance_Idelchik_t_Di = [0.0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024,
                                   0.03, 0.04, 0.05, 1.0] # last point infinity

entrance_distance_Idelchik_dat = [
    [0.5, 0.57, 0.63, 0.68, 0.73, 0.8, 0.86, 0.92, 0.97, 1, 1],
    [0.5, 0.54, 0.58, 0.63, 0.67, 0.74, 0.8, 0.86, 0.9, 0.94, 0.94],
    [0.5, 0.53, 0.55, 0.58, 0.62, 0.68, 0.74, 0.81, 0.85, 0.88, 0.88],
    [0.5, 0.52, 0.53, 0.55, 0.58, 0.63, 0.68, 0.75, 0.79, 0.83, 0.83],
    [0.5, 0.51, 0.51, 0.53, 0.55, 0.58, 0.64, 0.7, 0.74, 0.77, 0.77],
    [0.5, 0.51, 0.51, 0.52, 0.53, 0.55, 0.6, 0.66, 0.69, 0.72, 0.72],
    [0.5, 0.5, 0.5, 0.51, 0.52, 0.53, 0.58, 0.62, 0.65, 0.68, 0.68],
    [0.5, 0.5, 0.5, 0.51, 0.52, 0.52, 0.54, 0.57, 0.59, 0.61, 0.61],
    [0.5, 0.5, 0.5, 0.51, 0.51, 0.51, 0.51, 0.52, 0.52, 0.54, 0.54],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]


entrance_distance_Idelchik_tck = tck_interp2d_linear(entrance_distance_Idelchik_l_Di,
                                                     entrance_distance_Idelchik_t_Di,
            													entrance_distance_Idelchik_dat,
													          kx=1, ky=1)

entrance_distance_Idelchik_obj = lambda x, y: float(bisplev(x, y, entrance_distance_Idelchik_tck))
entrance_distance_Idelchik_obj = lambda x, y: bisplev(x, y, entrance_distance_Idelchik_tck)

entrance_distance_Harris_t_Di = [0.00322, 0.007255, 0.01223, 0.018015,
    0.021776, 0.029044, 0.039417, 0.049519, 0.058012, 0.066234,
    0.076747, 0.088337, 0.098714, 0.109497, 0.121762, 0.130655, 0.14036,
    0.148986, 0.159902, 0.17149, 0.179578, 0.189416, 0.200602, 0.208148,
    0.217716, 0.228232, 0.239821, 0.250063, 0.260845, 0.270818,
    0.280116, 0.289145]
entrance_distance_Harris_Ks = [0.894574, 0.832435, 0.749768, 0.671543,
    0.574442, 0.508432, 0.476283, 0.430261, 0.45027, 0.45474, 0.461993,
    0.457042, 0.458745, 0.464889, 0.471594, 0.461638, 0.467778,
    0.475024, 0.474509, 0.456239, 0.466258, 0.467959, 0.466336,
    0.459705, 0.454746, 0.478092, 0.468701, 0.467074, 0.468779,
    0.467151, 0.46441, 0.458894]

entrance_distance_Harris_tck = implementation_optimize_tck([
    [0.00322, 0.00322, 0.00322, 0.00322, 0.01223, 0.018015, 0.021776, 0.029044,
     0.039417, 0.049519, 0.058012, 0.066234, 0.076747, 0.088337, 0.098714,
     0.109497, 0.121762, 0.130655, 0.14036, 0.148986, 0.159902, 0.17149,
     0.179578, 0.189416, 0.200602, 0.208148, 0.217716, 0.228232, 0.239821,
     0.250063, 0.260845, 0.270818, 0.289145, 0.289145, 0.289145, 0.289145],
     [0.894574, 0.8607821362959746, 0.7418364422223542, 0.7071594764719331,
      0.5230593641637336, 0.5053866365045014, 0.4869380604512194,
      0.40993425463761973, 0.4588732899536263, 0.45115886608796796,
      0.4672085434114074, 0.45422360120010624, 0.45882234693051327,
      0.4633823025024543, 0.4785594597978615, 0.45603301615693537,
      0.46825191653436804, 0.4759245648612374, 0.4816400424293727,
      0.4467699156979281, 0.4713316096394432, 0.4667017151264001,
      0.4686302748435692, 0.4597796190662107, 0.445267522727416,
      0.491034205369033, 0.4641178520412072, 0.46721810151497395,
      0.46958841021674314, 0.4664976446563455, 0.46420067427943945,
      0.458894, 0.0, 0.0, 0.0, 0.0],
      3])

entrance_distance_Harris_obj = lambda x : float(splev(x, entrance_distance_Harris_tck))



entrance_distance_methods = ['Rennels', 'Miller', 'Idelchik', 'Harris',
                             'Crane']

entrance_distance_unrecognized_msg = 'Specified method not recognized; methods are %s' %(entrance_distance_methods)

def entrance_distance(Di, t=None, l=None, method='Rennels'):
    r'''Returns the loss coefficient for a sharp entrance to a pipe at a distance
    from the wall of a reservoir. This calculation has five methods available;
    all but 'Idelchik' require the pipe to be at least `Di/2` into the
    reservoir.

    The most conservative formulation is that of Rennels; with Miller being
    almost identical until `t/Di` reaches 0.05, when it continues settling to
    K = 0.53 compared to K = 0.57 for 'Rennels'. 'Idelchik' is offset lower
    by about 0.03 and settles to 0.50. The 'Harris' method is a straight
    interpolation from experimental results with smoothing, and it is the
    lowest at all points. The 'Crane' [6]_ method returns 0.78 for all cases.

    The Rennels [1]_ formula is:

    .. math::
        K = 1.12 - 22\frac{t}{d} + 216\left(\frac{t}{d}\right)^2 +
        80\left(\frac{t}{d}\right)^3

    .. figure:: fittings/sharp_edged_entrace_extended_mount.png
       :scale: 30 %
       :alt: sharp edged entrace, extended mount; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    t : float, optional
        Thickness of pipe wall, used in all but 'Crane' method, [m]
    l : float, optional
        The distance the pipe extends into the reservoir; used only in the
        'Idelchik' method, defaults to `Di`, [m]
    method : str, optional
        One of 'Rennels', 'Miller', 'Idelchik', 'Harris', 'Crane', [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    This type of inlet is also known as a Borda's mouthpiece.
    It is not of practical interest according to [1]_.

    The 'Idelchik' [3]_ data is recommended in [5]_; it also provides rounded
    values for the 'Harris. method.

    .. plot:: plots/entrance_distance_plot.py

    Examples
    --------
    >>> entrance_distance(Di=0.1, t=0.0005)
    1.0154100000000001
    >>> entrance_distance(Di=0.1, t=0.0005, method='Idelchik')
    0.9249999999999999
    >>> entrance_distance(Di=0.1, t=0.0005, l=.02, method='Idelchik')
    0.8474999999999999

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [3] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    .. [4] Harris, Charles William. The Influence of Pipe Thickness on
       Re-Entrant Intake Losses. Vol. 48. University of Washington, 1928.
    .. [5] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [6] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if method is None:
        method = 'Rennels'
    if method == 'Rennels':
        t_Di = t/Di
        if t_Di > 0.05:
            t_Di = 0.05
        return 1.12 + t_Di*(t_Di*(80.0*t_Di + 216.0) - 22.0)
    elif method == 'Miller':
        t_Di = t/Di
        if t_Di > 0.3:
            t_Di = 0.3
        return horner(entrance_distance_Miller_coeffs, 20.0/3.0*(t_Di - 0.15))
    elif method == 'Idelchik':
        if l is None:
            l = Di
        t_Di = min(t/Di, 1.0)
        l_Di = min(l/Di, 10.0)
        K = float(entrance_distance_Idelchik_obj(l_Di, t_Di))
        if K < 0.0:
            K = 0.0
        return K
    elif method == 'Harris':
        ratio = min(t/Di, 0.289145) # max value for interpolation - extrapolation looks bad
        K = float(entrance_distance_Harris_obj(ratio))
        return K
    elif method == 'Crane':
        return 0.78
    else:
        raise ValueError(entrance_distance_unrecognized_msg)


entrance_distance_45_Miller_coeffs = [1.866792110435199, -2.8873199398381075, -4.814715029513536,
                                      10.49562589373457, 1.40401776402922, -14.035912282651882,
                                      6.576826918678071, 7.854645523152614, -8.044860164646053,
                                      -1.1515885154512326, 4.145420152553604, -0.7994793202964967,
                                      -1.1034822877774095, 0.32764916637953573, 0.367065452438954,
                                      -0.2614447909010587, 0.29084476697430256]


def entrance_distance_45_Miller(Di, Di0):
    r'''Returns loss coefficient for a sharp entrance to a pipe at a distance
    from the wall of a reservoir with an initial 45 degree slope conical
    section of diameter `Di0` added to reduce the overall loss coefficient.

    This method is as shown in Miller's Internal Flow Systems
    [1]_. This method is a curve fit to a graph in [1]_ which was digitized.

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    Di0 : float
        Initial inner diameter of the welded conical section of the entrance
        of the distant (re-entrant) pipe, [m]

    Returns
    -------
    K : float
        Loss coefficient with respect to the main pipe diameter `Di`, [-]

    Notes
    -----
    The graph predicts an almost constant loss coefficient once the thickness
    of pipe wall to pipe diameter ratio becomes ~0.02.

    Examples
    --------
    >>> entrance_distance_45_Miller(Di=0.1, Di0=0.14)
    0.24407641818143339

    References
    ----------
    .. [1] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    '''
    t = 0.5*(Di0 - Di)
    t_Di = t/Di
    if t_Di > 0.3:
        t_Di = 0.3
    return horner(entrance_distance_45_Miller_coeffs, 6.66666666666666696*(t_Di-0.15))


entrance_angled_methods = ['Idelchik']

entrance_angled_methods_missing = ('Specified method not recognized; methods are %s'
                                   %(entrance_angled_methods))
def entrance_angled(angle, method='Idelchik'):
    r'''Returns loss coefficient for a sharp, angled entrance to a pipe
    flush with the wall of a reservoir. First published in [2]_, it has been
    recommended in [3]_ as well as in [1]_.

    .. math::
        K = 0.57 + 0.30\cos(\theta) + 0.20\cos(\theta)^2

    .. figure:: fittings/entrance_mounted_at_an_angle.png
       :scale: 30 %
       :alt: entrace mounted at an angle; after [1]_

    Parameters
    ----------
    angle : float
        Angle of inclination (90° = straight, 0° = parallel to pipe wall),
        [degrees]
    method : str, optional
        The method to use; only 'Idelchik' is supported

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Not reliable for angles under 20 degrees.
    Loss coefficient is the same for an upward or downward angled inlet.

    Examples
    --------
    >>> entrance_angled(30)
    0.9798076211353315

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    .. [3] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    if method == 'Idelchik' or method is None:
        cos_term = cos(deg2rad*angle)
        return 0.57 + cos_term*(0.2*cos_term + 0.3)
    else:
        raise ValueError(entrance_angled_methods_missing)


entrance_rounded_Miller_coeffs = [1.3127209945178038, 0.19963046592715727, -6.49081916725612,
                                  -0.10347409377743588, 12.68369791325003, -0.9435681020599904
                                  , -12.44320584089916, 1.328251365167716, 6.668390027065714,
                                  -0.4356382649470076, -2.209229212394282, -0.07222448354500295,
                                  0.6786898049825905, -0.18686362789567468, 0.020064570486606065,
                                  -0.013120241146656442, 0.061951596342059975]

entrance_rounded_ratios_Idelchik = [0, .01, .02, .03, .04, .05, .06, .08, .12,
                                    .16, .2]
entrance_rounded_Ks_Idelchik = [.5, .44, .37, .31, .26, .22, .2, .15, .09, .06,
                                .03]

entrance_rounded_Idelchik_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.015, 0.025, 0.035, 0.045, 0.055,
                                                              0.07, 0.1, 0.14, 0.2, 0.2, 0.2],
        [0.5, 0.46003224474143023, 0.3682580956033294, 0.30877401146621397, 0.2590978355993873,
         0.2166389749374616, 0.19717564973543905, 0.1332971654240214, 0.08659056691519569,
         0.05396118560777325, 0.03, 0.0, 0.0, 0.0],
         2])

entrance_rounded_Idelchik = lambda x : float(splev(x, entrance_rounded_Idelchik_tck))


entrance_rounded_ratios_Crane = [0.0, .02, .04, .06, .1, .15]
entrance_rounded_Ks_Crane = [.5, .28, .24, .15, .09, .04]

entrance_rounded_ratios_Harris = [0.0, .01, .02, .03, .04, .05, .06, .08, .12,
                                  .16]
entrance_rounded_Ks_Harris = [.44, .35, .28, .22, .17, .13, .1, .07, .03, 0.0]

entrance_rounded_Harris_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.015, 0.025, 0.035, 0.045,
                                                            0.055, 0.07, 0.1, 0.16, 0.16, 0.16],
    [0.44, 0.36435669860605086, 0.2790010365858813, 0.2187082142826953, 0.16874967771794716,
     0.1287937194096216, 0.09091157742799895, 0.06354756460434334, 0.01885121769782832,
     0.0, 0.0, 0.0, 0.0],
    2])

entrance_rounded_Harris = lambda x : float(splev(x, entrance_rounded_Harris_tck))

entrance_rounded_methods = ['Rennels', 'Crane', 'Miller', 'Idelchik', 'Harris',
                            'Swamee']
entrance_rounded_methods_error = ('Specified method not recognized; methods are %s'
                                  %(entrance_rounded_methods))

def entrance_rounded(Di, rc, method='Rennels'):
    r'''Returns loss coefficient for a rounded entrance to a pipe
    flush with the wall of a reservoir. This calculation has six methods
    available.

    The most conservative formulation is that of Rennels; with the Swammee
    correlation being 0.02-0.07 lower. They were published in 2012 and 2008
    respectively, and for this reason could be regarded as more reliable.

    The Idel'chik correlation appears based on the Hamilton data; and the
    Miller correlation as well, except a little more conservative. The Crane
    model trends similarly but only has a few points. The Harris data set is
    the lowest.

    The Rennels [1]_ formulas are:

    .. math::
        K = 0.0696\left(1 - 0.569\frac{r}{d}\right)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622\left(1 - 0.30\sqrt{\frac{r}{d}}
        - 0.70\frac{r}{d}\right)^4

    The Swamee [5]_ formula is:

    .. math::
        K = 0.5\left[1 + 36\left(\frac{r}{D}\right)^{1.2}\right]^{-1}

    .. figure:: fittings/flush_mounted_rounded_entrance.png
       :scale: 30 %
       :alt: rounded entrace mounted straight and flush; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    rc : float
        Radius of curvature of the entrance, [m]
    method : str, optional
        One of 'Rennels', 'Crane', 'Miller', 'Idelchik', 'Harris', or 'Swamee'.

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    For generously rounded entrance (rc/Di >= 1), the loss coefficient
    converges to 0.03 in the Rennels method.

    The Rennels formulation was derived primarily from data and theoretical
    analysis from different flow scenarios than a rounded pipe entrance; the
    only available data in [2]_ is quite old and [1]_ casts doubt on it.

    The Hamilton data set is available in [1]_ and [6]_.

    .. plot:: plots/entrance_rounded_plot.py


    Examples
    --------
    Point from Diagram 9.2 in [1]_, which was used to confirm the Rennels
    model implementation:

    >>> entrance_rounded(Di=0.1, rc=0.0235)
    0.09839534618360923

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Hamilton, James Baker. Suppression of Pipe Intake Losses by Various
       Degrees of Rounding. Seattle: Published by the University of Washington,
       1929. https://search.library.wisc.edu/catalog/999823652202121.
    .. [3] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [4] Harris, Charles William. Elimination of Hydraulic Eddy Current Loss
       at Intake, Agreement of Theory and Experiment. University of Washington,
       1930.
    .. [5] Swamee, Prabhata K., and Ashok K. Sharma. Design of Water Supply
       Pipe Networks. John Wiley & Sons, 2008.
    .. [6] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [7] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [8] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    '''
    if method is None:
        method = 'Rennels'
    ratio = rc/Di
    if method == 'Rennels':
        if ratio > 1.0:
            return 0.03

        lbd = (1.0 - 0.30*sqrt(ratio) - 0.70*ratio)
        lbd *= lbd
        lbd = 1.0 + 0.622*lbd*lbd
        return 0.0696*(1.0 - 0.569*ratio)*lbd*lbd + (lbd - 1.0)*(lbd - 1.0)
    elif method == 'Swamee':
        return 0.5/(1.0 + 36.0*(ratio)**1.2)
    elif method == 'Crane':
        if ratio < 0:
            return 0.5
        elif ratio > 0.15:
            return 0.04
        else:
            return interp(ratio, entrance_rounded_ratios_Crane,
                          entrance_rounded_Ks_Crane)
    elif method == 'Miller':
        if ratio > 0.3:
            ratio = 0.3
        return horner(entrance_rounded_Miller_coeffs, (20.0/3.0)*(ratio - 0.15))
    elif method == 'Harris':
        if ratio > .16:
            return 0.0
        return float(splev(ratio, entrance_rounded_Harris_tck))
    elif method == 'Idelchik':
        if ratio > .2:
            return entrance_rounded_Ks_Idelchik[-1]
        return float(splev(ratio, entrance_rounded_Idelchik_tck))
    else:
        raise ValueError(entrance_rounded_methods_error)


entrance_beveled_methods = ['Rennels', 'Idelchik']
entrance_beveled_methods_unknown_msg = 'Specified method not recognized; methods are %s' %entrance_beveled_methods

entrance_beveled_Idelchik_l_Di = [0.025, 0.05, 0.075, 0.1, 0.15, 0.6]
entrance_beveled_Idelchik_angles = [0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 100.0,
                                    140.0, 180.0]

entrance_beveled_Idelchik_dat = [
    [0.5, 0.47, 0.45, 0.43, 0.41, 0.4, 0.42, 0.45, 0.5],
    [0.5, 0.45, 0.41, 0.36, 0.33, 0.3, 0.35, 0.42, 0.5],
    [0.5, 0.42, 0.35, 0.3, 0.26, 0.23, 0.3, 0.4, 0.5],
    [0.5, 0.39, 0.32, 0.25, 0.22, 0.18, 0.27, 0.38, 0.5],
    [0.5, 0.37, 0.27, 0.2, 0.16, 0.15, 0.25, 0.37, 0.5],
    [0.5, 0.27, 0.18, 0.13, 0.11, 0.12, 0.23, 0.36, 0.5]]


entrance_beveled_Idelchik_tck = tck_interp2d_linear(entrance_beveled_Idelchik_angles,
                                                    entrance_beveled_Idelchik_l_Di,
                                                    entrance_beveled_Idelchik_dat,
                                                    kx=1, ky=1)
entrance_beveled_Idelchik_obj = lambda x, y : float(bisplev(x, y, entrance_beveled_Idelchik_tck))

def entrance_beveled(Di, l, angle, method='Rennels'):
    r'''Returns loss coefficient for a beveled or chamfered entrance to a pipe
    flush with the wall of a reservoir. This calculation has two methods
    available.

    The 'Rennels' and 'Idelchik' methods have similar trends, but the 'Rennels'
    formulation is centered around a straight loss coefficient of 0.57, so it
    is normally at least 0.07 higher.

    The Rennels [1]_ formulas are:

    .. math::
        K = 0.0696\left(1 - C_b\frac{l}{d}\right)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622\left[1-1.5C_b\left(\frac{l}{d}
        \right)^{\frac{1-(l/d)^{1/4}}{2}}\right]

    .. math::
        C_b = \left(1 - \frac{\theta}{90}\right)\left(\frac{\theta}{90}
        \right)^{\frac{1}{1+l/d}}

    .. figure:: fittings/flush_mounted_beveled_entrance.png
       :scale: 30 %
       :alt: Beveled entrace mounted straight; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    l : float
        Length of bevel measured parallel to the pipe length, [m]
    angle : float
        Angle of bevel with respect to the pipe length, [degrees]
    method : str, optional
        One of 'Rennels', or 'Idelchik', [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    A cheap way of getting a lower pressure drop.
    Little credible data is available.

    The table of data in [2]_ uses the angle for both bevels, so it runs from 0
    to 180 degrees; this function follows the convention in [1]_ which uses
    only one angle, with the angle varying from 0 to 90 degrees.

    .. plot:: plots/entrance_beveled_plot.py

    Examples
    --------
    >>> entrance_beveled(Di=0.1, l=0.003, angle=45)
    0.450868642219
    >>> entrance_beveled(Di=0.1, l=0.003, angle=45, method='Idelchik')
    0.399500000000

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    '''
    if method is None:
        method = 'Rennels'
    if method == 'Rennels':
        Cb = (1-angle/90.)*(angle/90.)**(1./(1 + l/Di ))
        lbd = 1 + 0.622*(1 - 1.5*Cb*(l/Di)**((1 - sqrt(sqrt(l/Di)))/2.))
        return 0.0696*(1 - Cb*l/Di)*lbd**2 + (lbd - 1.)**2
    elif method == 'Idelchik':
        return float(bisplev(angle*2.0, l/Di, entrance_beveled_Idelchik_tck))
    else:
        raise ValueError(entrance_beveled_methods_unknown_msg)


def entrance_beveled_orifice(Di, do, l, angle):
    r'''Returns loss coefficient for a beveled or chamfered orifice entrance to
    a pipe flush with the wall of a reservoir, as shown in [1]_.

    .. math::
        K = 0.0696\left(1 - C_b\frac{l}{d_o}\right)\lambda^2 + \left(\lambda
        -\left(\frac{d_o}{D_i}\right)^2\right)^2

    .. math::
        \lambda = 1 + 0.622\left[1-C_b\left(\frac{l}{d_o}\right)^{\frac{1-
        (l/d_o)^{0.25}}{2}}\right]

    .. math::
        C_b = \left(1 - \frac{\Psi}{90}\right)\left(\frac{\Psi}{90}
        \right)^{\frac{1}{1+l/d_o}}

    .. figure:: fittings/flush_mounted_beveled_orifice_entrance.png
       :scale: 30 %
       :alt: Beveled orifice entrace mounted straight; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    do : float
        Inside diameter of orifice, [m]
    l : float
        Length of bevel measured parallel to the pipe length, [m]
    angle : float
        Angle of bevel with respect to the pipe length, [degrees]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Examples
    --------
    >>> entrance_beveled_orifice(Di=0.1, do=.07, l=0.003, angle=45)
    1.2987552913818574

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    Cb = (1-angle/90.)*(angle/90.)**(1./(1 + l/do ))
    lbd = 1 + 0.622*(1 - Cb*(l/do)**((1 - sqrt(sqrt(l/do)))/2.))
    return 0.0696*(1 - Cb*l/do)*lbd**2 + (lbd - (do/Di)**2)**2


### Exits

def exit_normal():
    r'''Returns loss coefficient for any exit to a pipe
    as shown in [1]_ and in other sources.

    .. math::
        K = 1

    .. figure:: fittings/flush_mounted_exit.png
       :scale: 28 %
       :alt: Exit from a flush mounted wall; after [1]_

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
    return 1.0


### Bends


tck_bend_rounded_Miller = implementation_optimize_tck([[0.500967, 0.500967, 0.500967, 0.500967, 0.5572659504420276, 0.6220535279438968, 0.6876695918008857,
     0.8109956990835443, 0.8966138996017785, 1.0418136796591293, 1.2129808986390955, 1.4328097893561944,
     2.684491977649823, 3.496050493509287, 4.245254058334557, 10.0581, 10.0581, 10.0581, 10.0581],
  [10.0022, 10.0022, 10.0022, 10.0022, 26.661576730080427, 35.71142422728946, 46.22896414495794, 54.476944091380965,
     67.28681897720492, 79.96560467244989, 88.89484575805731, 104.37345376723293, 113.75217318286595, 121.36638011164008,
     139.53481668808192, 180.502, 180.502, 180.502, 180.502],
   [0.02844925354339322, 0.032368056788003474, 0.06341726367587057, 0.18372991235687228, 0.27828335685928296,
     0.4184452895626468, 0.5844709012848479, 0.8517327028006999, 1.0883889837806633, 1.003595822015052, 1.2959349743905006,
     1.3631701864169843, 3.2579960738248563, 8.188259745620396, 6.370167194425542, 0.026614405579949103, 0.03578575879432178,
     0.05399131725104529, 0.17357295746658216, 0.2597698136964017, 0.384398460262134, 0.5537955210508835, 0.842964805734998,
     1.1076060802420074, 1.0500502914944205, 1.2160489773171173, 1.2940140217639442, 2.5150913200614293, 5.987790923112488,
     4.791049223949247, 0.026866783841898684, 0.03061409809632371, 0.054698306220358, 0.14037162784411245, 0.23981090432386729,
     0.31617091309760137, 0.47435842573782666, 0.7484605121106159, 0.9223888516911868, 1.0345139221619066, 1.0709769967277933,
     1.1489283659291687, 1.4249255928619116, 2.6908421883082823, 2.3898833324508804, 0.019707980719056793, 0.03350958504709355,
     0.0457699204936841, 0.1180773988295937, 0.18163838540491214, 0.2955424583244998, 0.3178086095370295, 0.54907384767895,
     0.7497276995283433, 0.8353766950608585, 0.8907203653185313, 0.941376749552297, 0.8755423259796333, 0.8987849646797164,
     0.9905785504810203, 0.018632197087313764, 0.0275473376021632, 0.046686663726990756, 0.09334625398868963,
     0.15009471210360348, 0.21438462374865175, 0.310541469358518, 0.27652184608845864, 0.4703245212932829,
     0.5612926929410017, 0.6344189573543495, 0.6897616299237337, 0.8553230255854581, 0.8050040042565408,
     0.7800498994134173, 0.017040716941189974, 0.027163747207842776, 0.04233976165781228, 0.08546809847236579,
     0.11872359104267481, 0.1748602349243538, 0.248787221592314, 0.3166892465009758, 0.2894990945943436,
     0.35635089905047324, 0.3942719381041552, 0.4019846022857163, 0.4910888827789205, 0.4424331343990761,
     0.5367477778555589, 0.017232689797500957, 0.024595005629126976, 0.04235982677436609, 0.0748705682747817,
     0.11096283696103083, 0.13900984487771062, 0.18773056195495877, 0.2400721832034611, 0.28581377924973544,
     0.282839816159864, 0.2907117502580411, 0.3035848810896592, 0.31268019467513564, 0.3365050687225188, 0.2836774098946595,
     0.017462451480157917, 0.02373981127475937, 0.04248526591300313, 0.07305722078054935, 0.09424065630357203,
     0.13682400355164548, 0.15020534827616405, 0.2100221959547714, 0.23136495625582817, 0.24417894312621574,
     0.2505645472554214, 0.24143469557592281, 0.24722191256497117, 0.2195110087547775, 0.29557609063213136,
     0.017605444779345832, 0.026265210174737128, 0.0445497171166642, 0.07254637551095446, 0.08779690828578819,
     0.11992614224260065, 0.14501268843599757, 0.17386066713179812, 0.21657094190224363, 0.21594544490951023, 0.22661999176624517,
     0.23759356544596819, 0.23887614636323537, 0.25802515101229484, 0.20566480389514516, 0.01928450591486404, 0.03264367752872495,
     0.05391006363370407, 0.07430728218140033, 0.08818045730326454, 0.09978389535000864, 0.12544634357734885, 0.13365159719049172,
     0.15802979203343911, 0.17543365869590444, 0.17531453508236272, 0.1706085325985479, 0.15983319357859727, 0.16872558079206196,
     0.19799750352823683, 0.020835891827102552, 0.047105767455498285, 0.05307639179638059, 0.07839236342751181, 0.09519829368423402,
     0.10189528661430994, 0.12852821694010982, 0.13195311029179943, 0.1594822363328695, 0.15660304273110143, 0.15934161651984413,
     0.17702957118830723, 0.1892675345030034, 0.19710951153945122, 0.1897835097361326, 0.031571285288316195, 0.04810266172763896,
     0.05660304311192384, 0.09317293919692342, 0.08967028392412497, 0.12028974875677166, 0.1182836264474129, 0.13845925262729528,
     0.15739100571169004, 0.17649056196464383, 0.20171423738165223, 0.20947832805305883, 0.22837004534830094, 0.23661874048689152,
     0.24537433391842686, 0.042992073811512765, 0.045958026954244176, 0.08988351069774198, 0.08320361205549355, 0.1253881915447805,
     0.12765039447605908, 0.1632907944306065, 0.17922551055575348, 0.20436939408609628, 0.23133806857897737, 0.22837190631962206,
     0.2611718034649056, 0.30462224139228183, 0.3277471634644065, 0.3595577208662931, 0.042671097083349346, 0.06027193387363409,
     0.07182684474072856, 0.12072547771177115, 0.1331787059163636, 0.16137414417679433, 0.1780034002291815, 0.19820571860540606,
     0.2294059556234193, 0.23221403415772682, 0.2697708431035234, 0.2813760107306456, 0.28992333749905363, 0.3650401400682786,
     0.8993207970132076, 0.045660964207664585, 0.06299599466264151, 0.09193684371316964, 0.12747145786167088, 0.14606550538249963,
     0.172664884028299, 0.19152378303841075, 0.2212007207927944, 0.23752800077573005, 0.26289800433018995, 0.2772198641539113,
     0.2995308585350757, 0.3549459028594012, 0.8032461437896778, 3.330618601208751],
   3, 3])


bend_rounded_Miller_Kb = lambda rc_D, angle : float(bisplev(rc_D, angle, tck_bend_rounded_Miller))

tck_bend_rounded_Miller_C_Re = implementation_optimize_tck([[4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0],
                                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                                [2.177340320782947, 2.185952396281732, 2.185952396281732, 2.1775876405173977,
  0.6513348082098823, 0.7944713057222101, 0.7944713057222103, 1.0526247737400114,
  0.6030278030721317, 1.3741240162063968, 1.3741240162063992, 0.7693594604301893,
  -2.1663631289607883, -1.9474318981548622, -1.9474318981548622, 0.4196741237602154],
   3, 3])

bend_rounded_Miller_C_Re = lambda Re, rc_D : float(bisplev(log10(Re), rc_D, tck_bend_rounded_Miller_C_Re))
bend_rounded_Miller_C_Re_limit_1 = [2428087.757821312, -13637184.203693766, 28450331.830760233, -25496945.91463643, 8471761.477755375]


tck_bend_rounded_Miller_C_o_0_1 = implementation_optimize_tck([[9.975803953769495e-06, 9.975803953769495e-06, 9.975803953769495e-06,
        9.975803953769495e-06, 0.5259485989276764, 1.3157845547408782, 3.220104449183945, 6.133677908951886,
        30.260656153593906, 30.260656153593906, 30.260656153593906, 30.260656153593906],
        [0.6179524338907976, 0.6000479624108129, 0.49299050530751654, 0.4820011733402483, 0.5584830305084972,
        0.7496716557444135, 0.8977538553873484, 0.9987218804089956, 0.0, 0.0, 0.0, 0.0],
3])
tck_bend_rounded_Miller_C_o_0_15 = implementation_optimize_tck([[0.0025931401409935687, 0.0025931401409935687, 0.0025931401409935687,
        0.0025931401409935687, 0.26429667728434275, 0.5188174292838083, 1.469212480387932, 4.269571348168375,
        13.268280073552294, 26.28093462852014, 26.28093462852014, 26.28093462852014, 26.28093462852014],
        [0.8691924906711972, 0.8355177386350426, 0.7617588987656675, 0.5853012015918869, 0.5978128647571033,
        0.7366100253604377, 0.8229203841913866, 0.9484887080989913, 1.0003643259424702, 0.0, 0.0, 0.0, 0.0],
3])
tck_bend_rounded_Miller_C_o_0_2 = implementation_optimize_tck([[-0.001273275512351991, -0.001273275512351991, -0.001273275512351991, -
        0.001273275512351991, 0.36379835796750504, 0.7789151587713531, 1.7319487323386349, 3.559883175039053,
        22.10600230228466, 22.10600230228466, 22.10600230228466, 22.10600230228466],
        [1.2055892891232, 1.1810797953131011, 0.8556056552110055, 0.6595884323229468, 0.6669634037761268,
        0.8636791463334055, 0.8855712717206472, 0.9992625616471772, 0.0, 0.0, 0.0, 0.0],
3])
tck_bend_rounded_Miller_C_o_0_25 = implementation_optimize_tck([[0.0025931401409935687, 0.0025931401409935687, 0.0025931401409935687,
        0.0025931401409935687, 0.2765978180291006, 0.5010875816968301, 0.6395222359284018, 0.661563946104784,
        0.6887462820881093, 0.7312909084975013, 0.7605490601821624, 0.8078652661481783, 0.8553090397903271,
        1.024376958429362, 1.4748577103270428, 2.052843716337269, 3.9670225184835175, 6.951737782758053,
        16.770001745987884, 16.770001745987884, 16.770001745987884, 16.770001745987884],
        [2.7181584441006414, 2.6722855229796196, 2.510271857479865, 2.162580617260359, 1.8234805515473758,
        1.5274137403431902, 1.3876379087140025, 1.2712745614209848, 1.1478416325256429, 1.015542018903243,
        0.8445749706812837, 0.7368799268423506, 0.7061205857035833, 0.7381928947255646, 0.7960778489514514,
        0.878729192230999, 0.9281388590439098, 0.9825611959699471, 0.0, 0.0, 0.0, 0.0],
3])

tck_bend_rounded_Miller_C_o_1_0 = implementation_optimize_tck([[0.0025931401409935687, 0.0025931401409935687, 0.0025931401409935687,
        0.0025931401409935687, 0.4940382602529053, 0.7383107558560895, 0.8929948619544391, 0.9910262538499016,
        1.1035407055233972, 1.2685727302009009, 2.190931635360523, 3.718073594472333, 6.026458907878363,
        13.268280073552294, 13.268280073552294, 13.268280073552294, 13.268280073552294],
        [2.713127433391318, 2.6799201583608965, 2.4446034702691906, 2.0505313661892837, 1.7853408404592677,
        1.5802763594858027, 1.395503315683405, 1.0504150726350026, 0.9294800209596744, 0.8937523212160566,
        0.9339124388590752, 0.9769117997985829, 0.9948478073955791, 0.0, 0.0, 0.0, 0.0],
3])

tck_bend_rounded_Miller_C_os = (tck_bend_rounded_Miller_C_o_0_1, tck_bend_rounded_Miller_C_o_0_15,
                                tck_bend_rounded_Miller_C_o_0_2, tck_bend_rounded_Miller_C_o_0_25,
                                tck_bend_rounded_Miller_C_o_1_0)
bend_rounded_Miller_C_o_Kbs = [.1, .15, .2, .25, 1]
bend_rounded_Miller_C_o_limits = [30.260656153593906, 26.28093462852014, 22.10600230228466, 16.770001745987884, 13.268280073552294]
bend_rounded_Miller_C_o_limit_0_01 = [0.6169055099514943, 0.8663244713199465, 1.2029584898712695, 2.7143438886138744, 2.7115417734646114]


def Miller_bend_roughness_correction(Re, Di, roughness):
    # Section 9.2.4 - Roughness correction
    # Re limited to under 1E6 in friction factor falculations
    # Use a cached smooth fd value if Re too high
    Re_fd_min = min(1E6, Re)
    if Re_fd_min < 1E6:
        fd_smoth = friction_factor(Re=Re_fd_min, eD=0.0)
    else:
        fd_smoth = 0.011645040997991626
    fd_rough = friction_factor(Re=Re_fd_min, eD=roughness/Di)
    C_roughness = fd_rough/fd_smoth
    return C_roughness


def Miller_bend_unimpeded_correction(Kb, Di, L_unimpeded):
    """Limitations as follows:

    * Ratio not over 30
    * If ratio under 0.01, tabulated values are used near the limits
      (discontinuity in graph anyway)
    * If ratio for a tried curve larger than max value, max value is used
      instead of calculating it
    * Kb limited to between 0.1 and 1.0
    * When between two Kb curves, interpolate linearly after evaluating both
      splines appropriately
    """
    if Kb < 0.1:
        Kb_C_o = 0.1
    elif Kb > 1:
        Kb_C_o = 1.0
    else:
        Kb_C_o = Kb

    L_unimpeded_ratio = L_unimpeded/Di
    if L_unimpeded_ratio > 30:
        L_unimpeded_ratio = 30.0

    for i in range(len(bend_rounded_Miller_C_o_Kbs)):
        Kb_low, Kb_high = bend_rounded_Miller_C_o_Kbs[i], bend_rounded_Miller_C_o_Kbs[i+1]
        if Kb_low <= Kb_C_o <= Kb_high:
            if L_unimpeded_ratio >= bend_rounded_Miller_C_o_limits[i]:
                Co_low = 1.0
            elif L_unimpeded_ratio <= 0.01:
                Co_low = bend_rounded_Miller_C_o_limit_0_01[i]
            else:
                Co_low = float(splev(L_unimpeded_ratio, tck_bend_rounded_Miller_C_os[i]))
            if L_unimpeded_ratio >= bend_rounded_Miller_C_o_limits[i+1]:
                Co_high = 1.0
            elif L_unimpeded_ratio <= 0.01:
                Co_high = bend_rounded_Miller_C_o_limit_0_01[i+1]
            else:
                Co_high = float(splev(L_unimpeded_ratio, tck_bend_rounded_Miller_C_os[i+1]))
            C_o = Co_low + (Kb_C_o - Kb_low)*(Co_high - Co_low)/(Kb_high - Kb_low)
            return C_o


def bend_rounded_Miller(Di, angle, Re, rc=None, bend_diameters=None,
                        roughness=0.0, L_unimpeded=None):
    r'''Calculates the loss coefficient for a rounded pipe bend according to
    Miller [1]_. This is a sophisticated model which uses corrections for
    pipe roughness, the length of the pipe downstream before another
    interruption, and a correction for Reynolds number. It interpolates several
    times using several corrections graphs in [1]_.

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    angle : float
        Angle of bend, [degrees]
    Re : float
        Reynolds number of the pipe (no specification if inlet or outlet
        properties should be used), [m]
    rc : float, optional
        Radius of curvature of the entrance, [m]
    bend_diameters : float, optional
        Number of diameters of pipe making up the bend radius  (used if rc not
        provided; defaults to 5), [-]
    roughness : float, optional
        Roughness of bend wall, [m]
    L_unimpeded : float, optional
        The length of unimpeded pipe without any fittings, instrumentation,
        or flow disturbances downstream (assumed 20 diameters if not
        specified), [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    When inputting bend diameters, note that manufacturers often specify
    this as a multiplier of nominal diameter, which is different than actual
    diameter. Those require that rc be specified.

    `rc` is limited to 0.5 or above; which represents a sharp, square, inner
    edge - and an outer bend radius of 1.0. Losses are at a minimum when this
    value is large.

    This was developed for bend angles between 10 and 180 degrees; and r/D
    ratios between 0.5 and 10. Both smooth and rough data was used in its
    development from several sources.

    Note the loss coefficient includes the surface friction of the pipe as if
    it was straight.

    Examples
    --------
    >>> bend_rounded_Miller(Di=.6, bend_diameters=2, angle=90,  Re=2e6,
    ... roughness=2E-5, L_unimpeded=30*.6)
    0.15261820705145895

    References
    ----------
    .. [1] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    '''
    if rc is None:
        if bend_diameters is None:
            bend_diameters = 5.0
        rc = Di*bend_diameters

    radius_ratio = rc/Di

    if L_unimpeded is None:
        # Assumption - smooth outlet
        L_unimpeded = 20.0*Di

    # Graph is defined for angles 10 to 180 degrees, ratios 0.5 to 10
    if radius_ratio < 0.5:
        radius_ratio = 0.5
    if radius_ratio > 10.0:
        radius_ratio = 10.0
    if angle < 10.0:
        angle = 10.0

    # Curve fit in terms of degrees
    # Caching could work here - angle, radius ratio does not change often
    Kb = bend_rounded_Miller_Kb(radius_ratio, angle)

    C_roughness = Miller_bend_roughness_correction(Re=Re, Di=Di,
                                                   roughness=roughness)
    '''Section 9.2.2 - Reynolds Number Correction
    Allow some extrapolation up to 1E8 (1E7 max in graph but the trend looks good)
    '''
    Re_C_Re = min(max(Re, 1E4), 1E8)
    if radius_ratio >= 2.0:
        if Re_C_Re == 1E8:
            C_Re = 0.4196741237602154 # bend_rounded_Miller_C_Re(1e8, 2.0)
        elif Re_C_Re == 1E4:
            C_Re = 2.1775876405173977 # bend_rounded_Miller_C_Re(1e4, 2.0)
        else:
            C_Re = bend_rounded_Miller_C_Re(Re_C_Re, 2.0)
    elif radius_ratio <= 1.0:
        # newton(lambda x: bend_rounded_Miller_C_Re(x, 1.0)-1, 2e5) to get the boundary value
        C_Re_1 = bend_rounded_Miller_C_Re(Re_C_Re, 1.0) if Re_C_Re < 207956.58904584477 else 1.0
        if radius_ratio > 0.7 or Kb < 0.4:
            C_Re = C_Re_1
        else:
            C_Re = Kb/(Kb - 0.2*C_Re_1 + 0.2)
            if C_Re > 2.2 or C_Re < 0:
                C_Re = 2.2
    else:
        # regardless of ratio - 1
        if Re_C_Re > 1048884.4656835075:
            C_Re = 1.0
        elif Re_C_Re > horner(bend_rounded_Miller_C_Re_limit_1, radius_ratio):
            C_Re = 1.0
#            ps = np.linspace(1, 2)
#            qs = [secant(lambda x: bend_rounded_Miller_C_Re(x, i)-1, 2e5) for i in ps]
#            np.polyfit(ps, qs, 4).tolist()
            # Line of C_Re=1 as a function of r_d between 0 and 1
        else:
            C_Re = bend_rounded_Miller_C_Re(Re_C_Re, radius_ratio)
    C_o =  Miller_bend_unimpeded_correction(Kb=Kb, Di=Di, L_unimpeded=L_unimpeded)

#    print('Kb=%g, C Re=%g, C rough =%g, Co=%g' %(Kb, C_Re, C_roughness, C_o))
    return Kb*C_Re*C_roughness*C_o


bend_rounded_Crane_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0,
                             14.0, 16.0, 20.0]
bend_rounded_Crane_fds = [20.0, 14.0, 12.0, 12.0, 14.0, 17.0, 24.0, 30.0, 34.0,
                          38.0, 42.0, 50.0]

bend_rounded_Crane_coeffs = [111.75011378177442, -331.89911345404107, -27.841951521656483,
                             1066.8916917931147, -857.8702190626232, -1151.4621655498092,
                             1775.2416673594603, 216.37911821941805, -1458.1661519377653,
                             447.169127650163, 515.361158769082, -322.58377486107577,
                             -38.38349416327068, 71.12796602489138, -16.198233745350535,
                             19.377150177339015, 31.107110520349494]


def bend_rounded_Crane(Di, angle, rc=None, bend_diameters=None):
    r'''Calculates the loss coefficient for any rounded bend in a pipe
    according to the Crane TP 410M [1]_ method. This method effectively uses
    an interpolation from tabulated values in [1]_ for friction factor
    multipliers vs. curvature radius.

    .. figure:: fittings/bend_rounded.png
       :scale: 30 %
       :alt: rounded bend; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    angle : float
        Angle of bend, [degrees]
    rc : float, optional
        Radius of curvature of the entrance; specify either `rc` or
        `bend_diameters`, optional [m]
    bend_diameters : float, optional
        Number of diameters of pipe making up the bend radius; specify either
        `rc` or `bend_diameters`, [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    The Crane method does match the trend of increased pressure drop as
    roughness increases.

    The points in [1]_ are extrapolated to other angles via a well-fitting
    Chebyshev approximation, whose accuracy can be seen in the below plot.

    .. plot:: plots/bend_rounded_Crane_plot.py

    Examples
    --------
    >>> bend_rounded_Crane(Di=.4020, rc=.4*5, angle=30)
    0.09321910015613409

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if (rc is not None and bend_diameters is not None): # numba: delete
        if abs(Di*bend_diameters/rc - 1.0) > 1e-12: # numba: delete
            raise ValueError("Cannot specify both `rc` and `bend_diameters`") # numba: delete
    if rc is None:
        if bend_diameters is None:
            bend_diameters = 5.0
        rc = Di*bend_diameters
    fd = ft_Crane(Di)

    radius_ratio = rc/Di
    if radius_ratio < 1.0:
        radius_ratio = 1.0
    elif radius_ratio > 20.0:
        radius_ratio = 20.0
    factor = horner(bend_rounded_Crane_coeffs, 0.105263157894736836*(radius_ratio - 10.5))
    K = fd*factor
    K = (angle/90.0 - 1.0)*(0.25*pi*fd*radius_ratio + 0.5*K) + K
    return K


_Ito_angles = [45.0, 90.0, 180.0]
def bend_rounded_Ito(Di, angle, Re, rc=None, bend_diameters=None,
                     roughness=0.0):
    """Ito method as shown in Blevins.

    Curved friction factor as given in Blevins, with minor tweaks to be more
    accurate to the original methods.
    """
    if not rc:
        if bend_diameters is None:
            bend_diameters = 5.0
        rc = Di*bend_diameters

    radius_ratio = rc/Di
    angle_rad = radians(angle)
    De2 = Re*(Di/rc)**2.0
    if rc > 50.0*Di:
        alpha = 1.0
    else:
        # Alpha is up to 6, as ratio gets higher, can go down to 1
        alpha_45 = 1.0 + 5.13*(Di/rc)**1.47
        alpha_90 = 0.95 + 4.42*(Di/rc)**1.96 if rc/Di < 9.85 else 1.0
        alpha_180 = 1.0 + 5.06*(Di/rc)**4.52
        alpha = interp(angle, _Ito_angles, [alpha_45, alpha_90, alpha_180])

    if De2 <= 360.0:
        fc = friction_factor_curved(Re=Re, Di=Di, Dc=2.0*rc,
                                    roughness=roughness,
                                    Rec_method='Srinivasan',
                                    laminar_method='White',
                                    turbulent_method='Srinivasan turbulent')
        K = 0.0175*alpha*fc*angle*rc/Di
    else:
        K = 0.00431*alpha*angle*Re**-0.17*(rc/Di)**0.84
    return K

crane_standard_bend_angles = [45.0, 90.0, 180.0]
crane_standard_bend_losses = [16.0, 30.0, 50.0]

bend_rounded_methods = ['Rennels', 'Crane', 'Crane standard', 'Miller', 'Swamee', 'Ito']
bend_rounded_method_unknown = 'Specified method not recognized; methods are %s' %(bend_rounded_methods)

def bend_rounded(Di, angle, fd=None, rc=None, bend_diameters=None,
                 Re=None, roughness=0.0, L_unimpeded=None, method='Rennels'):
    r'''Returns loss coefficient for rounded bend in a pipe of diameter `Di`,
    `angle`, with a specified either radius of curvature `rc` or curvature
    defined by `bend_diameters`, Reynolds number `Re` and optionally pipe
    roughness, unimpeded length downstrean, and with the specified method.
    This calculation has six methods available.

    It is hard to describe one method as more conservative than another as
    depending on the conditions, the relative results change significantly.

    The 'Miller' method is the most complicated and slowest method; the 'Ito'
    method comprehensive as well and a source of original data, and the primary
    basis for the 'Rennels' method. The 'Swamee' method is very simple and
    generally does not match the other methods. The 'Crane' method may match
    or not match other methods depending on the inputs. There is also a
    'Crane standard' method for use with threaded fittings which have higher
    pressure drops. It is a linear interpolation of values at angles of
    45, 90, and 180 degrees.

    The Rennels [1]_ formula is:

    .. math::
        K = f\alpha\frac{r}{d} + (0.10 + 2.4f)\sin(\alpha/2)
        + \frac{6.6f(\sqrt{\sin(\alpha/2)}+\sin(\alpha/2))}
        {(r/d)^{\frac{4\alpha}{\pi}}}

    The Swamee [5]_ formula is:

    .. math::
        K = \left[0.0733 + 0.923 \left(\frac{d}{rc}\right)^{3.5} \right]
        \theta^{0.5}

    .. figure:: fittings/bend_rounded.png
       :scale: 30 %
       :alt: rounded bend; after [1]_

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    angle : float
        Angle of bend, [degrees]
    fd : float, optional
        Darcy friction factor; used only in Rennels method; calculated if not
        provided from Reynolds number, diameter, and roughness [-]
    rc : float, optional
        Radius of curvature of the entrance, optional [m]
    bend_diameters : float, optional (used if rc not provided)
        Number of diameters of pipe making up the bend radius [-]
    Re : float, optional
        Reynolds number of the pipe (used in Miller, Ito methods primarily, and
        Rennels method if no friction factor given), [-]
    roughness : float, optional
        Roughness of bend wall (used in Miller, Ito methods primarily, and
        Rennels method if no friction factor given), [m]
    L_unimpeded : float, optional
        The length of unimpeded pipe without any fittings, instrumentation,
        or flow disturbances downstream (assumed 20 diameters if not
        specified); used only in Miller method, [m]
    method : str, optional
        One of 'Rennels', 'Miller', 'Crane', 'Crane standard', 'Ito', or
        'Swamee', [-]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    When inputting bend diameters, note that manufacturers often specify
    this as a multiplier of nominal diameter, which is different than actual
    diameter. Those require that rc be specified.

    In the 'Rennels' method, `rc` is limited to 0.5 or above; which represents
    a sharp, square, inner edge - and an outer bend radius of 1.0. Losses are
    at a minimum when this value is large. Its first term represents surface
    friction loss; the second, secondary flows; and the third, flow separation.
    It encompasses the entire range of elbow and pipe bend configurations.
    It was developed for bend angles between 0 and 180 degrees; and r/D
    ratios above 0.5. Only smooth pipe data was used in its development.
    Note the loss coefficient includes the surface friction of the pipe as if
    it was straight.

    Examples
    --------
    >>> bend_rounded(Di=4.020, rc=4.0*5, angle=30, Re=1E5)
    0.11519070808085191

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [3] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [4] Swamee, Prabhata K., and Ashok K. Sharma. Design of Water Supply
       Pipe Networks. John Wiley & Sons, 2008.
    .. [5] Itō, H."Pressure Losses in Smooth Pipe Bends." Journal of Fluids
       Engineering 82, no. 1 (March 1, 1960): 131-40. doi:10.1115/1.3662501
    .. [6] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    if method is None:
        method = 'Rennels'
    if bend_diameters is None and rc is None:
        bend_diameters = 5.0
    if rc is None:
        rc = Di*bend_diameters

    if method == 'Rennels':
        angle = radians(angle)
        if fd is None:
            if Re is None:
                raise ValueError("The `Rennels` method requires either a "
                                 "specified friction factor or `Re`")
            fd = Clamond(Re=Re, eD=roughness/Di, fast=False)
        sin_term = sin(0.5*angle)
        return (fd*angle*rc/Di + (0.10 + 2.4*fd)*sin_term
        + 6.6*fd*(sqrt(sin_term) + sin_term)/(rc/Di)**(4.*angle/pi))
    elif method == 'Miller':
        if Re is None:
            raise ValueError('Miller method requires Reynolds number')
        return bend_rounded_Miller(Di=Di, angle=angle, Re=Re, rc=rc,
                                   bend_diameters=bend_diameters,
                                   roughness=roughness,
                                   L_unimpeded=L_unimpeded)
    elif method == 'Crane':
        return bend_rounded_Crane(Di=Di, angle=angle, rc=rc,
                                  bend_diameters=bend_diameters)
    elif method == 'Crane standard':
        return ft_Crane(Di)*interp(angle, crane_standard_bend_angles, crane_standard_bend_losses, extrapolate=True)
    elif method == 'Ito':
        if Re is None:
            raise ValueError("The `Iso` method requires`Re`")
        return bend_rounded_Ito(Di=Di, angle=angle, Re=Re, rc=rc, bend_diameters=bend_diameters,
                     roughness=roughness)
    elif method == 'Swamee':
        return (0.0733 + 0.923*(Di/rc)**3.5)*sqrt(radians(angle))
    else:
        raise ValueError(bend_rounded_method_unknown)


bend_miter_Miller_coeffs = [-12.050299402650126, -4.472433689233185, 50.51478860493546, 18.246302079077196,
                            -84.61426660754049, -28.9340865412371, 71.07345367553872, 21.354010992349565,
                            -30.239604839338, -5.855129345095336, 5.465131779316523, -1.0881363712712555,
                            -0.3635431075401224, 0.5120065303391261, 0.46818214491579246, 0.9789177645343993,
                            0.5080285124448385]

def bend_miter_Miller(Di, angle, Re, roughness=0.0, L_unimpeded=None):
    r'''Calculates the loss coefficient for a single miter bend according to
    Miller [1]_. This is a sophisticated model which uses corrections for
    pipe roughness, the length of the pipe downstream before another
    interruption, and a correction for Reynolds number. It interpolates several
    times using several corrections graphs in [1]_.

    Parameters
    ----------
    Di : float
        Inside diameter of pipe, [m]
    angle : float
        Angle of miter bend, [degrees]
    Re : float
        Reynolds number of the pipe (no specification if inlet or outlet
        properties should be used), [m]
    roughness : float, optional
        Roughness of bend wall, [m]
    L_unimpeded : float, optional
        The length of unimpeded pipe without any fittings, instrumentation,
        or flow disturbances downstream (assumed 20 diameters if not
        specified), [m]

    Returns
    -------
    K : float
        Loss coefficient [-]

    Notes
    -----
    Note the loss coefficient includes the surface friction of the pipe as if
    it was straight.

    Examples
    --------
    >>> bend_miter_Miller(Di=.6, angle=90, Re=2e6, roughness=2e-5,
    ... L_unimpeded=30*.6)
    1.1921574594947664

    References
    ----------
    .. [1] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    '''
    if L_unimpeded is None:
        L_unimpeded = 20.0*Di
    if angle > 120.0:
        angle = 120.0

    Kb = horner(bend_miter_Miller_coeffs, 1.0/60.0*(angle-60.0))

    C_o =  Miller_bend_unimpeded_correction(Kb=Kb, Di=Di, L_unimpeded=L_unimpeded)
    C_roughness = Miller_bend_roughness_correction(Re=Re, Di=Di,
                                                   roughness=roughness)
    Re_C_Re = min(max(Re, 1E4), 1E8)
    C_Re_1 = bend_rounded_Miller_C_Re(Re_C_Re, 1.0) if Re_C_Re < 207956.58904584477 else 1.0
    C_Re = Kb/(Kb - 0.2*C_Re_1 + 0.2)
    if C_Re > 2.2 or C_Re < 0:
        C_Re = 2.2
    return Kb*C_Re*C_roughness*C_o


bend_miter_Crane_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
bend_miter_Crane_fds = [2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0]

bend_miter_Blevins_angles = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 120.0]
bend_miter_Blevins_Ks = [0.0, .025, .055, .1, .2, .35, .5, .7, .9, 1.1, 1.5]

bend_miter_methods = ['Rennels', 'Miller', 'Crane', 'Blevins']
bend_miter_method_unknown_msg = 'Specified method not recognized; methods are %s' %(bend_miter_methods)

def bend_miter(angle, Di=None, Re=None, roughness=0.0, L_unimpeded=None,
               method='Rennels'):
    r'''Returns loss coefficient for any single-joint miter bend in a pipe
    of angle `angle`, diameter `Di`, Reynolds number `Re`, roughness
    `roughness` unimpeded downstream length `L_unimpeded`, and using the
    specified method. This calculation has four methods available.
    The 'Rennels' method is based on a formula and extends to angles up to
    150 degrees. The 'Crane' method extends only to 90 degrees; the 'Miller'
    and 'Blevins' methods extend to 120 degrees.

    The Rennels [1]_ formula is:

    .. math::
        K = 0.42\sin(\alpha/2) + 2.56\sin^3(\alpha/2)

    The 'Crane', 'Miller', and 'Blevins' methods are all in part graph or
    tabular based and do not have straightforward formulas.

    .. figure:: fittings/bend_miter.png
       :scale: 25 %
       :alt: Miter bend, one joint only; after [1]_

    Parameters
    ----------
    angle : float
        Angle of bend, [degrees]
    Di : float, optional
        Inside diameter of pipe, [m]
    Re : float, optional
        Reynolds number of the pipe (no specification if inlet or outlet
        properties should be used), [-]
    roughness : float, optional
        Roughness of bend wall, [m]
    L_unimpeded : float, optional
        The length of unimpeded pipe without any fittings, instrumentation,
        or flow disturbances downstream (assumed 20 diameters if not
        specified), [m]
    method : str, optional
        The specified method to use; one of 'Rennels', 'Miller', 'Crane',
        or 'Blevins', [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to either upstream or downstream
        diameter, [-]

    Notes
    -----
    This method is designed only for single-jointed miter bends. It is common
    for miter bends to have two or three sections, to further reduce the loss
    coefficient. Some methods exist in [2]_ for taking this into account.
    Because the additional configurations reduce the pressure loss, it is
    "common practice" to simply ignore their effect and accept the slight
    overdesign.

    The following figure illustrates the different methods.

    .. plot:: plots/bend_miter_plot.py

    Examples
    --------
    >>> bend_miter(150)
    2.7128147734758103
    >>> bend_miter(Di=.6, angle=45, Re=1e6, roughness=1e-5, L_unimpeded=20,
    ... method='Miller')
    0.2944060416245169

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [3] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [4] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    '''
    if method is None:
        method = 'Rennels'
    if method == 'Rennels':
        angle_rad = radians(angle)
        sin_half_angle = sin(angle_rad*0.5)
        return 0.42*sin_half_angle + 2.56*sin_half_angle*sin_half_angle*sin_half_angle
    elif method == 'Crane':
        factor = interp(angle, bend_miter_Crane_angles, bend_miter_Crane_fds)
        return ft_Crane(Di)*factor
    elif method == 'Miller':
        return bend_miter_Miller(Di=Di, angle=angle, Re=Re, roughness=roughness, L_unimpeded=L_unimpeded)
    elif method == 'Blevins':
        # data from Idelchik, Miller, an earlier ASME publication
        # For 90-120 degrees, a polynomial/spline would be better than a linear fit
        K_base = interp(angle, bend_miter_Blevins_angles, bend_miter_Blevins_Ks)
        return K_base*(2E5/Re)**0.2
    else:
        raise ValueError(bend_miter_method_unknown_msg)


def helix(Di, rs, pitch, N, fd):
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
    return N*(fd*sqrt((2*pi*rs)**2 + pitch**2)/Di + 0.20 + 4.8*fd)


def spiral(Di, rmax, rmin, pitch, fd):
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
    rmin : float
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
    return (rmax-rmin)/pitch*(fd*pi*(rmax+rmin)/Di + 0.20 + 4.8*fd) + 13.2*fd/(rmin/Di)**2

### Contractions

tck_contraction_abrupt_Miller = implementation_optimize_tck([
  [0.0, 0.0, 0.0, 0.0, 0.5553844358576507, 0.7193937784550933, 0.8144518359319883, 1.0, 1.0, 1.0, 1.0],
  [0.0, 0.0, 0.0, 0.0, 0.008318525134414716, 0.03421785904690331, 0.1, 0.1, 0.1, 0.1],
  [0.4994829280256306, 0.4879234090312588, 0.4255534701302917, 0.13986792857000196, 0.18065199312360336,
            0.08701863105570044, 0.440886271558411, 0.4243716649409474, 0.36030826702480984, 0.2117960027770777,
            0.11248601502220595, 0.08616608643911047, 0.4018850813314268, 0.3706136100344715, 0.26368725187530173,
            0.15316562777200723, 0.09856904494833027, 0.08399367477431015, 0.17005190739488515, 0.16023910724406945,
            0.1242906181281536, 0.06137573180850665, 0.05726821990215439, 0.04684229988854647, 0.03922553704852396,
            0.036955938945600654, 0.029450340285188167, 0.028656302938315878, 0.019588760093397686, 0.01950497484044149,
            0.006447273360860872, 0.006569278508667471, 0.0053786079483153885, -0.013158950566037957,
            0.010870991979047888, 0.0015100946100218284, -0.0005221250682760256, -0.0006447517875307877,
            -0.0007846123907797336, 0.0024459067063225485, -0.0019102888752274472, -0.0001356300464508266],
          3, 3])


def contraction_round_Miller(Di1, Di2, rc):
    r'''Returns loss coefficient for any round edged pipe contraction
    using the method of Miller [1]_. This method uses a spline fit to a graph
    with area ratios 0 to 1, and radius ratios (rc/Di2) from 0.1 to 0.

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]
    rc : float
        Radius of curvature of the contraction, [m]

    Returns
    -------
    K : float
        Loss coefficient in terms of the following pipe, [-]

    Notes
    -----
    This method normally gives lower losses than the Rennels formulation.

    Examples
    --------
    >>> contraction_round_Miller(Di1=1, Di2=0.4, rc=0.04)
    0.08565953051298639

    References
    ----------
    .. [1] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    '''
    A_ratio = Di2*Di2/(Di1*Di1)
    radius_ratio = rc/Di2
    if radius_ratio > 0.1:
        radius_ratio = 0.1
    Ks = float(bisplev(A_ratio, radius_ratio, tck_contraction_abrupt_Miller))
    # For some near-1 ratios, can get negative Ks due to the spline.
    if Ks < 0.0:
        Ks = 0.0
    return Ks


contraction_sharp_methods = ['Rennels', 'Hooper', 'Crane']
contraction_sharp_method_unknown = 'Specified method not recognized; methods are %s' %(contraction_sharp_methods)

def contraction_sharp(Di1, Di2, fd=None, Re=None, roughness=0.0,
                      method='Rennels'):
    r'''Returns loss coefficient for a sharp edged pipe contraction.

    This calculation has two methods available. The 'Rennels' [2]_ method is a
    fit for turbulent regimes, while the `Hooper` method is more complicated
    and claims to have full dependence on `Re` including a laminar transition
    at `Re` of 2500 (based on the original pipe diameter).

    The Rennels [1]_ formulas are:

    .. math::
        K_1 = 0.0696(1-\beta^5)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622(1-0.215\beta^2 -  0.785\beta^5)

    .. math::
        \beta = d_2/d_1

    The Hooper [1]_ formulas are:

    If :math:`{Re}_1 \le 2500`:

    .. math::
        K_1 = \left[1.2 + \frac{160}{\text{Re}_1}\right]
        \left[ \left(\frac{D_1} {D_2} \right)^4 -1 \right]

    If :math:`{Re}_1 > 2500`:

    .. math::
        K_1 = \left[0.6 + 0.48f_1\right]  \left(\frac{D_1} {D_2} \right)^2
        \left[ \left(\frac{D_1} {D_2} \right)^2 -1 \right]

    Converting the loss coefficient to a consistent basis:

    .. math::
        K_2 = K_1\frac{D_2^4}{D_1^4}

    For the Crane formula see `contraction_conical_Crane` with a length of zero.

    .. figure:: fittings/contraction_sharp.png
       :scale: 40 %
       :alt: Sharp contraction

    Parameters
    ----------
    Di1 : float
        Inside diameter of original (larger) pipe, [m]
    Di2 : float
        Inside diameter of following (smaller) pipe, [m]
    fd : float, optional
        Darcy friction factor in original pipe; used only in the Hooper method
        and will be  calculated from `Re` if not given, [-]
    Re : float, optional
        Reynolds number of the pipe (used in Hooper method, [m]
    roughness : float, optional
        Roughness of original pipe (used in Hooper method only if no friction
        factor given), [m]
    method : str
        The calculation method to use; one of 'Hooper', 'Rennels', or 'Crane' [-]

    Returns
    -------
    K : float
        Loss coefficient in terms of the following pipe [-]

    Notes
    -----
    A value of 0.506 or simply 0.5 is often used.

    Examples
    --------
    >>> contraction_sharp(Di1=1, Di2=0.4)
    0.5301269161
    >>> contraction_sharp(Di1=1, Di2=0.4, Re=1e5, method='Hooper')
    0.5112534765

    The Hooper method supports laminar flow, while `Rennels` is not even `Re`
    aware.

    >>> contraction_sharp(Di1=1, Di2=0.4, Re=1e3, method='Hooper')
    1.325184

    Crane offers similar results:

    >>> contraction_sharp(3.0, 2.0, method='Crane')
    0.2777777

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Hooper, William B. "Calculate Head Loss Caused by Change in Pipe
       Size." Chemical Engineering 95, no. 16 (November 7, 1988): 89.
    '''
    if method == 'Rennels':
        beta = Di2/Di1
        beta2 = beta*beta
        beta5 = beta2*beta2*beta
        lbd = 1.0 + 0.622*(1.0 - 0.215*beta2 - 0.785*beta5)
        return 0.0696*(1.0 - beta5)*lbd*lbd + (lbd - 1.0)*(lbd - 1.0)
    elif method == 'Hooper':
        if Re is None:
            raise ValueError("Hooper method requires `Re`")
        D1_D2 = Di1/Di2
        D1_D2_2 = D1_D2*D1_D2
        if Re <= 2500.0:
            K = (1.2 + 160.0/Re)*(D1_D2_2*D1_D2_2 - 1.0)
        else:
            if fd is None:
                fd = Clamond(Re=Re, eD=roughness/Di1)
            K = (0.6 + 0.48*fd)*D1_D2_2*(D1_D2_2 - 1.0)
        K = change_K_basis(K, Di1, Di2)
        return K
    elif method == 'Crane':
        return contraction_conical_Crane(Di1, Di2, l=0.0)
    else:
        raise ValueError(contraction_sharp_method_unknown)

contraction_round_Idelchik_ratios = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                     0.08, 0.12, 0.16, 0.2]
contraction_round_Idelchik_factors = [0.5, 0.43, 0.37, 0.31, 0.26, 0.22, 0.20,
                                      0.15, 0.09, 0.06, 0.03]
# Third factor is 0.36 in 1960 edition, 0.37 in Design Guide

contraction_round_methods = ['Rennels', 'Miller', 'Idelchik']
contraction_round_unknown_method = 'Specified method not recognized; methods are %s' %(contraction_round_methods)

def contraction_round(Di1, Di2, rc, method='Rennels'):
    r'''Returns loss coefficient for any any round edged pipe contraction.
    This calculation has three methods available. The 'Miller' [2]_ method is a
    bivariate spline digitization of a graph; the 'Idelchik' [3]_ method is an
    interpolation using a formula and a table of values.

    The most conservative formulation is that of Rennels; with fairly similar.
    The 'Idelchik' method is more conservative and less complex; it offers a
    straight-line curve where the others curves are curved.

    The Rennels [1]_ formulas are:

    .. math::
        K = 0.0696\left(1 - 0.569\frac{r}{d_2}\right)\left(1-\sqrt{\frac{r}
        {d_2}}\beta\right)(1-\beta^5)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622\left(1 - 0.30\sqrt{\frac{r}{d_2}}
        - 0.70\frac{r}{d_2}\right)^4 (1-0.215\beta^2-0.785\beta^5)

    .. math::
        \beta = d_2/d_1

    .. figure:: fittings/contraction_round.png
       :scale: 30 %
       :alt: Circular round contraction; after [1]_

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe, [m]
    Di2 : float
        Inside diameter of following pipe, [m]
    rc : float
        Radius of curvature of the contraction, [m]
    method : str
        The calculation method to use; one of 'Rennels', 'Miller', or
        'Idelchik', [-]

    Returns
    -------
    K : float
        Loss coefficient in terms of the following pipe [-]

    Notes
    -----
    Rounding radius larger than 0.14Di2 prevents flow separation from the wall.
    Further increase in rounding radius continues to reduce loss coefficient.

    .. plot:: plots/contraction_round_plot.py

    Examples
    --------
    >>> contraction_round(Di1=1, Di2=0.4, rc=0.04)
    0.1783332490866574

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [3] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    '''
    beta = Di2/Di1
    if method is None:
        method = 'Rennels'
    if method == 'Rennels':
        lbd = 1.0 + 0.622*(1.0 - 0.30*sqrt(rc/Di2) - 0.70*rc/Di2)**4*(1.0 - 0.215*beta**2 - 0.785*beta**5)
        return 0.0696*(1.0 - 0.569*rc/Di2)*(1.0 - sqrt(rc/Di2)*beta)*(1.0 - beta**5)*lbd*lbd + (lbd - 1.0)**2
    elif method == 'Miller':
        return contraction_round_Miller(Di1=Di1, Di2=Di2, rc=rc)
    elif method == 'Idelchik':
        # Di2, ratio defined in terms over diameter
        K0 = interp(rc/Di2, contraction_round_Idelchik_ratios,
                    contraction_round_Idelchik_factors)
        return K0*(1.0 - beta*beta)
    else:
        raise ValueError(contraction_round_unknown_method)


def contraction_conical_Crane(Di1, Di2, l=None, angle=None):
    r'''Returns loss coefficient for a conical pipe contraction
    as shown in Crane TP 410M [1]_ between 0 and 180 degrees.

    If :math:`\theta < 45^{\circ}`:

    .. math::
        K_2 = {0.8 \sin \frac{\theta}{2}(1 - \beta^2)}

    otherwise:

    .. math::
        K_2 = {0.5\sqrt{\sin \frac{\theta}{2}} (1 - \beta^2)}

    .. math::
        \beta = d_2/d_1

    Parameters
    ----------
    Di1 : float
        Inside pipe diameter of the larger, upstream, pipe, [m]
    Di2 : float
        Inside pipe diameter of the smaller, downstream, pipe, [m]
    l : float, optional
        Length of the contraction [m]
    angle : float, optional
        Angle of contraction [degrees]

    Returns
    -------
    K : float
        Loss coefficient in terms of the following (smaller) pipe [-]

    Notes
    -----
    Cheap and has substantial impact on pressure drop. Note that the
    nomenclature in [1]_ is somewhat different - the smaller pipe is called 1,
    and the larger pipe is called 2; and so the beta ratio is reversed, and the
    fourth power of beta used in their equation is not necessary.

    Examples
    --------
    >>> contraction_conical_Crane(Di1=0.0779, Di2=0.0525, l=0)
    0.2729017979998056

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if l is not None:
        if l == 0.0:
            angle_rad = pi
        else:
            angle_rad = 2.0*atan((Di1-Di2)/(2.0*l))
    elif angle is not None:
        angle_rad = deg2rad*angle
        #l = (Di1 - Di2)/(2.0*tan(0.5*angle)) # L is not needed in this calculation
    else:
        raise ValueError('One of `l` or `angle` must be specified')
    beta = Di2/Di1
    beta2 = beta*beta
    if angle_rad < 0.25*pi:
        # Formula 1
        K2 = 0.8*sin(0.5*angle_rad)*(1.0 - beta2)
    else:
        # Formula 2
        K2 = 0.5*(sqrt(sin(0.5*angle_rad))*(1.0 - beta2))
    return K2


contraction_conical_angles_Idelchik = [2, 3, 6, 8, 10, 12, 14, 16, 20]
contraction_conical_A_ratios_Idelchik = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

contraction_conical_friction_Idelchik = [
    [0.14, 0.1, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01],
    [0.14, 0.1, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.14, 0.1, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.14, 0.1, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.14, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.14, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.13, 0.09, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01],
    [0.12, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01],
    [0.11, 0.07, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01],
    [0.09, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]]

contraction_conical_frction_Idelchik_tck = tck_interp2d_linear(contraction_conical_angles_Idelchik,
                                                               contraction_conical_A_ratios_Idelchik,
                                                               contraction_conical_friction_Idelchik,
                                                               kx=1, ky=1)
contraction_conical_frction_Idelchik_obj = lambda x, y : float(bisplev(x, y, contraction_conical_frction_Idelchik_tck))

contraction_conical_l_ratios_Blevins = [0.0, 0.05, 0.1, 0.15, 0.6]
contraction_conical_A_ratios_Blevins = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
contraction_conical_Ks_Blevins = [[.08, .06, .04, .03, .03],
                                  [.17, .12, .09, .07, .06],
                                  [.25, .23, .17, .14, .06],
                                  [.33, .31, .27, .23, .08],
                                  [.4, .38, .35, .31, .18],
                                  [.45, .45, .41, .39, .27]]
contraction_conical_Blevins_tck = tck_interp2d_linear(contraction_conical_l_ratios_Blevins,
                                                      contraction_conical_A_ratios_Blevins,
                                                      contraction_conical_Ks_Blevins, kx=1, ky=1)
contraction_conical_Blevins_obj = lambda x, y: float(bisplev(x, y, contraction_conical_Blevins_tck))


contraction_conical_Miller_tck = implementation_optimize_tck([
        [
        -2.2990613088204293, -2.2990613088204293, -2.2990613088204293, -2.2990613088204293, -1.9345621970869704,
        -1.404550366067981, -1.1205580332553446, -0.7202074014540876, -0.18305354619604816, 0.5791478950190209,
        1.2576636025381396, 2.2907351590368092, 2.2907351590368092, 2.2907351590368092, 2.2907351590368092],
    [
        0.09564194294666524, 0.09564194294666524, 0.17553288711543455, 0.263895293813645, 0.3890819147022019,
        0.46277323951998217, 0.5504296236707121, 0.7265657737596892, 1.0772357648098938, 1.2566022106161683,
        1.3896885941879062, 1.3896885941879062],
    [
        -0.019518693251672135, 0.04439613867473242, 0.11549650174721836, 0.21325506677861075, 0.268179723158688,
        0.31125301421509866, 0.38394595875289805, 0.4808287074532006, 0.5205981039085685, 0.5444079315893322,
        -0.016435668699253902, 0.036132755789022385, 0.09344296094392814, 0.18264727448046977, 0.23460506265914166,
        0.2772896726095435, 0.3475409775384636, 0.45339837219176454, 0.49766916609817535, 0.533981552804865,
        -0.006524265764454468, 0.024107195694715193, 0.05862956870028131, 0.12122104285943507, 0.17207312024278762,
        0.2175356288866053, 0.282297563080016, 0.3995008583081823, 0.4563724107887528, 0.5175856070810377,
        0.00971345082784277, 0.025981390544674948, 0.0438578322196561, 0.08103403101086341, 0.11351528283253318,
        0.16873088559958743, 0.2347695003589526, 0.3428907161435351, 0.42017998591926276, 0.49784770602295325,
        0.022572122504756167, 0.0277671279384801, 0.033512283408629495, 0.05470423531298454, 0.06485563480390757,
        0.10483763206962131, 0.1802208799223503, 0.29075723837012296, 0.35502824385155335, 0.4460106883062252,
        0.030312717163327077, 0.03080869253188484, 0.03583128286874324, 0.04627567520803308, 0.050501484562613955,
        0.05683263025468022, 0.12297253802915259, 0.2415222338797251, 0.3025777968736861, 0.3724407040165538,
        0.03115993727503623, 0.03443665864698284, 0.03574452046031886, 0.03995718256281492, 0.04759698369059247,
        0.050404788737262694, 0.052375330859925545, 0.1356057568743366, 0.20463667731329582, 0.26043914743762864,
        0.02844193432840707, 0.0219797618956514, 0.013352154001094038, 0.018393840217638825, 0.02448602185526976,
        0.038812331325140816, 0.0522197430071833, 0.057132169238281294, 0.06871138075102912, 0.09334527259294226,
        0.04089985439478869, 0.07148502476706058, 0.06750266344761692, 0.038560772865945815, 0.020172054809734774,
        0.01596047961326318, 0.033338955878272625, 0.058808731166289874, 0.055802602927507314, 0.025265841939291166,
        0.11200365568168691, 0.11945663812857424, 0.10673570013847415, 0.07758458179796549, 0.055266607234870514,
        0.03072901347153607, 0.025790727504652375, 0.037031664564632104, 0.0601306808668177, 0.07612350738135039,
        0.0964900248905913, 0.11088549072803407, 0.10778442024110846, 0.09386482850507959, 0.06940476627270852,
        0.04434507143623664, 0.03331958878624311, 0.01854072032522763, 0.027553821071285824, 0.045426686375783926],
    3, 1])

contraction_conical_Miller_obj = lambda l_r2, A_ratio: max(min(float(bisplev(log(l_r2), log(A_ratio), contraction_conical_Miller_tck)), .5), 0)

contraction_conical_methods = ['Rennels', 'Idelchik', 'Crane', 'Swamee',
                               'Blevins', 'Miller', 'Hooper']
contraction_conical_method_unknown = 'Specified method not recognized; methods are %s' %(contraction_conical_methods)

def contraction_conical(Di1, Di2, fd=None, l=None, angle=None,
                        Re=None, roughness=0.0, method='Rennels'):
    r'''Returns the loss coefficient for any conical pipe contraction.
    This calculation has five methods available. The 'Idelchik' [2]_ and
    'Blevins' [3]_ methods use interpolation among tables of values; 'Miller'
    uses a 2d spline representation of a graph; and the
    'Rennels' [1]_, 'Crane' [4]_, 'Swamee' [5]_ and 'Hooper' methods use
    formulas for their calculations.

    The 'Rennels' [1]_ formulas are:

    .. math::
        K_2 = K_{fr,2} + K_{conv,2}

    .. math::
        K_{fr,2} = \frac{f_d ({1 - \beta^4})}{8\sin(\theta/2)}

    .. math::
        K_{conv,2} = 0.0696[1+C_B(\sin(\alpha/2)-1)](1-\beta^5)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622(\alpha/180)^{0.8}(1-0.215\beta^2-0.785\beta^5)

    .. math::
        \beta = d_2/d_1

    The 'Swamee' [5]_ formula is:

    .. math::
        K = 0.315 \theta^{1/3}

    The Hooper [7]_ formulas are:

    If :math:`{Re}_1 \le 2500`:

    .. math::
        K_{1,sharp} = \left[1.2 + \frac{160}{\text{Re}_1}\right]
        \left[ \left(\frac{D_1} {D_2} \right)^4 -1 \right]

    If :math:`{Re}_1 > 2500`:

    .. math::
        K_{1,sharp} = \left[0.6 + 0.48f_1\right]  \left(\frac{D_1} {D_2} \right)^2
        \left[ \left(\frac{D_1} {D_2} \right)^2 -1 \right]

    In both cases, a multiplier is added for the angle:

    For angles between 45 and 180 degrees:

    .. math::
        K_1 = K_{1,sharp} \sqrt{\sin \frac{\theta}{2}}

    For angles between 0 and 45 degrees:

    .. math::
        K_1 = K_{1,sharp} 1.6 \sin \frac{\theta}{2}

    Converting the Hooper loss coefficient to a consistent basis:

    .. math::
        K_2 = K_1\frac{D_2^4}{D_1^4}

    .. figure:: fittings/contraction_conical.png
       :scale: 30 %
       :alt: contraction conical; after [1]_

    Parameters
    ----------
    Di1 : float
        Inside pipe diameter of the larger, upstream, pipe, [m]
    Di2 : float
        Inside pipe diameter of the smaller, downstream, pipe, [m]
    fd : float, optional
        Darcy friction factor; used only in the `Rennels` and `Hooper` method
        and will be calculated from `Re` and `roughness` if not given, [-]
    l : float, optional
        Length of the contraction, optional [m]
    angle : float, optional
        Angle of contraction (180 = sharp, 0 = infinitely long contraction),
        optional [degrees]
    Re : float, optional
        Reynolds number of the pipe (used in `Rennels` and `Hooper` method only
        if no friction factor given), [m]
    roughness : float, optional
        Roughness of bend wall (used in Rennel method if no friction factor
        given), [m]
    method : str, optional
        The method to use for the calculation; one of 'Rennels', 'Idelchik',
        'Crane', 'Swamee' 'Hooper', or 'Blevins', [-]

    Returns
    -------
    K : float
        Loss coefficient in terms of the following pipe [-]

    Notes
    -----
    Cheap and has substantial impact on pressure drop.

    The 'Idelchik' method includes two tabular interpolations; its friction
    term is limited to angles between 2 and 20 degrees and area ratios 0.05 to
    0.6, while its main term is limited to length over diameter ratios 0.025 to
    0.6. This seems to give it high results for angles < 25 degrees.

    The 'Blevins' method is based on Idelchik data; it should not be used,
    because its data jumps around and its data is limited to area ratios .1 to
    0.83, and length over diameter ratios 0 to 0.6. The 'Miller' method jumps
    around as well. Unlike most of Miller's method, there is no correction for
    Reynolds number.

    There is quite a bit of variance in the predictions of the methods, as
    demonstrated by the following figure.

    .. plot:: plots/contraction_conical_plot.py

    Examples
    --------
    >>> contraction_conical(Di1=0.1, Di2=0.04, l=0.04, Re=1E6)
    0.15639885880609544

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    .. [3] Blevins, Robert D. Applied Fluid Dynamics Handbook. New York, N.Y.:
       Van Nostrand Reinhold Co., 1984.
    .. [4] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [5] Swamee, Prabhata K., and Ashok K. Sharma. Design of Water Supply
       Pipe Networks. John Wiley & Sons, 2008.
    .. [6] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [7] Hooper, William B. "Calculate Head Loss Caused by Change in Pipe
       Size." Chemical Engineering 95, no. 16 (November 7, 1988): 89.
    '''
    beta = Di2/Di1
    if angle is not None:
        angle_rad = angle*deg2rad
        l = (Di1 - Di2)/(2.0*tan(0.5*angle_rad))
    elif l is not None:
        if l != 0.0:
            angle_rad = 2.0*atan((Di1-Di2)/(2.0*l))
        else:
            angle_rad = pi
    else:
        raise ValueError('Either l or angle is required')
    if method == 'Rennels':
        if fd is None:
            if Re is None:
                raise ValueError("The `Rennels` method requires either a "
                                 "specified friction factor or `Re`")
            fd = Clamond(Re=Re, eD=roughness/Di2, fast=False)

        beta2 = beta*beta
        beta4 = beta2*beta2
        beta5 = beta4*beta
        lbd = 1.0 + 0.622*(angle_rad/pi)**0.8*(1.0 - 0.215*beta2 - 0.785*beta5)
        sin_half_angle = sin(0.5*angle_rad)
        K_fr2 = fd*(1.0 - beta4)/(8.0*sin_half_angle)
        K_conv2 = 0.0696*sin_half_angle*(1.0 - beta5)*lbd*lbd + (lbd - 1.0)**2
        return K_fr2 + K_conv2
    elif method == 'Crane':
        return contraction_conical_Crane(Di1=Di1, Di2=Di2, l=l, angle=angle_rad*rad2deg)
    elif method == 'Swamee':
        return 0.315*angle_rad**(1.0/3.0)
    elif method == 'Idelchik':
        # Diagram 3-6; already digitized for beveled entrance
        K0 = float(bisplev(angle_rad*rad2deg, l/Di2, entrance_beveled_Idelchik_tck))

        # Angles 0 to 20, ratios 0.05 to 0.06
        if angle_rad > 20.0*deg2rad:
            angle_fric = 20.0
        elif angle_rad < 2.0*deg2rad:
            angle_fric = 2.0
        else:
            angle_fric = angle_rad*rad2deg

        A_ratio = A_ratio_fric = Di2*Di2/(Di1*Di1)
        if A_ratio_fric < 0.05:
            A_ratio_fric = 0.05
        elif A_ratio_fric > 0.6:
            A_ratio_fric = 0.6

        K_fr = float(contraction_conical_frction_Idelchik_obj(angle_fric, A_ratio_fric))
        return K0*(1.0 - A_ratio) + K_fr
    elif method == 'Blevins':
        A_ratio = Di1*Di1/(Di2*Di2)
        if A_ratio < 1.2:
            A_ratio = 1.2
        elif A_ratio > 10.0:
            A_ratio = 10.0

        l_ratio = l/Di2
        if l_ratio > 0.6:
            l_ratio = 0.6
        return float(contraction_conical_Blevins_obj(l_ratio, A_ratio))
    elif method == 'Miller':
        A_ratio = Di1*Di1/(Di2*Di2)
        if A_ratio > 4.0:
            A_ratio = 4.0
        elif A_ratio < 1.1:
            A_ratio = 1.1
        l_ratio = l/(Di2*0.5)
        if l_ratio < 0.1:
            l_ratio = 0.1
        elif l_ratio > 10.0:
            l_ratio = 10.0
        # Turning on ofr off the limits - little difference in plot
        return contraction_conical_Miller_obj(l_ratio, A_ratio)
    elif method == 'Hooper':
        if Re is None:
            raise ValueError("Hooper method requires `Re`")
        D1_D2 = Di1/Di2
        D1_D2_2 = D1_D2*D1_D2
        if Re <= 2500.0:
            K = (1.2 + 160.0/Re)*(D1_D2_2*D1_D2_2 - 1.0)
        else:
            if fd is None:
                fd = Clamond(Re=Re, eD=roughness/Di1)
            K = (0.6 + 0.48*fd)*D1_D2_2*(D1_D2_2 - 1.0)

        if angle_rad > 0.25*pi:
            K *= sqrt(sin(0.5*angle_rad))
        else:
            K *= 1.6*sin(0.5*angle_rad)
        K = change_K_basis(K, Di1, Di2)
        return K
    else:
        raise ValueError(contraction_conical_method_unknown)


def contraction_beveled(Di1, Di2, l=None, angle=None):
    r'''Returns loss coefficient for any sharp beveled pipe contraction
    as shown in [1]_.

    .. math::
        K = 0.0696[1+C_B(\sin(\alpha/2)-1)](1-\beta^5)\lambda^2 + (\lambda-1)^2

    .. math::
        \lambda = 1 + 0.622\left[1+C_B\left(\left(\frac{\alpha}{180}
        \right)^{0.8}-1\right)\right](1-0.215\beta^2-0.785\beta^5)

    .. math::
        C_B = \frac{l}{d_2}\frac{2\beta\tan(\alpha/2)}{1-\beta}

    .. math::
        \beta = d_2/d_1

    .. figure:: fittings/contraction_beveled.png
       :scale: 30 %
       :alt: contraction beveled; after [1]_

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
        Loss coefficient in terms of the following pipe [-]

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
    angle = radians(angle)
    beta = Di2/Di1
    CB = l/Di2*2.0*beta*tan(0.5*angle)/(1.0 - beta)
    beta2 = beta*beta
    beta5 = beta2*beta2*beta
    lbd = 1.0 + 0.622*(1.0 + CB*((angle/pi)**0.8 - 1.0))*(1.0 - 0.215*beta2 - 0.785*beta5)
    return 0.0696*(1.0 + CB*(sin(0.5*angle) - 1.0))*(1.0 - beta5)*lbd*lbd + (lbd-1.0)**2

### Expansions (diffusers)

diffuser_sharp_methods = ['Rennels', 'Hooper']
diffuser_sharp_method_unknown = 'Specified method not recognized; methods are %s' %(diffuser_sharp_methods)

def diffuser_sharp(Di1, Di2, Re=None, fd=None, roughness=0.0, method='Rennels'):
    r'''Returns loss coefficient for any sudden pipe diameter expansion
    according to the specified method.

    The main theoretical formula is as follows, in [1]_ and in other sources
    and is implemented under the name `Rennels`.

    .. math::
        K_2 = (1-\beta^2)^2

    The Hooper [2]_ formulas are:

    If :math:`{Re}_1 \le 4000`:

    .. math::
        K_1 = 2 \left[1 - \left( \frac{D_1}{D_2} \right)^4 \right]

    else:

    .. math::
        K_1 = \left[1 + 0.8 f_{d,1}\right] \left\{
        \left[1 - \left( \frac{D_1}{D_2}\right)^2
        \right]^2 \right\}

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    Re : float, optional
        Reynolds number of the pipe for original (smaller) pipe, used in
        `Hooper` method [-]
    fd : float, optional
        Darcy friction factor for original (smaller) pipe [-]
    roughness : float, optional
        Roughness of pipe wall (used in `Hooper` method if no friction factor
        given), [m]
    method : str
        The method to use for the calculation; one of 'Rennels', 'Hooper' [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the original (smaller) pipe [-]

    Notes
    -----
    Highly accurate.

    Examples
    --------
    >>> diffuser_sharp(Di1=.5, Di2=1)
    0.5625
    >>> diffuser_sharp(Di1=.5, Di2=1, Re=1e5, fd=1e-7, method='Hooper')
    0.562500045

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Hooper, William B. "Calculate Head Loss Caused by Change in Pipe
       Size." Chemical Engineering 95, no. 16 (November 7, 1988): 89.
    '''
    beta = Di1/Di2
    if method == 'Rennels':
        r = 1.0 - beta*beta
        return r*r
    elif method == 'Hooper':
        if Re is None:
            raise ValueError("Method `Hooper` requires Reynolds number")
        if Re < 4000.0:
            return 2.0*(1.0 - beta*beta*beta*beta) # Not the same formula as Rennels
        if fd is None:
            fd = Clamond(Re=Re, eD=roughness/Di1)
        x = 1.0 - beta*beta
        return (1.0 + 0.8*fd)*x*x
    else:
        raise ValueError(diffuser_sharp_method_unknown)


def diffuser_conical_Crane(Di1, Di2, l=None, angle=None):
    beta = Di1/Di2
    beta2 = beta*beta
    if angle is not None:
        angle_rad = radians(angle)
        angle_deg = angle
    elif l is not None:
        if l != 0.0:
            angle_rad = 2.0*atan((Di1-Di2)/(2.0*l))
            angle_deg = degrees(angle_rad)
        else:
            angle_rad = pi
            angle_deg = 180.0
    else:
        raise ValueError('Either `l` or `angle` must be specified')

    if angle_deg < 45.0:
        # Formula 3
        K2 = 2.6*sin(0.5*angle_rad)*(1.0 - beta2)**2/(beta2*beta2)
    else:
        K2 = (1.0 - beta2)**2/(beta2*beta2)
        # formula 4
    K1 = K2*beta2*beta2 # Standard has become using upstream diameter
    return K1


tck_diffuser_conical_Miller = implementation_optimize_tck([
    [
        -2.307004845727645, -2.307004845727645, -2.307004845727645, -2.307004845727645, -0.852533937110498,
        -0.08240363489988907, 0.5915927994712962, 0.8982804334259539, 1.2315822114127628, 1.5343291978351532,
        1.9774792041044793, 2.990267368122924, 2.990267368122924, 2.990267368122924, 2.990267368122924
    ],
    [
        0.15265175024859737, 0.15265175024859737, 0.15265175024859737, 0.15265175024859737, 0.40701687154729443,
        0.6664564516122377, 0.8948974705226967, 1.0144777142876453, 1.0931592421107108, 1.1789561829062467,
        1.3141101898631344, 1.4016433190574298, 1.4016433190574298, 1.4016433190574298, 1.4016433190574298
    ],
    [
        0.06036297171599943, 0.08322477303304361, 0.1533018560180316, 0.23256231139725417, 0.3176212581983357,
        0.40020914174974515, 0.4385944607898857, 0.5200344894492758, 0.6068491969006803, 0.5644812620968174,
        0.5206931820307759, 0.05279258341151595, 0.06701886136626269, 0.15460022709300852, 0.22187392289400498,
        0.3163189969211137, 0.40236602598664045, 0.44217477520553994, 0.5224439320660155, 0.5978399391103398,
        0.6131809640282799, 0.6101286467987195, 0.05708355184742518, 0.06843627744908527, 0.08943713554460665,
        0.2666074936578441, 0.3093579837678418, 0.3920305705167829, 0.44503141066730906, 0.5320996705995045,
        0.5598015078960548, 0.9045290434928654, 1.1278543134986714, 0.004082132921064788, 0.08726673904790738,
        0.05768023021275458, 0.2018006237954987, 0.31496483541908044, 0.3856708355645899, 0.4432173742517448,
        0.5150555453757539, 0.5447727935474795, 0.8251456282600432, 0.996071097893787, -0.1110682037244921,
        0.07314890991840513, 0.06176280023793122, 0.14210338139570033, 0.221133551530109, 0.34303500384378116,
        0.40130996632027693, 0.49982098188910806, 0.5348917607889022, 0.6163719511180222, 0.6823385842053077,
        -0.2166378057986125, 0.03883937343819872, 0.06286476564404532, 0.10772310640543344, 0.16931893225970837,
        0.22920155110345403, 0.32189134044934775, 0.4091523406543155, 0.5122997879847003, 0.5557259511248352,
        0.5834892444785406, -0.2784258718931251, 0.01614983641474248, 0.06657175843926792, 0.06987287339424499,
        0.11347683852709868, 0.18271325237542604, 0.24381226992585622, 0.33699751608726225, 0.4328543409526461,
        0.4932084120786604, 0.5172902462503076, -0.3110304748285624, -0.02554857636053585, 0.04945754727786904,
        0.06935393005092971, 0.05644398696176074, 0.08533241552366327, 0.15458680076525846, 0.24566876577901098,
        0.35324686175439035, 0.4095605186012888, 0.4277661722408436, -0.27286175236092153, -0.15488345611240545,
        -0.09243246273089455, 0.03455782910023685, 0.0829563174865211, 0.05506682466210118, 0.07027248456489407,
        0.13458355260751956, 0.21084209763905942, 0.2971705194724395, 0.3194829528180993, -0.08063077687005854,
        -0.4253397307338264, -0.6215191566655465, -0.29467521770312016, 0.018448009119198257, 0.08412326971799582,
        0.08337420030229001, 0.131275821589702, 0.1623166890922024, 0.21352111168837065, 0.2394011632386149,
        0.14484414802505116, -0.781141319195365, -1.4412452429263252, -0.6266583715858592, 0.019328251090708078,
        0.07939124881757918, 0.07570115443982374, 0.10818570632561267, 0.14931529315415798, 0.1845260859797597,
        0.1975713897205575
    ], 3, 3
])

diffuser_conical_Idelchik_angles = [3, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 60, 90, 180]
diffuser_conical_Idelchik_A_ratios = [0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
diffuser_conical_Idelchik_data = [
    [0.03, 0.08, 0.11, 0.15, 0.19, 0.23, 0.27, 0.36, 0.47, 0.65, 0.92, 1.15, 1.1, 1.02],
    [0.03, 0.07, 0.1, 0.14, 0.16, 0.2, 0.24, 0.32, 0.42, 0.58, 0.83, 1.04, 0.99, 0.92],
    [0.03, 0.07, 0.09, 0.13, 0.16, 0.19, 0.23, 0.3, 0.4, 0.55, 0.79, 0.99, 0.95, 0.88],
    [0.03, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.29, 0.38, 0.52, 0.75, 0.93, 0.89, 0.83],
    [0.02, 0.06, 0.08, 0.11, 0.14, 0.17, 0.2, 0.26, 0.34, 0.46, 0.67, 0.84, 0.79, 0.74],
    [0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.23, 0.3, 0.41, 0.59, 0.74, 0.7, 0.65],
    [0.02, 0.05, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.26, 0.35, 0.47, 0.65, 0.62, 0.58],
    [0.02, 0.04, 0.05, 0.07, 0.09, 0.11, 0.13, 0.18, 0.23, 0.31, 0.4, 0.57, 0.54, 0.5],
    [0.01, 0.03, 0.04, 0.06, 0.07, 0.08, 0.1, 0.13, 0.17, 0.23, 0.33, 0.41, 0.39, 0.37],
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.12, 0.16, 0.23, 0.29, 0.28, 0.26],
    [0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.18, 0.17, 0.16]]

diffuser_conical_Idelchik_tck = implementation_optimize_tck([[0.0, 0.0, 0.0, 0.0, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.6, 0.6, 0.6],
     [3.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 30.0, 40.0, 60.0, 90.0, 180.0, 180.0],
     [0.03, 0.08000000000000002, 0.11, 0.15000000000000002, 0.19, 0.23000000000000004, 0.2700000000000001, 0.36000000000000004,
      0.4700000000000001, 0.6500000000000001, 0.9200000000000003, 1.1499999999999997,
      1.0999999999999999, 1.02, 0.031285899404876215, 0.06962481602354913, 0.12336980866449107,
      0.1503712244832664, 0.14378748320215035, 0.20742060216292338, 0.24836000991873095,
      0.35209826742177064, 0.43872500319959085, 0.6090878367568959, 0.8690773980930455,
      1.0742803401164671, 1.0021612593588036, 0.9451655708069392, 0.028714100595123804,
      0.06926407286533984, 0.08440796911328675, 0.1374065532945115, 0.17287918346451642,
      0.19813495339263237, 0.23719554563682488, 0.301235065911563, 0.4134972190226316,
      0.5698010521319933, 0.8164781574625106, 1.0379418821057562, 1.0011720739745302,
      0.9192788736375066, 0.03171453253983491, 0.07116642136473203, 0.09282641155265463,
      0.11549496597768823, 0.14338331093620021, 0.17489413621723082, 0.21614667989164066,
      0.28946435656236014, 0.37330000426612064, 0.5104504490091938, 0.7371031974573926,
      0.9040404534886205, 0.8645483458117367, 0.810220761075916, 0.01599798425801497,
      0.0600112625583925, 0.07849171306072822, 0.11003185192295382, 0.14431407179880976,
      0.1740127023740962, 0.20378359569975044, 0.2582633102962821, 0.33980922441927436,
      0.45585837012862357, 0.6659720355794456, 0.8470955557688615, 0.7909107314263772,
      0.7433823030652078, 0.021150220771741206, 0.04655749664043002, 0.0703397965060472,
      0.10328500351954951, 0.11954655404108269, 0.1488787675177576, 0.1662463204709797,
      0.231242192999296, 0.3007649420874127, 0.4151976547001982, 0.604782427849235,
      0.7361883438919813, 0.6970812056056823, 0.6428823350611119, 0.019401132655020165,
      0.053758750879887386, 0.06014910091508289, 0.07682813399884816, 0.09749971203685935,
      0.13047222755487306, 0.1512311224163308, 0.19676791770653376, 0.2571310072310745,
      0.3433510110705831, 0.45489825302361336, 0.6481510686632118, 0.6207644461508929,
      0.5850883566903438, 0.02185995392589747, 0.033290416160064826, 0.045368699473134086,
      0.06692723598046114, 0.08810622640302032, 0.10215235383204274, 0.1213618790128196,
      0.17665887566391483, 0.2219043695740277, 0.3007473976664318, 0.37586666240054567,
      0.5455594857191605, 0.5128931976706977, 0.4673228653399028, 1.2670378191600348e-05,
      0.03091333375994541, 0.03916320044367654, 0.06214899426206778, 0.062121072502719726,
      0.06871380729933241, 0.09367771591902911, 0.10605919242336995, 0.14532614492011708,
      0.196826752842303, 0.32944561762761065, 0.340669205008426, 0.32703730722467556,
      0.32918425374885374, 0.014993664810904203, 0.014543333120027308, 0.025418399778161738,
      0.026425502868966118, 0.04393946374864015, 0.0556430963503338, 0.05566114204048549,
      0.07947040378831506, 0.10483692753994148, 0.13908662357884857, 0.1752771911861948,
      0.26216539749578693, 0.2564813463876624, 0.22290787312557322,
      0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.18, 0.17, 0.16],
     3, 1])


diffuser_conical_Idelchik_obj = lambda x, y : float(bisplev(x, y, diffuser_conical_Idelchik_tck))

diffuser_conical_methods = ['Rennels', 'Crane', 'Miller', 'Swamee', 'Idelchik', 'Hooper']
diffuser_conical_method_unknown = 'Specified method not recognized; methods are %s' %(diffuser_conical_methods)

def diffuser_conical(Di1, Di2, l=None, angle=None, fd=None, Re=None,
                     roughness=0.0, method='Rennels'):
    r'''Returns the loss coefficient for any conical pipe diffuser.
    This calculation has six methods available.

    The 'Rennels' [1]_ formulas are as follows (three different formulas are
    used, depending on the angle and the ratio of diameters):

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

    The Swamee [5]_ formula is:

    .. math::
        K = \left\{\frac{0.25}{\theta^3}\left[1 + \frac{0.6}{r^{1.67}}
        \left(\frac{\pi-\theta}{\theta} \right) \right]^{0.533r - 2.6}
        \right\}^{-0.5}

    The Hooper [6]_ formulas are:

    If :math:`{Re}_1 \le 4000`:

    .. math::
        K_{sharp} = 2 \left[1 - \left( \frac{D_1}{D_2} \right)^4 \right]

    else:

    .. math::
        K_{sharp} = \left[1 + 0.8 f_{d,1}\right] \left\{
        \left[1 - \left( \frac{D_1}{D_2}\right)^2
        \right]^2 \right\}

    If the angle > 45 degrees, :math:`K = K_{sharp}` otherwise

    .. math::
        K = 2.6 \sin \left(\frac{\theta}{2}  \right)K_{sharp}


    .. figure:: fittings/diffuser_conical.png
       :scale: 60 %
       :alt: diffuser conical; after [1]_

    Parameters
    ----------
    Di1 : float
        Inside diameter of original pipe (smaller), [m]
    Di2 : float
        Inside diameter of following pipe (larger), [m]
    l : float, optional
        Length of the contraction along the pipe axis, optional, [m]
    angle : float, optional
        Angle of contraction, [degrees]
    fd : float, optional
        Darcy friction factor [-]
    Re : float, optional
        Reynolds number of the pipe (used in Rennels method only if no friction
        factor given), [m]
    roughness : float, optional
        Roughness of bend wall (used in Rennel method if no friction factor
        given), [m]
    method : str
        The method to use for the calculation; one of 'Rennels', 'Crane',
        'Miller', 'Swamee', 'Idelchik', or 'Hooper' [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to smaller, upstream diameter [-]

    Notes
    -----
    The Miller method changes around quite a bit.

    There is quite a bit of variance in the predictions of the methods, as
    demonstrated by the following figure.

    .. plot:: plots/diffuser_conical_plot.py

    Examples
    --------
    >>> diffuser_conical(Di1=1/3., Di2=1.0, angle=50.0, Re=1E6)
    0.8027721093415322

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    .. [2] Idel’chik, I. E. Handbook of Hydraulic Resistance: Coefficients of
       Local Resistance and of Friction (Spravochnik Po Gidravlicheskim
       Soprotivleniyam, Koeffitsienty Mestnykh Soprotivlenii i Soprotivleniya
       Treniya). National technical information Service, 1966.
    .. [3] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [4] Swamee, Prabhata K., and Ashok K. Sharma. Design of Water Supply
       Pipe Networks. John Wiley & Sons, 2008.
    .. [5] Miller, Donald S. Internal Flow Systems: Design and Performance
       Prediction. Gulf Publishing Company, 1990.
    .. [6] Hooper, William B. "Calculate Head Loss Caused by Change in Pipe
       Size." Chemical Engineering 95, no. 16 (November 7, 1988): 89.
    '''
    beta = Di1/Di2
    beta2 = beta*beta
    if l is not None:
        angle_rad = 2.0*atan(0.5*(Di2-Di1)/l)
        angle_deg = angle_rad*rad2deg
        l_calc = l
    elif angle is not None:
        angle_rad = angle*deg2rad
        angle_deg = angle
        l_calc = (Di2 - Di1)/(2.0*tan(0.5*angle_rad))
    else:
        raise ValueError('Either `l` or `angle` must be specified')
    if method is None:
        method = 'Rennels'
    if method == 'Rennels':
        if fd is None:
            if Re is None:
                raise ValueError("The `Rennels` method requires either a "
                                 "specified friction factor or `Re`")
            fd = Clamond(Re=Re, eD=roughness/Di2, fast=False)

        if 0.0 < angle_deg <= 20.0:
            K = 8.30*tan(0.5*angle_rad)**1.75*(1.0 - beta2)**2 + 0.125*fd*(1.0 - beta2*beta2)/sin(0.5*angle_rad)
        elif 20.0 < angle_deg <= 60.0 and 0.0 <= beta < 0.5:
            K = (1.366*sqrt(sin(2.0*pi*(angle_deg - 15.0)/180.)) - 0.170
            - 3.28*(0.0625-beta**4)*sqrt(0.025*(angle_deg-20.0)))*(1.0 - beta2)**2 + 0.125*fd*(1.0 - beta2*beta2)/sin(0.5*angle_rad)
        elif 20.0 < angle_deg <= 60.0 and beta >= 0.5:
            K = (1.366*sqrt(sin(2.0*pi*(angle_deg - 15.0)/180.0)) - 0.170)*(1.0 - beta2)**2 + 0.125*fd*(1.0 - beta2*beta2)/sin(0.5*angle_rad)
        elif 60.0 < angle_deg <= 180.0 and 0.0 <= beta < 0.5:
            beta4 = beta2*beta2
            K = (1.205 - 3.28*(0.0625 - beta4) - 12.8*beta4*beta2*sqrt((angle_deg - 60.0)/120.))*(1.0 - beta2)**2
        elif 60.0 < angle_deg <= 180.0 and beta >= 0.5:
            K = (1.205 - 0.20*sqrt((angle_deg - 60.0)/120.))*(1.0 - beta**2)**2
        else:
            raise ValueError('Conical diffuser inputs incorrect')
        return K
    elif method == 'Crane':
        return diffuser_conical_Crane(Di1=Di1, Di2=Di2, l=l_calc, angle=angle_deg)
    elif method == 'Miller':
        A_ratio = 1.0/beta2
        if A_ratio > 4.0:
            A_ratio = 4.0
        elif A_ratio < 1.1:
            A_ratio = 1.1

        l_R1_ratio = l_calc/(0.5*Di1)
        if l_R1_ratio < 0.1:
            l_R1_ratio = 0.1
        elif l_R1_ratio > 20.0:
            l_R1_ratio = 20.0
        Kd = max(float(bisplev(log(l_R1_ratio), log(A_ratio), tck_diffuser_conical_Miller)), 0)
        return Kd
    elif method == 'Idelchik':
        A_ratio = beta2
        # Angles 0 to 20, ratios 0.05 to 0.06
        if angle_deg > 20.0:
            angle_fric = 20.0
        elif angle_deg < 2.0:
            angle_fric = 2.0
        else:
            angle_fric = angle_deg

        A_ratio_fric = A_ratio
        if A_ratio_fric < 0.05:
            A_ratio_fric = 0.05
        elif A_ratio_fric > 0.6:
            A_ratio_fric = 0.6

        K_fr = float(contraction_conical_frction_Idelchik_obj(angle_fric, A_ratio_fric))
        K_exp = float(diffuser_conical_Idelchik_obj(min(0.6, A_ratio), max(3.0, angle_deg)))
        return K_fr + K_exp

    elif method == 'Swamee':
        # Really starting to thing Swamee uses a different definition of loss coefficient!
        r = Di2/Di1
        K = 1.0/sqrt(0.25*angle_rad**-3*(1.0 + 0.6*r**(-1.67)*(pi-angle_rad)/angle_rad)**(0.533*r - 2.6))
        return K
    elif method == 'Hooper':
        if Re is None:
            raise ValueError("Method `Hooper` requires Reynolds number")
        if Re < 4000.0:
            return 2.0*(1.0 - beta*beta*beta*beta) # Not the same formula as Rennels
        if fd is None:
            fd = Clamond(Re=Re, eD=roughness/Di1)
        x = 1.0 - beta*beta
        K = (1.0 + 0.8*fd)*x*x
        if angle_rad > 0.25*pi:
            return K
        return K*2.6*sin(0.5*angle_rad)
    else:
        raise ValueError(diffuser_conical_method_unknown)


def diffuser_conical_staged(Di1, Di2, DEs, ls, fd=None, method='Rennels'):
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
        Lengths of the various sections, size 1 more than `DEs`, [m]
    fd : float
        Darcy friction factor [-]
    method : str
        The method to use for the calculation; one of 'Rennels', 'Crane',
        'Miller', 'Swamee', or 'Idelchik' [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to smaller, upstream diameter [-]

    Notes
    -----
    Only lengths of sections currently allowed. This could be changed
    to understand angles also.

    Formula doesn't make much sense, as observed by the example comparing
    a series of conical sections. Use only for small numbers of segments of
    highly differing angles.

    Examples
    --------
    >>> diffuser_conical_staged(Di1=1., Di2=10., DEs=[2,3,4], ls=[1.1,1.2,1.3,1.4], fd=0.01)
    1.9317533188274658

    References
    ----------
    .. [1] Rennels, Donald C., and Hobart M. Hudson. Pipe Flow: A Practical
       and Comprehensive Guide. 1st edition. Hoboken, N.J: Wiley, 2012.
    '''
    K = 0.0
    K += diffuser_conical(Di1=Di1, Di2=DEs[0], l=ls[0], fd=fd, method=method)
    K += diffuser_conical(Di1=DEs[-1], Di2=Di2, l=ls[-1], fd=fd, method=method)
    for i in range(len(DEs)-1):
        K += diffuser_conical(Di1=DEs[i], Di2=DEs[i+1], l=ls[i+1], fd=fd, method=method)
    return K


def diffuser_curved(Di1, Di2, l):
    r'''Returns loss coefficient for any curved wall pipe expansion
    as shown in [1]_.

    .. math::
        K_1 = \phi(1.43-1.3\beta^2)(1-\beta^2)^2

    .. math::
        \phi = 1.01 - 0.624\frac{l}{d_1} + 0.30\left(\frac{l}{d_1}\right)^2
        - 0.074\left(\frac{l}{d_1}\right)^3 + 0.0070\left(\frac{l}{d_1}\right)^4

    .. figure:: fittings/curved_wall_diffuser.png
       :scale: 25 %
       :alt: diffuser curved; after [1]_

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
    return phi*(1.43 - 1.3*beta**2)*(1 - beta**2)**2


def diffuser_pipe_reducer(Di1, Di2, l, fd1, fd2=None):
    r'''Returns loss coefficient for any pipe reducer pipe expansion
    as shown in [1]. This is an approximate formula.

    .. math::
        K_f = f_1\frac{0.20l}{d_1} + \frac{f_1(1-\beta)}{8\sin(\alpha/2)}
        + f_2\frac{0.20l}{d_2}\beta^4

    .. math::
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
    if fd2 is None:
        fd2 = fd1
    beta = Di1/Di2
    angle = -2*atan((Di1-Di2)/1.20/l)
    K = fd1*0.20*l/Di1 + fd1*(1-beta)/8./sin(angle/2) + fd2*0.20*l/Di2*beta**4
    return K

### TODO: Tees

###  3 Darby 3K Method (with valves)
Darby = {}
'''Dictionary of coefficients for Darby's 3K fitting pressure drop method;
the tuple contains :math:`K_1` and :math:`K_i` and :math:`K_d` in that order.
'''
Darby['Elbow, 90°, threaded, standard, (r/D = 1)'] = (800.0, 0.14, 4.0)
Darby['Elbow, 90°, threaded, long radius, (r/D = 1.5)'] = (800.0, 0.071, 4.2)
Darby['Elbow, 90°, flanged, welded, bends, (r/D = 1)'] = (800.0, 0.091, 4.0)
Darby['Elbow, 90°, (r/D = 2)'] = (800.0, 0.056, 3.9)
Darby['Elbow, 90°, (r/D = 4)'] = (800.0, 0.066, 3.9)
Darby['Elbow, 90°, (r/D = 6)'] = (800.0, 0.075, 4.2)
Darby['Elbow, 90°, mitered, 1 weld, (90°)'] = (1000.00, 0.27, 4.0)
Darby['Elbow, 90°, 2 welds, (45°)'] = (800.0, 0.068, 4.1)
Darby['Elbow, 90°, 3 welds, (30°)'] = (800.0, 0.035, 4.2)
Darby['Elbow, 45°, threaded standard, (r/D = 1)'] = (500.0, 0.071, 4.2)
Darby['Elbow, 45°, long radius, (r/D = 1.5)'] = (500.0, 0.052, 4.0)
Darby['Elbow, 45°, mitered, 1 weld, (45°)'] = (500.0, 0.086, 4.0)
Darby['Elbow, 45°, mitered, 2 welds, (22.5°)'] = (500.0, 0.052, 4.0)
Darby['Elbow, 180°, threaded, close-return bend, (r/D = 1)'] = (1000.00, 0.23, 4.0)
Darby['Elbow, 180°, flanged, (r/D = 1)'] = (1000.00, 0.12, 4.0)
Darby['Elbow, 180°, all, (r/D = 1.5)'] = (1000.00, 0.1, 4.0)
Darby['Tee, Through-branch, (as elbow), threaded, (r/D = 1)'] = (500.0, 0.274, 4.0)
Darby['Tee, Through-branch,(as elbow), (r/D = 1.5)'] = (800.0, 0.14, 4.0)
Darby['Tee, Through-branch, (as elbow), flanged, (r/D = 1)'] = (800.0, 0.28, 4.0)
Darby['Tee, Through-branch, (as elbow), stub-in branch'] = (1000.00, 0.34, 4.0)
Darby['Tee, Run-through, threaded, (r/D = 1)'] = (200.0, 0.091, 4.0)
Darby['Tee, Run-through, flanged, (r/D = 1)'] = (150.0, 0.05, 4.0)
Darby['Tee, Run-through, stub-in branch'] = (100.0, 0.0, 0.0)
Darby['Valve, Angle valve, 45°, full line size, β = 1'] = (950.0, 0.25, 4.0)
Darby['Valve, Angle valve, 90°, full line size, β = 1'] = (1000.0, 0.69, 4.0)
Darby['Valve, Globe valve, standard, β = 1'] = (1500.0, 1.7, 3.6)
Darby['Valve, Plug valve, branch flow'] = (500.0, 0.41, 4.0)
Darby['Valve, Plug valve, straight through'] = (300.0, 0.084, 3.9)
Darby['Valve, Plug valve, three-way (flow through)'] = (300.0, 0.14, 4.0)
Darby['Valve, Gate valve, standard, β = 1'] = (300.0, 0.037, 3.9)
Darby['Valve, Ball valve, standard, β = 1'] = (300.0, 0.017, 3.5)
Darby['Valve, Diaphragm, dam type'] = (1000.00, 0.69, 4.9)
Darby['Valve, Swing check'] = (1500.0, 0.46, 4.0)
Darby['Valve, Lift check'] = (2000.00, 2.85, 3.8)

try:
    if IS_NUMBA: # type: ignore
        Darby_keys = tuple(Darby.keys())
        Darby_values = tuple(Darby.values())
except:
    pass


def Darby3K(NPS=None, Re=None, name=None, K1=None, Ki=None, Kd=None):
    r'''Returns loss coefficient for any various fittings, depending
    on the name input. Alternatively, the Darby constants K1, Ki and Kd
    may be provided and used instead. Source of data is [1]_.
    Reviews of this model are favorable.

    .. math::
        K_f = \frac{K_1}{Re} + K_i\left(1 + \frac{K_d}{D_{\text{NPS}}^{0.3}}
        \right)

    Note this model uses nominal pipe diameter in inches.

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
    1.1572523963562356
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
    if name is not None:
        K1 = None
        if name in Darby: # NUMBA: DELETE
            K1, Ki, Kd = Darby[name] # NUMBA: DELETE
        if K1 is None:
            try:
                K1, Ki, Kd = Darby_values[Darby_keys.index(name)]
            except:
                raise ValueError('Name of fitting is not in database')
    elif K1 is not None and Ki is not None and Kd is not None:
        pass
    else:
        raise ValueError('Name of fitting or constants are required')
    return K1/Re + Ki*(1. + Kd*NPS**-0.3)


### 2K Hooper Method

Hooper = {}
r'''Dictionary of coefficients for Hooper's 2K fitting pressure drop method;
the tuple contains :math:`K_1` and :math:`K_\infty` in that order.
'''
Hooper['Elbow, 90°, Standard (R/D = 1), Screwed'] = (800.0, 0.4)
Hooper['Elbow, 90°, Standard (R/D = 1), Flanged/welded'] = (800.0, 0.25)
Hooper['Elbow, 90°, Long-radius (R/D = 1.5), All types'] = (800.0, 0.2)
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 1 weld (90° angle)'] = (1000.0, 1.15)
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 2 weld (45° angle)'] = (800.0, 0.35)
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 3 weld (30° angle)'] = (800.0, 0.3)
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 4 weld (22.5° angle)'] = (800.0, 0.27)
Hooper['Elbow, 90°, Mitered (R/D = 1.5), 5 weld (18° angle)'] = (800.0, 0.25)
Hooper['Elbow, 45°, Standard (R/D = 1), All types'] = (500.0, 0.2)
Hooper['Elbow, 45°, Long-radius (R/D 1.5), All types'] = (500.0, 0.15)
Hooper['Elbow, 45°, Mitered (R/D=1.5), 1 weld (45° angle)'] = (500.0, 0.25)
Hooper['Elbow, 45°, Mitered (R/D=1.5), 2 weld (22.5° angle)'] = (500.0, 0.15)
Hooper['Elbow, 45°, Standard (R/D = 1), Screwed'] = (1000.0, 0.7)
Hooper['Elbow, 180°, Standard (R/D = 1), Flanged/welded'] = (1000.0, 0.35)
Hooper['Elbow, 180°, Long-radius (R/D = 1.5), All types'] = (1000.0, 0.3)
Hooper['Elbow, Used as, Standard, Screwed'] = (500.0, 0.7)
Hooper['Elbow, Elbow, Long-radius, Screwed'] = (800.0, 0.4)
Hooper['Elbow, Elbow, Standard, Flanged/welded'] = (800.0, 0.8)
Hooper['Elbow, Elbow, Stub-in type branch'] = (1000.0, 1.0)
Hooper['Tee, Run, Screwed'] = (200.0, 0.1)
Hooper['Tee, Through, Flanged or welded'] = (150.0, 0.05)
Hooper['Tee, Tee, Stub-in type branch'] = (100.0, 0.0)
Hooper['Valve, Gate, Full line size, Beta = 1'] = (300.0, 0.1)
Hooper['Valve, Ball, Reduced trim, Beta = 0.9'] = (500.0, 0.15)
Hooper['Valve, Plug, Reduced trim, Beta = 0.8'] = (1000.0, 0.25)
Hooper['Valve, Globe, Standard'] = (1500.0, 4.0)
Hooper['Valve, Globe, Angle or Y-type'] = (1000.0, 2.0)
Hooper['Valve, Diaphragm, Dam type'] = (1000.0, 2.0)
Hooper['Valve, Butterfly,'] = (800.0, 0.25)
Hooper['Valve, Check, Lift'] = (2000.0, 10.0)
Hooper['Valve, Check, Swing'] = (1500.0, 1.5)
Hooper['Valve, Check, Tilting-disc'] = (1000.0, 0.5)

try:
    if IS_NUMBA: # type: ignore
        Hooper_keys = tuple(Hooper.keys())
        Hooper_values = tuple(Hooper.values())
except:
    pass


def Hooper2K(Di, Re, name=None, K1=None, Kinfty=None):
    r'''Returns loss coefficient for any various fittings, depending
    on the name input. Alternatively, the Hooper constants K1, Kinfty
    may be provided and used instead. Source of data is [1]_.
    Reviews of this model are favorable less favorable than the Darby method
    but superior to the constant-K method.

    .. math::
        K = \frac{K_1}{Re} + K_\infty\left(1 + \frac{1\text{ inch}}{D_{in}}\right)

    Note this model uses actual inside pipe diameter in inches.

    Parameters
    ----------
    Di : float
        Actual inside diameter of the pipe, [in]
    Re : float
        Reynolds number, [-]
    name : str, optional
        String from Hooper dict representing a fitting
    K1 : float, optional
        K1 parameter of Hooper model, optional [-]
    Kinfty : float, optional
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
    if name is not None:
        K1 = None
        if name in Hooper: # NUMBA: DELETE
            K1, Kinfty = Hooper[name] # NUMBA: DELETE
        if K1 is None:
            try:
                K1, Kinfty = Hooper_values[Hooper_keys.index(name)]
            except:
                raise ValueError('Name of fitting is not in database')
    elif K1 is not None and Kinfty is not None:
        pass
    else:
        raise ValueError('Name of fitting or constants are required')
    return K1/Re + Kinfty*(1. + 1./Di)


### Valves



def Kv_to_Cv(Kv):
    r'''Convert valve flow coefficient from imperial to common metric units.

    .. math::
        C_v = 1.156 K_v

    Parameters
    ----------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop
        of 1 bar) [m^3/hr]

    Returns
    -------
    Cv : float
        Imperial Cv valve flow coefficient (flow rate of water at a pressure
        drop of 1 psi) [gallons/minute]

    Notes
    -----
    Kv = 0.865 Cv is in the IEC standard 60534-2-1.
    It has also been said that Cv = 1.17Kv; this is wrong by current standards.

    The conversion factor does not depend on the density of the fluid or the
    diameter of the valve. It is calculated with the definition of a US gallon
    as 231 cubic inches, and a psi as a pound-force per square inch.

    The exact conversion coefficient between Kv to Cv is 1.1560992283536566;
    it is rounded in the formula above.

    Examples
    --------
    >>> Kv_to_Cv(2)
    2.3121984567073133

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    return 1.1560992283536566*Kv


def Cv_to_Kv(Cv):
    r'''Convert valve flow coefficient from imperial to common metric units.

    .. math::
        K_v = C_v/1.156

    Parameters
    ----------
    Cv : float
        Imperial Cv valve flow coefficient (flow rate of water at a pressure
        drop of 1 psi) [gallons/minute]

    Returns
    -------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop
        of 1 bar) [m^3/hr]

    Notes
    -----
    Kv = 0.865 Cv is in the IEC standard 60534-2-1.
    It has also been said that Cv = 1.17Kv; this is wrong by current standards.

    The conversion factor does not depend on the density of the fluid or the
    diameter of the valve. It is calculated with the definition of a US gallon
    as 231 cubic inches, and a psi as a pound-force per square inch.

    The exact conversion coefficient between Kv to Cv is 1.1560992283536566;
    it is rounded in the formula above.

    Examples
    --------
    >>> Cv_to_Kv(2.312)
    1.9998283393826013

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    return Cv/1.1560992283536566


def Kv_to_K(Kv, D):
    r'''Convert valve flow coefficient from common metric units to regular
    loss coefficients.

    .. math::
        K = 1.6\times 10^9 \frac{D^4}{K_v^2}

    Parameters
    ----------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop
        of 1 bar) [m^3/hr]
    D : float
        Inside diameter of the valve [m]

    Returns
    -------
    K : float
        Loss coefficient, [-]

    Notes
    -----
    Crane TP 410 M (2009) gives the coefficient of 0.04 (with diameter in mm).

    It also suggests the density of water should be found between 5-40°C.
    Older versions specify the density should be found at 60 °F, which is
    used here, and the pessure for the appropriate density is back calculated.

    .. math::
        \Delta P = 1 \text{ bar} = \frac{1}{2}\rho V^2\cdot K

        V = \frac{\frac{K_v\cdot \text{ hour}}{3600 \text{ second}}}{\frac{\pi}{4}D^2}

        \rho = 999.29744568 \;\; kg/m^3  \text{ at } T=60° F, P = 703572 Pa

    The value of density is calculated with IAPWS-95; it is chosen as it makes
    the coefficient a very convenient round number. Others constants that have
    been used are 1.604E9, and 1.60045E9.

    Examples
    --------
    >>> Kv_to_K(2.312, .015)
    15.153374600399898

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    return 1.6E9*D**4*Kv**-2


def K_to_Kv(K, D):
    r'''Convert regular loss coefficient to valve flow coefficient.

    .. math::
        K_v = 4\times 10^4 \sqrt{ \frac{D^4}{K}}

    Parameters
    ----------
    K : float
        Loss coefficient, [-]
    D : float
        Inside diameter of the valve [m]

    Returns
    -------
    Kv : float
        Metric Kv valve flow coefficient (flow rate of water at a pressure drop
        of 1 bar) [m^3/hr]

    Notes
    -----
    Crane TP 410 M (2009) gives the coefficient of 0.04 (with diameter in mm).

    It also suggests the density of water should be found between 5-40°C.
    Older versions specify the density should be found at 60 °F, which is
    used here, and the pessure for the appropriate density is back calculated.

    .. math::
        \Delta P = 1 \text{ bar} = \frac{1}{2}\rho V^2\cdot K

        V = \frac{\frac{K_v\cdot \text{ hour}}{3600 \text{ second}}}{\frac{\pi}{4}D^2}

        \rho = 999.29744568 \;\; kg/m^3 \text{ at } T=60° F, P = 703572 Pa

    The value of density is calculated with IAPWS-95; it is chosen as it makes
    the coefficient a very convenient round number. Others constants that have
    been used are 1.604E9, and 1.60045E9.

    Examples
    --------
    >>> K_to_Kv(15.15337460039990, .015)
    2.312

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    return D*D*sqrt(1.6E9/K)


def K_to_Cv(K, D):
    r'''Convert regular loss coefficient to imperial valve flow coefficient.

    .. math::
        K_v = 1.156 \cdot 4\times 10^4 \sqrt{ \frac{D^4}{K}}

    Parameters
    ----------
    K : float
        Loss coefficient, [-]
    D : float
        Inside diameter of the valve [m]

    Returns
    -------
    Cv : float
        Imperial Cv valve flow coefficient (flow rate of water at a pressure
        drop of 1 psi) [gallons/minute]

    Notes
    -----
    The conversion factor does not depend on the density of the fluid or the
    diameter of the valve. It is calculated with the definition of a US gallon
    as 231 cubic inches, and a psi as a pound-force per square inch.

    The exact conversion coefficient between Kv to Cv is 1.1560992283536566;
    it is rounded in the formula above.

    Examples
    --------
    >>> K_to_Cv(16, .015)
    2.601223263795727

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    return 1.1560992283536566*D*D*sqrt(1.6E9/K)


def Cv_to_K(Cv, D):
    r'''Convert imperial valve flow coefficient from imperial units to regular
    loss coefficients.

    .. math::
        K = 1.6\times 10^9 \frac{D^4}{\left(\frac{C_v}{1.56}\right)^2}

    Parameters
    ----------
    Cv : float
        Imperial Cv valve flow coefficient (flow rate of water at a pressure
        drop of 1 psi) [gallons/minute]
    D : float
        Inside diameter of the valve [m]

    Returns
    -------
    K : float
        Loss coefficient, [-]

    Notes
    -----
    The exact conversion coefficient between Kv to Cv is 1.1560992283536566;
    it is rounded in the formula above.

    Examples
    --------
    >>> Cv_to_K(2.712, .015)
    14.719595348352

    References
    ----------
    .. [1] ISA-75.01.01-2007 (60534-2-1 Mod) Draft
    '''
    D2 = D*D
    term = (Cv*(1.0/1.1560992283536566))
    return 1.6E9*D2*D2/(term*term)


def K_gate_valve_Crane(D1, D2, angle, fd=None):
    r'''Returns loss coefficient for a gate valve of types wedge disc, double
    disc, or plug type, as shown in [1]_.

    If β = 1 and θ = 0:

    .. math::
        K = K_1 = K_2 = 8f_d

    If β < 1 and θ <= 45°:

    .. math::
        K_2 = \frac{K + \sin \frac{\theta}{2} \left[0.8(1-\beta^2)
        + 2.6(1-\beta^2)^2\right]}{\beta^4}

    If β < 1 and θ > 45°:

    .. math::
        K_2 = \frac{K + 0.5\sqrt{\sin\frac{\theta}{2}}(1-\beta^2)
        + (1-\beta^2)^2}{\beta^4}

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    angle : float
        Angle formed by the reducer in the valve, [degrees]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions [2]_.

    Examples
    --------
    Example 7-4 in [1]_; a 150 by 100 mm class 600 steel gate valve, conically
    tapered ports, length 550 mm, back of sear ring ~150 mm. The valve is
    connected to 146 mm schedule 80 pipe. The angle can be calculated to be
    13 degrees. The valve is specified to be operating in turbulent conditions.

    >>> K_gate_valve_Crane(D1=.1, D2=.146, angle=13.115)
    1.1466029421844073

    The calculated result is lower than their value of 1.22; the difference is
    due to Crane's generous intermediate rounding. A later, Imperial edition of
    Crane rounds differently - and comes up with K=1.06.

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    .. [2] Harvey Wilson. "Pressure Drop in Pipe Fittings and Valves |
       Equivalent Length and Resistance Coefficient." Katmar Software. Accessed
       July 28, 2017. http://www.katmarsoftware.com/articles/pipe-fitting-pressure-drop.htm.
    '''
    angle = radians(angle)
    beta = D1/D2
    if fd is None:
        fd = ft_Crane(D2)
    K1 = 8.0*fd # This does not refer to upstream loss per se
    if beta == 1.0 or angle == 0.0:
        return K1 # upstream and down
    else:
        beta2 = beta*beta
        one_m_beta2 = 1.0 - beta2
        if angle <= 0.7853981633974483:
            K = (K1 + sin(0.5*angle)*(one_m_beta2*(0.8 + 2.6*one_m_beta2)))/(beta2*beta2)
        else:
            K = (K1 + one_m_beta2*(0.5*sqrt(sin(0.5*angle)) + one_m_beta2))/(beta2*beta2)
    return K


def K_globe_valve_Crane(D1, D2, fd=None):
    r'''Returns the loss coefficient for all types of globe valve, (reduced
    seat or throttled) as shown in [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = 340 f_d

    Otherwise:

    .. math::
        K_2 = \frac{K + \left[0.5(1-\beta^2) + (1-\beta^2)^2\right]}{\beta^4}

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_globe_valve_Crane(.01, .02)
    135.9200548324305

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = D1/D2
    if fd is None:
        fd = ft_Crane(D2)
    K1 = 340.0*fd
    if beta == 1.0:
        return K1 # upstream and down
    else:
        beta2 = beta*beta
        one_m_beta = 1.0 - beta
        one_m_beta2 = 1.0 - beta2
        return (K1 + beta*(0.5*one_m_beta*one_m_beta
                           + one_m_beta2*one_m_beta2))/(beta2*beta2)


def K_angle_valve_Crane(D1, D2, fd=None, style=0):
    r'''Returns the loss coefficient for all types of angle valve, (reduced
    seat or throttled) as shown in [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    Otherwise:

    .. math::
        K_2 = \frac{K + \left[0.5(1-\beta^2) + (1-\beta^2)^2\right]}{\beta^4}

    For style 0 and 2, N = 55; for style 1, N=150.

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        One of 0, 1, or 2; refers to three different types of angle valves
        as shown in [1]_ [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_angle_valve_Crane(.01, .02)
    26.597361811128465

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = D1/D2
    if style not in (0, 1, 2):
        raise ValueError('Valve style should be 0, 1, or 2')
    if fd is None:
        fd = ft_Crane(D2)

    if style == 0 or style == 2:
        K1 = 55.0*fd
    else:
        K1 = 150.0*fd
    if beta == 1:
        return K1 # upstream and down
    else:
        return (K1 + beta*(0.5*(1-beta)**2 + (1-beta**2)**2))/beta**4


def K_swing_check_valve_Crane(D=None, fd=None, angled=True):
    r'''Returns the loss coefficient for a swing check valve as shown in [1]_.

    .. math::
        K_2 = N\cdot f_d

    For angled swing check valves N = 100; for straight valves, N = 50.

    Parameters
    ----------
    D : float, optional
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    angled : bool, optional
        If True, returns a value 2x the unangled value; the style of the valve
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_swing_check_valve_Crane(D=.02)
    2.3974274785373257

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if D is None and fd is None:
        raise ValueError('Either `D` or `fd` must be specified')
    if fd is None:
        fd = ft_Crane(D)
    if angled:
        return 100.*fd
    return 50.*fd


def K_lift_check_valve_Crane(D1, D2, fd=None, angled=True):
    r'''Returns the loss coefficient for a lift check valve as shown in [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    Otherwise:

    .. math::
        K_2 = \frac{K + \left[0.5(1-\beta^2) + (1-\beta^2)^2\right]}{\beta^4}

    For angled lift check valves N = 55; for straight valves, N = 600.

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    angled : bool, optional
        If True, returns a value 2x the unangled value; the style of the valve
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_lift_check_valve_Crane(.01, .02)
    28.597361811128465

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = D1/D2
    if fd is None:
        fd = ft_Crane(D2)
    if angled:
        K1 = 55*fd
        if beta == 1:
            return K1
        else:
            return (K1 + beta*(0.5*(1 - beta**2) + (1 - beta**2)**2))/beta**4
    else:
        K1 = 600.*fd
        if beta == 1:
            return K1
        else:
            return (K1 + beta*(0.5*(1 - beta**2) + (1 - beta**2)**2))/beta**4


def K_tilting_disk_check_valve_Crane(D, angle, fd=None):
    r'''Returns the loss coefficient for a tilting disk check valve as shown in
    [1]_. Results are specified in [1]_ to be for the disk's resting position
    to be at 5 or 25 degrees to the flow direction.  The model is implemented
    here so as to switch to the higher loss 15 degree coefficients at 10
    degrees, and use the lesser coefficients for any angle under 10 degrees.

    .. math::
        K = N\cdot f_d

    N is obtained from the following table:

    +--------+-------------+-------------+
    |        | angle = 5 ° | angle = 15° |
    +========+=============+=============+
    | 2-8"   | 40          | 120         |
    +--------+-------------+-------------+
    | 10-14" | 30          | 90          |
    +--------+-------------+-------------+
    | 16-48" | 20          | 60          |
    +--------+-------------+-------------+

    The actual change of coefficients happen at <= 9" and <= 15".

    Parameters
    ----------
    D : float
        Diameter of the pipe section the valve in mounted in; the
        same as the line size [m]
    angle : float
        Angle of the tilting disk to the flow direction; nominally 5 or 15
        degrees [degrees]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_tilting_disk_check_valve_Crane(.01, 5)
    1.1626516551826345

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D)
    if angle < 10:
        # 5 degree case
        if D <= 0.2286:
            # 2-8 inches, split at 9 inch
            return 40*fd
        elif D <= 0.381:
            # 10-14 inches, split at 15 inch
            return 30*fd
        else:
            # 16-18 inches
            return 20*fd
    else:
        # 15 degree case
        if D < 0.2286:
            # 2-8 inches
            return 120*fd
        elif D < 0.381:
            # 10-14 inches
            return 90*fd
        else:
            # 16-18 inches
            return 60*fd


globe_stop_check_valve_Crane_coeffs = {0: 400.0, 1: 300.0, 2: 55.0}


def K_globe_stop_check_valve_Crane(D1, D2, fd=None, style=0):
    r'''Returns the loss coefficient for a globe stop check valve as shown in
    [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    Otherwise:

    .. math::
        K_2 = \frac{K + \left[0.5(1-\beta^2) + (1-\beta^2)^2\right]}{\beta^4}

    Style 0 is the standard form; style 1 is angled, with a restrition to force
    the flow up through the valve; style 2 is also angled but with a smaller
    restriction forcing the flow up. N is 400, 300, and 55 for those cases
    respectively.

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        One of 0, 1, or 2; refers to three different types of angle valves
        as shown in [1]_ [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_globe_stop_check_valve_Crane(.1, .02, style=1)
    4.5235076518969795

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D2)
    if style == 0:
        K = 400.0*fd
    elif style == 1:
        K = 300.0*fd
    elif style == 2:
        K = 55.0*fd
    else:
        raise ValueError('Accepted valve styles are 0, 1, and 2 only')
    beta = D1/D2
    if beta == 1.0:
        return K
    else:
        return (K + beta*(0.5*(1 - beta**2) + (1 - beta**2)**2))/beta**4


angle_stop_check_valve_Crane_coeffs = {0: 200.0, 1: 350.0, 2: 55.0}


def K_angle_stop_check_valve_Crane(D1, D2, fd=None, style=0):
    r'''Returns the loss coefficient for a angle stop check valve as shown in
    [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    Otherwise:

    .. math::
        K_2 = \frac{K + \left[0.5(1-\beta^2) + (1-\beta^2)^2\right]}{\beta^4}

    Style 0 is the standard form; style 1 has a restrition to force
    the flow up through the valve; style 2 is has the clearest flow area with
    no guides for the angle valve. N is 200, 350, and 55 for those cases
    respectively.

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be smaller or equal to `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        One of 0, 1, or 2; refers to three different types of angle valves
        as shown in [1]_ [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_angle_stop_check_valve_Crane(.1, .02, style=1)
    4.525425593879809

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D2)
    if style == 0:
        K = 200.0*fd
    elif style == 1:
        K = 350.0*fd
    elif style == 2:
        K = 55.0*fd
    else:
        raise ValueError('Accepted valve styles are 0, 1, and 2 only')

    beta = D1/D2
    if beta == 1:
        return K
    else:
        return (K + beta*(0.5*(1.0 - beta**2) + (1.0 - beta**2)**2))/beta**4


def K_ball_valve_Crane(D1, D2, angle, fd=None):
    r'''Returns the loss coefficient for a ball valve as shown in [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = 3f_d

    If β < 1 and θ <= 45°:

    .. math::
        K_2 = \frac{K + \sin \frac{\theta}{2} \left[0.8(1-\beta^2)
        + 2.6(1-\beta^2)^2\right]} {\beta^4}

    If β < 1 and θ > 45°:

    .. math::
        K_2 = \frac{K + 0.5\sqrt{\sin\frac{\theta}{2}}(1-\beta^2)
        + (1-\beta^2)^2}{\beta^4}

    Parameters
    ----------
    D1 : float
        Diameter of the valve seat bore (must be equal to or smaller than
        `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    angle : float
        Angle formed by the reducer in the valve, [degrees]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_ball_valve_Crane(.01, .02, 50)
    14.051310974926592

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D2)
    beta = D1/D2
    K1 = 3*fd
    angle = radians(angle)
    if beta == 1:
        return K1
    else:
        if angle <= pi/4:
            return (K1 + sin(angle/2)*(0.8*(1-beta**2) + 2.6*(1-beta**2)**2))/beta**4
        else:
            return (K1 + 0.5*sqrt(sin(angle/2)) * (1 - beta**2) + (1-beta**2)**2)/beta**4


diaphragm_valve_Crane_coeffs = {0: 149.0, 1: 39.0}


def K_diaphragm_valve_Crane(D=None, fd=None, style=0):
    r'''Returns the loss coefficient for a diaphragm valve of either weir
    (`style` = 0) or straight-through (`style` = 1) as shown in [1]_.

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    For style 0 (weir), N = 149; for style 1 (straight through), N = 39.

    Parameters
    ----------
    D : float, optional
        Diameter of the pipe section the valve in mounted in; the
        same as the line size [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        Either 0 (weir type valve) or 1 (straight through weir valve) [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_diaphragm_valve_Crane(D=.1, style=0)
    2.4269804835982565

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if D is None and fd is None:
        raise ValueError('Either `D` or `fd` must be specified')
    if fd is None:
        fd = ft_Crane(D)
    if style == 0:
        K = 149.0*fd
    elif style == 1:
        K = 39.0*fd
    else:
        raise ValueError('Accepted valve styles are 0 (weir) or 1 (straight through) only')
    return K


foot_valve_Crane_coeffs = {0: 420.0, 1: 75.0}


def K_foot_valve_Crane(D=None, fd=None, style=0):
    r'''Returns the loss coefficient for a foot valve of either poppet disc
    (`style` = 0) or hinged-disk (`style` = 1) as shown in [1]_. Both valves
    are specified include the loss of the attached strainer.

    .. math::
        K = K_1 = K_2 = N\cdot f_d

    For style 0 (poppet disk), N = 420; for style 1 (hinged disk), N = 75.

    Parameters
    ----------
    D : float, optional
        Diameter of the pipe section the valve in mounted in; the
        same as the line size [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        Either 0 (poppet disk foot valve) or 1 (hinged disk foot valve) [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_foot_valve_Crane(D=0.2, style=0)
    5.912221498436275

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if D is None and fd is None:
        raise ValueError('Either `D` or `fd` must be specified')
    if fd is None:
        fd = ft_Crane(D)
    if style == 0:
        K = 420.0*fd
    elif style == 1:
        K = 75.0*fd
    else:
        raise ValueError('Accepted valve styles are 0 (poppet disk) or 1 (hinged disk) only')
    return K


butterfly_valve_Crane_coeffs = {0: (45.0, 35.0, 25.0), 1: (74.0, 52.0, 43.0),
                                2: (218.0, 96.0, 55.0)}


def K_butterfly_valve_Crane(D, fd=None, style=0):
    r'''Returns the loss coefficient for a butterfly valve as shown in
    [1]_. Three different types are supported; Centric (`style` = 0),
    double offset (`style` = 1), and triple offset (`style` = 2).

    .. math::
        K = N\cdot f_d

    N is obtained from the following table:

    +------------+---------+---------------+---------------+
    | Size range | Centric | Double offset | Triple offset |
    +============+=========+===============+===============+
    | 2" - 8"    | 45      | 74            | 218           |
    +------------+---------+---------------+---------------+
    | 10" - 14"  | 35      | 52            | 96            |
    +------------+---------+---------------+---------------+
    | 16" - 24"  | 25      | 43            | 55            |
    +------------+---------+---------------+---------------+

    The actual change of coefficients happen at <= 9" and <= 15".

    Parameters
    ----------
    D : float
        Diameter of the pipe section the valve in mounted in; the
        same as the line size [m]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        Either 0 (centric), 1 (double offset), or 2 (triple offset) [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_butterfly_valve_Crane(D=.1, style=2)
    3.5508841974793284

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D)
    if style == 0:
        c1, c2, c3 = 45.0, 35.0, 25.0
    elif style == 1:
        c1, c2, c3 = 74.0, 52.0, 43.0
    elif style == 2:
        c1, c2, c3 = 218.0, 96.0, 55.0
    else:
        raise ValueError('Accepted valve styles are 0 (centric), 1 (double offset), or 2 (triple offset) only.')
    if D <= 0.2286:
        # 2-8 inches, split at 9 inch
        return c1*fd
    elif D <= 0.381:
        # 10-14 inches, split at 15 inch
        return c2*fd
    else:
        # 16-18 inches
        return c3*fd


plug_valve_Crane_coeffs = {0: 18.0, 1: 30.0, 2: 90.0}


def K_plug_valve_Crane(D1, D2, angle, fd=None, style=0):
    r'''Returns the loss coefficient for a plug valve or cock valve as shown in
    [1]_.

    If β = 1:

    .. math::
        K = K_1 = K_2 = Nf_d

    Otherwise:

    .. math::
        K_2 = \frac{K + 0.5\sqrt{\sin\frac{\theta}{2}}(1-\beta^2)
        + (1-\beta^2)^2}{\beta^4}

    Three types of plug valves are supported. For straight-through plug valves
    (`style` = 0), N = 18. For 3-way, flow straight through (`style` = 1)
    plug valves, N = 30. For 3-way, flow 90° valves (`style` = 2) N = 90.

    Parameters
    ----------
    D1 : float
        Diameter of the valve plug bore (must be equal to or smaller than
        `D2`), [m]
    D2 : float
        Diameter of the pipe attached to the valve, [m]
    angle : float
        Angle formed by the reducer in the valve, [degrees]
    fd : float, optional
        Darcy friction factor calculated for the actual pipe flow in clean
        steel (roughness = 0.0018 inch) in the fully developed turbulent
        region; do not specify this to use the original Crane friction factor!,
        [-]
    style : int, optional
        Either 0 (straight-through), 1 (3-way, flow straight-through), or 2
        (3-way, flow 90°) [-]

    Returns
    -------
    K : float
        Loss coefficient with respect to the pipe inside diameter [-]

    Notes
    -----
    This method is not valid in the laminar regime and the pressure drop will
    be underestimated in those conditions.

    Examples
    --------
    >>> K_plug_valve_Crane(D1=.01, D2=.02, angle=50)
    19.80513692341617

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    if fd is None:
        fd = ft_Crane(D2)
    beta = D1/D2
    if style == 0:
        K = 18.0*fd
    elif style == 1:
        K = 30.0*fd
    elif style == 2:
        K = 90.0*fd
    else:
        raise ValueError('Accepted valve styles are 0 (straight-through), 1 (3-way, flow straight-through), or 2 (3-way, flow 90°)')
    angle = radians(angle)
    if beta == 1:
        return K
    else:
        return (K + 0.5*sqrt(sin(angle/2)) * (1 - beta**2) + (1-beta**2)**2)/beta**4


def v_lift_valve_Crane(rho, D1=None, D2=None, style='swing check angled'):
    r'''Calculates the approximate minimum velocity required to lift the disk
    or other controlling element of a check valve to a fully open, stable,
    position according to the Crane method [1]_.

    .. math::
        v_{min} = N\cdot \text{m/s} \cdot \sqrt{\frac{\text{kg/m}^3}{\rho}}

    .. math::
        v_{min} = N\beta^2 \cdot \text{m/s} \cdot \sqrt{\frac{\text{kg/m}^3}{\rho}}

    See the notes for the definition of values of N and which check valves use
    which formulas.

    Parameters
    ----------
    rho : float
        Density of the fluid [kg/m^3]
    D1 : float, optional
        Diameter of the valve bore (must be equal to or smaller than
        `D2`), [m]
    D2 : float, optional
        Diameter of the pipe attached to the valve, [m]
    style : str
        The type of valve; one of ['swing check angled', 'swing check straight',
        'swing check UL', 'lift check straight', 'lift check angled',
        'tilting check 5°', 'tilting check 15°', 'stop check globe 1',
        'stop check angle 1', 'stop check globe 2',  'stop check angle 2',
        'stop check globe 3', 'stop check angle 3', 'foot valve poppet disc',
        'foot valve hinged disc'], [-]

    Returns
    -------
    v_min : float
        Approximate minimum velocity required to keep the disc fully lifted,
        preventing chattering and wear [m/s]

    Notes
    -----
    This equation is not dimensionless.

    +--------------------------+-----+------+
    | Name/string              | N   | Full |
    +==========================+=====+======+
    | 'swing check angled'     | 45  | No   |
    +--------------------------+-----+------+
    | 'swing check straight'   | 75  | No   |
    +--------------------------+-----+------+
    | 'swing check UL'         | 120 | No   |
    +--------------------------+-----+------+
    | 'lift check straight'    | 50  | Yes  |
    +--------------------------+-----+------+
    | 'lift check angled'      | 170 | Yes  |
    +--------------------------+-----+------+
    | 'tilting check 5°'       | 100 | No   |
    +--------------------------+-----+------+
    | 'tilting check 15°'      | 40  | No   |
    +--------------------------+-----+------+
    | 'stop check globe 1'     | 70  | Yes  |
    +--------------------------+-----+------+
    | 'stop check angle 1'     | 95  | Yes  |
    +--------------------------+-----+------+
    | 'stop check globe 2'     | 75  | Yes  |
    +--------------------------+-----+------+
    | 'stop check angle 2'     | 75  | Yes  |
    +--------------------------+-----+------+
    | 'stop check globe 3'     | 170 | Yes  |
    +--------------------------+-----+------+
    | 'stop check angle 3'     | 170 | Yes  |
    +--------------------------+-----+------+
    | 'foot valve poppet disc' | 20  | No   |
    +--------------------------+-----+------+
    | 'foot valve hinged disc' | 45  | No   |
    +--------------------------+-----+------+

    Examples
    --------
    >>> v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight')
    1.0252301935349286

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    specific_volume = 1./rho
    if D1 is not None and D2 is not None:
        beta = D1/D2
        beta2 = beta*beta
    if style == 'swing check angled':
        return 45.0*sqrt(specific_volume)
    elif style == 'swing check straight':
        return 75.0*sqrt(specific_volume)
    elif style == 'swing check UL':
        return 120.0*sqrt(specific_volume)
    elif style == 'lift check straight':
        return 50.0*beta2*sqrt(specific_volume)
    elif style == 'lift check angled':
        return 170.0*beta2*sqrt(specific_volume)
    elif style == 'tilting check 5°':
        return 100.0*sqrt(specific_volume)
    elif style == 'tilting check 15°':
        return 40.0*sqrt(specific_volume)
    elif style == 'stop check globe 1':
        return 70.0*beta2*sqrt(specific_volume)
    elif style == 'stop check angle 1':
        return 95.0*beta2*sqrt(specific_volume)
    elif style in ('stop check globe 2', 'stop check angle 2'):
        return 75.0*beta2*sqrt(specific_volume)
    elif style in ('stop check globe 3', 'stop check angle 3'):
        return 170.0*beta2*sqrt(specific_volume)
    elif style == 'foot valve poppet disc':
        return 20.0*sqrt(specific_volume)
    elif style == 'foot valve hinged disc':
        return 45.0*sqrt(specific_volume)


branch_converging_Crane_Fs = [1.74, 1.41, 1.0, 0.0]
branch_converging_Crane_angles = [30.0, 45.0, 60.0, 90.0]


def K_branch_converging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90.0):
    r'''Returns the loss coefficient for the branch of a converging tee or wye
    according to the Crane method [1]_.

    .. math::
        K_{branch} = C\left[1 + D\left(\frac{Q_{branch}}{Q_{comb}\cdot
        \beta_{branch}^2}\right)^2 - E\left(1 - \frac{Q_{branch}}{Q_{comb}}
        \right)^2 - \frac{F}{\beta_{branch}^2} \left(\frac{Q_{branch}}
        {Q_{comb}}\right)^2\right]

    .. math::
        \beta_{branch} = \frac{D_{branch}}{D_{comb}} = \frac{D_{branch}}{D_{run}}

    In the above equation, D = 1, E = 2. See the notes for definitions of F and
    C. The run and combined diameter are assumed the same in this model.

    Parameters
    ----------
    D_run : float
        Diameter of the straight-through inlet portion of the tee or wye [m]
    D_branch : float
        Diameter of the pipe attached at an angle to the straight-through, [m]
    Q_run : float
        Volumetric flow rate in the straight-through inlet of the tee or wye,
        [m^3/s]
    Q_branch : float
        Volumetric flow rate in the pipe attached at an angle to the straight-
        through, [m^3/s]
    angle : float, optional
        Angle the branch makes with the straight-through (tee=90, wye<90)
        [degrees]

    Returns
    -------
    K : float
        Loss coefficient of branch with respect to the velocity and inside
        diameter of the combined flow outlet [-]

    Notes
    -----
    F is linearly interpolated from the table of angles below. There is no
    cutoff to prevent angles from being larger or smaller than 30 or 90
    degrees.

    +-----------+------+
    | Angle [°] |      |
    +===========+======+
    | 30        | 1.74 |
    +-----------+------+
    | 45        | 1.41 |
    +-----------+------+
    | 60        | 1    |
    +-----------+------+
    | 90        | 0    |
    +-----------+------+

    If :math:`\beta_{branch}^2 \le 0.35`, C = 1

    If :math:`\beta_{branch}^2 > 0.35` and :math:`Q_{branch}/Q_{comb} > 0.4`,
    C = 0.55.

    If neither of the above conditions are met:

    .. math::
        C = 0.9\left(1 - \frac{Q_{branch}}{Q_{comb}}\right)

    Note that there is an error in the text of [1]_; the errata can be obtained
    here: http://www.flowoffluids.com/publications/tp-410-errata.aspx

    Examples
    --------
    Example 7-35 of [1]_. A DN100 schedule 40 tee has 1135 liters/minute of
    water passing through the straight leg, and 380 liters/minute of water
    converging with it through a 90° branch. Calculate the loss coefficient in
    the branch. The calculated value there is -0.04026.

    >>> K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633)
    -0.0404410851362

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = (D_branch/D_run)
    beta2 = beta*beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch/Q_comb
    if beta2 <= 0.35:
        C = 1.
    elif Q_ratio <= 0.4:
        C = 0.9*(1 - Q_ratio)
    else:
        C = 0.55
    D, E = 1., 2.
    F = interp(angle, branch_converging_Crane_angles, branch_converging_Crane_Fs)
    K = C*(1. + D*(Q_ratio/beta2)**2 - E*(1. - Q_ratio)**2 - F/beta2*Q_ratio**2)
    return K


run_converging_Crane_Fs = [1.74, 1.41, 1.0]
run_converging_Crane_angles = [30.0, 45.0, 60.0]

def K_run_converging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90):
    r'''Returns the loss coefficient for the run of a converging tee or wye
    according to the Crane method [1]_.

    .. math::
        K_{branch} = C\left[1 + D\left(\frac{Q_{branch}}{Q_{comb}\cdot
        \beta_{branch}^2}\right)^2 - E\left(1 - \frac{Q_{branch}}{Q_{comb}}
        \right)^2 - \frac{F}{\beta_{branch}^2} \left(\frac{Q_{branch}}
        {Q_{comb}}\right)^2\right]

    .. math::
        \beta_{branch} = \frac{D_{branch}}{D_{comb}} = \frac{D_{branch}}{D_{run}}

    In the above equation, C=1, D=0, E=1. See the notes for definitions of F
    and also the special case of 90°. The run and combined diameter are assumed
    the same in this model.

    Parameters
    ----------
    D_run : float
        Diameter of the straight-through inlet portion of the tee or wye
        [m]
    D_branch : float
        Diameter of the pipe attached at an angle to the straight-through, [m]
    Q_run : float
        Volumetric flow rate in the straight-through inlet of the tee or wye,
        [m^3/s]
    Q_branch : float
        Volumetric flow rate in the pipe attached at an angle to the straight-
        through, [m^3/s]
    angle : float, optional
        Angle the branch makes with the straight-through (tee=90, wye<90)
        [degrees]

    Returns
    -------
    K : float
        Loss coefficient of run with respect to the velocity and inside
        diameter of the combined flow outlet [-]

    Notes
    -----
    F is linearly interpolated from the table of angles below. There is no
    cutoff to prevent angles from being larger or smaller than 30 or 60
    degrees. The switch to the special 90° happens at 75°.

    +-----------+------+
    | Angle [°] |      |
    +===========+======+
    | 30        | 1.74 |
    +-----------+------+
    | 45        | 1.41 |
    +-----------+------+
    | 60        | 1    |
    +-----------+------+

    For the special case of 90°, the formula used is as follows.

    .. math::
        K_{run} = 1.55\left(\frac{Q_{branch}}{Q_{comb}} \right)
        - \left(\frac{Q_{branch}}{Q_{comb}}\right)^2

    Examples
    --------
    Example 7-35 of [1]_. A DN100 schedule 40 tee has 1135 liters/minute of
    water passing through the straight leg, and 380 liters/minute of water
    converging with it through a 90° branch. Calculate the loss coefficient in
    the run. The calculated value there is 0.03258.

    >>> K_run_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633)
    0.32575847854551254

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = (D_branch/D_run)
    beta2 = beta*beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch/Q_comb
    if angle < 75.0:
        C = 1.0
    else:
        return 1.55*(Q_ratio) - Q_ratio*Q_ratio

    D, E = 0.0, 1.0
    F = interp(angle, run_converging_Crane_angles, run_converging_Crane_Fs)
    K = C*(1. + D*(Q_ratio/beta2)**2 - E*(1. - Q_ratio)**2 - F/beta2*Q_ratio**2)
    return K


def K_branch_diverging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90):
    r'''Returns the loss coefficient for the branch of a diverging tee or wye
    according to the Crane method [1]_.

    .. math::
        K_{branch} = G\left[1 + H\left(\frac{Q_{branch}}{Q_{comb}
        \beta_{branch}^2}\right)^2 - J\left(\frac{Q_{branch}}{Q_{comb}
        \beta_{branch}^2}\right)\cos\theta\right]

    .. math::
        \beta_{branch} = \frac{D_{branch}}{D_{comb}} = \frac{D_{branch}}{D_{run}}

    See the notes for definitions of H, J, and G. The run and combined diameter
    are assumed the same in this model.

    Parameters
    ----------
    D_run : float
        Diameter of the straight-through inlet portion of the tee or wye [m]
    D_branch : float
        Diameter of the pipe attached at an angle to the straight-through, [m]
    Q_run : float
        Volumetric flow rate in the straight-through outlet of the tee or wye,
        [m^3/s]
    Q_branch : float
        Volumetric flow rate in the pipe attached at an angle to the straight-
        through, [m^3/s]
    angle : float, optional
        Angle the branch makes with the straight-through (tee=90, wye<90)
        [degrees]

    Returns
    -------
    K : float
        Loss coefficient of branch with respect to the velocity and inside
        diameter of the combined flow inlet [-]

    Notes
    -----
    If :math:`\beta_{branch} = 1, \theta = 90^\circ`, H = 0.3 and J = 0.
    Otherwise H = 1 and J = 2.

    G is determined according to the following pseudocode:

    .. code-block:: python

        if angle < 75:
            if beta2 <= 0.35:
                if Q_ratio <= 0.4:
                    G = 1.1 - 0.7*Q_ratio
                else:
                    G = 0.85
            else:
                if Q_ratio <= 0.6:
                    G = 1.0 - 0.6*Q_ratio
                else:
                    G = 0.6
        else:
            if beta2 <= 2/3.:
                G = 1
            else:
                G = 1 + 0.3*Q_ratio*Q_ratio

    Note that there are several errors in the text of [1]_; the errata can be
    obtained here: http://www.flowoffluids.com/publications/tp-410-errata.aspx

    Examples
    --------
    Example 7-36 of [1]_. A DN150 schedule 80 wye has 1515 liters/minute of
    water exiting the straight leg, and 950 liters/minute of water
    exiting it through a 45° branch. Calculate the loss coefficient in
    the branch. The calculated value there is 0.4640.

    >>> K_branch_diverging_Crane(0.146, 0.146, 0.02525, 0.01583, angle=45)
    0.4639895627496694

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = (D_branch/D_run)
    beta2 = beta*beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch/Q_comb

    if angle < 60 or beta <= 2/3.:
        H, J = 1., 2.
    else:
        H, J = 0.3, 0
    if angle < 75:
        if beta2 <= 0.35:
            if Q_ratio <= 0.4:
                G = 1.1 - 0.7*Q_ratio
            else:
                G = 0.85
        else:
            if Q_ratio <= 0.6:
                G = 1.0 - 0.6*Q_ratio
            else:
                G = 0.6
    else:
        if beta2 <= 2/3.:
            G = 1
        else:
            G = 1 + 0.3*Q_ratio*Q_ratio
    angle_rad = radians(angle)
    K_branch = G*(1 + H*(Q_ratio/beta2)**2 - J*(Q_ratio/beta2)*cos(angle_rad))
    return K_branch


def K_run_diverging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90):
    r'''Returns the loss coefficient for the run of a converging tee or wye
    according to the Crane method [1]_.

    .. math::
        K_{run} = M \left(\frac{Q_{branch}}{Q_{comb}}\right)^2

    .. math::
        \beta_{branch} = \frac{D_{branch}}{D_{comb}} = \frac{D_{branch}}{D_{run}}

    See the notes for the definition of M. The run and combined diameter are
    assumed the same in this model.

    Parameters
    ----------
    D_run : float
        Diameter of the straight-through inlet portion of the tee or wye [m]
    D_branch : float
        Diameter of the pipe attached at an angle to the straight-through, [m]
    Q_run : float
        Volumetric flow rate in the straight-through outlet of the tee or wye,
        [m^3/s]
    Q_branch : float
        Volumetric flow rate in the pipe attached at an angle to the straight-
        through, [m^3/s]
    angle : float, optional
        Angle the branch makes with the straight-through (tee=90, wye<90)
        [degrees]

    Returns
    -------
    K : float
        Loss coefficient of run with respect to the velocity and inside
        diameter of the combined flow inlet [-]

    Notes
    -----
    M is calculated according to the following pseudocode:

    .. code-block:: python

        if beta*beta <= 0.4:
            M = 0.4
        elif Q_branch/Q_comb <= 0.5:
            M = 2*(2*Q_branch/Q_comb - 1)
        else:
            M = 0.3*(2*Q_branch/Q_comb - 1)

    Examples
    --------
    Example 7-36 of [1]_. A DN150 schedule 80 wye has 1515 liters/minute of
    water exiting the straight leg, and 950 liters/minute of water
    exiting it through a 45° branch. Calculate the loss coefficient in
    the branch. The calculated value there is -0.06809.

    >>> K_run_diverging_Crane(0.146, 0.146, 0.02525, 0.01583, angle=45)
    -0.06810067607153049

    References
    ----------
    .. [1] Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe. Crane,
       2009.
    '''
    beta = (D_branch/D_run)
    beta2 = beta*beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch/Q_comb
    if beta2 <= 0.4:
        M = 0.4
    elif Q_ratio <= 0.5:
        M = 2.*(2.*Q_ratio - 1.)
    else:
        M = 0.3*(2.*Q_ratio - 1.)
    return M*Q_ratio*Q_ratio


