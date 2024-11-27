"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a model for a jet pump, also known as an eductor or an
ejector.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/fluids/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Interfaces
----------
.. autofunction:: liquid_jet_pump

Objective Function
------------------
.. autofunction:: liquid_jet_pump_ancillary

Vacuum Air Leakage Estimation
-----------------------------
.. autofunction:: vacuum_air_leakage_HEI2633
.. autofunction:: vacuum_air_leakage_Ryans_Croll
.. autofunction:: vacuum_air_leakage_Coker_Worthington
.. autofunction:: vacuum_air_leakage_Seider

"""

from math import exp, log, pi, sqrt

from fluids.constants import foot_cubed_inv, hour_inv, inchHg, lb, mmHg_inv, torr_inv
from fluids.numerics import brenth, secant, SolverInterface
from fluids.numerics import numpy as np

__all__ = ['liquid_jet_pump', 'liquid_jet_pump_ancillary',
           'vacuum_air_leakage_Seider', 'vacuum_air_leakage_Coker_Worthington',
           'vacuum_air_leakage_HEI2633', 'vacuum_air_leakage_Ryans_Croll']


def liquid_jet_pump_ancillary(rhop, rhos, Kp, Ks, d_nozzle=None, d_mixing=None,
                              Qp=None, Qs=None, P1=None, P2=None):
    r'''Calculates the remaining variable in a liquid jet pump when solving for
    one if the inlet variables only and the rest of them are known. The
    equation comes from conservation of energy and momentum in the mixing
    chamber.

    The variable to be solved for must be one of `d_nozzle`, `d_mixing`,
    `Qp`, `Qs`, `P1`, or `P2`.

    .. math::
        P_1 - P_2 = \frac{1}{2}\rho_pV_n^2(1+K_p)
        - \frac{1}{2}\rho_s V_3^2(1+K_s)

    Rearrange to express V3 in terms of Vn, and using the density ratio `C`,
    the expression becomes:

    .. math::
        P_1 - P_2 = \frac{1}{2}\rho_p V_n^2\left[(1+K_p) - C(1+K_s)
        \left(\frac{MR}{1-R}\right)^2\right]

    Using the primary nozzle area and flow rate:

    .. math::
        P_1 - P_2 = \frac{1}{2}\rho_p \left(\frac{Q_p}{A_n}\right)^2
        \left[(1+K_p) - C(1+K_s) \left(\frac{MR}{1-R}\right)^2\right]

    For `P`, `P2`, `Qs`, and `Qp`, the equation can be rearranged explicitly
    for them. For `d_mixing` and `d_nozzle`, a bounded solver is used searching
    between 1E-9 m and 20 times the other diameter which was specified.

    Parameters
    ----------
    rhop : float
        The density of the primary (motive) fluid, [kg/m^3]
    rhos : float
        The density of the secondary fluid (drawn from the vacuum chamber),
        [kg/m^3]
    Kp : float
        The primary nozzle loss coefficient, [-]
    Ks : float
        The secondary inlet loss coefficient, [-]
    d_nozzle : float, optional
        The inside diameter of the primary fluid's nozle, [m]
    d_mixing : float, optional
        The diameter of the mixing chamber, [m]
    Qp : float, optional
        The volumetric flow rate of the primary fluid, [m^3/s]
    Qs : float, optional
        The volumetric flow rate of the secondary fluid, [m^3/s]
    P1 : float, optional
        The pressure of the primary fluid entering its nozzle, [Pa]
    P2 : float, optional
        The pressure of the secondary fluid at the entry of the ejector, [Pa]

    Returns
    -------
    solution : float
        The parameter not specified (one of `d_nozzle`, `d_mixing`,
        `Qp`, `Qs`, `P1`, or `P2`), (units of `m`, `m`, `m^3/s`, `m^3/s`,
        `Pa`, or `Pa` respectively)

    Notes
    -----
    The following SymPy code was used to obtain the analytical formulas (
    they are not shown here due to their length):

    >>> from sympy import * # doctest: +SKIP
    >>> A_nozzle, A_mixing, Qs, Qp, P1, P2, rhos, rhop, Ks, Kp = symbols('A_nozzle, A_mixing, Qs, Qp, P1, P2, rhos, rhop, Ks, Kp') # doctest: +SKIP
    >>> R = A_nozzle/A_mixing # doctest: +SKIP
    >>> M = Qs/Qp # doctest: +SKIP
    >>> C = rhos/rhop # doctest: +SKIP
    >>> rhs = rhop/2*(Qp/A_nozzle)**2*((1+Kp) - C*(1 + Ks)*((M*R)/(1-R))**2 ) # doctest: +SKIP
    >>> new = Eq(P1 - P2,  rhs) # doctest: +SKIP
    >>> solve(new, Qp) # doctest: +SKIP
    >>> solve(new, Qs) # doctest: +SKIP
    >>> solve(new, P1) # doctest: +SKIP
    >>> solve(new, P2) # doctest: +SKIP

    Examples
    --------
    Calculating primary fluid nozzle inlet pressure P1:

    >>> liquid_jet_pump_ancillary(rhop=998., rhos=1098., Ks=0.11, Kp=.04,
    ... P2=133600, Qp=0.01, Qs=0.01, d_mixing=0.045, d_nozzle=0.02238)
    426434.60314398

    References
    ----------
    .. [1] Ejectors and Jet Pumps. Design and Performance for Incompressible
       Liquid Flow. 85032. ESDU International PLC, 1985.
    '''
    unknowns = sum(i is None for i in (d_nozzle, d_mixing, Qs, Qp, P1, P2))
    if unknowns > 1:
        raise ValueError('Too many unknowns')
    elif unknowns < 1:
        raise ValueError('Overspecified')
    C = rhos/rhop

    if Qp is not None and Qs is not None:
        M = Qs/Qp
    if d_nozzle is not None:
        A_nozzle = pi/4*d_nozzle*d_nozzle
        if d_mixing is not None:
            A_mixing = pi/4*d_mixing*d_mixing
            R = A_nozzle/A_mixing

    if P1 is None:
        return rhop/2*(Qp/A_nozzle)**2*((1+Kp) - C*(1 + Ks)*((M*R)/(1-R))**2 ) + P2
    elif P2 is None:
        return -rhop/2*(Qp/A_nozzle)**2*((1+Kp) - C*(1 + Ks)*((M*R)/(1-R))**2 ) + P1
    elif Qs is None:
        try:
            return sqrt((-2*A_nozzle**2*P1 + 2*A_nozzle**2*P2 + Kp*Qp**2*rhop + Qp**2*rhop)/(C*rhop*(Ks + 1)))*(A_mixing - A_nozzle)/A_nozzle
        except ValueError:
            return -1j
    elif Qp is None:
        return A_nozzle*sqrt((2*A_mixing**2*P1 - 2*A_mixing**2*P2 - 4*A_mixing*A_nozzle*P1 + 4*A_mixing*A_nozzle*P2 + 2*A_nozzle**2*P1 - 2*A_nozzle**2*P2 + C*Ks*Qs**2*rhop + C*Qs**2*rhop)/(rhop*(Kp + 1)))/(A_mixing - A_nozzle)
    elif d_nozzle is None:
        def err(d_nozzle):
            return P1 - liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Kp=Kp, Ks=Ks, d_nozzle=d_nozzle, d_mixing=d_mixing, Qp=Qp, Qs=Qs,
                              P1=None, P2=P2)
        return brenth(err, 1E-9, d_mixing*20)
    elif d_mixing is None:
        def err(d_mixing):
            return P1 - liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Kp=Kp, Ks=Ks, d_nozzle=d_nozzle, d_mixing=d_mixing, Qp=Qp, Qs=Qs,
                              P1=None, P2=P2)
        try:
            return brenth(err, 1E-9, d_nozzle*20)
        except:
            return secant(err, d_nozzle*2)




def liquid_jet_pump_pressure_ratio(rhop, rhos, Km, Kd, Ks, Kp,
                     d_nozzle=None, d_mixing=None, d_diffuser=None,
                     Qp=None, Qs=None, P1=None, P2=None, P5=None,
                     nozzle_retracted=True):
    C = rhos/rhop
    if nozzle_retracted:
        j = 0.0
    else:
        j = 1.0

    R = d_nozzle**2/d_mixing**2
    alpha = d_mixing**2/d_diffuser**2
    M = Qs/Qp

    M2, R2, alpha2 = M*M, R*R, alpha*alpha
    num = 2.0*R + 2*C*M2*R2/(1.0 - R)
    num -= R2*(1.0 + C*M)*(1.0 + M)*(1.0 + Km + Kd + alpha2)
    num -= C*M2*R2/(1.0 - R)**2*(1.0 + Ks)

    den = (1.0 + Kp) - 2.0*R - 2.0*C*M2*R2/(1.0 - R)
    den += R2*(1.0 + C*M)*(1.0 + M)*(1.0 + Km + Kd + alpha2)
    den += (1.0 - j)*(C*M2/((1.0 - R)/R)**2)*(1.0 - Ks)
    N = num/den
    if P1 is None:
        P1 = (-P2 + P5*N + P5)/N
    elif P2 is None:
        P2 = -P1*N + P5*N + P5
    elif P5 is None:
        P5 = (P1*N + P2)/(N + 1.0)
    else:
        return N - (P5 - P2)/(P1 - P5)

    solution = {}
    solution['P1'] = P1
    solution['P2'] = P2
    solution['P5'] = P5
#    solution['d_nozzle'] = d_nozzle
#    solution['d_mixing'] = d_mixing
#    solution['d_diffuser'] = d_diffuser
#    solution['Qs'] = Qs
#    solution['Qp'] = Qp
#    solution['N'] = N
#    solution['M'] = M
#    solution['R'] = R
#    solution['alpha'] = alpha
#    solution['efficiency'] = M*N
    return solution


def liquid_jet_pump(rhop, rhos, Kp=0.0, Ks=0.1, Km=.15, Kd=0.1,
                    d_nozzle=None, d_mixing=None, d_diffuser=None,
                    Qp=None, Qs=None, P1=None, P2=None, P5=None,
                    nozzle_retracted=True, max_variations=100):
    r'''Calculate the remaining two variables in a liquid jet pump, using a
    model presented in [1]_ as well as [2]_, [3]_, and [4]_.

    .. math::
        N = \frac{2R + \frac{2 C M^2 R^2}{1-R} - R^2 (1+CM) (1+M) (1 + K_m
        + K_d + \alpha^2) - \frac{CM^2R^2}{(1-R)^2} (1+K_s)}
        {(1+K_p) - 2R - \frac{2CM^2R^2}{1-R} + R^2(1+CM)(1+M)(1+K_m + K_d
        + \alpha^2) + (1-j)\left(\frac{CM^2R^2}{({1-R})^2} \right)(1+K_s)}

    .. math::
        P_1 - P_2 = \frac{1}{2}\rho_p \left(\frac{Q_p}{A_n}\right)^2
        \left[(1+K_p) - C(1+K_s) \left(\frac{MR}{1-R}\right)^2\right]


    .. math::
        \text{Pressure ratio} = N = \frac{P_5 - P_2}{P_1 - P_5}

    .. math::
        \text{Volume flow ratio} = M =  \frac{Q_s}{Q_p}

    .. math::
        \text{Jet pump efficiency} = \eta = M\cdot N =
        \frac{Q_s(P_5-P_2)}{Q_p(P_1 - P_5)}

    .. math::
        R = \frac{A_n}{A_m}

    .. math::
        C = \frac{\rho_s}{\rho_p}

    There is no guarantee a solution will be found for the provided variable
    values, but every combination of two missing variables are supported.

    Parameters
    ----------
    rhop : float
        The density of the primary (motive) fluid, [kg/m^3]
    rhos : float
        The density of the secondary fluid (drawn from the vacuum chamber),
        [kg/m^3]
    Kp : float, optional
        The primary nozzle loss coefficient, [-]
    Ks : float, optional
        The secondary inlet loss coefficient, [-]
    Km : float, optional
        The mixing chamber loss coefficient, [-]
    Kd : float, optional
        The diffuser loss coefficient, [-]
    d_nozzle : float, optional
        The inside diameter of the primary fluid's nozle, [m]
    d_mixing : float, optional
        The diameter of the mixing chamber, [m]
    d_diffuser : float, optional
        The diameter of the diffuser at its exit, [m]
    Qp : float, optional
        The volumetric flow rate of the primary fluid, [m^3/s]
    Qs : float, optional
        The volumetric flow rate of the secondary fluid, [m^3/s]
    P1 : float, optional
        The pressure of the primary fluid entering its nozzle, [Pa]
    P2 : float, optional
        The pressure of the secondary fluid at the entry of the ejector, [Pa]
    P5 : float, optional
        The pressure at the exit of the diffuser, [Pa]
    nozzle_retracted : bool, optional
        Whether or not the primary nozzle's exit is before the mixing chamber,
        or somewhat inside it, [-]
    max_variations : int, optional
        When the initial guesses do not lead to a converged solution, try this
        many more guesses at converging the problem, [-]

    Returns
    -------
    solution : dict
        Dictionary of calculated parameters, [-]

    Notes
    -----
    The assumptions of the model are:

    * The flows are one dimensional except in the mixing chamber.
    * The mixing chamber has constant cross-sectional area.
    * The mixing happens entirely in the mixing chamber, prior to entry into
      the diffuser.
    * The primary nozzle is in a straight line with the middle of the mixing
      chamber.
    * Both fluids are incompressible, and have no excess volume on mixing.
    * Primary and secondary flows both enter the mixing throat with their
      own uniform velocity distribution; the mixed stream leaves with a uniform
      velocity profile.
    * If the secondary fluid is a gas, it undergoes isothermal compression in
      the throat and diffuser.
    * If the secondary fluid is a gas or contains a bubbly gas, it is
      homogeneously distributed in a continuous liquid phase.
    * Heat transfer between the fluids is negligible - there is no change in
      density due to temperature changes
    * The change in the solubility of a dissolved gas, if there is one, is
      negigibly changed by temperature or pressure changes.

    The model can be derived from the equations in
    :py:func:`~.liquid_jet_pump_ancillary` and the following:

    Conservation of energy at the primary nozzle, secondary inlet, and diffuser exit:

    .. math::
        P_1 = P_3 + \frac{1}{2}\rho_p V_n^2 + K_p\left(\frac{1}{2}\rho_p V_n^2\right)

    .. math::
        P_2 = P_3 + \frac{1}{2}\rho_s V_3^2 + K_s\left(\frac{1}{2}\rho_s V_3^2\right)

    .. math::
        P_5 = P_4 + \frac{1}{2}\rho_d V_4^2 - K_d\left(\frac{1}{2}\rho_d V_4^2\right)


    The mixing chamber loss coefficient should be obtained through the following
    expression, using the mixing chamber exit velocity to obtain the friction
    factor.

    .. math::
        K_m = \frac{4f_d L}{D}

    .. math::
        K_d = \frac{P_4 - P_5}{0.5 \rho_d V_4^2} = 1 - \left(\frac{A_4}{A_5}
        \right)^2 - C_{pr}

    .. math::
        K_s = \frac{P_2 - P_3}{0.5\rho_s V_3^2} - 1

    .. math::
        K_p = \frac{P_1 - P_n}{0.5\rho_p V_n^2} - 1


    Continuity of the ejector:

    .. math::
        \rho_p Q_p + \rho_s Q_s = \rho_d Q_d

    Examples
    --------
    >>> ans = liquid_jet_pump(rhop=998., rhos=1098., Km=.186, Kd=0.12, Ks=0.11,
    ... Kp=0.04, d_mixing=0.045, Qs=0.01, Qp=.01, P2=133600,
    ... P5=200E3, nozzle_retracted=False, max_variations=10000)
    >>> s = []
    >>> for key, value in ans.items():
    ...     s.append('%s: %g' %(key, value))
    >>> sorted(s)
    ['M: 1', 'N: 0.293473', 'P1: 426256', 'P2: 133600', 'P5: 200000', 'Qp: 0.01', 'Qs: 0.01', 'R: 0.247404', 'alpha: 1e-06', 'd_diffuser: 45', 'd_mixing: 0.045', 'd_nozzle: 0.0223829', 'efficiency: 0.293473']

    References
    ----------
    .. [1] Karassik, Igor J., Joseph P. Messina, Paul Cooper, and Charles C.
       Heald. Pump Handbook. 4th edition. New York: McGraw-Hill Education, 2007.
    .. [2] Winoto S. H., Li H., and Shah D. A. "Efficiency of Jet Pumps."
       Journal of Hydraulic Engineering 126, no. 2 (February 1, 2000): 150-56.
       https://doi.org/10.1061/(ASCE)0733-9429(2000)126:2(150).
    .. [3] Elmore, Emily, Khalid Al-Mutairi, Bilal Hussain, and A. Sheriff
       El-Gizawy. "Development of Analytical Model for Predicting Dual-Phase
       Ejector Performance," November 11, 2016, V007T09A013.
    .. [4] Ejectors and Jet Pumps. Design and Performance for Incompressible
       Liquid Flow. 85032. ESDU International PLC, 1985.
    '''
    from random import uniform
    solution_vars = ['d_nozzle', 'd_mixing', 'Qp', 'Qs', 'P1', 'P2', 'P5']
    unknown_vars = []
    for i in solution_vars:
         if locals()[i] is None:
             unknown_vars.append(i)

    if len(unknown_vars) > 2:
        raise ValueError('Too many unknowns')
    elif len(unknown_vars) < 2:
        raise ValueError('Overspecified')


    vals = {'d_nozzle': d_nozzle, 'd_mixing': d_mixing, 'Qp': Qp,
            'Qs': Qs, 'P1': P1, 'P2': P2, 'P5': P5}
    var_guesses = []
    # Initial guess algorithms for each variable here
    # No clever algorithms invented yet
    for v in unknown_vars:
        if v == 'd_nozzle':
            try:
                var_guesses.append(d_mixing*0.4)
            except:
                var_guesses.append(0.01)
        if v == 'd_mixing':
            try:
                var_guesses.append(d_nozzle*2)
            except:
                var_guesses.append(0.02)
        elif v == 'P1':
            try:
                var_guesses.append(P2*5)
            except:
                var_guesses.append(P5*5)
        elif v == 'P2':
            try:
                var_guesses.append((P1 + P5)*0.5)
            except:
                try:
                    var_guesses.append(P1/1.1)
                except:
                    var_guesses.append(P5*1.25)
        elif v == 'P5':
            try:
                var_guesses.append(P1*1.12)
            except:
                var_guesses.append(P2*1.12)
        elif v == 'Qp':
            try:
                var_guesses.append(Qs*1.04)
            except:
                var_guesses.append(0.01)
        elif v == 'Qs':
            try:
                var_guesses.append(Qp*0.5)
            except:
                var_guesses.append(0.01)

    C = rhos/rhop
    if nozzle_retracted:
        j = 0.0
    else:
        j = 1.0
    # The diffuser diameter, if not specified, is set to a very large diameter
    # so as to not alter the results
    if d_diffuser is None:
        if d_mixing is not None:
            d_diffuser = d_mixing*1E3
        elif d_nozzle is not None:
            d_diffuser = d_nozzle*1E3
        else:
            d_diffuser = 1000.0
    vals['d_diffuser'] = d_diffuser


    def obj_err(val):
        # Use the dictionary `vals` to keep track of the currently iterating
        # variables
        for i, v in zip(unknown_vars, val):
            vals[i] = abs(float(v))

        # Keep the pressure limits sane
#        if 'P1' in unknown_vars:
#            if 'P5' not in unknown_vars:
#                vals['P1'] = max(vals['P1'], 1.001*vals['P5'])
#            elif 'P2' not in unknown_vars:
#                vals['P1'] = max(vals['P1'], 1.001*vals['P2'])
#        if 'P2' in unknown_vars:
#            if 'P1' not in unknown_vars:
#                vals['P2'] = min(vals['P2'], 0.999*vals['P1'])
#            if 'P5' not in unknown_vars:
#                vals['P2'] = max(vals['P2'], 1.001*vals['P2'])

        # Prelimary numbers
        A_nozzle = pi/4*vals['d_nozzle']**2
        alpha = vals['d_mixing']**2/d_diffuser**2
        R = vals['d_nozzle']**2/vals['d_mixing']**2
        M = vals['Qs']/vals['Qp']

        err1 = liquid_jet_pump_pressure_ratio(rhop=rhop, rhos=rhos, Km=Km, Kd=Kd,
                                              Ks=Ks, Kp=Kp, d_nozzle=vals['d_nozzle'],
                                              d_mixing=vals['d_mixing'],
                                              Qs=vals['Qs'], Qp=vals['Qp'],
                                              P2=vals['P2'], P1=vals['P1'],
                                              P5=vals['P5'],
                                              nozzle_retracted=nozzle_retracted,
                                              d_diffuser=d_diffuser)

        rhs = rhop/2.0*(vals['Qp']/A_nozzle)**2*((1.0 + Kp) - C*(1.0 + Ks)*((M*R)/(1.0 - R))**2 )

        err2 = rhs  - (vals['P1'] - vals['P2'])

        vals['N'] = N = (vals['P5'] - vals['P2'])/(vals['P1']-vals['P5'])
        vals['M'] = M
        vals['R'] = R
        vals['alpha'] = alpha
        vals['efficiency'] = M*N

        if vals['efficiency'] < 0:
            if err1 < 0:
                err1 -= abs(vals['efficiency'])
            else:
                err1 += abs(vals['efficiency'])
            if err2 < 0:
                err2 -= abs(vals['efficiency'])
            else:
                err2 += abs(vals['efficiency'])

#        elif vals['N'] < 0:
#            err1, err2 =  abs(vals['N']) + err1,  abs(vals['N']) + err2
#        print(err1, err2)
        return err1, err2

    # Only one unknown var
    if 'P5' in unknown_vars:
        ancillary = liquid_jet_pump_ancillary(rhop=rhop, rhos=rhos, Kp=Kp,
                                              Ks=Ks, d_nozzle=d_nozzle,
                                              d_mixing=d_mixing, Qp=Qp, Qs=Qs,
                                              P1=P1, P2=P2)
        if unknown_vars[0] == 'P5':
            vals[unknown_vars[1]] = ancillary
        else:
            vals[unknown_vars[0]] = ancillary

        vals['P5'] = liquid_jet_pump_pressure_ratio(rhop=rhop, rhos=rhos, Km=Km, Kd=Kd, Ks=Ks, Kp=Kp, d_nozzle=vals['d_nozzle'],
                               d_mixing=vals['d_mixing'], Qs=vals['Qs'], Qp=vals['Qp'], P2=vals['P2'],
                               P1=vals['P1'], P5=None,
                               nozzle_retracted=nozzle_retracted, d_diffuser=d_diffuser)['P5']
        # Compute the remaining parameters
        obj_err([vals[unknown_vars[0]], vals[unknown_vars[1]]])
        return vals

    with np.errstate(all='ignore'):
        def solve_with_newton(var_guesses):
            solver = SolverInterface(method='newton_system_line_search_progress', objf=obj_err, xtol=3E-7, maxiter=100, jacobian_perturbation=1e-7)
            solution = solver.solve(var_guesses)
            errs = obj_err(solution)

            if (abs(errs[0]) + abs(errs[1])) > 1E-5:
                raise ValueError('Could not solve')

            for u, v in zip(unknown_vars, solution):
                vals[u] = abs(v)
            return vals

        try:
            return solve_with_newton(var_guesses)
        except:
            pass
        # Just do variations on this until it works
        for _ in range(int(max_variations/8)):
            for idx in [0, 1]:
                for r in [(1, 10), (0.1, 1)]:
                    i = uniform(*r)
                    try:
                        l = list(var_guesses)
                        l[idx] = l[idx]*i
                        return solve_with_newton(l)
                    except:
                        pass
        # Vary both parameters at once
        for _ in range(int(max_variations/8)):
            for r in [(1, 10), (0.1, 1)]:
                i = uniform(*r)
                for s in [(1, 10), (0.1, 1)]:
                    j = uniform(*s)
                    try:
                        l = list(var_guesses)
                        l[0] = l[0]*i
                        l[1] = l[1]*j
                        return solve_with_newton(l)
                    except:
                        pass
        raise ValueError('Could not solve')


def vacuum_air_leakage_Ryans_Croll(V, P, P_atm=101325.0):
    r'''Calculates an estimated leakage of air into a vessel using
    a correlation from Ryans and Croll (1981) [1]_ as given in [2]_ and [3]_.

    if P < 10 torr:

    .. math::
        W = 0.026P^{0.34}V^{0.6}

    if P < 100 torr:

    .. math::
        W = 0.032P^{0.26}V^{0.6}

    else:

    .. math::
        W = 0.106V^{0.6}


    In the above equation, the units are lb/hour, torr (vacuum), and cubic feet;
    they are converted in this function.

    Parameters
    ----------
    V : float
        Vessel volume, [m^3]
    P : float
        Vessel actual absolute operating pressure - less than `P_atm`!, [Pa]
    P_atm : float, optional
        The atmospheric pressure surrounding the vessel, [Pa]

    Returns
    -------
    m : float
        Air leakage flow rate, [kg/s]

    Notes
    -----
    No limits are applied to this function.

    Examples
    --------
    >>> vacuum_air_leakage_Ryans_Croll(10, 10000)
    0.0004512

    References
    ----------
    .. [1] Ryans, J. L. and Croll, S. "Selecting Vacuum Systems," 1981.
    .. [2] Coker, Kayode. Ludwig's Applied Process Design for Chemical and
       Petrochemical Plants. 4 edition. Amsterdam ; Boston: Gulf Professional
       Publishing, 2007.
    .. [3] Govoni, Patrick. "An Overview of Vacuum System Design"
       Chemical Engineering Magazine, September 2017.
    '''
    V *= foot_cubed_inv
    P *= torr_inv
    P_atm *= torr_inv
    P_vacuum = P_atm - P
    if P_vacuum < 10:
        air_leakage = 0.026*P_vacuum**0.34*V**0.6
    elif P_vacuum < 100:
        air_leakage = 0.032*P_vacuum**0.26*V**0.6
    else:
        air_leakage = 0.106*V**0.6
    leakage = air_leakage*lb*hour_inv
    return leakage

def vacuum_air_leakage_Seider(V, P, P_atm=101325.0):
    r'''Calculates an estimated leakage of air into a vessel using
    a correlation from Seider [1]_.

    .. math::
        W = 5 + \left[
        0.0298 + 0.03088\ln P - 0.0005733(\ln P)^2
        \right]V^{0.66}

    In the above equation, the units are lb/hour, torr (vacuum), and cubic feet;
    they are converted in this function.

    Parameters
    ----------
    V : float
        Vessel volume, [m^3]
    P : float
        Vessel actual absolute operating pressure - less than `P_atm`!, [Pa]
    P_atm : float, optional
        The atmospheric pressure surrounding the vessel, [Pa]

    Returns
    -------
    m : float
        Air leakage flow rate, [kg/s]

    Notes
    -----
    This formula is rough.

    Examples
    --------
    >>> vacuum_air_leakage_Seider(10, 10000)
    0.0018775547

    References
    ----------
    .. [1] Seider, Warren D., J. D. Seader, and Daniel R. Lewin.
       Product and Process Design Principles: Synthesis, Analysis,
       and Evaluation. 2nd edition. New York: Wiley, 2003.
    '''
    P *= torr_inv
    P_atm *= torr_inv
    P_vacuum = P_atm - P
    V *= foot_cubed_inv
    lnP = log(P_vacuum)
    leakage_lb_hr = 5.0 + (0.0289 + 0.03088*lnP - 0.0005733*lnP*lnP)*V**0.66
    leakage = leakage_lb_hr*lb*hour_inv
    return leakage

def vacuum_air_leakage_HEI2633(V, P, P_atm=101325.0):
    r'''Calculates an estimated leakage of air into a vessel using
    fits to a graph of HEI-2633-00 for air leakage in commercially `tight`
    vessels [1]_.

    There are 5 fits, for < 1 mmHg; 1-3 mmHg; 3-20 mmHg, 20-90 mmHg, and
    90 mmHg to atmospheric. The fits are for `maximum` air leakage.

    Actual values may be significantly larger or smaller depending on the
    condition of the seals, manufacturing defects, and the application.


    Parameters
    ----------
    V : float
        Vessel volume, [m^3]
    P : float
        Vessel actual absolute operating pressure - less than `P_atm`!, [Pa]
    P_atm : float, optional
        The atmospheric pressure surrounding the vessel, [Pa]

    Returns
    -------
    m : float
        Air leakage flow rate, [kg/s]

    Notes
    -----
    The volume is capped to 10 ft^3 on the low end, but extrapolation past
    the maximum size of 10000 ft^3 is allowed.

    It is believed :obj:`vacuum_air_leakage_Seider` was derived from this data,
    so this function should be used in preference to it.

    Examples
    --------
    >>> vacuum_air_leakage_HEI2633(10, 10000)
    0.001186252403781038

    References
    ----------
    .. [1] "Standards for Steam Jet Vacuum Systems", 5th Edition
    '''
    P_atm *= mmHg_inv
    P *= mmHg_inv
    P_vacuum = P_atm - P
    V *= foot_cubed_inv
    if V < 10:
        V = 10.0
    logV = log(V)

    if P_vacuum <= 1:
        c0, c1 = 0.6667235169997174, -3.71246576520232
    elif P_vacuum <= 3:
        c0, c1 = 0.664489357445796, -3.0147277548691274
    elif P_vacuum <= 20:
        c0, c1 = 0.6656780453394583, -2.34007321331419
    elif P_vacuum <= 90:
        c0, c1 = 0.663080000739313, -1.9278288516732665
    else:
        c0, c1 = 0.6658471905826482, -1.6641585778506027
    leakage_lb_hr = exp(c1 + logV*c0)
    leakage = leakage_lb_hr*lb*hour_inv
    return leakage

def vacuum_air_leakage_Coker_Worthington(P, P_atm=101325.0, conservative=True):
    r'''Calculates an estimated leakage of air into a vessel using
    a tabular lookup from Coker cited as being from Worthington Corp's
    1955 Steam-Jet Ejector Application Handbook, Bulletin W-205-E21 [1]_.

    Parameters
    ----------
    P : float
        Vessel actual absolute operating pressure - less than `P_atm`!, [Pa]
    P_atm : float, optional
        The atmospheric pressure surrounding the vessel, [Pa]
    conservative : bool
        Whether to use the high values or low values in the table, [-]

    Returns
    -------
    m : float
        Air leakage flow rate, [kg/s]

    Notes
    -----

    Examples
    --------
    >>> vacuum_air_leakage_Coker_Worthington(10000)
    0.005039915222222222

    References
    ----------
    .. [1] Coker, Kayode. Ludwig's Applied Process Design for Chemical and
       Petrochemical Plants. 4 edition. Amsterdam ; Boston: Gulf Professional
       Publishing, 2007.
    '''
    P /= inchHg # convert to inch Hg
    P_atm /= inchHg # convert to inch Hg
    P_vacuum = P_atm - P
    if conservative:
        if P_vacuum > 8:
            leakage = 40
        elif P_vacuum > 5:
            leakage = 30
        elif P_vacuum > 3:
            leakage = 25
        else:
            leakage = 20
    else:
        if P_vacuum > 8:
            leakage = 30
        elif P_vacuum > 5:
            leakage = 25
        elif P_vacuum > 3:
            leakage = 20
        else:
            leakage = 10
    leakage = leakage*lb*hour_inv
    return leakage

