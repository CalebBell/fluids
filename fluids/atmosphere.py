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
from math import exp

__all__ = ['ATMOSPHERE_1976']

H_std = [0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852]
T_grad = [-6.5E-3, 0, 1E-3, 2.8E-3, 0, -2.8E-3, -2E-3, 0]
T_std = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
P_std = [101325, 22632.06397346291, 5474.8886696777745, 868.0186847552279,
        110.90630555496608, 66.93887311868738, 3.956420428040732,
        0.3733835899762159]

r0 = 6356766.0
P0 = 101325.0
M0 = 28.9644
g0 = 9.80665
R = 8314.32
gamma = 1.400


class ATMOSPHERE_1976(object):
    r'''US Standard Atmosphere 1976 class, which calculates `T`, `P`,
    `rho`, `v_sonic`, `mu`, `k`, and `g` as a function of altitude above 
    sea level. 
    
    Parameters
    ----------
    Z : float
        Elevation, [m]
    dT : float, optional
        Temperature difference from standard conditions used in determining
        the properties of the atmosphere, [K]
    
    Examples
    --------
    >>> five_km = ATMOSPHERE_1976(5000)
    >>> five_km.P, five_km.rho, five_km.mu
    (54048.28614576141, 0.7364284207799743, 1.628248135362207e-05)
    >>> five_km.k, five_km.g, five_km.v_sonic
    (0.02273190295142526, 9.791241076982665, 320.5455196704035)
    '''        

    @staticmethod
    def get_ind_from_H(H):
        r'''Method defined in the US Standard Atmosphere 1976 for determining
        the index of the layer a specified elevation is above. Levels are 
        0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852 meters respectively.
        '''
        if H <= 0:
            return 0
        for ind, Hi in enumerate(H_std):
            if Hi >= H :
                return ind-1
        return 7 # case for > 84852 m.
    
    @staticmethod
    def thermal_conductivity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        thermal conductivity of air as a function of `T` only. 
        
        .. math::
            k_g = \frac{2.64638\times10^{-3}T^{1.5}}
            {T + 245.4\cdot 10^{-12./T}}
    
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        kg : float
            Thermal conductivity, [W/m/K]
        '''        
        return 2.64638E-3*T**1.5/(T + 245.4*10**(-12./T))
    
    @staticmethod
    def viscosity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        viscosity of air as a function of `T` only. 
        
        .. math::
            \mu_g = \frac{1.458\times10^{-6}T^{1.5}}{T+110.4}
    
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        mug : float
            Viscosity, [Pa*s]
        '''        
        return 1.458E-6*T**1.5/(T+110.4)
    
    @staticmethod
    def density(T, P):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        density of air as a function of `T` and `P`. MW is defined as 28.9644
        g/mol, and R as 8314.32 J/kmol/K
        
        .. math::
            \rho_g = \frac{P\cdot MW}{T\cdot R\cdot 1000}
    
        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]

        Returns
        -------
        rho : float
            Mass density, [kg/m^3]
        '''        
        # 0.00348367635597379 = M0/R
        return P*0.00348367635597379/T
    
    @staticmethod
    def sonic_velocity(T):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        the speed of sound in air as a function of `T` only. 
        
        .. math::
            c = \left(\frac{\gamma R T}{MW}\right)^{0.5}
    
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        c : float
            Speed of sound, [m/s]
        '''        
        # 401.87... = gamma*R/MO
        return (401.87430086589046*T)**0.5
    
    @staticmethod
    def gravity(Z):
        r'''Method defined in the US Standard Atmosphere 1976 for calculating
        the gravitational acceleration above earth as a function of elevation
        only.
        
        .. math::
            g = g_0\left(\frac{r_0}{r_0+Z}\right)^2
    
        Parameters
        ----------
        Z : float
            Elevation above sea level, [m]

        Returns
        -------
        g : float
            Acceleration due to gravity, [m/s^2]
        '''        
        return g0*(r0/(r0+Z))**2


    def __init__(self, Z, dT=0):
        self.Z = Z
        self.H = r0*self.Z/(r0+self.Z)

        i = self.get_ind_from_H(self.H)
        self.T_layer = T_std[i]
        self.T_increase = T_grad[i]
        self.P_layer = P_std[i]
        self.H_layer = H_std[i]

        self.H_above_layer = self.H - self.H_layer
        self.T = self.T_layer + self.T_increase*self.H_above_layer

        if self.T_increase == 0:
            self.P = self.P_layer*exp(-g0*M0*(self.H_above_layer)/R/self.T_layer)
        else:
            self.P = self.P_layer*(self.T_layer/self.T)**(g0*M0/R/self.T_increase)

        if dT: # Affects only the following properties
            self.T += dT
            
        self.rho = self.density(self.T, self.P)
        self.v_sonic = self.sonic_velocity(self.T)
        self.mu = self.viscosity(self.T)
        self.k = self.thermal_conductivity(self.T)
        self.g = self.gravity(self.Z)
    