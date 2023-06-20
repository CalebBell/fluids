#from fluids import *
from fluids.constants import *
from fluids.numerics import IS_PYPY

#IS_PYPY = True

if not IS_PYPY:
    import numba

    import fluids.numba
import inspect
from datetime import datetime

import numpy as np
import pytz


def also_numba(f):
    if not IS_PYPY:
        f.duplicate_with_numba = True
    return f


from fluids.atmosphere import ATMOSPHERE_1976, ATMOSPHERE_NRLMSISE00, airmass, earthsun_distance, solar_irradiation, solar_position, sunrise_sunset

if not IS_PYPY:
    ATMOSPHERE_1976_numba = fluids.numba.ATMOSPHERE_1976
    airmass_numba = fluids.numba.atmosphere.airmass

    @numba.njit
    def numba_int_airmass(Z):
        return fluids.numba.atmosphere.ATMOSPHERE_1976(Z, 0).rho

class BaseTimeSuite:
    def setup(self):
        if not IS_PYPY:
            for k in dir(self.__class__):
                if 'time' in k and 'numba' in k:
                    c = getattr(self, k)
                    c()

class TimeAtmosphereSuite(BaseTimeSuite):
    def setup(self):
        if not IS_PYPY:
            self.time_ATMOSPHERE_1976_numba()
        self.date_test_es = datetime(2020, 6, 6, 10, 0, 0, 0)
        self.tz_dt = pytz.timezone('Australia/Perth').localize(datetime(2020, 6, 6, 7, 10, 57))
        self.tz_dt2 = pytz.timezone('America/Edmonton').localize(datetime(2018, 4, 15, 13, 43, 5))



    def time_ATMOSPHERE_1976(self):
        ATMOSPHERE_1976(5000.0)

    def time_ATMOSPHERE_1976_numba(self):
        ATMOSPHERE_1976_numba(5000.0)

    def time_ATMOSPHERE_1976_pressure_integral(self):
        ATMOSPHERE_1976.pressure_integral(288.6, 84100.0, 147.0)

    def time_ATMOSPHERE_NRLMSISE00(self):
        ATMOSPHERE_NRLMSISE00(1E3, 45, 45, 150)

    def time_airmass(self):
        airmass(lambda Z : ATMOSPHERE_1976(Z).rho, 90.0)

    def time_airmass_numba(self):
        airmass_numba(numba_int_airmass, 90.0)



    def time_earthsun_distance(self):
        earthsun_distance(self.date_test_es)

    def time_solar_position(self):
        solar_position(self.tz_dt, -31.95265, 115.85742)

    def time_sunrise_sunset(self):
        sunrise_sunset(self.tz_dt, 51.0486, -114.07)

    def time_solar_irradiation(self):
        solar_irradiation(Z=1100.0, latitude=51.0486, longitude=-114.07, linke_turbidity=3, moment=self.tz_dt2, surface_tilt=41.0, surface_azimuth=180.0)


from fluids import P_isothermal_critical_flow, isentropic_efficiency, isentropic_work_compression, isothermal_gas

if not IS_PYPY:
    isentropic_work_compression_numba = fluids.numba.isentropic_work_compression
    isentropic_efficiency_numba = fluids.numba.isentropic_efficiency
    P_isothermal_critical_flow_numba = fluids.numba.P_isothermal_critical_flow
    isothermal_gas_numba = fluids.numba.isothermal_gas


class TimeCompressibleSuite(BaseTimeSuite):
    def setup(self):
        if not IS_PYPY:
            self.time_isentropic_work_compression_numba()
            self.time_isentropic_efficiency_numba()
            #self.time_P_isothermal_critical_flow_numba()
            self.time_compressible_D_numba()
            #self.time_compressible_P1_numba()
            #self.time_compressible_P2_numba()

    def time_compressible_D(self):
        isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, m=145.48475726, D=None)

    def time_compressible_D_numba(self):
        isothermal_gas_numba(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, m=145.48475726, D=None)

    def time_compressible_P1(self):
        isothermal_gas(rho=11.3, fd=0.00185, P2=9E5, L=1000, m=145.48475726, D=0.5)

    #def time_compressible_P1_numba(self):
        #isothermal_gas_numba(rho=11.3, fd=0.00185, P2=9E5, L=1000, m=145.48475726, D=0.5)

    def time_compressible_P2(self):
        isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, L=1000, m=145.48475726, D=0.5)

    #def time_compressible_P2_numba(self):
        #isothermal_gas_numba(rho=11.3, fd=0.00185, P1=1E6, L=1000, m=145.48475726, D=0.5)

    def time_isentropic_work_compression(self):
        isentropic_work_compression(P1=1E5, P2=1E6, T1=300.0, k=1.4, eta=0.78)

    def time_isentropic_work_compression_numba(self):
        isentropic_work_compression_numba(P1=1E5, P2=1E6, T1=300.0, k=1.4, eta=0.78)

    def time_isentropic_efficiency(self):
        isentropic_efficiency(1E5, 1E6, 1.4, eta_p=0.78)

    def time_isentropic_efficiency_numba(self):
        isentropic_efficiency_numba(1E5, 1E6, 1.4, eta_p=0.78)

    def time_P_isothermal_critical_flow(self):
        P_isothermal_critical_flow(P=1E6, fd=0.00185, L=1000., D=0.5)

    #def time_P_isothermal_critical_flow_numba(self):
        #P_isothermal_critical_flow_numba(P=1E6, fd=0.00185, L=1000., D=0.5)


from fluids import control_valve_noise_g_2011, control_valve_noise_l_2015, size_control_valve_g, size_control_valve_l

if not IS_PYPY:
    control_valve_noise_l_2015_numba = fluids.numba.control_valve_noise_l_2015
    control_valve_noise_g_2011_numba = fluids.numba.control_valve_noise_g_2011

class TimeControlValveSuite(BaseTimeSuite):
    def time_size_control_valve_g(self):
        size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,  Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)

    def time_size_control_valve_l(self):
        size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.15, D2=0.15, d=0.15, FL=0.9, Fd=0.46)


    def time_control_valve_noise_l_2015(self):
        control_valve_noise_l_2015(m=40, P1=1E6, P2=6.5E5, Psat=2.32E3, rho=997, c=1400, Kv=77.848, d=0.1, Di=0.1071, FL=0.92, Fd=0.42, t_pipe=0.0036, rho_pipe=7800.0, c_pipe=5000.0,rho_air=1.293, c_air=343.0, An=-4.6)

    def time_control_valve_noise_l_2015_numba(self):
        control_valve_noise_l_2015_numba(m=40, P1=1E6, P2=6.5E5, Psat=2.32E3, rho=997, c=1400, Kv=77.848, d=0.1, Di=0.1071, FL=0.92, Fd=0.42, t_pipe=0.0036, rho_pipe=7800.0, c_pipe=5000.0,rho_air=1.293, c_air=343.0, An=-4.6)


    def time_control_valve_noise_g_2011(self):
        control_valve_noise_g_2011(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3, gamma=1.22, MW=19.8, Kv=77.85,  d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98, Fd=0.296, t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0, rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)

    def time_control_valve_noise_g_2011_numba(self):
        control_valve_noise_g_2011_numba(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3, gamma=1.22, MW=19.8, Kv=77.85,  d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98, Fd=0.296, t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0, rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)


from fluids import drag_sphere, integrate_drag_sphere, v_terminal

if not IS_PYPY:
    drag_sphere_numba = fluids.numba.drag_sphere
    v_terminal_numba = fluids.numba.v_terminal
#integrate_drag_sphere_numba = fluids.numba.integrate_drag_sphere


class TimeDragSuite(BaseTimeSuite):

    def time_drag_sphere(self):
        drag_sphere(20000.0, 'Barati_high')

    def time_drag_sphere_numba(self):
        drag_sphere_numba(20000.0, 'Barati_high')

    def time_v_terminal(self):
        v_terminal(D=70E-6, rhop=2600., rho=1000., mu=1E-3)

    def time_v_terminal_numba(self):
        v_terminal_numba(D=70E-6, rhop=2600., rho=1000., mu=1E-3)

    def time_integrate_drag_sphere(self):
        integrate_drag_sphere(D=0.001, rhop=2200., rho=1.2, mu=1.78E-5, t=0.5, V=30, distance=True)

from fluids import Darby3K, Hooper2K, K_angle_valve_Crane, K_branch_converging_Crane, change_K_basis, entrance_distance, v_lift_valve_Crane

if not IS_PYPY:

    change_K_basis_numba = fluids.numba.change_K_basis
    entrance_distance_numba = fluids.numba.entrance_distance
    Darby3K_numba = fluids.numba.Darby3K
    Hooper2K_numba = fluids.numba.Hooper2K
    K_angle_valve_Crane_numba = fluids.numba.K_angle_valve_Crane
    v_lift_valve_Crane_numba = fluids.numba.v_lift_valve_Crane
    K_branch_converging_Crane_numba = fluids.numba.K_branch_converging_Crane


class TimeFittingsSuite(BaseTimeSuite):
    #def setup(self):
        #pass
        #self.time_change_K_basis_numba()
        #self.time_entrance_distance_idelchik_numba()
        #self.time_entrance_distance_harris_numba()
        #self.time_Darby3K_numba()
        #self.time_Hooper2K_numba()

    def time_change_K_basis(self):
        change_K_basis(K1=32.68875692997804, D1=.01, D2=.02)

    def time_change_K_basis_numba(self):
        change_K_basis_numba(K1=32.68875692997804, D1=.01, D2=.02)

    def time_entrance_distance_idelchik(self):
        entrance_distance(Di=0.1, t=0.0005, l=.02, method='Idelchik')

    def time_entrance_distance_idelchik_numba(self):
        entrance_distance_numba(Di=0.1, t=0.0005, l=.02, method='Idelchik')

    def time_entrance_distance_harris(self):
        entrance_distance(Di=0.1, t=0.0005, l=.02, method='Harris')

    def time_entrance_distance_harris_numba(self):
        entrance_distance_numba(Di=0.1, t=0.0005, l=.02, method='Harris')

    def time_Darby3K(self):
        Darby3K(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')

    def time_Darby3K_numba(self):
        Darby3K_numba(NPS=2., Re=10000., name='Valve, Angle valve, 45°, full line size, β = 1')

    def time_Hooper2K(self):
         Hooper2K(Di=2., Re=10000., name='Valve, Globe, Standard')

    def time_Hooper2K_numba(self):
         Hooper2K_numba(Di=2., Re=10000., name='Valve, Globe, Standard')

    def time_K_angle_valve_Crane(self):
         K_angle_valve_Crane(.01, .02)

    def time_K_angle_valve_Crane_numba(self):
         K_angle_valve_Crane_numba(.01, .02)

    def time_v_lift_valve_Crane(self):
        v_lift_valve_Crane(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight')

    def time_v_lift_valve_Crane_numba(self):
        v_lift_valve_Crane_numba(rho=998.2, D1=0.0627, D2=0.0779, style='lift check straight')

    def time_K_branch_converging_Crane(self):
        K_branch_converging_Crane(0.1023, 0.1023, 0.018917, 0.00633)

    def time_K_branch_converging_Crane_numba(self):
        K_branch_converging_Crane_numba(0.1023, 0.1023, 0.018917, 0.00633)


from fluids import C_Reader_Harris_Gallagher, differential_pressure_meter_solver, dP_venturi_tube

if not IS_PYPY:
    C_Reader_Harris_Gallagher_numba = fluids.numba.C_Reader_Harris_Gallagher
    differential_pressure_meter_solver_numba = fluids.numba.differential_pressure_meter_solver
    dP_venturi_tube_numba = fluids.numba.dP_venturi_tube

class TimeFlowMeterSuite(BaseTimeSuite):

    def time_C_Reader_Harris_Gallagher(self):
        C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')

    def time_C_Reader_Harris_Gallagher_numba(self):
        C_Reader_Harris_Gallagher_numba(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')

    def time_dP_venturi_tube(self):
        dP_venturi_tube(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0)

    def time_dP_venturi_tube_numba(self):
        dP_venturi_tube_numba(D=0.07366, Do=0.05, P1=200000.0, P2=183000.0)

    def time_differential_pressure_meter_solver_m(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_m(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_P2(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_P2(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_P1(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_P1(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_D2(self):
        differential_pressure_meter_solver(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_D2(self):
        differential_pressure_meter_solver_numba(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='ISO 5167 orifice', taps='D')



    def time_differential_pressure_meter_solver_m_Hollingshead(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_m_Hollingshead(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_P2_Hollingshead(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_P2_Hollingshead(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_P1_Hollingshead(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_P1_Hollingshead(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_D2_Hollingshead(self):
        differential_pressure_meter_solver(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')

    def time_differential_pressure_meter_solver_numba_D2_Hollingshead(self):
        differential_pressure_meter_solver_numba(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Hollingshead orifice', taps='D')




    def time_differential_pressure_meter_solver_m_Miller_orifice(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_numba_m_Miller_orifice(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, P2=183000.0, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_P2_Miller_orifice(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_numba_P2_Miller_orifice(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P1=200000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_P1_Miller_orifice(self):
        differential_pressure_meter_solver(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_numba_P1_Miller_orifice(self):
        differential_pressure_meter_solver_numba(D=0.07366, D2=0.05, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_D2_Miller_orifice(self):
        differential_pressure_meter_solver(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')

    def time_differential_pressure_meter_solver_numba_D2_Miller_orifice(self):
        differential_pressure_meter_solver_numba(D=0.07366, P1=200000.0, P2=183000.0, m=7.702338035732167, rho=999.1, mu=0.0011, k=1.33, meter_type='Miller orifice', taps='corner')


from fluids import friction_factor, friction_factor_curved, friction_plate_Kumar, ft_Crane, roughness_Farshad

if not IS_PYPY:
    ft_Crane_numba = fluids.numba.ft_Crane
    friction_factor_numba = fluids.numba.friction_factor
    friction_factor_curved_numba = fluids.numba.friction_factor_curved
    friction_plate_Kumar_numba = fluids.numba.friction_plate_Kumar
    roughness_Farshad_numba = fluids.numba.roughness_Farshad


class TimeFrictionSuite(BaseTimeSuite):
    def time_ft_Crane(self):
        ft_Crane(.1)

    def time_ft_Crane_numba(self):
        ft_Crane_numba(.1)

    def time_friction_factor(self):
        friction_factor(Re=1E5, eD=1E-4)

    def time_friction_factor_numba(self):
        friction_factor_numba(Re=1E5, eD=1E-4)

    def time_friction_factor_S2(self):
        friction_factor(Re=2.9E5, eD=1E-5, Method='Serghides_2')

    def time_friction_factor_S2_numba(self):
        friction_factor_numba(Re=2.9E5, eD=1E-5, Method='Serghides_2')

    def time_friction_factor_curved(self):
        friction_factor_curved(Re=1E5, Di=0.02, Dc=0.5)

    def time_friction_factor_curved_numba(self):
        friction_factor_curved_numba(Re=1E5, Di=0.02, Dc=0.5)

    def time_friction_plate_Kumar(self):
        friction_plate_Kumar(Re=2000.0, chevron_angle=30.0)

    def time_friction_plate_Kumar_numba(self):
        friction_plate_Kumar_numba(Re=2000.0, chevron_angle=30.0)

    def time_roughness_Farshad(self):
        roughness_Farshad('Cr13, bare', 0.05)

    def time_roughness_Farshad_numba(self):
        roughness_Farshad_numba('Cr13, bare', 0.05)



from fluids import (
    SA_from_h,
    SA_partial_horiz_ellipsoidal_head,
    SA_partial_horiz_guppy_head,
    SA_partial_horiz_spherical_head,
    SA_partial_horiz_torispherical_head,
    SA_tank,
    V_from_h,
    V_horiz_conical,
    V_horiz_spherical,
    V_horiz_torispherical,
    V_tank,
    V_vertical_torispherical_concave,
)

if not IS_PYPY:
    V_horiz_conical_numba = fluids.numba.V_horiz_conical
    V_horiz_spherical_numba = fluids.numba.V_horiz_spherical
    V_horiz_torispherical_numba = fluids.numba.V_horiz_torispherical
    V_vertical_torispherical_concave_numba = fluids.numba.V_vertical_torispherical_concave
    SA_tank_numba = fluids.numba.SA_tank
    V_tank_numba = fluids.numba.V_tank
    SA_partial_horiz_spherical_head_numba = fluids.numba.SA_partial_horiz_spherical_head
    SA_partial_horiz_ellipsoidal_head_numba = fluids.numba.SA_partial_horiz_ellipsoidal_head
    SA_partial_horiz_guppy_head_numba = fluids.numba.SA_partial_horiz_guppy_head
    SA_partial_horiz_torispherical_head_numba = fluids.numba.SA_partial_horiz_torispherical_head
    V_from_h_numba = fluids.numba.V_from_h
    SA_from_h_numba = fluids.numba.SA_from_h


class TimeGeometrySuite(BaseTimeSuite):
    def time_V_horiz_conical(self):
        V_horiz_conical(D=108., L=156., a=42., h=36)

    def time_V_horiz_conical_numba(self):
        V_horiz_conical_numba(D=108., L=156., a=42., h=36)


    def time_V_horiz_spherical(self):
        V_horiz_spherical(D=108., L=156., a=0.1, h=36.0)

    def time_V_horiz_spherical_numba(self):
        V_horiz_spherical_numba(D=108., L=156., a=0.1, h=36.0)


    def time_V_horiz_torispherical_1(self):
        V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=1.0)

    def time_V_horiz_torispherical_1_numba(self):
        V_horiz_torispherical_numba(D=108., L=156., f=1., k=0.06, h=1.0)

    def time_V_horiz_torispherical_2(self):
        V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=84.0)

    def time_V_horiz_torispherical_2_numba(self):
        V_horiz_torispherical_numba(D=108., L=156., f=1., k=0.06, h=84.0)

    def time_V_horiz_torispherical_3(self):
        V_horiz_torispherical(D=108., L=156., f=1., k=0.06, h=106.0)

    def time_V_horiz_torispherical_3_numba(self):
        V_horiz_torispherical_numba(D=108., L=156., f=1., k=0.06, h=106.0)


    def time_V_vertical_torispherical_concave(self):
        V_vertical_torispherical_concave(D=113., f=0.71, k=0.081, h=15.0)

    def time_V_vertical_torispherical_concave_numba(self):
        V_vertical_torispherical_concave_numba(D=113., f=0.71, k=0.081, h=15.0)


    def time_SA_tank_1(self):
        SA_tank(D=2.54, L=5, sideA='torispherical', sideB='torispherical', sideA_f=1.039370079, sideA_k=0.062362205, sideB_f=1.039370079, sideB_k=0.062362205)

    def time_SA_tank_1_numba(self):
        SA_tank_numba(D=2.54, L=5, sideA='torispherical', sideB='torispherical', sideA_f=1.039370079, sideA_k=0.062362205, sideB_f=1.039370079, sideB_k=0.062362205)


    def time_V_tank_1(self):
        V_tank(D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)

    def time_V_tank_1_numba(self):
        V_tank_numba(D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)


    def time_SA_partial_horiz_spherical_head(self):
        SA_partial_horiz_spherical_head(D=72., a=48.0, h=24.0)

    def time_SA_partial_horiz_spherical_head_numba(self):
        SA_partial_horiz_spherical_head_numba(D=72., a=48.0, h=24.0)


    def time_SA_partial_horiz_ellipsoidal_head(self):
        SA_partial_horiz_ellipsoidal_head(D=72., a=48.0, h=24.0)

    def time_SA_partial_horiz_ellipsoidal_head_numba(self):
        SA_partial_horiz_ellipsoidal_head_numba(D=72., a=48.0, h=24.0)


    def time_SA_partial_horiz_guppy_head(self):
        SA_partial_horiz_guppy_head(D=72., a=48.0, h=24.0)

    def time_SA_partial_horiz_guppy_head_numba(self):
        SA_partial_horiz_guppy_head_numba(D=72., a=48.0, h=24.0)


    def time_SA_partial_horiz_torispherical_head_1(self):
        SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=1.0)

    def time_SA_partial_horiz_torispherical_head_1_numba(self):
        SA_partial_horiz_torispherical_head_numba(D=72., f=1, k=.06, h=1.0)

    def time_SA_partial_horiz_torispherical_head_2(self):
        SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=20.0)

    def time_SA_partial_horiz_torispherical_head_2_numba(self):
        SA_partial_horiz_torispherical_head_numba(D=72., f=1, k=.06, h=20.0)

    def time_SA_partial_horiz_torispherical_head_3(self):
        SA_partial_horiz_torispherical_head(D=72., f=1, k=.06, h=66.0)

    def time_SA_partial_horiz_torispherical_head_3_numba(self):
        SA_partial_horiz_torispherical_head_numba(D=72., f=1, k=.06, h=66.0)


    def time_V_from_h(self):
        V_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)

    def time_V_from_h_numba(self):
        V_from_h_numba(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)


    def time_SA_from_h(self):
        SA_from_h(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)

    def time_SA_from_h_numba(self):
        SA_from_h_numba(h=7, D=1.5, L=5., horizontal=False, sideA='conical', sideB='conical', sideA_a=2., sideB_a=1.)


    # TODO test classes in geometry


from fluids import Q_weir_V_Shen

if not IS_PYPY:
    Q_weir_V_Shen_numba = fluids.numba.Q_weir_V_Shen


class TimeOpenFlowSuite(BaseTimeSuite):
    def time_Q_weir_V_Shen(self):
        Q_weir_V_Shen(0.6, angle=45)

    def time_Q_weir_V_Shen_numba(self):
        Q_weir_V_Shen_numba(0.6, angle=45)


from fluids import dP_packed_bed

if not IS_PYPY:
    dP_packed_bed_numba = fluids.numba.dP_packed_bed


class TimePackedBedSuite(BaseTimeSuite):
    def time_dP_packed_bed(self):
        dP_packed_bed(dp=0.05, voidage=0.492, vs=0.1, rho=1E3, mu=1E-3, Dt=0.015, Method='Guo, Sun, Zhang, Ding & Liu')

    def time_dP_packed_bed_numba(self):
        dP_packed_bed_numba(dp=0.05, voidage=0.492, vs=0.1, rho=1E3, mu=1E-3, Dt=0.015, Method='Guo, Sun, Zhang, Ding & Liu')


from fluids import Stichlmair_flood, Stichlmair_wet

if not IS_PYPY:
    Stichlmair_wet_numba = fluids.numba.Stichlmair_wet
    Stichlmair_flood_numba = fluids.numba.Stichlmair_flood


class TimePackedTowerSuite(BaseTimeSuite):
    def time_Stichlmair_wet(self):
        Stichlmair_wet(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)

    def time_Stichlmair_wet_numba(self):
        Stichlmair_wet_numba(Vg=0.4, Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)

    def time_Stichlmair_flood(self):
        Stichlmair_flood(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)

    def time_Stichlmair_flood_numba(self):
        Stichlmair_flood_numba(Vl = 5E-3, rhog=5., rhol=1200., mug=5E-5, voidage=0.68, specific_area=260., C1=32., C2=7., C3=1.)


from fluids import ParticleSizeDistribution

if not IS_PYPY:
    ParticleSizeDistribution_numba = fluids.numba.ParticleSizeDistribution




class TimeParticleSizeDistributionSuite(BaseTimeSuite):

    # TODO optimize these; maybe add numba support in the future - or increase support for numpy inputs
    #
    def time_ParticleSizeDistribution_init(self):
        ds = 1E-6*np.array([240, 360, 450, 562.5, 703, 878, 1097, 1371, 1713, 2141, 2676, 3345, 4181, 5226, 6532])
        numbers = [65, 119, 232, 410, 629, 849, 990, 981, 825, 579, 297, 111, 21, 1]
        psd = ParticleSizeDistribution(ds=ds, fractions=numbers, order=0)


from fluids import gauge_from_t, nearest_pipe, t_from_gauge

if not IS_PYPY:
    nearest_pipe_numba = fluids.numba.nearest_pipe


class TimePipingSuite(BaseTimeSuite):
    def time_nearest_pipe(self):
        nearest_pipe(Di=0.021)

    def time_gauge_from_t(self):
        gauge_from_t(.5, SI=False, schedule='BWG')

    def time_t_from_gauge(self):
        t_from_gauge(.2, False, 'BWG')

from fluids import CSA_motor_efficiency, VFD_efficiency

if not IS_PYPY:
    VFD_efficiency_numba = fluids.numba.VFD_efficiency
    CSA_motor_efficiency_numba = fluids.numba.CSA_motor_efficiency


class TimePumpSuite(BaseTimeSuite):
    def time_VFD_efficiency(self):
        VFD_efficiency(100*hp, load=0.2)

    def time_VFD_efficiency_numba(self):
        VFD_efficiency_numba(100*hp, load=0.2)


    def time_CSA_motor_efficiency(self):
        CSA_motor_efficiency(100*hp, closed=True, poles=6, high_efficiency=True)

    def time_CSA_motor_efficiency_numba(self):
        CSA_motor_efficiency_numba(100*hp, closed=True, poles=6, high_efficiency=True)

from fluids import API520_B, API520_SH, API520_A_g

if not IS_PYPY:
    API520_A_g_numba = fluids.numba.API520_A_g
    API520_B_numba = fluids.numba.API520_B
    API520_SH_numba = fluids.numba.API520_SH


class TimeSafetyValveSuite(BaseTimeSuite):
    def time_API520_A_g(self):
        API520_A_g(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)

    def time_API520_A_g_numba(self):
        API520_A_g_numba(m=24270/3600., T=348., Z=0.90, MW=51., k=1.11, P1=670E3, Kb=1, Kc=1)


    def time_API520_B(self):
        API520_B(1E6, 5E5)

    def time_API520_B_numba(self):
        API520_B_numba(1E6, 5E5)


    def time_API520_SH(self):
        API520_SH(593+273.15, 1066.325E3)

    def time_API520_SH_numba(self):
        API520_SH_numba(593+273.15, 1066.325E3)


from fluids import K_separator_Watkins

if not IS_PYPY:
    K_separator_Watkins_numba = fluids.numba.K_separator_Watkins

class TimeSeparatorSuite(BaseTimeSuite):
    def time_K_separator_Watkins(self):
        K_separator_Watkins(0.88, 985.4, 1.3, horizontal=True)

    def time_K_separator_Watkins_numba(self):
        K_separator_Watkins_numba(0.88, 985.4, 1.3, horizontal=True)


from fluids import gas_liquid_viscosity, liquid_gas_voidage

if not IS_PYPY:
    liquid_gas_voidage_numba = fluids.numba.liquid_gas_voidage
    gas_liquid_viscosity_numba = fluids.numba.gas_liquid_viscosity

class TimeTwoPhaseVoidageSuite(BaseTimeSuite):

    def time_liquid_gas_voidage(self):
        liquid_gas_voidage(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05)

    def time_liquid_gas_voidage_numba(self):
        liquid_gas_voidage_numba(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05)


    def time_gas_liquid_viscosity(self):
        gas_liquid_viscosity(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2, Method='Duckler')

    def time_gas_liquid_viscosity_numba(self):
        gas_liquid_viscosity_numba(x=0.4, mul=1E-3, mug=1E-5, rhol=850, rhog=1.2, Method='Duckler')

from fluids import Mandhane_Gregory_Aziz_regime, Taitel_Dukler_regime, two_phase_dP

if not IS_PYPY:
    two_phase_dP_numba = fluids.numba.two_phase_dP
    Taitel_Dukler_regime_numba = fluids.numba.Taitel_Dukler_regime
    Mandhane_Gregory_Aziz_regime_numba = fluids.numba.Mandhane_Gregory_Aziz_regime

class TimeTwoPhaseSuite(BaseTimeSuite):
    def time_two_phase_dP(self):
        two_phase_dP(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)

    def time_two_phase_dP_numba(self):
        two_phase_dP_numba(m=0.6, x=0.1, rhol=915., rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.0487, D=0.05, L=1.0)


    def time_Taitel_Dukler_regime(self):
        Taitel_Dukler_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, angle=0)

    def time_Taitel_Dukler_regime_numba(self):
        Taitel_Dukler_regime_numba(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, D=0.05, roughness=0.0, angle=0)


    def time_Mandhane_Gregory_Aziz_regime(self):
        Mandhane_Gregory_Aziz_regime(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)

    def time_Mandhane_Gregory_Aziz_regime_numba(self):
        Mandhane_Gregory_Aziz_regime_numba(m=0.6, x=0.112, rhol=915.12, rhog=2.67, mul=180E-6, mug=14E-6, sigma=0.065, D=0.05)

suites = [TimeAtmosphereSuite, TimeCompressibleSuite, TimeControlValveSuite,
          TimeDragSuite, TimeFittingsSuite, TimeFlowMeterSuite,
          TimeFrictionSuite, TimeGeometrySuite, TimeOpenFlowSuite,
          TimePackedBedSuite, TimePackedTowerSuite, TimeParticleSizeDistributionSuite,
          TimePipingSuite, TimePumpSuite, TimeSafetyValveSuite,
          TimeSeparatorSuite, TimeTwoPhaseVoidageSuite, TimeTwoPhaseSuite,
          ]



for suite in suites:
    continue
    # asv requires inspect to work :(
    # Do I want to write a file that writes this benchmark file?
    glbs, lcls = {}, {}
    for k in dir(suite):
        if 'time' in k:
            f = getattr(suite, k)
            if hasattr(f, 'duplicate_with_numba'):
                source = inspect.getsource(f)
                source = '\n'.join([s[4:] for s in source.split('\n')[1:]])
                orig_function = k.replace('time_', '')
                numba_function = orig_function + '_numba'
                new_function_name = k + '_numba'
                new_source = source.replace(orig_function, numba_function)
                exec(new_source, glbs, lcls)
                setattr(suite, new_function_name, lcls[new_function_name])

if IS_PYPY:
    for s in suites:
        for k in dir(s):
            if 'time' in k and 'numba' in k:
                delattr(s, k)

