#from fluids import *
import fluids.numba
import numba
from datetime import datetime
import pytz



from fluids.atmosphere import ATMOSPHERE_1976, ATMOSPHERE_NRLMSISE00, airmass, solar_position, earthsun_distance, sunrise_sunset, solar_irradiation

ATMOSPHERE_1976_numba = fluids.numba.ATMOSPHERE_1976
airmass_numba = fluids.numba.atmosphere.airmass

@numba.njit
def numba_int_airmass(Z):
    return fluids.numba.atmosphere.ATMOSPHERE_1976(Z, 0).rho


class TimeAtmosphereSuite:
    def setup(self):
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


from fluids import isothermal_gas, isentropic_work_compression
isentropic_work_compression_numba = fluids.numba.isentropic_work_compression


class TimeCompressibleSuite:
    def setup(self):
        self.time_isentropic_work_compression_numba()

    def time_compressible(self):
        isothermal_gas(rho=11.3, fd=0.00185, P1=1E6, P2=9E5, L=1000, m=145.48475726, D=None)
        
    def time_isentropic_work_compression(self):
        isentropic_work_compression(P1=1E5, P2=1E6, T1=300.0, k=1.4, eta=0.78)
        
    def time_isentropic_work_compression_numba(self):
        isentropic_work_compression_numba(P1=1E5, P2=1E6, T1=300.0, k=1.4, eta=0.78)
        
from fluids import control_valve_noise_g_2011, control_valve_noise_l_2015, size_control_valve_l, size_control_valve_g

class TimeControlValveSuite:
    def setup(self):
        pass
    
    def time_size_control_valve_g(self):
        size_control_valve_g(T=433., MW=44.01, mu=1.4665E-4, gamma=1.30,  Z=0.988, P1=680E3, P2=310E3, Q=38/36., D1=0.08, D2=0.1, d=0.05, FL=0.85, Fd=0.42, xT=0.60)

    def time_size_control_valve_l(self):
        size_control_valve_l(rho=965.4, Psat=70.1E3, Pc=22120E3, mu=3.1472E-4, P1=680E3, P2=220E3, Q=0.1, D1=0.15, D2=0.15, d=0.15, FL=0.9, Fd=0.46)
        
    def time_control_valve_noise_l_2015(self):
        control_valve_noise_l_2015(m=40, P1=1E6, P2=6.5E5, Psat=2.32E3, rho=997, c=1400, Kv=77.848, d=0.1, Di=0.1071, FL=0.92, Fd=0.42, t_pipe=0.0036, rho_pipe=7800.0, c_pipe=5000.0,rho_air=1.293, c_air=343.0, An=-4.6)
        
    def time_control_valve_noise_g_2011(self):
        control_valve_noise_g_2011(m=2.22, P1=1E6, P2=7.2E5, T1=450, rho=5.3, gamma=1.22, MW=19.8, Kv=77.85,  d=0.1, Di=0.2031, FL=None, FLP=0.792, FP=0.98, Fd=0.296, t_pipe=0.008, rho_pipe=8000.0, c_pipe=5000.0, rho_air=1.293, c_air=343.0, An=-3.8, Stp=0.2)
        
        
from fluids import C_Reader_Harris_Gallagher
C_Reader_Harris_Gallagher_numba = fluids.numba.C_Reader_Harris_Gallagher

class TimeFlowMeterSuite:
    def setup(self):
        C_Reader_Harris_Gallagher_numba(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')

    def time_C_Reader_Harris_Gallagher(self):
        C_Reader_Harris_Gallagher(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')
    
    def time_C_Reader_Harris_Gallagher_numba(self):
        C_Reader_Harris_Gallagher_numba(D=0.07391, Do=0.0222, rho=1.165, mu=1.85E-5, m=0.12, taps='flange')
