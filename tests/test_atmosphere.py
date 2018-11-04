# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from numpy.testing import assert_allclose
from fluids.atmosphere import ATMOSPHERE_1976, MagnetosphereIGRF, hwm93, hwm14, airmass
import numpy as np


def test_ATMOSPHERE_1976():
    # Test values from 'Atmosphere to 86 Km by 2 Km (SI units)', from 
    # http://ckw.phys.ncku.edu.tw/public/pub/Notes/Languages/Fortran/FORSYTHE/www.pdas.com/m1.htm
    # as provided in atmtabs.html in http://www.pdas.com/atmosdownload.html
    H_1 = [-2000, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000, 84000, 86000]
    T_1 = [301.15, 288.15, 275.15, 262.17, 249.19, 236.22, 223.25, 216.65, 216.65, 216.65, 216.65, 216.65, 218.57, 220.56, 222.54, 224.53, 226.51, 228.49, 233.74, 239.28, 244.82, 250.35, 255.88, 261.4, 266.92, 270.65, 270.65, 269.03, 263.52, 258.02, 252.52, 247.02, 241.53, 236.04, 230.55, 225.07, 219.58, 214.26, 210.35, 206.45, 202.54, 198.64, 194.74, 190.84, 186.95]
    P_1 = [127780, 101320, 79501, 61660, 47218, 35652, 26500, 19399, 14170, 10353, 7565.2, 5529.3, 4047.5, 2971.7, 2188.4, 1616.2, 1197, 889.06, 663.41, 498.52, 377.14, 287.14, 219.97, 169.5, 131.34, 102.3, 79.779, 62.215, 48.338, 37.362, 28.724, 21.959, 16.689, 12.606, 9.4609, 7.0529, 5.2209, 3.8363, 2.8009, 2.0333, 1.4674, 1.0525, 0.75009, 0.53104, 0.37338]
    rho_1 = [1.4782, 1.225, 1.0066, 0.81935, 0.66011, 0.52579, 0.41351, 0.31194, 0.22786, 0.16647, 0.12165, 0.08891, 0.06451, 0.046938, 0.034257, 0.025076, 0.01841, 0.013555, 0.0098874, 0.0072579, 0.0053666, 0.0039957, 0.0029948, 0.0022589, 0.0017141, 0.0013167, 0.0010269, 0.00080562, 0.000639, 0.00050445, 0.00039626, 0.00030968, 0.00024071, 0.00018605, 0.00014296, 0.00010917, 0.000082829, 0.000062373, 0.000046385, 0.000034311, 0.000025239, 0.000018458, 0.000013418, 9.6939E-006, 6.9578E-006]
    c_1 = [347.89, 340.29, 332.53, 324.59, 316.45, 308.11, 299.53, 295.07, 295.07, 295.07, 295.07, 295.07, 296.38, 297.72, 299.06, 300.39, 301.71, 303.02, 306.49, 310.1, 313.67, 317.19, 320.67, 324.12, 327.52, 329.8, 329.8, 328.81, 325.43, 322.01, 318.56, 315.07, 311.55, 307.99, 304.39, 300.75, 297.06, 293.44, 290.75, 288.04, 285.3, 282.54, 279.75, 276.94, 274.1]
    mu_1 = [0.000018515, 0.000017894, 0.00001726, 0.000016612, 0.000015949, 0.000015271, 0.000014577, 0.000014216, 0.000014216, 0.000014216, 0.000014216, 0.000014216, 0.000014322, 0.00001443, 0.000014538, 0.000014646, 0.000014753, 0.000014859, 0.00001514, 0.000015433, 0.000015723, 0.000016009, 0.000016293, 0.000016573, 0.000016851, 0.000017037, 0.000017037, 0.000016956, 0.00001668, 0.000016402, 0.000016121, 0.000015837, 0.000015551, 0.000015262, 0.00001497, 0.000014675, 0.000014377, 0.000014085, 0.000013868, 0.00001365, 0.00001343, 0.000013208, 0.000012985, 0.00001276, 0.000012533]


    Ts = [ATMOSPHERE_1976(Z).T for Z in H_1]
    assert_allclose(Ts, T_1, atol=0.005)
    Ps = [ATMOSPHERE_1976(Z).P for Z in H_1]
    assert_allclose(Ps, P_1, rtol=5E-5)
    rhos = [ATMOSPHERE_1976(Z).rho for Z in H_1]
    assert_allclose(rhos, rho_1, rtol=5E-5)
    cs = [ATMOSPHERE_1976(Z).v_sonic for Z in H_1]
    assert_allclose(cs, c_1, rtol=5E-5)
    mus = [ATMOSPHERE_1976(Z).mu for Z in H_1]
    assert_allclose(mus, mu_1, rtol=5E-5)
    
    assert_allclose(ATMOSPHERE_1976(1000, dT=1).T, 282.6510223716947)
    
    # Check thermal conductivity with: http://www.aerospaceweb.org/design/scripts/atmosphere/
    assert_allclose(ATMOSPHERE_1976(1000).k, 0.0248133634493)
    # Other possible additions: 
    # mean air particle speed; mean collision frequency; mean free path; mole volume; total number density


    delta_P = ATMOSPHERE_1976.pressure_integral(288.6, 84100.0, 147.0)
    assert_allclose(delta_P, 1451.9583061008857)
    
    
def test_airmass():
    m = airmass(lambda Z : ATMOSPHERE_1976(Z).rho, 90)
    assert_allclose(m, 10356.127665863998) # vs 10356
    m = airmass(lambda Z : ATMOSPHERE_1976(Z).rho, 60)
    assert_allclose(m, 11954.138271601627) # vs 11954
    
    m = airmass(lambda Z : ATMOSPHERE_1976(Z).rho, 5)
    assert_allclose(m, 106861.56335489497) # vs 106837
    
    m = airmass(lambda Z : ATMOSPHERE_1976(Z).rho, .1)
    assert_allclose(m, 379082.24065519444) # vs 378596
    
    # airmass(lambda Z : ATMOSPHERE_1976(Z).rho, .1, RI=1.0016977377367)
    # As refractive index increases, the atmospheric mass increases drastically. An exception is being raised numerically, not sure why
    # 7966284.95792788 - that's an 800x atmospheric increase.
        
    
    
def test_hwm93():
    # pass on systems without f2py for now
    try:
        custom = hwm93(5E5, 45, 50, 365)
        assert_allclose(custom, [-73.00312042236328, 0.1485661268234253])
        
        
        # Test from pyhwm93
        ans = hwm93(Z=150E3, latitude=65, longitude=-148, day=90, seconds=12*3600, f107=100., f107_avg=100., geomagnetic_disturbance_index=4)
        assert_allclose(ans, [-110.16133880615234, -12.400712013244629])
    except:
        pass


def test_hwm14():
    # Data in checkhwm14.f90; all checks out.
    # Disturbance wind model checks are not separately implemented.
    try:
    
        # Height profile
        HEIGHTS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
        HEIGHT_PROFILE_MER = [0.031, 2.965, -6.627, 2.238, -14.253, 37.403, 42.789, 20.278, 25.027, 34.297, 40.408, 44.436, 47.092, 48.843, 49.997, 50.758, 51.259]
        HEIGHT_PROFILE_ZON = [6.271, 25.115, 96.343, 44.845, 31.59, 11.628, -33.319, -49.984, -68.588, -80.022, -87.56, -92.53, -95.806, -97.965, -99.389, -100.327, -100.946]
        
        winds = [hwm14(alt*1000, latitude=-45.0, longitude=-85.0, day=150, seconds=12*3600, geomagnetic_disturbance_index=80) for alt in HEIGHTS]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, HEIGHT_PROFILE_MER)
        assert_allclose(ZON_CALC, HEIGHT_PROFILE_ZON)
    
    
        # Latitude profile
        LATS = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        LAT_PROFILE_MER = [-124.197, -150.268, -124.54, -23.132, 31.377, 39.524, 56.305, 60.849, 58.117, 56.751, 51.048, 35.653, 14.832, 1.068, -2.749, -27.112, -91.199, -186.757, -166.717]
        LAT_PROFILE_ZON = [177.174, 63.864, -71.971, -105.913, -28.176, 36.532, 32.79, 34.341, 72.676, 110.18, 111.472, 90.547, 77.736, 74.993, 41.972, -140.557, 3.833, 179.951, 235.447]
        
        winds = [hwm14(250E3, latitude=LAT, longitude=30, day=305, seconds=18*3600, geomagnetic_disturbance_index=48) for LAT in LATS]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, LAT_PROFILE_MER)
        assert_allclose(ZON_CALC, LAT_PROFILE_ZON)
    
    
        # Time of day profile: Note the data is specified in terms of local time
        TIMES_LT = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, 16.5, 18, 19.5, 21, 22.5, 24]
        TIMES = [(lt_hour+70/15.)*3600 for lt_hour in TIMES_LT]
        TIME_PROFILE_MER = [6.564, 28.79, 22.316, -4.946, -23.175, -11.278, 17.57, 34.192, 26.875, 9.39, -1.362, -7.168, -21.035, -41.123, -46.702, -27.048, 6.566]
        TIME_PROFILER_ZON = [-40.187, -54.899, -57.187, -47.936, -41.468, -43.648, -49.691, -44.868, -22.542, 2.052, 4.603, -24.13, -66.38, -83.942, -60.262, -36.616, -40.145]
        
        winds = [hwm14(125E3, latitude=45, longitude=-70, day=75, seconds=TIME, geomagnetic_disturbance_index=30) for TIME in TIMES]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, TIME_PROFILE_MER)
        assert_allclose(ZON_CALC, TIME_PROFILER_ZON)
    
        # Longitude profile
        LONGS = [-180, -160, -140, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        LONG_PROFILE_MER = [-0.757, -0.592, 0.033, 0.885, 1.507, 1.545, 1.041, 0.421, 0.172, 0.463, 1.049, 1.502, 1.552, 1.232, 0.757, 0.288, -0.146, -0.538, -0.757]
        LONG_PROFILE_ZON = [-16.835, -18.073, -20.107, -22.166, -22.9, -21.649, -19.089, -16.596, -14.992, -13.909, -12.395, -10.129, -7.991, -7.369, -8.869, -11.701, -14.359, -15.945, -16.835]
        
        winds = [hwm14(40E3, latitude=-5, longitude=LONG, day=330, seconds=6*3600, geomagnetic_disturbance_index=4) for LONG in LONGS]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, LONG_PROFILE_MER)
        assert_allclose(ZON_CALC, LONG_PROFILE_ZON)
        
        # Day of year profile
        DAYS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360]
        DAY_PROFILE_MER = [1.57, -5.43, -13.908, -22.489, -30.844, -39.415, -48.717, -58.582, -67.762, -74.124, -75.371, -70.021, -58.19, -41.813, -24.159, -8.838, 1.319, 5.064, 2.908]
        DAY_PROFILE_ZON = [-42.143, -36.947, -29.927, -23.077, -17.698, -14.016, -11.35, -8.72, -5.53, -2.039, 0.608, 0.85, -2.529, -9.733, -19.666, -30.164, -38.684, -43.208, -42.951]
        
        winds = [hwm14(200E3, latitude=-65, longitude=-135, day=DAY, seconds=21*3600, geomagnetic_disturbance_index=15) for DAY in DAYS]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, DAY_PROFILE_MER)
        assert_allclose(ZON_CALC, DAY_PROFILE_ZON)
        
        # Magnetic strength profile 
        APS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
        AP_PROFILE_MER = [18.63, 11.026, -0.395, -9.121, -13.965, -16.868, -18.476, -19.38, -19.82, -19.887, -19.685, -19.558, -19.558, -19.558]
        AP_PROFILE_ZON = [-71.801, -69.031, -83.49, -96.899, -104.811, -109.891, -112.984, -114.991, -116.293, -116.99, -117.22, -117.212, -117.212, -117.212]
        
        winds = [hwm14(350E3, latitude=38, longitude=125, day=280, seconds=21*3600, geomagnetic_disturbance_index=AP) for AP in APS]
        
        winds = [[round(i, 3) for i in j] for j in winds]
        
        MER_CALC = [i[0] for i in winds]
        ZON_CALC = [i[1] for i in winds]
        
        assert_allclose(MER_CALC, AP_PROFILE_MER)
        assert_allclose(ZON_CALC, AP_PROFILE_ZON)
    except:
        pass


def test_igrf_datetime():
    # Test two calls get different results
    import datetime
    ans1 = MagnetosphereIGRF(-45, 50, 0, datetime.datetime.now()).total_intensity
    ans2 = MagnetosphereIGRF(-45, 50, 0, datetime.datetime.now()).total_intensity
    assert ans1 != ans2
    
    
def test_igrf():
    # 1 Height, km
    # 2 B_total, nT
    # 3 B_north, nT
    # 4 B_east, nT
    # 5 B_down, nT
    # 6 DIP_Inclination, deg
    # 7 DIP_Declination, deg.
    
    txt = '''0.  37750.8  11326.4 -13042.3 -33566.9  -62.8  -49.0
         100.  36345.2  10998.5 -12199.3 -32422.0  -63.1  -48.0
         200.  35003.3  10677.2 -11425.9 -31315.7  -63.5  -46.9
         300.  33721.6  10363.3 -10715.3 -30247.8  -63.8  -46.0
         400.  32497.1  10056.9 -10061.4 -29217.9  -64.0  -45.0
         500.  31326.7   9758.4  -9458.7 -28225.3  -64.3  -44.1
         600.  30207.7   9468.0  -8902.5 -27269.2  -64.5  -43.2
         700.  29137.5   9185.7  -8388.3 -26348.7  -64.7  -42.4
         800.  28113.8   8911.7  -7912.4 -25462.9  -64.9  -41.6
         900.  27134.1   8645.8  -7471.4 -24610.7  -65.1  -40.8
        1000.  26196.4   8388.2  -7062.0 -23791.2  -65.3  -40.1'''
    to_1000_km = np.fromstring(txt, sep=' ')
    to_1000_km = to_1000_km.reshape(11, 7)
    
    txt = '''1000.  19247.8  17621.0  -1444.8  -7608.6  -23.3   -4.7
        3900.   6655.0   6457.6   -755.0  -1420.8  -12.3   -6.7
        6800.   3147.9   3095.4   -413.9   -395.3   -7.2   -7.6
        9700.   1744.5   1721.9   -245.6   -134.4   -4.4   -8.1
       12600.   1068.4   1055.7   -156.3    -50.2   -2.7   -8.4
       15500.    701.7    693.6   -105.1    -18.8   -1.5   -8.6
       18400.    485.6    479.9    -73.9     -6.0   -0.7   -8.7
       21300.    350.0    345.8    -53.8     -0.5   -0.1   -8.8
       24200.    260.5    257.3    -40.4      1.8    0.4   -8.9
       27100.    199.1    196.7    -31.1      2.7    0.8   -9.0
       30000.    155.6    153.6    -24.4      3.0    1.1   -9.0'''
    to_30000_km = np.fromstring(txt, sep=' ')
    to_30000_km = to_30000_km.reshape(11, 7)
    
    txt = '''0.  33328.9  28850.0   4601.2  16041.2   28.8    9.1
           0.  33327.4  28848.8   4601.1  16040.3   28.8    9.1
           0.  33325.8  28847.5   4600.9  16039.4   28.8    9.1
           0.  33324.3  28846.3   4600.7  16038.6   28.8    9.1
           0.  33322.8  28845.0   4600.6  16037.7   28.8    9.1
           1.  33321.3  28843.7   4600.4  16036.9   28.8    9.1
           1.  33319.7  28842.5   4600.2  16036.0   28.8    9.1
           1.  33318.2  28841.2   4600.1  16035.2   28.8    9.1
           1.  33316.7  28839.9   4599.9  16034.3   28.8    9.1
           1.  33315.1  28838.7   4599.7  16033.4   28.8    9.1
           1.  33313.6  28837.4   4599.6  16032.6   28.8    9.1'''
    to_1_km = np.fromstring(txt, sep=' ')
    to_1_km = to_1_km.reshape(11, 7)
    
    # One bug found (at least vs. reference fortran code)
    #    -90.00  56594.5  -7540.3 -14452.7 -54196.0  -73.3  -62.4
    # -80.00  57699.8  -3291.2 -14982.0 -55623.5  -74.6  -77.6
    txt = '''
       -70.00  58317.1    457.0 -14882.8 -56384.2  -75.2  -88.2
       -60.00  58437.3   4566.8 -13966.8 -56559.7  -75.4  -71.9
       -50.00  57619.3   9312.7 -12284.3 -55518.9  -74.5  -52.8
       -40.00  55785.5  14631.1 -10067.3 -52882.9  -71.4  -34.5
       -30.00  53064.8  20673.4  -7604.5 -48276.9  -65.5  -20.2
       -20.00  49452.1  27404.5  -5215.9 -40832.6  -55.7  -10.8
       -10.00  45240.4  33980.3  -3247.8 -29689.8  -41.0   -5.5
         0.00  41736.0  38928.0  -1922.1 -14926.9  -21.0   -2.8
        10.00  40981.5  40909.8  -1155.4   2129.9    3.0   -1.6
        20.00  43884.8  39377.8   -614.8  19361.8   26.2   -0.9
        30.00  49111.8  34769.6     -0.2  34684.9   44.9    0.0
        40.00  54570.1  28145.2    747.5  46745.9   58.9    1.5
        50.00  58722.6  20584.6   1483.1  54976.5   69.4    4.1
        60.00  60643.2  13045.6   2006.6  59189.4   77.4    8.7
        70.00  60084.6   6700.8   2190.3  59669.6   83.3   18.1
        80.00  57982.3   2724.6   2072.6  57881.2   86.6   37.3
        90.00  56396.8   1161.1   1848.2  56354.5   87.8   57.9'''
    latitude_variation = np.fromstring(txt, sep=' ')
    # 90 row breaks
    latitude_variation = latitude_variation.reshape(17, 7)
    
    
    def validate_row(atm, row):
        assert_allclose(round(atm.declination, 1), row[6])
        assert_allclose(round(atm.inclination, 1), row[5])
        assert_allclose(round(atm.total_intensity, 1), row[1], rtol=1e-4)
        assert_allclose(round(atm.north_intensity, 1), row[2], rtol=1e-4)
        assert_allclose(round(atm.east_intensity, 1), row[3], rtol=1e-4, atol=.1)
        assert_allclose(round(atm.vertical_intensity, 1), row[4], rtol=1e-4, atol=.1)
    
    def ccmc_validate_H(lat, lon, year, Hs, array):
        for H, row in zip(Hs, array):
            atm = MagnetosphereIGRF(lat, lon, H, year)
            validate_row(atm, row)
    
    def ccmc_validate_lat(lats, lon, year, H, array):
        for lat, row in zip(lats, array):
            atm = MagnetosphereIGRF(lat, lon, H, year)
            validate_row(atm, row)

    ccmc_validate_H(-45, 50, 2018, np.linspace(0, 1e6, 11).tolist(), to_1000_km )
    ccmc_validate_H(0, 5, 2016, np.linspace(1000*1000, 30000*1000, 11).tolist(), to_30000_km)
    ccmc_validate_H(15, 200, 2012, np.linspace(0, 1000, 11).tolist(), to_1_km )
    
    ccmc_validate_lat(np.linspace(-90, 90, 19).tolist()[2:],
                     90, 1990, 0, latitude_variation)