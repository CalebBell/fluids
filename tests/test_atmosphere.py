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

from numpy.testing import assert_allclose
from fluids.atmosphere import ATMOSPHERE_1976, hwm93, hwm14



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
    # Disturbance wind model checks are not seperately implemented.
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