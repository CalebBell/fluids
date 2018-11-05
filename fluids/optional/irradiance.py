"""
"""
from __future__ import division
import os
import time
from datetime import datetime
import math
from math import degrees, sin, cos, tan, radians, asin, atan2, radians, exp, isnan


import numpy as np


from math import degrees, acos
from collections import OrderedDict


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    projection = (
        cos(radians(surface_tilt)) * cos(radians(solar_zenith)) +
        sin(radians(surface_tilt)) * sin(radians(solar_zenith)) *
        cos(radians(solar_azimuth) - surface_azimuth))
    return projection


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    projection = aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
    aoi_value = degrees(acos(projection))
    return aoi_value


def poa_components(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
    poa_direct = max(dni * cos(radians(aoi)), 0)
    poa_diffuse = poa_sky_diffuse + poa_ground_diffuse
    poa_global = poa_direct + poa_diffuse

    irrads = OrderedDict()
    irrads['poa_global'] = poa_global
    irrads['poa_direct'] = poa_direct
    irrads['poa_diffuse'] = poa_diffuse
    irrads['poa_sky_diffuse'] = poa_sky_diffuse
    irrads['poa_ground_diffuse'] = poa_ground_diffuse

    return irrads

def get_ground_diffuse(surface_tilt, ghi, albedo=.25, surface_type=None):
    diffuse_irrad = ghi*albedo*(1.0 - cos(radians(surface_tilt)))*0.5
    return diffuse_irrad

def get_sky_diffuse(surface_tilt, surface_azimuth,
                    solar_zenith, solar_azimuth,
                    dni, ghi, dhi, dni_extra=None, airmass=None,
                    model='isotropic',
                    model_perez='allsitescomposite1990'):    
    if model == 'perez':
        return perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                    solar_zenith, solar_azimuth, airmass,
                    model=model_perez)
    if model == 'isotropic':
        sky = isotropic(surface_tilt, dhi)    
    else:
        from pvlib import get_sky_diffuse
        return get_sky_diffuse(surface_tilt, surface_azimuth,
                    solar_zenith, solar_azimuth,
                    dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass,
                    model=model,
                    model_perez=model_perez)
        
        
def get_absolute_airmass(airmass_relative, pressure=101325.):
    airmass_absolute = airmass_relative*pressure/101325.
    return airmass_absolute

def get_relative_airmass(zenith, model='kastenyoung1989'):
    # need to filter first because python 2.7 does not support raising a
    # negative number to a negative power.
#     z = np.where(zenith > 90, np.nan, zenith)
    z = zenith
    zenith_rad = radians(z)

    if 'kastenyoung1989' == model:
        am = (1.0 / (cos(zenith_rad) +
              0.50572*(((6.07995 + (90.0 - z))**-1.6364))))
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)

    return am

coeffdict = {
    'allsitescomposite1990': [
        [-0.0080,    0.5880,   -0.0620,   -0.0600,    0.0720,   -0.0220],
        [0.1300,    0.6830,   -0.1510,   -0.0190,    0.0660,   -0.0290],
        [0.3300,    0.4870,   -0.2210,    0.0550,   -0.0640,   -0.0260],
        [0.5680,    0.1870,   -0.2950,    0.1090,   -0.1520,   -0.0140],
        [0.8730,   -0.3920,   -0.3620,    0.2260,   -0.4620,    0.0010],
        [1.1320,   -1.2370,   -0.4120,    0.2880,   -0.8230,    0.0560],
        [1.0600,   -1.6000,   -0.3590,    0.2640,   -1.1270,    0.1310],
        [0.6780,   -0.3270,   -0.2500,    0.1560,   -1.3770,    0.2510]],
    'allsitescomposite1988': [
        [-0.0180,    0.7050,   -0.071,   -0.0580,    0.1020,   -0.0260],
        [0.1910,    0.6450,   -0.1710,    0.0120,    0.0090,   -0.0270],
        [0.4400,    0.3780,   -0.2560,    0.0870,   -0.1040,   -0.0250],
        [0.7560,   -0.1210,   -0.3460,    0.1790,   -0.3210,   -0.0080],
        [0.9960,   -0.6450,   -0.4050,    0.2600,   -0.5900,    0.0170],
        [1.0980,   -1.2900,   -0.3930,    0.2690,   -0.8320,    0.0750],
        [0.9730,   -1.1350,   -0.3780,    0.1240,   -0.2580,    0.1490],
        [0.6890,   -0.4120,   -0.2730,    0.1990,   -1.6750,    0.2370]],
    'sandiacomposite1988': [
        [-0.1960,    1.0840,   -0.0060,   -0.1140,    0.1800,   -0.0190],
        [0.2360,    0.5190,   -0.1800,   -0.0110,    0.0200,   -0.0380],
        [0.4540,    0.3210,   -0.2550,    0.0720,   -0.0980,   -0.0460],
        [0.8660,   -0.3810,   -0.3750,    0.2030,   -0.4030,   -0.0490],
        [1.0260,   -0.7110,   -0.4260,    0.2730,   -0.6020,   -0.0610],
        [0.9780,   -0.9860,   -0.3500,    0.2800,   -0.9150,   -0.0240],
        [0.7480,   -0.9130,   -0.2360,    0.1730,   -1.0450,    0.0650],
        [0.3180,   -0.7570,    0.1030,    0.0620,   -1.6980,    0.2360]],
    'usacomposite1988': [
        [-0.0340,    0.6710,   -0.0590,   -0.0590,    0.0860,   -0.0280],
        [0.2550,    0.4740,   -0.1910,    0.0180,   -0.0140,   -0.0330],
        [0.4270,    0.3490,   -0.2450,    0.0930,   -0.1210,   -0.0390],
        [0.7560,   -0.2130,   -0.3280,    0.1750,   -0.3040,   -0.0270],
        [1.0200,   -0.8570,   -0.3850,    0.2800,   -0.6380,   -0.0190],
        [1.0500,   -1.3440,   -0.3480,    0.2800,   -0.8930,    0.0370],
        [0.9740,   -1.5070,   -0.3700,    0.1540,   -0.5680,    0.1090],
        [0.7440,   -1.8170,   -0.2560,    0.2460,   -2.6180,    0.2300]],
    'france1988': [
        [0.0130,    0.7640,   -0.1000,   -0.0580,    0.1270,   -0.0230],
        [0.0950,    0.9200,   -0.1520,         0,    0.0510,   -0.0200],
        [0.4640,    0.4210,   -0.2800,    0.0640,   -0.0510,   -0.0020],
        [0.7590,   -0.0090,   -0.3730,    0.2010,   -0.3820,    0.0100],
        [0.9760,   -0.4000,   -0.4360,    0.2710,   -0.6380,    0.0510],
        [1.1760,   -1.2540,   -0.4620,    0.2950,   -0.9750,    0.1290],
        [1.1060,   -1.5630,   -0.3980,    0.3010,   -1.4420,    0.2120],
        [0.9340,   -1.5010,   -0.2710,    0.4200,   -2.9170,    0.2490]],
    'phoenix1988': [
        [-0.0030,    0.7280,   -0.0970,   -0.0750,    0.1420,   -0.0430],
        [0.2790,    0.3540,   -0.1760,    0.0300,   -0.0550,   -0.0540],
        [0.4690,    0.1680,   -0.2460,    0.0480,   -0.0420,   -0.0570],
        [0.8560,   -0.5190,   -0.3400,    0.1760,   -0.3800,   -0.0310],
        [0.9410,   -0.6250,   -0.3910,    0.1880,   -0.3600,   -0.0490],
        [1.0560,   -1.1340,   -0.4100,    0.2810,   -0.7940,   -0.0650],
        [0.9010,   -2.1390,   -0.2690,    0.1180,   -0.6650,    0.0460],
        [0.1070,    0.4810,    0.1430,   -0.1110,   -0.1370,    0.2340]],
    'elmonte1988': [
        [0.0270,    0.7010,   -0.1190,   -0.0580,    0.1070,  -0.0600],
        [0.1810,    0.6710,   -0.1780,   -0.0790,    0.1940,  -0.0350],
        [0.4760,    0.4070,   -0.2880,    0.0540,   -0.0320,  -0.0550],
        [0.8750,   -0.2180,   -0.4030,    0.1870,   -0.3090,  -0.0610],
        [1.1660,   -1.0140,   -0.4540,    0.2110,   -0.4100,  -0.0440],
        [1.1430,   -2.0640,   -0.2910,    0.0970,   -0.3190,   0.0530],
        [1.0940,   -2.6320,   -0.2590,    0.0290,   -0.4220,   0.1470],
        [0.1550,    1.7230,    0.1630,   -0.1310,   -0.0190,   0.2770]],
    'osage1988': [
        [-0.3530,    1.4740,   0.0570,   -0.1750,    0.3120,   0.0090],
        [0.3630,    0.2180,  -0.2120,    0.0190,   -0.0340,  -0.0590],
        [-0.0310,    1.2620,  -0.0840,   -0.0820,    0.2310,  -0.0170],
        [0.6910,    0.0390,  -0.2950,    0.0910,   -0.1310,  -0.0350],
        [1.1820,   -1.3500,  -0.3210,    0.4080,   -0.9850,  -0.0880],
        [0.7640,    0.0190,  -0.2030,    0.2170,   -0.2940,  -0.1030],
        [0.2190,    1.4120,   0.2440,    0.4710,   -2.9880,   0.0340],
        [3.5780,   22.2310, -10.7450,    2.4260,    4.8920,  -5.6870]],
    'albuquerque1988': [
        [0.0340,    0.5010,  -0.0940,   -0.0630,    0.1060,  -0.0440],
        [0.2290,    0.4670,  -0.1560,   -0.0050,   -0.0190,  -0.0230],
        [0.4860,    0.2410,  -0.2530,    0.0530,   -0.0640,  -0.0220],
        [0.8740,   -0.3930,  -0.3970,    0.1810,   -0.3270,  -0.0370],
        [1.1930,   -1.2960,  -0.5010,    0.2810,   -0.6560,  -0.0450],
        [1.0560,   -1.7580,  -0.3740,    0.2260,   -0.7590,   0.0340],
        [0.9010,   -4.7830,  -0.1090,    0.0630,   -0.9700,   0.1960],
        [0.8510,   -7.0550,  -0.0530,    0.0600,   -2.8330,   0.3300]],
    'capecanaveral1988': [
        [0.0750,    0.5330,   -0.1240,  -0.0670,   0.0420,  -0.0200],
        [0.2950,    0.4970,   -0.2180,  -0.0080,   0.0030,  -0.0290],
        [0.5140,    0.0810,   -0.2610,   0.0750,  -0.1600,  -0.0290],
        [0.7470,   -0.3290,   -0.3250,   0.1810,  -0.4160,  -0.0300],
        [0.9010,   -0.8830,   -0.2970,   0.1780,  -0.4890,   0.0080],
        [0.5910,   -0.0440,   -0.1160,   0.2350,  -0.9990,   0.0980],
        [0.5370,   -2.4020,    0.3200,   0.1690,  -1.9710,   0.3100],
        [-0.8050,    4.5460,    1.0720,  -0.2580,  -0.9500,    0.7530]],
    'albany1988': [
        [0.0120,    0.5540,   -0.0760, -0.0520,   0.0840,  -0.0290],
        [0.2670,    0.4370,   -0.1940,  0.0160,   0.0220,  -0.0360],
        [0.4200,    0.3360,   -0.2370,  0.0740,  -0.0520,  -0.0320],
        [0.6380,   -0.0010,   -0.2810,  0.1380,  -0.1890,  -0.0120],
        [1.0190,   -1.0270,   -0.3420,  0.2710,  -0.6280,   0.0140],
        [1.1490,   -1.9400,   -0.3310,  0.3220,  -1.0970,   0.0800],
        [1.4340,   -3.9940,   -0.4920,  0.4530,  -2.3760,   0.1170],
        [1.0070,   -2.2920,   -0.4820,  0.3900,  -3.3680,   0.2290]], }

def _get_perez_coefficients(perezmodel):
    
    dat = coeffdict[perezmodel]
    F1coeffs = [i[:3] for i in dat]
    F2coeffs = [i[3:] for i in dat]
    return F1coeffs, F2coeffs

#     array = np.array(coeffdict[perezmodel])
#     F1coeffs = array[:, 0:3]
#     F2coeffs = array[:, 3:7]
#     return F1coeffs, F2coeffs


def get_total_irradiance(surface_tilt, surface_azimuth,
                         solar_zenith, solar_azimuth,
                         dni, ghi, dhi, dni_extra=None, airmass=None,
                         albedo=.25, surface_type=None,
                         model='isotropic',
                         model_perez='allsitescomposite1990', **kwargs):
    poa_sky_diffuse = get_sky_diffuse(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass, model=model,
        model_perez=model_perez)

    poa_ground_diffuse = get_ground_diffuse(surface_tilt, ghi, albedo,
                                            surface_type)
    aoi_ = aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    irrads = poa_components(aoi_, dni, poa_sky_diffuse, poa_ground_diffuse)
    return irrads


def isotropic(surface_tilt, dhi):
    sky_diffuse = dhi * (1 + cos(radians(surface_tilt))) * 0.5

    return sky_diffuse


def ineichen(apparent_zenith, airmass_absolute, linke_turbidity,
             altitude=0, dni_extra=1364., perez_enhancement=False):
    if isnan(airmass_absolute) or isnan(apparent_zenith):
        return {'ghi': 0.0, 'dni': 0.0, 'dhi': 0.0}
#    print(apparent_zenith, airmass_absolute, linke_turbidity, altitude, dni_extra, perez_enhancement)
    # ghi is calculated using either the equations in [1] by setting
    # perez_enhancement=False (default behavior) or using the model
    # in [2] by setting perez_enhancement=True.

    # The NaN handling is a little subtle. The AM input is likely to
    # have NaNs that we'll want to map to 0s in the output. However, we
    # want NaNs in other inputs to propagate through to the output. This
    # is accomplished by judicious use and placement of np.maximum,
    # np.minimum, and np.fmax

    # use max so that nighttime values will result in 0s instead of
    # negatives. propagates nans.
    cos_zenith = cos(radians(apparent_zenith))
    if cos_zenith < 0.0:
        cos_zenith = 0.0

    tl = linke_turbidity
    fh1 = exp(-altitude/8000.)
    fh2 = exp(-altitude/1250.)
    cg1 = 5.09e-5*altitude + 0.868
    cg2 = 3.92e-5*altitude + 0.0387
    ghi = exp(-cg2*airmass_absolute*(fh1 + fh2*(tl - 1.0)))

    # https://github.com/pvlib/pvlib-python/issues/435
    if perez_enhancement:
        ghi *= exp(0.01*airmass_absolute**1.8)

    # use fmax to map airmass nans to 0s. multiply and divide by tl to
    # reinsert tl nans
    if ghi > 0.0:
        ghi = cg1 * dni_extra * cos_zenith * tl / tl * ghi
    else:
        ghi = 0.0

    # BncI = "normal beam clear sky radiation"
    b = 0.664 + 0.163/fh1
    bnci = b * exp(-0.09 * airmass_absolute * (tl - 1))    
    if bnci > 0.0:
        bnci = dni_extra * bnci
    else:
        bnci = 0.0
        
    # "empirical correction" SE 73, 157 & SE 73, 312.
    bnci_2 = ((1.0 - (0.1 - 0.2*exp(-tl))/(0.1 + 0.882/fh1)) /
              cos_zenith)
    
    multiplier = (bnci_2 if bnci_2 > 0.0 else bnci_2)
    multiplier = 1e20 if multiplier > 1e20 else multiplier
    
    bnci_2 = ghi*multiplier

    dni = bnci if bnci < bnci_2 else bnci_2

    dhi = ghi - dni*cos_zenith


#     return {'ghi': ghi, 'dni': dni, 'dhi': dhi}
    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi
    return irrads



def perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
          solar_zenith, solar_azimuth, airmass,
          model='allsitescomposite1990', return_components=False):
    kappa = 1.041  # for solar_zenith in radians
    z = radians(solar_zenith)  # convert to radians

    # delta is the sky's "brightness"
    delta = dhi * airmass / dni_extra

    # epsilon is the sky's "clearness"
    z3 = z*z*z
    eps = ((dhi + dni) / dhi + kappa * (z3)) / (1.0 + kappa * (z3))

    # Perez et al define clearness bins according to the following
    # rules. 1 = overcast ... 8 = clear (these names really only make
    # sense for small zenith angles, but...) these values will
    # eventually be used as indicies for coeffecient look ups
    ebin = np.digitize(eps, (0., 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2))
    ebin[np.isnan(eps)] = 0

    # correct for 0 indexing in coeffecient lookup
    # later, ebin = -1 will yield nan coefficients
    ebin -= 1

    # The various possible sets of Perez coefficients are contained
    # in a subfunction to clean up the code.
    F1c, F2c = _get_perez_coefficients(model)

    # results in invalid eps (ebin = -1) being mapped to nans
    nans = np.array([np.nan, np.nan, np.nan])
    F1c = np.vstack((F1c, nans))
    F2c = np.vstack((F2c, nans))

    F1 = (F1c[ebin, 0] + F1c[ebin, 1] * delta + F1c[ebin, 2] * z)
    F1 = np.maximum(F1, 0)

    F2 = (F2c[ebin, 0] + F2c[ebin, 1] * delta + F2c[ebin, 2] * z)
    F2 = np.maximum(F2, 0)

    A = aoi_projection(surface_tilt, surface_azimuth,
                       solar_zenith, solar_azimuth)
    A = np.maximum(A, 0)

    B = cos(radians(solar_zenith))
    B = np.maximum(B, tools.cosd(85))

    # Calculate Diffuse POA from sky dome
    term1 = 0.5 * (1 - F1) * (1 + cos(radians(surface_tilt)))
    term2 = F1 * A / B
    term3 = F2 * sin(radians(surface_tilt))

    sky_diffuse = np.maximum(dhi * (term1 + term2 + term3), 0)

    # we've preserved the input type until now, so don't ruin it!
    sky_diffuse = np.where(np.isnan(airmass), 0, sky_diffuse)

    if return_components:
        diffuse_components = OrderedDict()
        diffuse_components['sky_diffuse'] = sky_diffuse

        # Calculate the different components
        diffuse_components['isotropic'] = dhi * term1
        diffuse_components['circumsolar'] = dhi * term2
        diffuse_components['horizon'] = dhi * term3

        # Set values of components to 0 when sky_diffuse is 0
        mask = sky_diffuse == 0
        diffuse_components = {k: np.where(mask, 0, v) for k, v in
                              diffuse_components.items()}
        return diffuse_components
    else:
        return sky_diffuse