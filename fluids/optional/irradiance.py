# -*- coding: utf-8 -*-
"""
irradiance.py from pvlib
========================
Extremely stripped down, reimplementation/vendorized version from:
https://github.com/pvlib/pvlib-python/

The rational for not including this library as a strict dependency is to avoid
including a dependency on pandas, keeping load time low, and PyPy compatibility

Most of the functions will import pvlib and use it for calculations, except
for one case which allows this to be used without `pvlib`

For a full list of contributors to this file, see the `pvlib` repository.


The copyright notice (BSD-3 clause) is as follows:
    
BSD 3-Clause License

Copyright (c) 2013-2018, Sandia National Laboratories and pvlib python Development Team
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the {organization} nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division
import os
import time
from datetime import datetime
import math
from math import degrees, sin, cos, tan, radians, asin, atan2, radians, exp, isnan
nan = float("nan")

from math import degrees, acos


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    projection = (
        cos(radians(surface_tilt)) * cos(radians(solar_zenith)) +
        sin(radians(surface_tilt)) * sin(radians(solar_zenith)) *
        cos(radians(solar_azimuth - surface_azimuth)))
    return projection

def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    projection = aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
    aoi_value = degrees(acos(projection))
    return aoi_value

def poa_components(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
    poa_direct = max(dni * cos(radians(aoi)), 0.0)
    
    if poa_sky_diffuse is not None:
        poa_diffuse = poa_sky_diffuse + poa_ground_diffuse
    else:
        poa_diffuse = poa_ground_diffuse
    poa_global = poa_direct + poa_diffuse

    irrads = {}
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
    if model == 'isotropic':
        return isotropic(surface_tilt, dhi)    
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
    z = zenith
    zenith_rad = radians(z)

    if 'kastenyoung1989' == model:
        try:
            am = (1.0 / (cos(zenith_rad) +
                  0.50572*(((6.07995 + (90.0 - z))**-1.6364))))
        except:
            am = nan
        if isinstance(am, complex):
            am = nan
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)
    return am


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
    try:
        bnci_2 = ((1.0 - (0.1 - 0.2*exp(-tl))/(0.1 + 0.882/fh1)) /
                  cos_zenith)
    except:
        bnci_2 = 1e20
    
    multiplier = (bnci_2 if bnci_2 > 0.0 else bnci_2)
    multiplier = 1e20 if multiplier > 1e20 else multiplier
    
    bnci_2 = ghi*multiplier

    dni = bnci if bnci < bnci_2 else bnci_2

    dhi = ghi - dni*cos_zenith

    return {'ghi': ghi, 'dni': dni, 'dhi': dhi}



