#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 07:28:17 2020

@author: caleb
"""

import test_drag
import test_control_valve
import test_geometry
import test_two_phase
import test_two_phase_voidage
import test_separator
import test_saltation
import test_piping
import test_packed_bed
import test_packed_tower
import test_compressible
import test_core
import test_particle_size_distribution
import test_safety_valve
import test_open_flow
import test_filters
import test_flow_meter
import test_atmosphere
import test_pump
import test_friction
import test_fittings
import test_jet_pump
import test_mixing

to_test = [test_drag, test_control_valve, test_geometry, test_two_phase, 
           test_two_phase_voidage, test_separator, test_piping, test_packed_bed,
           test_compressible, test_core, test_particle_size_distribution,
           test_safety_valve, test_open_flow, test_filters, test_flow_meter,
           test_atmosphere, test_pump, test_friction, test_fittings, test_jet_pump,
           test_packed_tower, test_saltation, test_mixing]

for mod in to_test:
    for s in dir(mod):
        obj = getattr(mod, s)
        if callable(obj) and hasattr(obj, '__name__') and obj.__name__.startswith('test'):
#            print(s)
            try:
                obj()
            except Exception as e:
                print('FAILED TEST %s with error:' %s)
                print(e)