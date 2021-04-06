#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import fluids.numerics
try:
    import test_drag
except:
    print('run this from the tests directory')
    exit()
#import test_numerics
import test_numerics_special
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
import test_nrlmsise00_full

to_test = [#test_numerics, 
    test_numerics_special, 
           test_drag, test_control_valve, test_two_phase,
           test_two_phase_voidage, test_separator, test_piping, test_packed_bed,
           test_compressible, test_core,
           test_safety_valve, test_open_flow, test_filters, test_flow_meter,
           test_atmosphere, test_pump, test_friction, test_fittings,
           test_packed_tower, test_saltation, test_mixing, test_nrlmsise00_full]
#to_test.append([test_particle_size_distribution, test_jet_pump, test_geometry])

if fluids.numerics.is_micropython or fluids.numerics.is_ironpython:
    skip_marks = ['slow', 'fuzz', 'scipy', 'numpy', 'f2py', 'pytz', 'numba']
else:
    skip_marks = ['slow', 'fuzz']
# pytz loads but doesn't work right in ironpython
skip_marks_set = set(skip_marks)
if len(sys.argv) >= 2:
    #print(sys.argv)
    # Run modules specified by user
    to_test = [globals()[i] for i in sys.argv[1:]]
for mod in to_test:
    print(mod)
    for s in dir(mod):
        skip = False
        obj = getattr(mod, s)
        if callable(obj) and hasattr(obj, '__name__') and obj.__name__.startswith('test'):
            try:
                for bad in skip_marks:
                    if bad in obj.__dict__:
                        skip = True
                if 'pytestmark' in obj.__dict__:
                    marked_names = [i.name for i in obj.__dict__['pytestmark']]
                    for mark_name in marked_names:
                        if mark_name in skip_marks_set:
                            skip = True
            except Exception as e:
                #print(e)
                pass
            if not skip:
                try:
                    #print(obj)
                    obj()
                except Exception as e:
                    print('FAILED TEST %s with error:' %s)
                    print(e)
