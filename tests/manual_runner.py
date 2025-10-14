#!/usr/bin/env python3
"""
How to run this test file using micropython:

1. Install required modules using mip (MicroPython's package manager):
   micropython -c "import mip; mip.install('github:josverl/micropython-stubs/mip/typing_mpy.json')"
   micropython -c "import mip; mip.install('datetime')"
   micropython -c "import mip; mip.install('__future__')"

2. Run the test file with increased heap size:
   micropython -X heapsize=4M tests/manual_runner.py

Note: These modules will be installed to ~/.micropython/lib and only need to be installed once.
"""
import sys

import fluids.numerics
import test_drag
try:
    import test_drag
except:
    print("run this from the tests directory")
    sys.exit()
#import test_numerics
import test_atmosphere
import test_compressible
import test_control_valve
import test_core
import test_filters
import test_fittings
import test_flow_meter
import test_friction
import test_mixing
import test_nrlmsise00_full
import test_open_flow
import test_packed_bed
import test_packed_tower
import test_piping
import test_pump
import test_safety_valve
import test_saltation
import test_separator
import test_two_phase
import test_two_phase_voidage

to_test = [#test_numerics,
    #test_numerics_special,
           test_drag, test_control_valve, test_two_phase,
           test_two_phase_voidage, test_separator, test_piping, test_packed_bed,
           test_compressible, test_core,
           test_safety_valve, test_open_flow, test_filters, test_flow_meter,
           test_atmosphere, test_pump, test_friction, test_fittings,
           test_packed_tower, test_saltation, test_mixing, test_nrlmsise00_full]
#to_test.append([test_particle_size_distribution, test_jet_pump, test_geometry])

if fluids.numerics.is_micropython:
    skip_marks = ["slow", "fuzz", "scipy", "numpy", "f2py", "pytz", "numba"]
else:
    skip_marks = ["slow", "fuzz"]
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
        if callable(obj) and hasattr(obj, "__name__") and obj.__name__.startswith("test"):
            try:
                for bad in skip_marks:
                    if bad in obj.__dict__:
                        skip = True
                if "pytestmark" in obj.__dict__:
                    marked_names = [i.name for i in obj.__dict__["pytestmark"]]
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
                    print("FAILED TEST {} with error:".format(s))
                    print(e)

