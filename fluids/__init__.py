"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Caleb Bell
<Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.
"""
# ruff: noqa: F403
import os

from . import constants, numerics

if not numerics.is_micropython: # type: ignore
    from . import (
        atmosphere,
        compressible,
        control_valve,
        core,
        drag,
        filters,
        fittings,
        flow_meter,
        friction,
        geometry,
        jet_pump,
        mixing,
        open_flow,
        packed_bed,
        packed_tower,
        particle_size_distribution,
        piping,
        pump,
        safety_valve,
        saltation,
        separator,
        two_phase,
        two_phase_voidage,
    )
    from .atmosphere import *
    from .compressible import *
    from .control_valve import *
    from .core import *
    from .drag import *
    from .filters import *
    from .fittings import *
    from .flow_meter import *
    from .friction import *
    from .geometry import *
    from .jet_pump import *
    from .mixing import *
    from .open_flow import *
    from .packed_bed import *
    from .packed_tower import *
    from .particle_size_distribution import *
    from .piping import *
    from .pump import *
    from .safety_valve import *
    from .saltation import *
    from .separator import *
    from .two_phase import *
    from .two_phase_voidage import *


    __all__ = [
        "atmosphere",
        "compressible",
        "control_valve",
        "core",
        "drag",
        "filters",
        "fittings",
        "flow_meter",
        "friction",
        "geometry",
        "jet_pump",
        "mixing",
        "open_flow",
        "packed_bed",
        "packed_tower",
        "particle_size_distribution",
        "piping",
        "pump",
        "safety_valve",
        "saltation",
        "separator",
        "two_phase",
        "two_phase_voidage",
    ]

    __all__.extend(atmosphere.__all__)
    __all__.extend(compressible.__all__)
    __all__.extend(control_valve.__all__)
    __all__.extend(core.__all__)
    __all__.extend(filters.__all__)

    __all__.extend(fittings.__all__)
    __all__.extend(friction.__all__)
    __all__.extend(geometry.__all__)
    __all__.extend(mixing.__all__)
    __all__.extend(open_flow.__all__)
    __all__.extend(flow_meter.__all__)
    __all__.extend(packed_bed.__all__)
    __all__.extend(piping.__all__)
    __all__.extend(pump.__all__)
    __all__.extend(safety_valve.__all__)
    __all__.extend(packed_tower.__all__)
    __all__.extend(two_phase.__all__)
    __all__.extend(two_phase_voidage.__all__)
    __all__.extend(drag.__all__)
    __all__.extend(saltation.__all__)
    __all__.extend(separator.__all__)
    __all__.extend(particle_size_distribution.__all__)
    __all__.extend(jet_pump.__all__)

    submodules = [atmosphere, compressible, core, friction, filters, fittings,
                  flow_meter, geometry, mixing, open_flow, packed_bed, piping, pump,
                  safety_valve, packed_tower, two_phase_voidage, two_phase, drag,
                  saltation, separator, particle_size_distribution, jet_pump,
                  control_valve,]

    def __getattr__(name):
        if name == "vectorized":
            import fluids.vectorized
            globals()[name] = fluids.vectorized
            return fluids.vectorized
        if name == "numba":
            import fluids.numba
            globals()[name] = fluids.numba
            return fluids.numba
        if name == "units":
            import fluids.units
            globals()[name] = fluids.units
            return fluids.units
        if name == "numba_vectorized":
            import fluids.numba_vectorized
            globals()[name] = fluids.numba_vectorized
            return fluids.numba_vectorized
        raise AttributeError(f"module {__name__} has no attribute {name}")

def all_submodules(with_numerics=True):
    import fluids.nrlmsise00.nrlmsise_00
    import fluids.nrlmsise00.nrlmsise_00_data
    import fluids.nrlmsise00.nrlmsise_00_header
    import fluids.optional
    import fluids.optional.irradiance
    import fluids.optional.spa
    from fluids.numerics import arrays, polynomial_evaluation, polynomial_roots, polynomial_utils
    new_submodules =  submodules + [fluids.optional, fluids.optional.irradiance, fluids.optional.spa,
                         fluids.nrlmsise00.nrlmsise_00_data, fluids.nrlmsise00.nrlmsise_00, fluids.nrlmsise00.nrlmsise_00_header,
                         ]
    if with_numerics:
        new_submodules += [arrays, polynomial_roots, polynomial_evaluation, polynomial_utils, constants, numerics]
    return new_submodules


__version__ = "1.2.6"

try:
    fluids_dir = os.path.dirname(__file__)
    fluids_data_dir = os.path.join(fluids_dir, "data")
except:
    pass
