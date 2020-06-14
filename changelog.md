# Changelog

## [Unreleased]

## [0.1.82] - 2020-06-14

### Added
- Expose C2K, K2C, F2C, C2F, F2K, K2F, C2R, K2R, F2R, R2C, R2K, R2F to public interface of fluids.core so they can be used with numba, fluids.vectorized
- Speed up some unit tests
- Add numba interface which offers speed boosts and allows interoperability with programs using numba
- Added methods friction_factor_methods, friction_factor_curved_methods, dP_packed_bed_methods, drag_sphere_methods, two_phase_dP_methods, gas_liquid_viscosity_methods, liquid_gas_voidage_methods to easily find which correlations are applicable to the a given parameter range.

### Removed
- The AvailableMethods argument in the functions which had it is now deprecated! Changing the return type of a function does not work with numba, and will limit the future ability of fluids to possibly provide typing information i.e. for MyPy. It is also considered bad practice by some, although it is quite popular and used in libraries like SciPy.


## [0.1.81] - 2020-06-06
### Added
- Updated solar_irradiation, solar_position, earthsun_distance, sunrise_sunset to take and return timezone information.

### Changed
- Updated tutorial with new results; almost all are small changes.
- Preliminary work on a numba interface.
- Tentative adaptive `quad` function implementation to perform integration fast with PyPy
- Misc try/excepts which allow the library to be loaded in micropython
- Misc try/excepts which allow the library to be loaded in IronPython

### Removed
- Nothing

### Fixed
- solar_irradiation, and solar_position incorrectly were assuming the timezone of the computer the calculation was run on. This has been remedied; the documentation has been updated to show the timezone now can be in either UTC time zone or with a time zone information provided.