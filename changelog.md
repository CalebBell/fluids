# Changelog

## [Unreleased]
- Expose C2K, K2C, F2C, C2F, F2K, K2F, C2R, K2R, F2R, R2C, R2K, R2F to public interface of fluids.core so they can be used with numba, fluids.vectorized
- Speed up some unit tests

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