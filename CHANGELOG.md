# Changelog

All notable changes to this project will be documented in this file.

## Version 0.3

### Added

### Changed

### Deprecated

### Removed

### Fixed
optimization button in demo environment

## Version 0.2

### Added
Memoization functionality for `TopFarmProblem.optimize()` method

### Changed
seperated functionality in multiple files;

- simplified controller class (in file app.py): now only contains views and button-related methods. 
- All static wind-farm/PyWake related functions moved to [`wind_farm.py`](lib/wind_farm.py)
- constants moved to seperate [`constants.py`](lib/constants.py) file
- parametrization class moved to seperate [`parametrization.py`](lib/parametrization.py) file
- all files, except for [`app.py`](app.py), grouped in [lib](lib) folder


### Deprecated

### Removed
Old way of saving optimized positions and aep data to lib folder and recovering them for optimized positions plot in the last step. Memoization described above is used instead.

### Fixed

## Version 0.1

### Added
First version of wind farm optimization with PyWake!

### Changed

### Deprecated

### Removed

### Fixed