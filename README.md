# poisson-solver

## Description
`poisson-solver` is a modern GPU-based library for solving the equation $`\Delta \phi = b`$ where $`\phi`$ and $`b`$ are functions of 1, 2 or 3 spatial dimensions
It is based on the [PoisFFT](https://github.com/LadaF/PoisFFT) library and provides the same functionalities.

## Requirements
`poisson-solver` always requires `VkFFT` for local FFTs and, if the distributed version is used, also `MPI` and `heFFTe`.
The FFT libraries are fetched automatically at CMake configure time.

By default, `poisson-solver` requires both a C++ and Fortran compiler.
If not building the Fortran binding (see [How to Build](#how-to-build)), Fortran compiler isn't required.

## How to Build
Navigate to the project root directory.
It is recommended to use one of the presents in `CMakePresets.json` (e.g., `cmake --preset release`).
More fine grained control can be achieved by altering some of the options, notably:
1. `BUILD_FORTRAN` to disable the build of Fortran binding and
2. `BUILD_SHARED_LIBS` as shared libs are built by default.
E.g., `cmake --preset release -DBUILD_FORTRAN=OFF -DBUILD_SHARED_LIBS=OFF`.

After generating the `Makefile`s, you can build the target
```bash
cmake --build build/
```
This creates a library inside the `build/lib/` directory (the debug library has a `_debug` suffix).
If Fortran binding is built, the `.mod` files are in `build/modules/`.

## Other Targets
### Tests
Apart from the default `poisson-solver` library target, there is also `tests` target for development and (maybe) illustratory purposes.

#### Acknowledgments
The Fortran tests are adapted from the [PoisFFT](https://github.com/LadaF/PoisFFT) library.
They have been modified to use the `poisson-solver` API and perform additional tests.
When using this library for your research please cite the `PoisFFT` article.

Vladimir Fuka: PoisFFT - a free parallel fast Poisson solver, Applied Mathematics and Computation, 2015

available online: http://dx.doi.org/10.1016/j.amc.2015.03.011 

preprint: http://arxiv.org/abs/1409.8116

#### How to Run Tests
Navigate to the `build/` directory and run `ctest -V`.
The type of tests performed depend on the chosen build type (local or distributed).
Beware that the tests can take several minutes to finish and in case of local tests won't output anything for roughly half of their runtime (`doctest` doesn't output unless a test fails).

### Docs
Next target is `doc` which can be used to generate the documentation for the current configuration.
There are differences in documentation for configurations with `USE_MPI ON` vs those with `USE_MPI OFF`.

The documentation will be in `build/docs/html/`.

## How to Use
TODO - How to use the built library.
