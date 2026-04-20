# poisson-solver

## Description
TODO

## Requirements
`poisson-solver` requires a library for FFT computations.
In the local version that is `VkFFT` and in the distributed version `heFFTe`.
The distributed version also requires `MPI`.

## How to Build
### Generate Makefile
First, generate the `Makefile`s by running
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=<Yes|No> -DUSE_MPI=<Yes|No> -B build-release/
```
You can choose between shared and static libraries with `DBUILD_SHARED_LIBS`.
You can choose between local and distributed build with `DUSE_MPI`.

It is possible to also generate a debug build by changing `Release` to `Debug` in `-DCMAKE_BUILD_TYPE`.
In that case, it may be a good idea to rename the target directory to `build-debug/`.

## Make
After generating the `Makefile`s, you can make the target
```bash
make -C build-release/
```
(replace `build-release` with `build-debug` if you decided to generate a debug build).
This creates a shared/static library inside the `lib/` directory (the debug library has a `_debug` suffix).

## Other Targets
Apart from the default `poisson-solver` library target (generated in previous steps), there are other targets for development purposes.
Namely:
```
residuum-tests
fortran-residuum-tests
benchmarks
simulation
```
Running them is simplified with the `run_residuum_tests.sh`, `run_fortran_residuum_tests.sh` and `run_benchmarks.sh` that can be executed immediately after the `Makefile` generation.
TODO - Create runner for simulation.

## How to Use
TODO - How to use the built library.
