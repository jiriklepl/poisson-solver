# poisson-solver

## Description
TODO

## Requirements
`poisson-solver` requires a library for FFT computations.
In the local version that is `VkFFT` and in the distributed version `heFFTe` (these libraries are installed during the build process).
The distributed version also requires `MPI`.

## How to Build
### Generate Makefile
First, generate the `Makefile`s.
You can use one of the presets from `CMakePresets.json` or generate them manually:
```bash
cmake -DCMAKE_BUILD_TYPE=<Release|Debug> -DBUILD_SHARED_LIBS=<Yes|No> -DUSE_MPI=<Yes|No> -B build/
```

## Make
After generating the `Makefile`s, you can make the target
```bash
make -C build/
```
This creates a shared or static library inside the `lib/` directory (the debug library has a `_debug` suffix).

## Other Targets
Apart from the default `poisson-solver` library target (generated in previous steps), there are other targets for development purposes.
Namely:
```
residuum-tests
fortran-residuum-tests
benchmarks
simulation
```

### Runners
You can run the `residuum-tests` and `fortran-residuum-tests` with `ctest --test-dir <build-dir>`.
For benchmarks there is the `run_benchmarks.sh` runner.
TODO - Create runner for simulation.

## How to Use
TODO - How to use the built library.
