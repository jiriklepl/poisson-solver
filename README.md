# poisson-solver

## Description
TODO

## How to Build
### Generate Makefile
First, generate the `Makefile`s by running
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=<Yes|No> -B build-release/
```
You can choose between shared and static libraries with `DBUILD_SHARED_LIBS`.

It is possible to also generate a debug build by omitting `-DCMAKE_BUILD_TYPE=Release`.
In that case, it may be a good idea to rename the target directory to `build-debug/`.

## Make
After generating the `Makefile`s, you can make the target
```bash
make -C build-release/
```
(replace `build-release` with `build-debug` if you decided to generate a debug build).
This creates a shared/static library inside the `lib/` directory.

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
