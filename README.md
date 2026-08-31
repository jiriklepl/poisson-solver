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
There are differences in documentation for configurations with `POISSON_SOLVER_USE_MPI ON` vs those with `POISSON_SOLVER_USE_MPI OFF`.

The documentation will be in `build/docs/html/`.

## How to Use

The library supports C++, C and Fortran.
In all languages, the typical workflow consists of three steps:
1. Initialization
2. Execution
3. Finalization

Throughout the examples it is assumed that all required headers apart from those provided by `poisson-solver` are included.

### C++ API

#### Local Solver
Include the header.
```cpp
#include <poisson-solver/solver.hpp>
```

Configure your grid.
```cpp
using Prec = double;
using namespace poisson_solver;

constexpr auto P = BC::Periodic;

GridSizes<3> ns = {64, 64, 64};
std::array<Prec, 3> lens = {1.0, 1.0, 1.0};
BCs<3> bcs = {BCPair{P, P}, BCPair{P, P}, BCPair{P, P}};
```

Initialize the solver.
```cpp
Solver<Prec, 3> solver(ns, lens, bcs, FiniteDifference2Approximation<Prec>{});
```

Prepare your data.
```cpp
std::vector<Prec> phi(64 * 64 * 64, 0.0);

// Filled with your RHS data
std::vector<Prec> rhs(64 * 64 * 64, 1.0);

// No ghost cells in this example
std::array<std::size_t, 3> ngs = {0, 0, 0};
```

Execute the solver on your grid.
```cpp
solver.Execute(phi, rhs, ngs, ngs);
```

In C++, the finalization is handled via RAII.

#### Distributed Solver (MPI)
Apart from `solver.hpp`, also include `mpi_utils.hpp` and create the communicator for `poisson-solver`.
```cpp
#include <poisson-solver/mpi_utils.hpp>

// Uses 1x2x2 decomposition, the X dimension contiguous.
MPI_Comm cart_comm = mpi_utils::InitMPIGrid(MPI_COMM_WORLD, {2, 2});
```

After `MPI_Init` but before constructing a distributed solver, call `SelectLocalDevice()` once per process.
You should skip this step if you already employ another mechanism by which you select devices for processes.
```cpp
mpi_utils::SelectLocalDevice(MPI_COMM_WORLD);
```

Get local grid dimensions and offsets.
```cpp
GridSizes<3> global_ns = {64, 64, 64};
auto local_info = mpi_utils::GetLocalGridInfo(global_ns, cart_comm);
```

The remaining steps are similar to the local solver use-case.
```cpp
Solver<Prec, 3> solver(
    local_info.ns, lens, bcs,
    global_ns, local_info.offs, cart_comm,
    FiniteDifference2Approximation<Prec>{}
);

solver.Execute(local_phi, local_rhs, ngs, ngs);
```

#### GPU API
The library is also able to execute directly on device buffers which saves host->device copy (but device->device copy may take place due to the slicing of ghost cells).
```cpp
// Assumes that `cu_phi` and `cu_rhs` are device pointers.
solver.ExecuteGPU(cu_phi, cu_rhs, ngs_phi, ngs_rhs);
```

### C API
There is a different header file for the C API and functions using single-precision data are suffixed with `Float`.

```c
#include <poisson-solver/c_binding.h>

PoissonSolver solver;
int ns[3] = {64, 64, 64};
double lens[3] = {1.0, 1.0, 1.0};
BoundaryCondition bcs[6] = {Periodic, Periodic, Periodic, Periodic, Periodic, Periodic};
int ngs[3] = {0, 0, 0};

PoissonSolverNew(&solver, 3, ns, lens, bcs, FD2, NULL, NULL, NULL);

PoissonSolverExecute(&solver, phi, rhs, ngs, ngs);
// Or PoissonSolverExecuteGPU(&solver, cu_phi, cu_rhs, ngs, ngs);

PoissonSolverFinalize(&solver);
```

### Fortran API
The Fortran module `poisson_solver` exposes the `PoissonSolver` type with overloaded subroutines that automatically infer dimensionality.
Single-precision versions are suffixed with `_float`.

```fortran
use poisson_solver

type(PoissonSolver) :: solver
integer :: ns(3) = [64, 64, 64]
real(c_double) :: lens(3) = [1.0, 1.0, 1.0]
integer(c_int) :: bcs(6) = [ &
    PoisSolver_Periodic, PoisSolver_Periodic, &
    PoisSolver_Periodic, PoisSolver_Periodic, &
    PoisSolver_Periodic, PoisSolver_Periodic]

call poisson_solver_new(solver, 3, ns, lens, bcs, PoisSolver_FiniteDifference2)

call poisson_solver_execute(solver, phi, rhs)
! Or call poisson_solver_execute_gpu(solver, cu_phi, cu_rhs)

call poisson_solver_finalize(solver)
```

### NonuniformZSolver
Both local and MPI builds also feature a solver for grids with non-uniform z axis.
The usage is very similar to that of uniform solvers.
