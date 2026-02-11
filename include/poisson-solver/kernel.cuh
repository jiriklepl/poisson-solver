#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"

#include <cuda_runtime.h>

namespace poisson_solver {
void RunSolver2DFullPeriodic(cufftComplex *cmplx_workspace,
                             const cufftReal *denomx, const cufftReal *denomy,
                             std::size_t nx, std::size_t ny,
                             dim3 block_size_2d);
}  // namespace poisson_solver

#endif  // KERNEL_CUH
