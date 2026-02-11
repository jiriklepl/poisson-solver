#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"

#include <cuda_runtime.h>

namespace poisson_solver {
void RunSolver2DFullPeriodic(cufftComplex *complex_workspace,
                             const cufftReal *denom_x, const cufftReal *denom_y,
                             std::size_t n_x, std::size_t n_y,
                             dim3 block_size_2d);
}  // namespace poisson_solver

#endif  // KERNEL_CUH
