#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"

#include <cuda_runtime.h>

namespace poisson_solver {
void RunSolver2DFullPeriodic(cufftComplex *complex_workspace,
                             const cufftReal *kDenom_x,
                             const cufftReal *kDenom_y, std::size_t n_x,
                             std::size_t n_y, std::size_t block_size);
}  // namespace poisson_solver

#endif  // KERNEL_CUH
