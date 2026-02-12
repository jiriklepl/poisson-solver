#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"

#include <cuda_runtime.h>

namespace poisson_solver {
// Initialize the complex workspace `cmplx_workspace` of size `nx * ny` with
// values from `rhs`
void RunInitCmplxWorkspace(cufftComplex *cmplx_workspace, const cufftReal *rhs,
                           std::size_t nx, std::size_t ny, dim3 grid_size_2d,
                           dim3 block_size_2d);

void RunDivideCmplxWorkspaceByDenoms(cufftComplex *cmplx_workspace,
                                     std::size_t nx, std::size_t ny,
                                     const cufftReal *denomx,
                                     const cufftReal *denomy, dim3 grid_size_2d,
                                     dim3 block_size_2d);

// Fill `phi` with normalized values of `cmplx_workspace`
void RunNormalization(cufftReal *phi, const cufftComplex *cmplx_workspace,
                      std::size_t nx, std::size_t ny, std::size_t norm_factor,
                      dim3 grid_size_2d, dim3 block_size_2d);
}  // namespace poisson_solver

#endif  // KERNEL_CUH
