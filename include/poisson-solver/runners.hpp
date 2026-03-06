#ifndef RUNNERS_HPP
#define RUNNERS_HPP

#include <array>
#include <cuda_runtime.h>
#include <cufft.h>

namespace poisson_solver {
template <std::size_t Dim>
// Initializes the complex workspace `cmplx_workspace` with values from `rhs`
void RunInitCmplxWorkspace(cufftDoubleComplex *cmplx_workspace,
                           const cufftDoubleReal *rhs,
                           std::array<std::size_t, Dim> ns, dim3 grid_size,
                           dim3 block_size);

template <std::size_t Dim>
void RunDivideCmplxWorkspaceByDenoms(
    cufftDoubleComplex *cmplx_workspace, std::array<std::size_t, Dim> ns,
    std::array<const cufftDoubleReal *, Dim> denoms, dim3 grid_size,
    dim3 block_size);

// Fills `phi` with normalized values of `cmplx_workspace`
template <std::size_t Dim>
void RunNormalization(cufftDoubleReal *phi,
                      const cufftDoubleComplex *cmplx_workspace,
                      std::array<std::size_t, Dim> ns, double norm_factor,
                      dim3 grid_size, dim3 block_size);
}  // namespace poisson_solver
#endif  // RUNNERS_HPP
