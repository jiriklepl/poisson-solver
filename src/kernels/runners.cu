#include "poisson-solver/kernel.cuh"
#include "poisson-solver/runners.hpp"

namespace poisson_solver {
template <std::size_t Dim>
void RunInitCmplxWorkspace(cufftDoubleComplex *cmplx_workspace,
                           const cufftDoubleReal *rhs,
                           std::array<std::size_t, Dim> ns, dim3 grid_size,
                           dim3 block_size) {
    InitCmplxWorkspace<<<grid_size, block_size>>>(cmplx_workspace, rhs, ns);
}

template <std::size_t Dim>
void RunDivideCmplxWorkspaceByDenoms(
    cufftDoubleComplex *cmplx_workspace, std::array<std::size_t, Dim> ns,
    std::array<const cufftDoubleReal *, Dim> denoms, dim3 grid_size,
    dim3 block_size) {
    DivideCmplxWorkspaceByDenoms<<<grid_size, block_size>>>(cmplx_workspace, ns,
                                                            denoms);
}

template <std::size_t Dim>
void RunNormalization(cufftDoubleReal *phi,
                      const cufftDoubleComplex *cmplx_workspace,
                      std::array<std::size_t, Dim> ns, double norm_factor,
                      dim3 grid_size, dim3 block_size) {
    NormalizeCmplxWorkspaceToPhi<<<grid_size, block_size>>>(
        phi, cmplx_workspace, ns, norm_factor);
}

// Explicit instantiations
template void RunInitCmplxWorkspace<1>(cufftDoubleComplex *cmplx_workspace,
                                       const cufftDoubleReal *rhs,
                                       std::array<std::size_t, 1> ns,
                                       dim3 grid_size, dim3 block_size);
template void RunInitCmplxWorkspace<2>(cufftDoubleComplex *cmplx_workspace,
                                       const cufftDoubleReal *rhs,
                                       std::array<std::size_t, 2> ns,
                                       dim3 grid_size, dim3 block_size);
template void RunInitCmplxWorkspace<3>(cufftDoubleComplex *cmplx_workspace,
                                       const cufftDoubleReal *rhs,
                                       std::array<std::size_t, 3> ns,
                                       dim3 grid_size, dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<1>(
    cufftDoubleComplex *cmplx_workspace, std::array<std::size_t, 1> ns,
    std::array<const cufftDoubleReal *, 1> denoms, dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<2>(
    cufftDoubleComplex *cmplx_workspace, std::array<std::size_t, 2> ns,
    std::array<const cufftDoubleReal *, 2> denoms, dim3 grid_size,
    dim3 block_size);
template void RunDivideCmplxWorkspaceByDenoms<3>(
    cufftDoubleComplex *cmplx_workspace, std::array<std::size_t, 3> ns,
    std::array<const cufftDoubleReal *, 3> denoms, dim3 grid_size,
    dim3 block_size);
template void RunNormalization<1>(cufftDoubleReal *phi,
                                  const cufftDoubleComplex *cmplx_workspace,
                                  std::array<std::size_t, 1> ns,
                                  double norm_factor, dim3 grid_size,
                                  dim3 block_size);
template void RunNormalization<2>(cufftDoubleReal *phi,
                                  const cufftDoubleComplex *cmplx_workspace,
                                  std::array<std::size_t, 2> ns,
                                  double norm_factor, dim3 grid_size,
                                  dim3 block_size);
template void RunNormalization<3>(cufftDoubleReal *phi,
                                  const cufftDoubleComplex *cmplx_workspace,
                                  std::array<std::size_t, 3> ns,
                                  double norm_factor, dim3 grid_size,
                                  dim3 block_size);
}  // namespace poisson_solver
