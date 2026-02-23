#include "poisson-solver/kernel.cuh"
#include "poisson-solver/runners.hpp"

namespace poisson_solver {
template <std::size_t Dim>
void RunInitCmplxWorkspace(cufftComplex *cmplx_workspace, const cufftReal *rhs,
                           std::array<std::size_t, Dim> ns, dim3 grid_size,
                           dim3 block_size) {
    InitCmplxWorkspace<<<grid_size, block_size>>>(cmplx_workspace, rhs, ns);
}

template <std::size_t Dim>
void RunDivideCmplxWorkspaceByDenoms(cufftComplex *cmplx_workspace,
                                     std::array<std::size_t, Dim> ns,
                                     std::array<const cufftReal *, Dim> denoms,
                                     dim3 grid_size, dim3 block_size) {
    DivideCmplxWorkspaceByDenoms<<<grid_size, block_size>>>(cmplx_workspace, ns,
                                                            denoms);
}

template <std::size_t Dim>
void RunNormalization(cufftReal *phi, const cufftComplex *cmplx_workspace,
                      std::array<std::size_t, Dim> ns, float norm_factor,
                      dim3 grid_size, dim3 block_size) {
    NormalizeCmplxWorkspaceToPhi<<<grid_size, block_size>>>(
        phi, cmplx_workspace, ns, norm_factor);
}

// Explicit instantiations
void RunInitCmplxWorkspace(cufftComplex *cmplx_workspace, const cufftReal *rhs,
                           std::array<std::size_t, 2> ns, dim3 grid_size,
                           dim3 block_size);

void RunDivideCmplxWorkspaceByDenoms(cufftComplex *cmplx_workspace,
                                     std::array<std::size_t, 2> ns,
                                     std::array<const cufftReal *, 2> denoms,
                                     dim3 grid_size, dim3 block_size);

void RunNormalization(cufftReal *phi, const cufftComplex *cmplx_workspace,
                      std::array<std::size_t, 2> ns, float norm_factor,
                      dim3 grid_size, dim3 block_size);
}  // namespace poisson_solver
