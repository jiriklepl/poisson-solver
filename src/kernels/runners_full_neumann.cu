#include "poisson-solver/kernels/full_neumann.cuh"
#include "poisson-solver/kernels/runners_full_neumann.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullNeumann(T *real_workspace,
                                           std::array<std::size_t, Dim> ns,
                                           std::array<const T *, Dim> denoms,
                                           dim3 grid_size,
                                           dim3 block_size) {
    DivideWorkspaceByDenomsFullNeumann<<<grid_size, block_size>>>(
        real_workspace, ns, denoms);
}

// Explicit instantiations
// Doubles
template void RunDivideWorkspaceByDenomsFullNeumann<double, 1>(
    double *real_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const double *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullNeumann<double, 2>(
    double *real_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const double *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullNeumann<double, 3>(
    double *real_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const double *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);

// Floats
template void RunDivideWorkspaceByDenomsFullNeumann<float, 1>(
    float *real_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const float *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullNeumann<float, 2>(
    float *real_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const float *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullNeumann<float, 3>(
    float *real_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const float *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);
}  // namespace poisson_solver
