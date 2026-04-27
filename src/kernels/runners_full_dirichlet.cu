#include "poisson-solver/kernels/full_dirichlet.cuh"
#include "poisson-solver/kernels/runners_full_dirichlet.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullDirichlet(T *real_workspace,
                                             std::array<std::size_t, Dim> ns,
                                             std::array<const T *, Dim> denoms,
                                             dim3 grid_size,
                                             dim3 block_size) {
    DivideWorkspaceByDenomsFullDirichlet<<<grid_size, block_size>>>(
        real_workspace, ns, denoms);
}

template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const T *real_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size) {
    NormalizeWorkspaceToPhiFullDirichlet<<<grid_size, block_size>>>(
        phi, real_workspace, ns, norm_factor);
}

// Explicit instantiations
// Doubles
template void RunDivideWorkspaceByDenomsFullDirichlet<double, 1>(
    double *real_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const double *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullDirichlet<double, 2>(
    double *real_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const double *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullDirichlet<double, 3>(
    double *real_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const double *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<double, 1>(double *phi,
                                          const double *real_workspace,
                                          std::array<std::size_t, 1> ns,
                                          double norm_factor,
                                          dim3 grid_size,
                                          dim3 block_size);
template void RunNormalization<double, 2>(double *phi,
                                          const double *real_workspace,
                                          std::array<std::size_t, 2> ns,
                                          double norm_factor,
                                          dim3 grid_size,
                                          dim3 block_size);
template void RunNormalization<double, 3>(double *phi,
                                          const double *real_workspace,
                                          std::array<std::size_t, 3> ns,
                                          double norm_factor,
                                          dim3 grid_size,
                                          dim3 block_size);

// Floats
template void RunDivideWorkspaceByDenomsFullDirichlet<float, 1>(
    float *real_workspace,
    std::array<std::size_t, 1> ns,
    std::array<const float *, 1> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullDirichlet<float, 2>(
    float *real_workspace,
    std::array<std::size_t, 2> ns,
    std::array<const float *, 2> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunDivideWorkspaceByDenomsFullDirichlet<float, 3>(
    float *real_workspace,
    std::array<std::size_t, 3> ns,
    std::array<const float *, 3> denoms,
    dim3 grid_size,
    dim3 block_size);
template void RunNormalization<float, 1>(float *phi,
                                         const float *real_workspace,
                                         std::array<std::size_t, 1> ns,
                                         float norm_factor,
                                         dim3 grid_size,
                                         dim3 block_size);
template void RunNormalization<float, 2>(float *phi,
                                         const float *real_workspace,
                                         std::array<std::size_t, 2> ns,
                                         float norm_factor,
                                         dim3 grid_size,
                                         dim3 block_size);
template void RunNormalization<float, 3>(float *phi,
                                         const float *real_workspace,
                                         std::array<std::size_t, 3> ns,
                                         float norm_factor,
                                         dim3 grid_size,
                                         dim3 block_size);
}  // namespace poisson_solver
