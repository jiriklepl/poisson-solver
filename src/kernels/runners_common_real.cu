#include "poisson-solver/kernels/common_real.cuh"
#include "poisson-solver/kernels/runners_common_real.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const T *real_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size) {
    NormalizeRealWorkspaceToPhi<<<grid_size, block_size>>>(
        phi, real_workspace, ns, norm_factor);
}

// Explicit instantiations
// Doubles
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
