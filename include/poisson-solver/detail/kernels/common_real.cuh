#ifndef COMMON_REAL_CUH
#define COMMON_REAL_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

// Functionality shared among all real solvers (e.g. full dirichlet, full
// neumann)

namespace poisson_solver::detail::kernels {
template <std::floating_point T, std::size_t Dim>
__global__ void Normalize(T *phi,
                          const T *real_workspace,
                          GridSizes<Dim> ns,
                          T norm_factor) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        phi[idx] = real_workspace[idx] / norm_factor;
    });
}
}  // namespace poisson_solver::detail::kernels

#endif  // COMMON_REAL_CUH
