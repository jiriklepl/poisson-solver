#ifndef COMMON_REAL_CUH
#define COMMON_REAL_CUH

#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

// Functionality shared among all real solvers (e.g. full dirichlet, full
// neumann)

namespace poisson_solver::detail::kernels {
template <std::floating_point T, std::size_t Dim>
__global__ void Normalize(T *phi,
                          const T *real_workspace,
                          std::array<std::size_t, Dim> ns,
                          T norm_factor) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        phi[idx] = real_workspace[idx] / norm_factor;
    });
}
}  // namespace poisson_solver::detail::kernels

#endif  // COMMON_REAL_CUH
