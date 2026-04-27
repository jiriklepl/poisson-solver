#ifndef FULL_DIRICHLET_CUH
#define FULL_DIRICHLET_CUH

#include "poisson-solver/helpers.hpp"
#include "poisson-solver/kernels/common.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
__global__ void DivideWorkspaceByDenomsFullDirichlet(
    T *real_workspace,
    std::array<std::size_t, Dim> ns,
    std::array<const T *, Dim> denoms) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);

        T denom = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            denom += denoms[i][idxs[i]];
        }
        real_workspace[idx] /= denom;
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void NormalizeWorkspaceToPhiFullDirichlet(
    T *phi,
    const T *real_workspace,
    std::array<std::size_t, Dim> ns,
    T norm_factor) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        phi[idx] = real_workspace[idx] / norm_factor;
    });
}
}  // namespace poisson_solver

#endif  // FULL_DIRICHLET_CUH
