#ifndef FULL_NEUMANN_CUH
#define FULL_NEUMANN_CUH

#include "poisson-solver/helpers.hpp"
#include "poisson-solver/kernels/common.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
__global__ void DivideWorkspaceByDenomsFullNeumann(
    T *real_workspace,
    std::array<std::size_t, Dim> ns,
    std::array<const T *, Dim> denoms) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);

        if (idx == 0) {
            real_workspace[idx] = 0;
            return;
        }

        T denom = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            denom += denoms[i][idxs[i]];
        }
        real_workspace[idx] /= denom;
    });
}
}  // namespace poisson_solver

#endif  // FULL_NEUMANN_CUH
