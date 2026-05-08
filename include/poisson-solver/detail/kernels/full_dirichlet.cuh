#ifndef FULL_DIRICHLET_CUH
#define FULL_DIRICHLET_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::full_dirichlet {
template <std::floating_point T, std::size_t Dim>
__global__ void DivideByDenoms(T *real_workspace,
                               GridSizes<Dim> ns,
                               std::array<const T *, Dim> denoms) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        real_workspace[idx] /= SumIndexedDenoms(denoms, idxs);
    });
}
}  // namespace poisson_solver::detail::kernels::full_dirichlet

#endif  // FULL_DIRICHLET_CUH
