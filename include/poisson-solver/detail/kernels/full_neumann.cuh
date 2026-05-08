#ifndef FULL_NEUMANN_CUH
#define FULL_NEUMANN_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::full_neumann {
template <std::floating_point T, std::size_t Dim>
__global__ void DivideByDenoms(T *real_workspace,
                               GridSizes<Dim> ns,
                               DenomConstPtrs<T, Dim> denoms) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);

        if (idx == 0) {
            real_workspace[idx] = 0;
            return;
        }

        real_workspace[idx] /= SumIndexedDenoms(denoms, midx);
    });
}
}  // namespace poisson_solver::detail::kernels::full_neumann

#endif  // FULL_NEUMANN_CUH
