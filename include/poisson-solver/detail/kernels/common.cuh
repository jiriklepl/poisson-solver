#ifndef COMMON_CUH
#define COMMON_CUH

#include "poisson-solver/aliases.hpp"

#include <array>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels {
// Returns indices of the current thread
template <std::size_t Dim>
__device__ std::array<std::size_t, Dim> GetIdxs() {
    if constexpr (Dim == 1) {
        return {blockIdx.x * blockDim.x + threadIdx.x};
    } else if constexpr (Dim == 2) {
        return {blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y};
    } else if constexpr (Dim == 3) {
        return {blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y,
                blockIdx.z * blockDim.z + threadIdx.z};
    }
}

// Returns strides for each dimension
template <std::size_t Dim>
__device__ std::array<std::size_t, Dim> GetStrides() {
    if constexpr (Dim == 1) {
        return {gridDim.x * blockDim.x};
    } else if constexpr (Dim == 2) {
        return {gridDim.x * blockDim.x, gridDim.y * blockDim.y};
    } else if constexpr (Dim == 3) {
        return {gridDim.x * blockDim.x,
                gridDim.y * blockDim.y,
                gridDim.z * blockDim.z};
    }
}

// Applies the function `f` to indices in `Dim` nested loops.
// Each dimensions has:
// 1. A starting index in `starts`.
// 2. Size in `ns`.
// 3. A stride (in cast there are too few threads overall) in `strides`.
template <std::size_t Dim, std::size_t CurDim, typename Func>
__device__ void ApplyStrideLoopImpl(std::array<std::size_t, Dim> &indices,
                                    std::array<std::size_t, Dim> starts,
                                    GridSizes<Dim> ns,
                                    std::array<std::size_t, Dim> strides,
                                    Func f) {
    for (indices[CurDim] = starts[CurDim]; indices[CurDim] < ns[CurDim];
         indices[CurDim] += strides[CurDim]) {
        if constexpr (CurDim + 1 < Dim) {
            ApplyStrideLoopImpl<Dim, CurDim + 1>(
                indices, starts, ns, strides, f);
        } else {
            f(indices);
        }
    }
}

// Applies the function `f` to indices in `Dim` nested loops where each
// dimension has the size according to `ns`.
template <std::size_t Dim, typename Func>
__device__ void ApplyStrideLoop(GridSizes<Dim> ns, Func f) {
    std::array<std::size_t, Dim> indices{};
    auto starts = GetIdxs<Dim>();
    auto strides = GetStrides<Dim>();
    ApplyStrideLoopImpl<Dim, 0>(indices, starts, ns, strides, f);
}

template <std::floating_point T, std::size_t Dim>
__device__ T SumIndexedDenoms(std::array<const T *, Dim> denoms,
                              std::array<std::size_t, Dim> idxs) {
    T res = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        res += denoms[d][idxs[d]];
    }
    return res;
}
}  // namespace poisson_solver::detail::kernels

#endif  // COMMON_CUH
