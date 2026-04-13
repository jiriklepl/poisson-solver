#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "poisson-solver/cuch.hpp"
#include "poisson-solver/helpers.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

namespace poisson_solver {
template <std::floating_point T>
__device__ CuComplexTrait_t<T> MakeCuComplex(T r, T i);

template <>
__device__ inline CuComplexTrait_t<float> MakeCuComplex(float r, float i) {
    return make_cuFloatComplex(r, i);
}

template <>
__device__ inline CuComplexTrait_t<double> MakeCuComplex(double r, double i) {
    return make_cuDoubleComplex(r, i);
}

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
                                    std::array<std::size_t, Dim> ns,
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
__device__ void ApplyStrideLoop(std::array<std::size_t, Dim> ns, Func f) {
    std::array<std::size_t, Dim> indices{};
    auto starts = GetIdxs<Dim>();
    auto strides = GetStrides<Dim>();
    ApplyStrideLoopImpl<Dim, 0>(indices, starts, ns, strides, f);
}

template <std::floating_point T, std::size_t Dim>
__global__ void InitCmplxWorkspace(CuComplexTrait_t<T> *cmplx_workspace,
                                   const T *rhs,
                                   std::array<std::size_t, Dim> ns) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        cmplx_workspace[idx] = MakeCuComplex<T>(rhs[idx], 0);
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void DivideCmplxWorkspaceByDenoms(
    CuComplexTrait_t<T> *cmplx_workspace,
    std::array<std::size_t, Dim> ns,
    std::array<const T *, Dim> denoms) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);

        if (idx == 0) {
            cmplx_workspace[idx] = MakeCuComplex<T>(0, 0);
            return;
        }

        T denom = 0;
        for (std::size_t i = 0; i < Dim; ++i) {
            denom += denoms[i][idxs[i]];
        }
        cmplx_workspace[idx].x /= denom;
        cmplx_workspace[idx].y /= denom;
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void NormalizeCmplxWorkspaceToPhi(
    T *phi,
    const CuComplexTrait_t<T> *cmplx_workspace,
    std::array<std::size_t, Dim> ns,
    T norm_factor) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        phi[idx] = cmplx_workspace[idx].x / norm_factor;
    });
}
}  // namespace poisson_solver

#endif  // KERNEL_CUH
