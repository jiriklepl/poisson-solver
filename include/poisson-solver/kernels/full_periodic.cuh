#ifndef FULL_PERIODIC_CUH
#define FULL_PERIODIC_CUH

#include "poisson-solver/helpers.hpp"
#include "poisson-solver/kernels/common.cuh"

#include <array>
#include <concepts>
#include <cstddef>
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

template <std::floating_point T, std::size_t Dim>
__global__ void InitWorkspaceFullPeriodic(CuComplexTrait_t<T> *cmplx_workspace,
                                          const T *rhs,
                                          std::array<std::size_t, Dim> ns) {
    ApplyStrideLoop(ns, [&](std::array<std::size_t, Dim> idxs) {
        const std::size_t idx = FlatIndex(idxs, ns);
        cmplx_workspace[idx] = MakeCuComplex<T>(rhs[idx], 0);
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void DivideWorkspaceByDenomsFullPeriodic(
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
__global__ void NormalizeWorkspaceToPhiFullPeriodic(
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

#endif  // FULL_PERIODIC_CUH
