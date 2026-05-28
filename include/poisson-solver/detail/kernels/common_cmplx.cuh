#ifndef COMMON_CMPLX_CUH
#define COMMON_CMPLX_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::common_cmplx {
template <std::floating_point T>
__device__ CuComplexTrait<T> MakeCuComplex(T r, T i);

template <>
__device__ inline CuComplexTrait<float> MakeCuComplex(float r, float i) {
    return make_cuFloatComplex(r, i);
}

template <>
__device__ inline CuComplexTrait<double> MakeCuComplex(double r, double i) {
    return make_cuDoubleComplex(r, i);
}

template <std::floating_point T, std::size_t Dim>
__global__ void Init(CuComplexTrait<T> *cmplx_workspace,
                     const T *real_workspace,
                     GridSizes<Dim> ns) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        cmplx_workspace[idx] = MakeCuComplex<T>(real_workspace[idx], 0);
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void DivideByDenoms(CuComplexTrait<T> *cmplx_workspace,
                               GridSizes<Dim> ns,
                               DenomConstPtrs<T, Dim> denoms) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        T denom_sum = SumIndexedDenoms(denoms, midx);
        cmplx_workspace[idx].x /= denom_sum;
        cmplx_workspace[idx].y /= denom_sum;
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void DivideByDenomsAndZero(CuComplexTrait<T> *cmplx_workspace,
                                      GridSizes<Dim> ns,
                                      DenomConstPtrs<T, Dim> denoms) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);

        if (idx == 0) {
            cmplx_workspace[idx] = MakeCuComplex<T>(0, 0);
            return;
        }

        T denom_sum = SumIndexedDenoms(denoms, midx);
        cmplx_workspace[idx].x /= denom_sum;
        cmplx_workspace[idx].y /= denom_sum;
    });
}

template <std::floating_point T, std::size_t Dim>
__global__ void Normalize(T *res,
                          const CuComplexTrait<T> *cmplx_workspace,
                          GridSizes<Dim> ns,
                          T norm_factor) {
    ApplyStrideLoop(ns, [&](MultidimIdx<Dim> midx) {
        const std::size_t idx = FlatIndex(midx, ns);
        res[idx] = cmplx_workspace[idx].x / norm_factor;
    });
}
}  // namespace poisson_solver::detail::kernels::common_cmplx

#endif  // COMMON_CMPLX_CUH
