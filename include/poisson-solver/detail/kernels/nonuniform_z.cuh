#ifndef NONUNIFORM_Z_CUH
#define NONUNIFORM_Z_CUH

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::kernels::nonuniform_z {
template <std::floating_point T>
__global__ void TridiagZReal(T *real_workspace,
                             T *cp_workspace,
                             GridSizes<3> ns,
                             DenomConstPtrs<T, 2> denoms,
                             const T *mat_a,
                             const T *mat_b,
                             const T *mat_c) {
    // TODO Implement
}

template <std::floating_point T>
__global__ void TridiagZCmplx(CuComplexTrait<T> *real_workspace,
                              T *cp_workspace,
                              GridSizes<3> ns,
                              DenomConstPtrs<T, 2> denoms,
                              const T *mat_a,
                              const T *mat_b,
                              const T *mat_c) {
    // TODO Implement
}

}  // namespace poisson_solver::detail::kernels::nonuniform_z

#endif  // NONUNIFORM_Z_CUH
