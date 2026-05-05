#include "poisson-solver/kernels/full_periodic.cuh"
#include "poisson-solver/kernels/runners_full_periodic.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunInitWorkspaceFullPeriodic(CuComplexTrait_t<T> *cmplx_workspace,
                                  const T *rhs,
                                  std::array<std::size_t, Dim> ns,
                                  dim3 grid_size,
                                  dim3 block_size) {
    InitWorkspaceFullPeriodic<<<grid_size, block_size>>>(
        cmplx_workspace, rhs, ns);
}

template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullPeriodic(
    CuComplexTrait_t<T> *cmplx_workspace,
    std::array<std::size_t, Dim> ns,
    std::array<const T *, Dim> denoms,
    dim3 grid_size,
    dim3 block_size) {
    DivideWorkspaceByDenomsFullPeriodic<<<grid_size, block_size>>>(
        cmplx_workspace, ns, denoms);
}

template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const CuComplexTrait_t<T> *cmplx_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size) {
    NormalizeWorkspaceToPhiFullPeriodic<<<grid_size, block_size>>>(
        phi, cmplx_workspace, ns, norm_factor);
}

// === Explicit Instantiations ===
#define INSTANTIATE_INITIALIZATION(TYPE, DIM)              \
    template void RunInitWorkspaceFullPeriodic<TYPE, DIM>( \
        CuComplexTrait_t<TYPE> * cmplx_workspace,          \
        const TYPE *rhs,                                   \
        std::array<std::size_t, DIM> ns,                   \
        dim3 grid_size,                                    \
        dim3 block_size);

#define INSTANTIATE_DIVISION(TYPE, DIM)                              \
    template void RunDivideWorkspaceByDenomsFullPeriodic<TYPE, DIM>( \
        CuComplexTrait_t<TYPE> * cmplx_workspace,                    \
        std::array<std::size_t, DIM> ns,                             \
        std::array<const TYPE *, DIM> denoms,                        \
        dim3 grid_size,                                              \
        dim3 block_size);

#define INSTANTIATE_NORMALIZATION(TYPE, DIM)           \
    template void RunNormalization<TYPE, DIM>(         \
        TYPE * phi,                                    \
        const CuComplexTrait_t<TYPE> *cmplx_workspace, \
        std::array<std::size_t, DIM> ns,               \
        TYPE norm_factor,                              \
        dim3 grid_size,                                \
        dim3 block_size);

#define INSTANTIATE_INIT_DIV_NORM(TYPE, DIM) \
    INSTANTIATE_INITIALIZATION(TYPE, DIM)    \
    INSTANTIATE_DIVISION(TYPE, DIM)          \
    INSTANTIATE_NORMALIZATION(TYPE, DIM)

#define INSTANTIATE_INIT_DIV_NORM_ALL_DIMS(TYPE) \
    INSTANTIATE_INIT_DIV_NORM(TYPE, 1)           \
    INSTANTIATE_INIT_DIV_NORM(TYPE, 2)           \
    INSTANTIATE_INIT_DIV_NORM(TYPE, 3)

INSTANTIATE_INIT_DIV_NORM_ALL_DIMS(float)
INSTANTIATE_INIT_DIV_NORM_ALL_DIMS(double)

#undef INSTANTIATE_INIT_DIV_NORM_ALL_DIMS
#undef INSTANTIATE_INIT_DIV_NORM
#undef INSTANTIATE_NORMALIZATION
#undef INSTANTIATE_DIVISION
#undef INSTANTIATE_INITIALIZATION
}  // namespace poisson_solver
