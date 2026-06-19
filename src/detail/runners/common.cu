#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/common.cuh"
#include "poisson-solver/detail/runners/common.hpp"

#include <concepts>

namespace poisson_solver::detail::runners {
namespace ker = ::poisson_solver::detail::kernels;

template <std::floating_point T, std::size_t Dim, ValidWorkspace<T> Workspace>
void DivideByDenoms(Workspace *workspace,
                    GridSizes<Dim> ns,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size) {
    ker::DivideByDenoms<<<cu_grid_size, cu_block_size>>>(workspace, ns, denoms);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim, ValidWorkspace<T> Workspace>
void DivideByDenomsAndZero(Workspace *workspace,
                           GridSizes<Dim> ns,
                           DenomConstPtrs<T, Dim> denoms,
                           dim3 cu_grid_size,
                           dim3 cu_block_size) {
    ker::DivideByDenomsAndZero<<<cu_grid_size, cu_block_size>>>(
        workspace, ns, denoms);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===

#define INSTANTIATE_DIVISION_REAL(TYPE, DIM)                       \
    template void DivideByDenoms<TYPE, DIM, TYPE>(                 \
        TYPE * workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<DIM> ns,                                         \
        DenomConstPtrs<TYPE, DIM> denoms,                          \
        dim3 cu_grid_size,                                         \
        dim3 cu_block_size);

#define INSTANTIATE_DIVISION_COMPLEX(TYPE, DIM)                                \
    template void                                                              \
    DivideByDenoms<TYPE,                                                       \
                   DIM,                                                        \
                   CuComplexTrait<TYPE>>(/* NOLINT(bugprone-macro-parentheses) \
                                          */                                   \
                                         CuComplexTrait<TYPE> * workspace,     \
                                         GridSizes<DIM> ns,                    \
                                         DenomConstPtrs<TYPE, DIM> denoms,     \
                                         dim3 cu_grid_size,                    \
                                         dim3 cu_block_size);

#define INSTANTIATE_DIVISION_AND_ZEROING_REAL(TYPE, DIM)           \
    template void DivideByDenomsAndZero<TYPE, DIM, TYPE>(          \
        TYPE * workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<DIM> ns,                                         \
        DenomConstPtrs<TYPE, DIM> denoms,                          \
        dim3 cu_grid_size,                                         \
        dim3 cu_block_size);

#define INSTANTIATE_DIVISION_AND_ZEROING_COMPLEX(TYPE, DIM)            \
    template void DivideByDenomsAndZero<                               \
        TYPE,                                                          \
        DIM,                                                           \
        CuComplexTrait<TYPE>>(/* NOLINT(bugprone-macro-parentheses) */ \
                              CuComplexTrait<TYPE> * workspace,        \
                              GridSizes<DIM> ns,                       \
                              DenomConstPtrs<TYPE, DIM> denoms,        \
                              dim3 cu_grid_size,                       \
                              dim3 cu_block_size);

#define INSTANTIATE_DIV_REAL_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_REAL(TYPE, 1)      \
    INSTANTIATE_DIVISION_REAL(TYPE, 2)      \
    INSTANTIATE_DIVISION_REAL(TYPE, 3)

#define INSTANTIATE_DIV_COMPLEX_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 1)      \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 2)      \
    INSTANTIATE_DIVISION_COMPLEX(TYPE, 3)

#define INSTANTIATE_DIV_AND_ZERO_REAL_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_AND_ZEROING_REAL(TYPE, 1)   \
    INSTANTIATE_DIVISION_AND_ZEROING_REAL(TYPE, 2)   \
    INSTANTIATE_DIVISION_AND_ZEROING_REAL(TYPE, 3)

#define INSTANTIATE_DIV_AND_ZERO_COMPLEX_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_AND_ZEROING_COMPLEX(TYPE, 1)   \
    INSTANTIATE_DIVISION_AND_ZEROING_COMPLEX(TYPE, 2)   \
    INSTANTIATE_DIVISION_AND_ZEROING_COMPLEX(TYPE, 3)

INSTANTIATE_DIV_REAL_ALL_DIMS(float)
INSTANTIATE_DIV_REAL_ALL_DIMS(double)
INSTANTIATE_DIV_COMPLEX_ALL_DIMS(float)
INSTANTIATE_DIV_COMPLEX_ALL_DIMS(double)
INSTANTIATE_DIV_AND_ZERO_REAL_ALL_DIMS(float)
INSTANTIATE_DIV_AND_ZERO_REAL_ALL_DIMS(double)
INSTANTIATE_DIV_AND_ZERO_COMPLEX_ALL_DIMS(float)
INSTANTIATE_DIV_AND_ZERO_COMPLEX_ALL_DIMS(double)

#undef INSTANTIATE_DIV_AND_ZERO_COMPLEX_ALL_DIMS
#undef INSTANTIATE_DIV_AND_ZERO_REAL_ALL_DIMS
#undef INSTANTIATE_DIV_COMPLEX_ALL_DIMS
#undef INSTANTIATE_DIV_REAL_ALL_DIMS
#undef INSTANTIATE_DIVISION_AND_ZEROING_COMPLEX
#undef INSTANTIATE_DIVISION_AND_ZEROING_REAL
#undef INSTANTIATE_DIVISION_COMPLEX
#undef INSTANTIATE_DIVISION_REAL
}  // namespace poisson_solver::detail::runners
