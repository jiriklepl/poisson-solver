#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/kernels/common_real.cuh"
#include "poisson-solver/detail/runners/common_real.hpp"

#include <concepts>

namespace poisson_solver::detail::runners {
namespace ker = kernels;

template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(T *real_workspace,
                    GridSizes<Dim> ns,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size) {
    ker::DivideByDenoms<<<cu_grid_size, cu_block_size>>>(
        real_workspace, ns, denoms);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim>
void Normalize(T *real_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 cu_grid_size,
               dim3 cu_block_size) {
    ker::Normalize<<<cu_grid_size, cu_block_size>>>(
        real_workspace, ns, norm_factor);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_DIVISION(TYPE, DIM)                                       \
    template void DivideByDenoms<TYPE, DIM>(TYPE * real_workspace,            \
                                            GridSizes<DIM> ns,                \
                                            DenomConstPtrs<TYPE, DIM> denoms, \
                                            dim3 cu_grid_size,                \
                                            dim3 cu_block_size);

#define INSTANTIATE_NORMALIZATION(TYPE, DIM)                  \
    template void Normalize<TYPE, DIM>(TYPE * real_workspace, \
                                       GridSizes<DIM> ns,     \
                                       TYPE norm_factor,      \
                                       dim3 cu_grid_size,     \
                                       dim3 cu_block_size);

#define INSTANTIATE_DIVISION_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION(TYPE, 1)           \
    INSTANTIATE_DIVISION(TYPE, 2)           \
    INSTANTIATE_DIVISION(TYPE, 3)

#define INSTANTIATE_NORMALIZATION_ALL_DIMS(TYPE) \
    INSTANTIATE_NORMALIZATION(TYPE, 1)           \
    INSTANTIATE_NORMALIZATION(TYPE, 2)           \
    INSTANTIATE_NORMALIZATION(TYPE, 3)

#define INSTANTIATE_NORM_AND_DIV_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION_ALL_DIMS(TYPE)         \
    INSTANTIATE_NORMALIZATION_ALL_DIMS(TYPE)

INSTANTIATE_NORM_AND_DIV_ALL_DIMS(float)
INSTANTIATE_NORM_AND_DIV_ALL_DIMS(double)

#undef INSTANTIATE_DIVISION_ALL_DIMS
#undef INSTANTIATE_NORMALIZATION_ALL_DIMS
#undef INSTANTIATE_DIVISION
#undef INSTANTIATE_NORMALIZATION
}  // namespace poisson_solver::detail::runners
