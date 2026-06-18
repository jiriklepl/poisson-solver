#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/kernels/full_neumann.cuh"
#include "poisson-solver/detail/runners/full_neumann.hpp"

#include <concepts>

namespace poisson_solver::detail::runners::full_neumann {
namespace fnk = kernels::full_neumann;

template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(T *real_workspace,
                    GridSizes<Dim> ns,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size) {
    fnk::DivideByDenoms<<<cu_grid_size, cu_block_size>>>(
        real_workspace, ns, denoms);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_DIVISION(TYPE, DIM)                                 \
    template void DivideByDenoms<TYPE, DIM>(                            \
        TYPE * real_workspace, /* NOLINT(bugprone-macro-parentheses) */ \
        GridSizes<DIM> ns,                                              \
        DenomConstPtrs<TYPE, DIM> denoms,                               \
        dim3 cu_grid_size,                                              \
        dim3 cu_block_size);

#define INSTANTIATE_DIVISION_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION(TYPE, 1)           \
    INSTANTIATE_DIVISION(TYPE, 2)           \
    INSTANTIATE_DIVISION(TYPE, 3)

INSTANTIATE_DIVISION_ALL_DIMS(float)
INSTANTIATE_DIVISION_ALL_DIMS(double)

#undef INSTANTIATE_DIVISION_ALL_DIMS
#undef INSTANTIATE_DIVISION
}  // namespace poisson_solver::detail::runners::full_neumann
