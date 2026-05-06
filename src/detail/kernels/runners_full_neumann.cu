#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/kernels/full_neumann.cuh"
#include "poisson-solver/detail/kernels/runners_full_neumann.hpp"

#include <concepts>

namespace poisson_solver::detail::kernels::full_neumann {
template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullNeumann(T *real_workspace,
                                           std::array<std::size_t, Dim> ns,
                                           std::array<const T *, Dim> denoms,
                                           dim3 grid_size,
                                           dim3 block_size) {
    DivideWorkspaceByDenomsFullNeumann<<<grid_size, block_size>>>(
        real_workspace, ns, denoms);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_DIVISION(TYPE, DIM)                             \
    template void RunDivideWorkspaceByDenomsFullNeumann<TYPE, DIM>( \
        TYPE * real_workspace,                                      \
        std::array<std::size_t, DIM> ns,                            \
        std::array<const TYPE *, DIM> denoms,                       \
        dim3 grid_size,                                             \
        dim3 block_size);

#define INSTANTIATE_DIVISION_ALL_DIMS(TYPE) \
    INSTANTIATE_DIVISION(TYPE, 1)           \
    INSTANTIATE_DIVISION(TYPE, 2)           \
    INSTANTIATE_DIVISION(TYPE, 3)

INSTANTIATE_DIVISION_ALL_DIMS(float)
INSTANTIATE_DIVISION_ALL_DIMS(double)

#undef INSTANTIATE_DIVISION_ALL_DIMS
#undef INSTANTIATE_DIVISION
}  // namespace poisson_solver::detail::kernels::full_neumann
