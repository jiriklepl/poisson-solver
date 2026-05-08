#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/kernels/common_real.cuh"
#include "poisson-solver/detail/runners/common_real.hpp"

#include <concepts>

namespace poisson_solver::detail::runners {
namespace ker = kernels;

template <std::floating_point T, std::size_t Dim>
void Normalize(T *phi,
               const T *real_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 grid_size,
               dim3 block_size) {
    ker::Normalize<<<grid_size, block_size>>>(
        phi, real_workspace, ns, norm_factor);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_NORMALIZATION(TYPE, DIM)                       \
    template void Normalize<TYPE, DIM>(TYPE * phi,                 \
                                       const TYPE *real_workspace, \
                                       GridSizes<DIM> ns,          \
                                       TYPE norm_factor,           \
                                       dim3 grid_size,             \
                                       dim3 block_size);

#define INSTANTIATE_NORMALIZATION_ALL_DIMS(TYPE) \
    INSTANTIATE_NORMALIZATION(TYPE, 1)           \
    INSTANTIATE_NORMALIZATION(TYPE, 2)           \
    INSTANTIATE_NORMALIZATION(TYPE, 3)

INSTANTIATE_NORMALIZATION_ALL_DIMS(float)
INSTANTIATE_NORMALIZATION_ALL_DIMS(double)

#undef INSTANTIATE_NORMALIZATION_ALL_DIMS
#undef INSTANTIATE_NORMALIZATION
}  // namespace poisson_solver::detail::runners
