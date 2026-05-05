#include "poisson-solver/cuch.hpp"
#include "poisson-solver/kernels/common_real.cuh"
#include "poisson-solver/kernels/runners_common_real.hpp"

#include <concepts>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const T *real_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size) {
    NormalizeRealWorkspaceToPhi<<<grid_size, block_size>>>(
        phi, real_workspace, ns, norm_factor);
    CUCH(cudaGetLastError());
}

// === Explicit Instantiations ===
#define INSTANTIATE_NORMALIZATION(TYPE, DIM)                                   \
    template void RunNormalization<TYPE, DIM>(TYPE * phi,                      \
                                              const TYPE *real_workspace,      \
                                              std::array<std::size_t, DIM> ns, \
                                              TYPE norm_factor,                \
                                              dim3 grid_size,                  \
                                              dim3 block_size);

#define INSTANTIATE_NORMALIZATION_ALL_DIMS(TYPE) \
    INSTANTIATE_NORMALIZATION(TYPE, 1)           \
    INSTANTIATE_NORMALIZATION(TYPE, 2)           \
    INSTANTIATE_NORMALIZATION(TYPE, 3)

INSTANTIATE_NORMALIZATION_ALL_DIMS(float)
INSTANTIATE_NORMALIZATION_ALL_DIMS(double)

#undef INSTANTIATE_NORMALIZATION_ALL_DIMS
#undef INSTANTIATE_NORMALIZATION
}  // namespace poisson_solver
