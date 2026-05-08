#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/cuch.hpp"
#include "poisson-solver/detail/kernels/full_periodic.cuh"
#include "poisson-solver/detail/runners/full_periodic.hpp"

#include <array>
#include <concepts>

namespace poisson_solver::detail::runners::full_periodic {
namespace fpk = kernels::full_periodic;

template <std::floating_point T, std::size_t Dim>
void Init(CuComplexTrait<T> *cmplx_workspace,
          const T *rhs,
          GridSizes<Dim> ns,
          dim3 grid_size,
          dim3 block_size) {
    fpk::Init<<<grid_size, block_size>>>(cmplx_workspace, rhs, ns);
    CUCH(cudaGetLastError());
}

template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(CuComplexTrait<T> *cmplx_workspace,
                    GridSizes<Dim> ns,
                    std::array<const T *, Dim> denoms,
                    dim3 grid_size,
                    dim3 block_size) {
    fpk::DivideByDenoms<<<grid_size, block_size>>>(cmplx_workspace, ns, denoms);
}

template <std::floating_point T, std::size_t Dim>
void Normalize(T *phi,
               const CuComplexTrait<T> *cmplx_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 grid_size,
               dim3 block_size) {
    fpk::Normalize<<<grid_size, block_size>>>(
        phi, cmplx_workspace, ns, norm_factor);
}

// === Explicit Instantiations ===
#define INSTANTIATE_INITIALIZATION(TYPE, DIM)                             \
    template void Init<TYPE, DIM>(CuComplexTrait<TYPE> * cmplx_workspace, \
                                  const TYPE *rhs,                        \
                                  GridSizes<DIM> ns,                      \
                                  dim3 grid_size,                         \
                                  dim3 block_size);

#define INSTANTIATE_DIVISION(TYPE, DIM)         \
    template void DivideByDenoms<TYPE, DIM>(    \
        CuComplexTrait<TYPE> * cmplx_workspace, \
        GridSizes<DIM> ns,                      \
        std::array<const TYPE *, DIM> denoms,   \
        dim3 grid_size,                         \
        dim3 block_size);

#define INSTANTIATE_NORMALIZATION(TYPE, DIM)         \
    template void Normalize<TYPE, DIM>(              \
        TYPE * phi,                                  \
        const CuComplexTrait<TYPE> *cmplx_workspace, \
        GridSizes<DIM> ns,                           \
        TYPE norm_factor,                            \
        dim3 grid_size,                              \
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
}  // namespace poisson_solver::detail::runners::full_periodic
