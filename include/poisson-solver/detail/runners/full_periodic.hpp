#ifndef RUNNERS_FULL_PERIODIC_HPP
#define RUNNERS_FULL_PERIODIC_HPP

#include "poisson-solver/aliases.hpp"

#include <array>
#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::full_periodic {
// Initializes the complex workspace `cmplx_workspace` with values from `rhs`
template <std::floating_point T, std::size_t Dim>
void Init(CuComplexTrait<T> *cmplx_workspace,
          const T *rhs,
          GridSizes<Dim> ns,
          dim3 grid_size,
          dim3 block_size);

template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(CuComplexTrait<T> *cmplx_workspace,
                    GridSizes<Dim> ns,
                    std::array<const T *, Dim> denoms,
                    dim3 grid_size,
                    dim3 block_size);

// Fills `phi` with normalized values of `cmplx_workspace`
template <std::floating_point T, std::size_t Dim>
void Normalize(T *phi,
               const CuComplexTrait<T> *cmplx_workspace,
               GridSizes<Dim> ns,
               T norm_factor,
               dim3 grid_size,
               dim3 block_size);
}  // namespace poisson_solver::detail::runners::full_periodic

#endif  // RUNNERS_FULL_PERIODIC_HPP
