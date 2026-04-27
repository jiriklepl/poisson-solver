#ifndef RUNNERS_FULL_PERIODIC_HPP
#define RUNNERS_FULL_PERIODIC_HPP

#include <array>
#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver {
// Initializes the complex workspace `cmplx_workspace` with values from `rhs`
template <std::floating_point T, std::size_t Dim>
void RunInitWorkspaceFullPeriodic(CuComplexTrait_t<T> *cmplx_workspace,
                                  const T *rhs,
                                  std::array<std::size_t, Dim> ns,
                                  dim3 grid_size,
                                  dim3 block_size);

template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullPeriodic(
    CuComplexTrait_t<T> *cmplx_workspace,
    std::array<std::size_t, Dim> ns,
    std::array<const T *, Dim> denoms,
    dim3 grid_size,
    dim3 block_size);

// Fills `phi` with normalized values of `cmplx_workspace`
template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const CuComplexTrait_t<T> *cmplx_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size);
}  // namespace poisson_solver

#endif  // RUNNERS_FULL_PERIODIC_HPP
