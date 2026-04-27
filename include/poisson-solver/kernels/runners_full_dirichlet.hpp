#ifndef RUNNERS_FULL_DIRICHLET_HPP
#define RUNNERS_FULL_DIRICHLET_HPP

#include <array>
#include <concepts>
#include <cuda_runtime.h>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
void RunDivideWorkspaceByDenomsFullDirichlet(T *real_workspace,
                                             std::array<std::size_t, Dim> ns,
                                             std::array<const T *, Dim> denoms,
                                             dim3 grid_size,
                                             dim3 block_size);

// Fills `phi` with normalized values of `real_workspace`
template <std::floating_point T, std::size_t Dim>
void RunNormalization(T *phi,
                      const T *real_workspace,
                      std::array<std::size_t, Dim> ns,
                      T norm_factor,
                      dim3 grid_size,
                      dim3 block_size);
}  // namespace poisson_solver

#endif  // RUNNERS_FULL_DIRICHLET_HPP
