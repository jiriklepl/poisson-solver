#ifndef RUNNERS_COMMON_REAL_HPP
#define RUNNERS_COMMON_REAL_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners {
// Fills `phi` with normalized values of `real_workspace`
template <std::floating_point T, std::size_t Dim>
void Normalize(T *phi,
               const T *real_workspace,
               std::array<std::size_t, Dim> ns,
               T norm_factor,
               dim3 grid_size,
               dim3 block_size);
}  // namespace poisson_solver::detail::runners

#endif  // RUNNERS_COMMON_REAL_HPP
