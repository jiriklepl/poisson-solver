#ifndef RUNNERS_FULL_DIRICHLET_HPP
#define RUNNERS_FULL_DIRICHLET_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::full_dirichlet {
template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(T *real_workspace,
                    std::array<std::size_t, Dim> ns,
                    std::array<const T *, Dim> denoms,
                    dim3 grid_size,
                    dim3 block_size);
}  // namespace poisson_solver::detail::runners::full_dirichlet

#endif  // RUNNERS_FULL_DIRICHLET_HPP
