#ifndef RUNNERS_FULL_DIRICHLET_HPP
#define RUNNERS_FULL_DIRICHLET_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners::full_dirichlet {
template <std::floating_point T, std::size_t Dim>
void DivideByDenoms(T *real_workspace,
                    GridSizes<Dim> ns,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size);
}  // namespace poisson_solver::detail::runners::full_dirichlet

#endif  // RUNNERS_FULL_DIRICHLET_HPP
