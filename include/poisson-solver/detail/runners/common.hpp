#ifndef COMMON_HPP
#define COMMON_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>

namespace poisson_solver::detail::runners {
template <std::floating_point T, std::size_t Dim, ValidWorkspace<T> Workspace>
void DivideByDenoms(Workspace *workspace,
                    GridSizes<Dim> ns,
                    GridOffsets<Dim> offs,
                    DenomConstPtrs<T, Dim> denoms,
                    dim3 cu_grid_size,
                    dim3 cu_block_size,
                    bool zero_origin = false);
}  // namespace poisson_solver::detail::runners

#endif  // COMMON_HPP
