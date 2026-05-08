#ifndef ALIASES_HPP
#define ALIASES_HPP

#include <array>
#include <cstddef>

// Publicly exposed aliases.

namespace poisson_solver {
template <std::size_t Dim>
using GridSizes = std::array<std::size_t, Dim>;
}

#endif  // ALIASES_HPP
