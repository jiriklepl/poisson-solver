#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/boundary_conditions.hpp"

#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
std::size_t CalculateNormalizationFactor(
    std::size_t grid_n, std::pair<BoundaryCondition, BoundaryCondition> bcs);

// f(x) = -(2x)^2
template <typename T>
constexpr T EigFnSpectral(T x) {
    return static_cast<T>(-4.0) * x * x;
}

void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::vector<float> &eigenvalues);
}  // namespace poisson_solver

#endif  // HELPERS_HPP
