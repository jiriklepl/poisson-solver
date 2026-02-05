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

float EigFnSpectral(float x);

void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::vector<float> &eigenvalues);
}  // namespace poisson_solver

#endif  // HELPERS_HPP
