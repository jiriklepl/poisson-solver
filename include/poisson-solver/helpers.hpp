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

// Returns the space between two grid points on one axis.
// Valid BC combinations are:
// 1. {P, P}
// 2. {D, D}
// 3. {N, N}
// 4. {*S, *S}
// 5. {D, N}
template <typename T>
T GetGridDx(T grid_len, std::size_t grid_n,
            std::pair<BoundaryCondition, BoundaryCondition> bcs) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        return grid_len / grid_n;
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        return grid_len / (grid_n + 1);
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        return grid_len / (grid_n - 1);
    } else if (IsStag(bc1) && IsStag(bc2)) {
        return grid_len / grid_n;
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        return grid_len / grid_n;
    }

    throw std::invalid_argument("Invalid boundary conditions.");
}

void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::pair<BoundaryCondition, BoundaryCondition> bcs,
                          std::vector<float> &eigenvalues);
}  // namespace poisson_solver

#endif  // HELPERS_HPP
