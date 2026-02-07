#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/approximations.hpp"

#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
std::size_t CalculateNormalizationFactor(
    std::size_t grid_n, std::pair<BoundaryCondition, BoundaryCondition> bcs);

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

template <typename T>
void CalculateEigenvalues(T grid_len, std::size_t grid_n,
                          std::pair<BoundaryCondition, BoundaryCondition> bcs,
                          const IApproximation<T> &approximation,
                          std::vector<T> &eigenvalues) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    eigenvalues.resize(grid_n);

    T grid_dx = GetGridDx(grid_len, grid_n, bcs);
    T dkx = std::numbers::pi_v<T> * grid_dx / grid_len;
    T dkx_half = dkx / 2;

    T inv_grid_dx_squared = 1.0 / (grid_dx * grid_dx);

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        std::size_t grid_n_half = grid_n / 2;
        for (std::size_t i = 0; i < grid_n; ++i) {
            if (i < grid_n_half) {
                eigenvalues[i] = approximation(i * dkx);
            } else {
                eigenvalues[i] = approximation((grid_n - i) * dkx);
            }
        }
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] = approximation((i + 1) * dkx_half);
        }
    } else if (bc1 == BC::kDirichletStag && bc2 == BC::kDirichletStag) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] = approximation((i + 1) * dkx_half);
        }
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] = approximation(i * dkx_half);
        }
    } else if (bc1 == BC::kNeumannStag && bc2 == BC::kNeumannStag) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] = approximation(i * dkx_half);
        }
    } else if ((bc1 == BC::kDirichletStag && bc2 == BC::kNeumannStag) ||
               (bc1 == BC::kNeumannStag && bc2 == BC::kDirichletStag)) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] =
                approximation((i + 1 - static_cast<T>(0.5)) * dkx_half);
        }
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        for (std::size_t i = 0; i < grid_n; ++i) {
            eigenvalues[i] =
                approximation((i + 1 - static_cast<T>(0.5)) * dkx_half);
        }
    }

    for (std::size_t i = 0; i < grid_n; ++i) {
        eigenvalues[i] *= inv_grid_dx_squared;
    }
}
}  // namespace poisson_solver

#endif  // HELPERS_HPP
