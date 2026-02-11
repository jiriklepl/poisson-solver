#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"

#include <concepts>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
std::size_t CalculateNormalizationFactor(
    std::size_t n, std::pair<BoundaryCondition, BoundaryCondition> bcs);

constexpr bool IsStag(BoundaryCondition bc) {
    return bc == BoundaryCondition::kDirichletStag ||
           bc == BoundaryCondition::kNeumannStag;
}

// Returns the space between two grid points on one axis.
// Valid BC combinations are:
// 1. {P, P}
// 2. {D, D}
// 3. {N, N}
// 4. {*S, *S}
// 5. {D, N}
template <std::floating_point T>
T GetGridDx(T len, std::size_t n,
            std::pair<BoundaryCondition, BoundaryCondition> bcs) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        return len / n;
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        return len / (n + 1);
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        return len / (n - 1);
    } else if (IsStag(bc1) && IsStag(bc2)) {
        return len / n;
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        return len / n;
    }

    throw std::invalid_argument("Invalid boundary conditions.");
}

template <std::floating_point T>
void CalculateEigenvalues(T len, std::size_t n,
                          std::pair<BoundaryCondition, BoundaryCondition> bcs,
                          const IApproximation<T> &approx,
                          std::vector<T> &eigenvals) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    eigenvals.resize(n);

    T grid_dx = GetGridDx(len, n, bcs);
    T dkx = std::numbers::pi_v<T> * grid_dx / len;

    T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        for (std::size_t i = 0; i < n; ++i) {
            if (i < (n / 2)) {
                eigenvals[i] = approx(i * dkx);
            } else {
                eigenvals[i] = approx((n - i) * dkx);
            }
        }
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx((i + 1) * (dkx / 2));
        }
    } else if (bc1 == BC::kDirichletStag && bc2 == BC::kDirichletStag) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx((i + 1) * (dkx / 2));
        }
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(i * (dkx / 2));
        }
    } else if (bc1 == BC::kNeumannStag && bc2 == BC::kNeumannStag) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(i * (dkx / 2));
        }
    } else if ((bc1 == BC::kDirichletStag && bc2 == BC::kNeumannStag) ||
               (bc1 == BC::kNeumannStag && bc2 == BC::kDirichletStag)) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx((i + 1 - static_cast<T>(0.5)) * (dkx / 2));
        }
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx((i + 1 - static_cast<T>(0.5)) * (dkx / 2));
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        eigenvals[i] *= inv_grid_dx_squared;
    }
}

template <std::floating_point T>
bool IsBelowThreshold(T residuum, std::size_t nx, std::size_t ny) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    return residuum <
           nx * ny * fudge_factor * std::numeric_limits<T>::epsilon();
}
}  // namespace poisson_solver

#endif  // HELPERS_HPP
