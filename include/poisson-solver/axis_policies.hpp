#ifndef AXIS_POLICIES_HPP
#define AXIS_POLICIES_HPP

#include "poisson-solver/boundary_conditions.hpp"

#include <concepts>
#include <cstddef>

// This file groups together certain calculations that are based on boundary
// conditions of an axis. Valid boundary conditions on an axis are explicitly
// enumerated as specializations of the `AxisPolicy` class.

namespace poisson_solver {
template <BCPair BCs, std::floating_point T>
class AxisPolicy {
    // Static assert needs to depend on template parameter (so cannot just use
    // `false`). Otherwise will be called even when not instantiated.
    static_assert(sizeof(T) == -1, "Unsupported pair of boundary conditions.");
};

template <BCPair BCs, std::floating_point T>
    requires(BCs.first == BoundaryCondition::kPeriodic &&
             BCs.second == BoundaryCondition::kPeriodic)
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / n; }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return i < (n / 2) ? i * dkx : (n - i) * dkx;
    }
    static T CalculateNormalizationFactor(std::size_t n) { return n; }
};

template <BCPair BCs, std::floating_point T>
    requires(BCs.first == BoundaryCondition::kDirichlet &&
             BCs.second == BoundaryCondition::kDirichlet)
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / (n + 1); }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return (i + 1) * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * (n + 1); }
};

template <BCPair BCs, std::floating_point T>
    requires(BCs.first == BoundaryCondition::kNeumann &&
             BCs.second == BoundaryCondition::kNeumann)
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / (n - 1); }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return i * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * (n - 1); }
};

template <BCPair BCs, std::floating_point T>
    requires(BCs.first == BoundaryCondition::kDirichletStag &&
             BCs.second == BoundaryCondition::kDirichletStag)
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / n; }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return (i + 1) * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * n; }
};

template <BCPair BCs, std::floating_point T>
    requires(BCs.first == BoundaryCondition::kNeumannStag &&
             BCs.second == BoundaryCondition::kNeumannStag)
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / n; }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return i * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * n; }
};

template <BCPair BCs, std::floating_point T>
    requires((BCs.first == BoundaryCondition::kDirichletStag &&
              BCs.second == BoundaryCondition::kNeumannStag) ||
             (BCs.first == BoundaryCondition::kNeumannStag &&
              BCs.second == BoundaryCondition::kDirichletStag))
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / n; }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return (i + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * n; }
};

template <BCPair BCs, std::floating_point T>
    requires((BCs.first == BoundaryCondition::kDirichlet &&
              BCs.second == BoundaryCondition::kNeumann) ||
             (BCs.first == BoundaryCondition::kNeumann &&
              BCs.second == BoundaryCondition::kDirichlet))
class AxisPolicy<BCs, T> {
public:
    static T GetGridDx(T len, std::size_t n) { return len / n; }
    static T GetApproximationInput(std::size_t i, std::size_t n, T dkx) {
        return (i + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }
    static T CalculateNormalizationFactor(std::size_t n) { return 2 * n; }
};
}  // namespace poisson_solver

#endif  // AXIS_POLICIES_HPP
