#ifndef AXIS_POLICIES_HPP
#define AXIS_POLICIES_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>

// This file groups together certain calculations that are based on boundary
// conditions of an axis. Valid boundary conditions on an axis are explicitly
// enumerated in the `GetAxisPolicy` function.

namespace poisson_solver {
template <std::floating_point T>
class IAxisPolicy {
public:
    virtual T GetGridDx(T len, std::size_t n) const = 0;
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const = 0;
    virtual T CalculateNormalizationFactor(std::size_t n) const = 0;

    virtual ~IAxisPolicy() = default;
};

template <std::floating_point T>
class PeriodicPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override { return len / n; }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return i < (n / 2) ? i * dkx : (n - i) * dkx;
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return n;
    }
};

template <std::floating_point T>
class DirichletPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / (n + 1);
    }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * (n + 1);
    }
};

template <std::floating_point T>
class NeumannPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / (n - 1);
    }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * (n - 1);
    }
};

template <std::floating_point T>
class DirichletStagPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override { return len / n; }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class NeumannStagPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override { return len / n; }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class MixedPolicy : public IAxisPolicy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override { return len / n; }
    virtual T GetApproximationInput(std::size_t i, std::size_t n,
                                    T dkx) const override {
        return (i + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }
    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
static std::unique_ptr<IAxisPolicy<T>> GetAxisPolicy(BCPair bcs) {
    using BC = BoundaryCondition;
    constexpr BC P = BC::kPeriodic;
    constexpr BC D = BC::kDirichlet;
    constexpr BC N = BC::kNeumann;
    constexpr BC DS = BC::kDirichletStag;
    constexpr BC NS = BC::kNeumannStag;

    if (bcs.first == P && bcs.second == P) {
        return std::make_unique<PeriodicPolicy<T>>();
    } else if (bcs.first == D && bcs.second == D) {
        return std::make_unique<DirichletPolicy<T>>();
    } else if (bcs.first == N && bcs.second == N) {
        return std::make_unique<NeumannPolicy<T>>();
    } else if (bcs.first == DS && bcs.second == DS) {
        return std::make_unique<DirichletStagPolicy<T>>();
    } else if (bcs.first == NS && bcs.second == NS) {
        return std::make_unique<NeumannStagPolicy<T>>();
    } else if (((bcs.first == D && bcs.second == N) ||
                (bcs.first == N && bcs.second == D)) ||
               (bcs.first == DS && bcs.second == NS) ||
               (bcs.first == NS && bcs.second == DS)) {
        return std::make_unique<MixedPolicy<T>>();
    } else {
        throw std::invalid_argument("Invalid boundary condition combination.");
    }
}
}  // namespace poisson_solver

#endif  // AXIS_POLICIES_HPP
