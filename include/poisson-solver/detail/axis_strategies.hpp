#ifndef AXIS_STRATEGIES_HPP
#define AXIS_STRATEGIES_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>

// This file groups together certain calculations that are based on boundary
// conditions of an axis. Valid boundary conditions on an axis are explicitly
// enumerated in the `GetAxisStrategy` function.

namespace poisson_solver::detail {
template <std::floating_point T>
class IAxisStrategy {
public:
    virtual T GetGridDx(T len, std::size_t n) const = 0;
    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const = 0;
    virtual T CalculateNormalizationFactor(std::size_t n) const = 0;

    virtual ~IAxisStrategy() = default;
};

template <std::floating_point T>
class PeriodicStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i < (n / 2) ? i * dkx : (n - i) * dkx;
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return n;
    }
};

template <std::floating_point T>
class DirichletStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / (n + 1);
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * (n + 1);
    }
};

template <std::floating_point T>
class NeumannStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / (n - 1);
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * (n - 1);
    }
};

template <std::floating_point T>
class DirichletStagStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class NeumannStagStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class MixedStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(T len, std::size_t n) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }

    virtual T CalculateNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
static std::unique_ptr<IAxisStrategy<T>> GetAxisStrategy(BCPair bcs) {
    using BC = BoundaryCondition;
    constexpr BC P = BC::kPeriodic;
    constexpr BC D = BC::kDirichlet;
    constexpr BC N = BC::kNeumann;
    constexpr BC DS = BC::kDirichletStag;
    constexpr BC NS = BC::kNeumannStag;

    if (bcs.first == P && bcs.second == P) {
        return std::make_unique<PeriodicStrategy<T>>();
    } else if (bcs.first == D && bcs.second == D) {
        return std::make_unique<DirichletStrategy<T>>();
    } else if (bcs.first == N && bcs.second == N) {
        return std::make_unique<NeumannStrategy<T>>();
    } else if (bcs.first == DS && bcs.second == DS) {
        return std::make_unique<DirichletStagStrategy<T>>();
    } else if (bcs.first == NS && bcs.second == NS) {
        return std::make_unique<NeumannStagStrategy<T>>();
    } else if (((bcs.first == D && bcs.second == N) ||
                (bcs.first == N && bcs.second == D)) ||
               (bcs.first == DS && bcs.second == NS) ||
               (bcs.first == NS && bcs.second == DS)) {
        return std::make_unique<MixedStrategy<T>>();
    } else {
        throw std::invalid_argument("Invalid boundary condition combination.");
    }
}
}  // namespace poisson_solver::detail

#endif  // AXIS_STRATEGIES_HPP
