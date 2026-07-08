#ifndef AXIS_STRATEGIES_HPP
#define AXIS_STRATEGIES_HPP

#include <concepts>
#include <cstddef>

// This file groups together certain calculations that are based on boundary
// conditions of an axis. Valid boundary conditions on an axis are explicitly
// enumerated in the `GetAxisStrategy` function.

namespace poisson_solver::detail {
template <std::floating_point T>
class IAxisStrategy {
public:
    virtual auto GetGridDx(std::size_t n, T len) const -> T = 0;
    virtual auto GetApproximationInput(std::size_t i,
                                       std::size_t offset,
                                       std::size_t gn,
                                       T dkx) const -> T = 0;
    virtual auto ComputeNormalizationFactor(std::size_t n) const -> T = 0;

    virtual ~IAxisStrategy() = default;
};

template <std::floating_point T>
class PeriodicStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / n;
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return gi <= (gn / 2) ? gi * dkx : (gn - gi) * dkx;
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return n;
    }
};

template <std::floating_point T>
class DirichletStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / (n + 1);
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return (gi + 1) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return 2 * (n + 1);
    }
};

template <std::floating_point T>
class NeumannStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / (n - 1);
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return gi * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return 2 * (n - 1);
    }
};

template <std::floating_point T>
class DirichletStagStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / n;
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return (gi + 1) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return 2 * n;
    }
};

template <std::floating_point T>
class NeumannStagStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / n;
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return gi * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return 2 * n;
    }
};

template <std::floating_point T>
class MixedStrategy : public IAxisStrategy<T> {
public:
    auto GetGridDx(std::size_t n, T len) const -> T override {
        return len / n;
    }

    auto GetApproximationInput(std::size_t i,
                               std::size_t offset,
                               [[maybe_unused]] std::size_t gn,
                               T dkx) const -> T override {
        auto gi = i + offset;
        return (gi + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }

    auto ComputeNormalizationFactor(std::size_t n) const -> T override {
        return 2 * n;
    }
};
}  // namespace poisson_solver::detail

#endif  // AXIS_STRATEGIES_HPP
