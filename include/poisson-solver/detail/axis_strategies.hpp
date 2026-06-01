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
    virtual T GetGridDx(std::size_t n, T len) const = 0;
    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const = 0;
    virtual T ComputeNormalizationFactor(std::size_t n) const = 0;

    virtual ~IAxisStrategy() = default;
};

template <std::floating_point T>
class PeriodicStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i < (n / 2) ? i * dkx : (n - i) * dkx;
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return n;
    }
};

template <std::floating_point T>
class DirichletStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / (n + 1);
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return 2 * (n + 1);
    }
};

template <std::floating_point T>
class NeumannStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / (n - 1);
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return 2 * (n - 1);
    }
};

template <std::floating_point T>
class DirichletStagStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1) * (dkx / 2);
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class NeumannStagStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return i * (dkx / 2);
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};

template <std::floating_point T>
class MixedStrategy : public IAxisStrategy<T> {
public:
    virtual T GetGridDx(std::size_t n, T len) const override {
        return len / n;
    }

    virtual T GetApproximationInput(std::size_t i,
                                    std::size_t n,
                                    T dkx) const override {
        return (i + 1 - static_cast<T>(0.5)) * (dkx / 2);
    }

    virtual T ComputeNormalizationFactor(std::size_t n) const override {
        return 2 * n;
    }
};
}  // namespace poisson_solver::detail

#endif  // AXIS_STRATEGIES_HPP
