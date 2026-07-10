#ifndef AXIS_STRATEGIES_HPP
#define AXIS_STRATEGIES_HPP

#include <concepts>
#include <cstddef>

// This file groups together certain calculations that are based on boundary
// conditions of an axis. Valid boundary conditions on an axis are explicitly
// enumerated in the `ISolverPlan::GetAxisStrategy` function.

namespace poisson_solver::detail::axis_strategy_internals {
/**
 * HeFFTe is used as an MPI FFT library and its unscaled results
 * do not match those of VkFFT (the local FFT library).
 * It is therefore necessary to adjust the norm_factor in those cases.
 *
 * @param norm_factor
 * @param times Multiplied by this.
 * @return Adjusted norm factor.
 */
inline auto AdjustNormFactorIfHeFFTeUsed(std::size_t norm_factor,
                                         [[maybe_unused]] std::size_t times)
    -> std::size_t {
#ifdef USE_MPI
    return norm_factor * times;
#else
    return norm_factor;
#endif  // USE_MPI
}
}  // namespace poisson_solver::detail::axis_strategy_internals

namespace poisson_solver::detail {
template <std::floating_point T>
class IAxisStrategy {
public:
    virtual auto GetGridDx(std::size_t n, T len) const -> T = 0;
    virtual auto GetApproximationInput(std::size_t i,
                                       std::size_t offset,
                                       std::size_t gn,
                                       T dkx) const -> T = 0;
    virtual auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t = 0;

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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
        using axis_strategy_internals::AdjustNormFactorIfHeFFTeUsed;
        return AdjustNormFactorIfHeFFTeUsed(2 * (n - 1), 2);
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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
        using axis_strategy_internals::AdjustNormFactorIfHeFFTeUsed;
        return AdjustNormFactorIfHeFFTeUsed(2 * n, 2);
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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
        using axis_strategy_internals::AdjustNormFactorIfHeFFTeUsed;
        return AdjustNormFactorIfHeFFTeUsed(2 * n, 2);
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

    auto ComputeNormalizationFactor(std::size_t n) const
        -> std::size_t override {
        return 2 * n;
    }
};
}  // namespace poisson_solver::detail

#endif  // AXIS_STRATEGIES_HPP
