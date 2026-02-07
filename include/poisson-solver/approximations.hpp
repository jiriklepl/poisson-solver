#ifndef APPROXIMATION_HPP
#define APPROXIMATION_HPP

#include <concepts>
#include <cmath>

namespace poisson_solver {
template <std::floating_point T>
class IApproximation {
public:
    virtual ~IApproximation() = default;
    virtual T operator()(T x) const = 0;
};

template <std::floating_point T>
class SpectralApproximation : public IApproximation<T> {
public:
    // f(x) = -(2x)^2
    virtual T operator()(T x) const override {
        return static_cast<T>(-4.0) * x * x;
    }
};

template <std::floating_point T>
class FiniteDifference2Approximation : public IApproximation<T> {
public:
    // f(x) = -(2 * sin(x))^2
    virtual T operator()(T x) const override {
        return static_cast<T>(-4.0) * std::sin(x) * std::sin(x);
    }
};

template <std::floating_point T>
class FiniteDifference4Approximation : public IApproximation<T> {
    // TODO - Not implemented
};
}  // namespace poisson_solver

#endif  // APPROXIMATION_HPP
