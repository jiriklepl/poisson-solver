#ifndef C_SOLVER_WRAPPER_HPP
#define C_SOLVER_WRAPPER_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/solver.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <span>

namespace poisson_solver {
template <std::floating_point T>
class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void Execute(std::span<T> phi, std::span<const T> rhs) = 0;
    virtual std::size_t GetTotalGridSize() = 0;
};

template <std::floating_point T,
          std::size_t Dim,
          template <std::floating_point> typename FFTPolicy>
class SolverWrapper : public ISolver<T> {
public:
    SolverWrapper(std::array<std::size_t, Dim> ns,
                  std::array<T, Dim> lens,
                  std::array<BCPair, Dim> bcs,
                  const IApproximation<T> &approx)
        : solver_(ns, lens, bcs, approx) {}

    virtual void Execute(std::span<T> phi, std::span<const T> rhs) override {
        solver_.Execute(phi, rhs);
    }

    virtual std::size_t GetTotalGridSize() {
        return solver_.GetTotalGridSize();
    }

private:
    Solver<T, Dim, FFTPolicy> solver_;
};
}  // namespace poisson_solver

#endif  // C_SOLVER_WRAPPER_HPP
