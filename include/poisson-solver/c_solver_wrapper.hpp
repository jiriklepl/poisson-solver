#ifndef C_SOLVER_WRAPPER_HPP
#define C_SOLVER_WRAPPER_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/solver.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace poisson_solver {
class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void Execute(std::span<double> phi,
                         std::span<const double> rhs) = 0;
    virtual std::size_t GetTotalGridSize() = 0;
};

template <std::size_t Dim, std::array<BCPair, Dim> BCs>
class SolverWrapper : public ISolver {
public:
    SolverWrapper(std::array<double, Dim> lens, std::array<std::size_t, Dim> ns,
                  const IApproximation<double> &approx)
        : solver_(lens, ns, approx) {}

    virtual void Execute(std::span<double> phi,
                         std::span<const double> rhs) override {
        solver_.Execute(phi, rhs);
    }

    virtual std::size_t GetTotalGridSize() {
        return solver_.GetTotalGridSize();
    }

private:
    Solver<Dim, BCs> solver_;
};
}  // namespace poisson_solver

#endif  // C_SOLVER_WRAPPER_HPP
