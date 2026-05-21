#ifndef C_SOLVER_WRAPPER_HPP
#define C_SOLVER_WRAPPER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/solver.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <span>

namespace poisson_solver::detail {
template <std::floating_point T>
class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void Execute(std::span<T> phi,
                         std::span<const T> rhs,
                         std::span<const std::size_t> ngs_phi_span,
                         std::span<const std::size_t> ngs_rhs_span) = 0;
    virtual std::size_t GetTotalGridSize() = 0;
};

template <std::floating_point T, std::size_t Dim>
class SolverWrapper : public ISolver<T> {
public:
    SolverWrapper(GridSizes<Dim> ns,
                  std::array<T, Dim> lens,
                  BCs<Dim> bcs,
                  const IApproximation<T> &approx)
        : solver_(ns, lens, bcs, approx) {}

    virtual void Execute(std::span<T> phi,
                         std::span<const T> rhs,
                         std::span<const std::size_t> ngs_phi_span,
                         std::span<const std::size_t> ngs_rhs_span) override {
        auto ngs_phi = CopyIntoGridSizes(ngs_phi_span);
        auto ngs_rhs = CopyIntoGridSizes(ngs_rhs_span);
        solver_.Execute(phi, rhs, ngs_phi, ngs_rhs);
    }

    virtual std::size_t GetTotalGridSize() override {
        return solver_.GetTotalGridSize();
    }

private:
    Solver<T, Dim> solver_;

    GridSizes<Dim> CopyIntoGridSizes(std::span<const std::size_t> s) {
        GridSizes<Dim> g{};
        for (std::size_t d = 0; d < Dim; ++d) {
            g[d] = s[d];
        }
        return g;
    }
};
}  // namespace poisson_solver::detail

#endif  // C_SOLVER_WRAPPER_HPP
