#include "poisson-solver/solver.hpp"

namespace poisson_solver {
Solver2D::Solver2D(float len_x, float len_y, std::size_t n_x, std::size_t n_y,
                   BoundaryConditions<2> bcs, Approximation approx)
    : len_x(len_x),
      len_y(len_y),
      n_x(n_x),
      n_y(n_y),
      cnt(n_x * n_y),
      bcs(bcs),
      denom_x(n_x),
      denom_y(n_y),
      approx(approx),
      real_workspace(n_x, n_y) {
    if (std::ranges::any_of(bcs.Flatten(), [](BoundaryCondition bc) {
            return bc != BoundaryCondition::kNeumann;
        })) {
        throw std::invalid_argument(
            "Error: Only Neumann BCs are allowed currently!\n");
    }

    if (approx != Approximation::kSpectral) {
        throw std::invalid_argument(
            "Error: Only spectral approximation is allowed currently!\n");
    }

    norm_factor = CalculateNormalizationFactor(n_x, bcs.x) *
                  CalculateNormalizationFactor(n_y, bcs.y);

    CalculateEigenvalues(len_x, n_x, bcs.x, denom_x);
    CalculateEigenvalues(len_y, n_y, bcs.y, denom_y);
}

void Solver2D::Execute() { RunVectorAdd(); }
}  // namespace poisson_solver
