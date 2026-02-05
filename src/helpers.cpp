#include "poisson-solver/helpers.hpp"

namespace poisson_solver {
// Valid BC combinations are:
// 1. {P, P}
// 2. {D, D}
// 3. {N, N}
// 4. {*S, *S}
// 5. {D, N}
std::size_t CalculateNormalizationFactor(
    std::size_t grid_n, std::pair<BoundaryCondition, BoundaryCondition> bcs) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        return grid_n;
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        return 2 * (grid_n + 1);
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        return 2 * (grid_n - 1);
    } else if (IsStag(bc1) && IsStag(bc2)) {
        return 2 * grid_n;
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        return 2 * grid_n;
    }

    throw std::invalid_argument("Invalid boundary conditions.");
}

// TODO: Works only for all bcs == Neumann and approximation == Spectral
void CalculateEigenvalues(float grid_len, std::size_t grid_n,
                          std::pair<BoundaryCondition, BoundaryCondition> bcs,
                          std::vector<float> &eigenvalues) {
    eigenvalues.resize(grid_n);

    float grid_dx = GetGridDx(grid_len, grid_n, bcs);
    float dkx = std::numbers::pi * grid_dx / grid_len;
    float dkx_half = dkx / 2;

    float inv_grid_dx_squared = 1.0f / (grid_dx * grid_dx);

    for (std::size_t i = 0; i < grid_n; ++i) {
        eigenvalues[i] = EigFnSpectral(i * dkx_half);

        eigenvalues[i] *= inv_grid_dx_squared;
    }
}
}  // namespace poisson_solver
