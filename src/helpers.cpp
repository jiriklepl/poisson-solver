#include "poisson-solver/helpers.hpp"

namespace poisson_solver {
// Valid BC combinations are:
// 1. {P, P}
// 2. {D, D}
// 3. {N, N}
// 4. {*S, *S}
// 5. {D, N}
std::size_t CalculateNormalizationFactor(
    std::size_t n, std::pair<BoundaryCondition, BoundaryCondition> bcs) {
    using BC = BoundaryCondition;

    auto [bc1, bc2] = bcs;

    if (bc1 == BC::kPeriodic && bc2 == BC::kPeriodic) {
        return n;
    } else if (bc1 == BC::kDirichlet && bc2 == BC::kDirichlet) {
        return 2 * (n + 1);
    } else if (bc1 == BC::kNeumann && bc2 == BC::kNeumann) {
        return 2 * (n - 1);
    } else if (IsStag(bc1) && IsStag(bc2)) {
        return 2 * n;
    } else if ((bc1 == BC::kDirichlet && bc2 == BC::kNeumann) ||
               (bc1 == BC::kNeumann && bc2 == BC::kDirichlet)) {
        return 2 * n;
    }

    throw std::invalid_argument("Invalid boundary conditions.");
}
}  // namespace poisson_solver
