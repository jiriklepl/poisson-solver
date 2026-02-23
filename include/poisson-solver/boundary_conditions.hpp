#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

namespace poisson_solver {
enum class BoundaryCondition {
    kPeriodic,
    kDirichlet,
    kNeumann,
    kDirichletStag,
    kNeumannStag
};

struct BCPair {
    const BoundaryCondition first;
    const BoundaryCondition second;
};
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
