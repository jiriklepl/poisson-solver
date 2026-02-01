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
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
