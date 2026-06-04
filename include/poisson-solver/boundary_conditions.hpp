#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include <array>
#include <cstddef>

namespace poisson_solver {
// === Enums ===
enum class BoundaryCondition {
    Periodic,
    Dirichlet,
    Neumann,
    DirichletStag,
    NeumannStag
};

// === Structs ===
struct BCPair {
    const BoundaryCondition first{};
    const BoundaryCondition second{};

    auto operator==(const BCPair &) const -> bool = default;
};

// === Aliases ===
template <std::size_t Dim>
using BCs = std::array<BCPair, Dim>;

// === Functions ===
constexpr auto IsPeriodic(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Periodic && bcp.second == BC::Periodic;
}

constexpr auto IsDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Dirichlet && bcp.second == BC::Dirichlet;
}

constexpr auto IsNeumann(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Neumann && bcp.second == BC::Neumann;
}

constexpr auto IsDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::DirichletStag && bcp.second == BC::DirichletStag;
}

constexpr auto IsNeumannStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::NeumannStag && bcp.second == BC::NeumannStag;
}

constexpr auto IsDirichletNeumann(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Dirichlet && bcp.second == BC::Neumann;
}

constexpr auto IsNeumannDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Neumann && bcp.second == BC::Dirichlet;
}

constexpr auto IsDirichletStagNeumannStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::DirichletStag && bcp.second == BC::NeumannStag;
}

constexpr auto IsNeumannStagDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::NeumannStag && bcp.second == BC::DirichletStag;
}

// (D, D) or (DS, DS)
constexpr auto IsDirichletOrStag(BCPair bcp) -> bool {
    return IsDirichlet(bcp) || IsDirichletStag(bcp);
}

// (N, N) or (NS, NS)
constexpr auto IsNeumannOrStag(BCPair bcp) -> bool {
    return IsNeumann(bcp) || IsNeumannStag(bcp);
}

constexpr auto HasPeriodic(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Periodic || bcp.second == BC::Periodic;
}

constexpr auto HasDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::Dirichlet || bcp.second == BC::Dirichlet;
}

constexpr auto HasDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::DirichletStag || bcp.second == BC::DirichletStag;
}

constexpr auto HasDirichletOrStag(BCPair bcp) -> bool {
    return HasDirichlet(bcp) || HasDirichletStag(bcp);
}
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
