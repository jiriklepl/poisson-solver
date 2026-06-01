#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include <array>
#include <cstddef>

namespace poisson_solver {
// === Enums ===
enum class BoundaryCondition {
    kPeriodic,
    kDirichlet,
    kNeumann,
    kDirichletStag,
    kNeumannStag
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
    return bcp.first == BC::kPeriodic && bcp.second == BC::kPeriodic;
}

constexpr auto IsDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichlet && bcp.second == BC::kDirichlet;
}

constexpr auto IsNeumann(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumann && bcp.second == BC::kNeumann;
}

constexpr auto IsDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichletStag && bcp.second == BC::kDirichletStag;
}

constexpr auto IsNeumannStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumannStag && bcp.second == BC::kNeumannStag;
}

constexpr auto IsDirichletNeumann(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichlet && bcp.second == BC::kNeumann;
}

constexpr auto IsNeumannDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumann && bcp.second == BC::kDirichlet;
}

constexpr auto IsDirichletStagNeumannStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichletStag && bcp.second == BC::kNeumannStag;
}

constexpr auto IsNeumannStagDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumannStag && bcp.second == BC::kDirichletStag;
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
    return bcp.first == BC::kPeriodic || bcp.second == BC::kPeriodic;
}

constexpr auto HasDirichlet(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichlet || bcp.second == BC::kDirichlet;
}

constexpr auto HasDirichletStag(BCPair bcp) -> bool {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichletStag || bcp.second == BC::kDirichletStag;
}

constexpr auto HasDirichletOrStag(BCPair bcp) -> bool {
    return HasDirichlet(bcp) || HasDirichletStag(bcp);
}
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
