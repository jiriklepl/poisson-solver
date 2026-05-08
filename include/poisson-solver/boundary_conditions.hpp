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
    const BoundaryCondition first;
    const BoundaryCondition second;

    bool operator==(const BCPair &) const = default;
};

// === Aliases ===
template <std::size_t Dim>
using BCs = std::array<BCPair, Dim>;

// === Functions ===
constexpr bool IsPeriodic(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kPeriodic && bcp.second == BC::kPeriodic;
}

constexpr bool IsDirichlet(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichlet && bcp.second == BC::kDirichlet;
}

constexpr bool IsNeumann(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumann && bcp.second == BC::kNeumann;
}

constexpr bool IsDirichletStag(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichletStag && bcp.second == BC::kDirichletStag;
}

constexpr bool IsNeumannStag(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kNeumannStag && bcp.second == BC::kNeumannStag;
}

// (D, D) or (DS, DS)
constexpr bool IsDirichletOrStag(BCPair bcp) {
    return IsDirichlet(bcp) || IsDirichletStag(bcp);
}

// (N, N) or (NS, NS)
constexpr bool IsNeumannOrStag(BCPair bcp) {
    return IsNeumann(bcp) || IsNeumannStag(bcp);
}

constexpr bool HasPeriodic(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kPeriodic || bcp.second == BC::kPeriodic;
}

constexpr bool HasDirichlet(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichlet || bcp.second == BC::kDirichlet;
}

constexpr bool HasDirichletStag(BCPair bcp) {
    using BC = BoundaryCondition;
    return bcp.first == BC::kDirichletStag || bcp.second == BC::kDirichletStag;
}

constexpr bool HasDirichletOrStag(BCPair bcp) {
    return HasDirichlet(bcp) || HasDirichletStag(bcp);
}
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
