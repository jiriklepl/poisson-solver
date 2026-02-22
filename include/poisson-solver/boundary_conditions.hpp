#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include <array>
#include <utility>

namespace poisson_solver {
enum class BoundaryCondition {
    kPeriodic,
    kDirichlet,
    kNeumann,
    kDirichletStag,
    kNeumannStag
};

struct BCPair {
    BoundaryCondition first;
    BoundaryCondition second;
};

// Stores boundary conditions for specific dimensions.
// Only specializations should be used as not every dimension is valid.
template <std::size_t Dim>
struct BCStorage {
    static_assert(Dim >= 1 && Dim <= 3,
                  "Boundary conditions have to be at least one dimensional and "
                  "at most three dimensional.");
};

template <>
struct BCStorage<1> {
    BCPair x;
    constexpr BCStorage(BCPair x) : x(x) {}
};

template <>
struct BCStorage<2> : public BCStorage<1> {
    BCPair y;
    constexpr BCStorage(BCPair x, BCPair y) : BCStorage<1>(x), y(y) {}
};

template <>
struct BCStorage<3> : public BCStorage<2> {
    BCPair z;
    constexpr BCStorage(BCPair x, BCPair y, BCPair z)
        : BCStorage<2>(x, y), z(z) {}
};

// Encompasses the logic of boundary conditions.
template <std::size_t Dim>
struct BoundaryConditions : public BCStorage<Dim> {
    // Inherit ctors
    using BCStorage<Dim>::BCStorage;

    // Flattens the boundary conditions into a 1D array.
    // Could be used for example when working with ranges.
    constexpr auto Flatten() const {
        constexpr std::size_t arr_size = Dim * 2;

        std::array<BoundaryCondition, arr_size> flat_bcs{};

        flat_bcs[0] = this->x.first;
        flat_bcs[1] = this->x.second;

        if constexpr (Dim >= 2) {
            flat_bcs[2] = this->y.first;
            flat_bcs[3] = this->y.second;
        }

        if constexpr (Dim >= 3) {
            flat_bcs[4] = this->z.first;
            flat_bcs[5] = this->z.second;
        }

        return flat_bcs;
    }
};
}  // namespace poisson_solver

#endif  // BOUNDARY_CONDITIONS_HPP
