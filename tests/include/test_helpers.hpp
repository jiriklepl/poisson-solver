#include "poisson-solver/aliases.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/helpers.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <random>
#include <span>

namespace poisson_solver::tests {
namespace psd = ::poisson_solver::detail;

template <std::floating_point T>
using RHSAdjustFunction = std::function<void(std::span<T>)>;

// === Forward Declarations ===
template <std::floating_point T>
auto MakeNopCorrector() -> RHSAdjustFunction<T>;

// === Structs and Classes ===

/**
 * struct TestConfig - Data required to execute a residuum test.
 *
 * @tparam T Floating point type.
 * @tparam Dim Dimension of the tested solver.
 */
template <std::floating_point T, std::size_t Dim>
struct TestConfig {
public:
    TestConfig(BCs<Dim> bcs,
               std::array<std::size_t, Dim> dx_divisors,
               RHSAdjustFunction<T> rhs_adjust = MakeNopCorrector<T>())
        : bcs(bcs),
          dx_divisors(dx_divisors),
          rhs_adjust(rhs_adjust) {}

    BCs<Dim> bcs{};
    std::array<std::size_t, Dim> dx_divisors{};
    RHSAdjustFunction<T> rhs_adjust = nullptr;

    auto Name() const -> std::string {
        std::string name;
        for (std::size_t i{0}; const auto &bcp : bcs) {
            name += ToString(bcp.first) + " | " + ToString(bcp.second);
            if (i < Dim - 1) {
                name += " | ";
            }
            ++i;
        }
        return name;
    }
};

template <std::floating_point T>
struct ResiduumResult {
public:
    bool success;
    T max_residuum;
};

// === Functions ===
template <std::floating_point T>
void Initialize(std::span<T> phi, std::span<T> rhs, unsigned seed) {
    assert(rhs.size() == phi.size());

    const std::size_t size = rhs.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    T rhs_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        phi[i] = dist(rng);
        T rhs_val = dist(rng);
        rhs[i] = rhs_val;
        rhs_sum += rhs_val;
    }

    T avg = rhs_sum / (size);
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] -= avg;
    }
}

template <std::size_t Dim>
auto FlatToMultidimIdx(std::size_t idx,
                       GridSizes<Dim> ns) -> psd::MultidimIdx<Dim> {
    psd::MultidimIdx<Dim> midx{};
    for (std::size_t i = 0; i < Dim; ++i) {
        midx[i] = idx % ns[i];
        idx /= ns[i];
    }
    return midx;
};

template <std::floating_point T, std::size_t Dim>
auto ResiduumBelowThreshold(T residuum,
                            GridSizes<Dim> ns) -> ResiduumResult<T> {
    constexpr T fudge_factor = static_cast<T>(100.0);
    T max_residuum =
        std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>()) *
        fudge_factor * std::numeric_limits<T>::epsilon();
    bool success = residuum < max_residuum;
    return {.success = success, .max_residuum = max_residuum};
}

// == Correctors ==

/**
 * @brief Makes a corrector that does no-op.
 */
template <std::floating_point T>
auto MakeNopCorrector() -> RHSAdjustFunction<T> {
    return [](std::span<T>) {};
}

template <std::floating_point T>
auto MakePPDRhsCorrector() -> RHSAdjustFunction<T> {
    using std::ranges::for_each;
    return [](std::span<T> rhs) {
        for_each(rhs, [](T &val) {
            val += static_cast<T>(1.0);
        });
    };
}

template <std::floating_point T, std::size_t Dim>
auto MakeNeumannRhsCorrector(GridSizes<Dim> ns) -> RHSAdjustFunction<T> {
    return [ns](std::span<T> rhs) {
        T sum = 0.0;
        std::size_t num_cells = 1.0;

        // Calculate the total number of "cells" (N-1 per dimension)
        for (std::size_t d = 0; d < Dim; ++d) {
            num_cells *= ns[d] - 1;
        }

        for (std::size_t i = 0; i < rhs.size(); ++i) {
            T p = rhs[i];

            auto midx = FlatToMultidimIdx<Dim>(i, ns);

            // Halve the value for every dimension's boundary it touches
            for (std::size_t d = 0; d < Dim; ++d) {
                if (midx[d] == 0 || midx[d] == ns[d] - 1) {
                    p /= static_cast<T>(2.0);
                }
            }

            sum += p;
        }

        T mean = sum / num_cells;

        // Subtract the calculated mean from every node
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            rhs[i] -= mean;
        }
    };
}

template <std::floating_point T>
auto MakeNeumannStagRhsCorrector() -> RHSAdjustFunction<T> {
    return [](std::span<T> rhs) {
        T sum = std::accumulate(rhs.begin(), rhs.end(), T{0.0});
        std::ranges::for_each(rhs, [sum](T &elem) {
            elem -= sum;
        });
    };
}

}  // namespace poisson_solver::tests
