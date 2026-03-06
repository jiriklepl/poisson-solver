#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <numeric>
#include <random>

namespace ps = poisson_solver;

namespace {
void Initialize(std::span<float> phi, std::span<float> rhs, unsigned seed) {
    if (rhs.size() != phi.size()) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }

    const std::size_t size = rhs.size();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float rhs_sum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        phi[i] = dist(rng);
        float rhs_val = dist(rng);
        rhs[i] = rhs_val;
        rhs_sum += rhs_val;
    }

    float avg = rhs_sum / (size);
    for (std::size_t i = 0; i < size; ++i) {
        rhs[i] -= avg;
    }
}

// Coeffs order: [west, east, south, north, bottom, top]
template <std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
float GetMaxResiduumFullPeriodic(std::span<float> phi, std::span<float> rhs,
                                 std::array<std::size_t, Dim> ns,
                                 std::array<float, 2 * Dim> coeffs) {
    if (rhs.size() != phi.size()) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }
    const std::size_t size = rhs.size();

    auto multidim_to_flat_idx = [=](std::array<std::size_t, Dim> idxs) {
        return ps::FlatIndex<Dim>(idxs, ns);
    };

    // Calculates multidimensional idx from a flat one
    auto flat_to_multidim_idx = [=](std::size_t idx) {
        std::array<std::size_t, Dim> multidim_idx;
        for (std::size_t i = Dim; i > 0; --i) {
            multidim_idx[i - 1] = idx % ns[i - 1];
            idx /= ns[i - 1];
        }
        return multidim_idx;
    };

    // Gets the multidim_idxs of neighbors on each axis.
    // Adjusts for wrapping around.
    auto get_neighbor_idxs = [=](std::array<std::size_t, Dim> multidim_idx) {
        // Pairs of mutlidim idxs for each axis
        std::array<std::array<std::array<std::size_t, Dim>, 2>, Dim>
            neighbor_idxs{};
        for (std::size_t i = 0; i < Dim; ++i) {
            auto left_neighbor_idx = multidim_idx;
            left_neighbor_idx[i] =
                (multidim_idx[i] == 0) ? (ns[i] - 1) : (multidim_idx[i] - 1);

            auto right_neighbor_idx = multidim_idx;
            right_neighbor_idx[i] =
                (multidim_idx[i] == ns[i] - 1) ? 0 : (multidim_idx[i] + 1);

            neighbor_idxs[i] =
                std::array{left_neighbor_idx, right_neighbor_idx};
        }
        return neighbor_idxs;
    };

    float max_residuum = 0.0f;
    const float coeff_center =
        std::accumulate(coeffs.begin(), coeffs.end(), 0.0f);

    for (std::size_t i = 0; i < size; ++i) {
        float neighbor_sum = 0.0f;
        auto multidim_idx = flat_to_multidim_idx(i);
        auto neighbor_idxs = get_neighbor_idxs(multidim_idx);

        for (std::size_t d = 0; d < Dim; ++d) {
            const auto& [left_neigh_idx, right_neigh_idx] = neighbor_idxs[d];
            std::size_t flat_left_idx = multidim_to_flat_idx(left_neigh_idx);
            std::size_t flat_right_idx = multidim_to_flat_idx(right_neigh_idx);
            neighbor_sum += phi[flat_left_idx] * coeffs[2 * d];
            neighbor_sum += phi[flat_right_idx] * coeffs[2 * d + 1];
        }

        neighbor_sum -= rhs[i];
        const float local_residuum =
            std::abs(coeff_center * phi[i] - neighbor_sum);
        max_residuum = std::max(max_residuum, local_residuum);
    }

    return max_residuum;
}

template <std::floating_point T, std::size_t Dim>
bool ResiduumBelowThreshold(T residuum, std::array<std::size_t, Dim> ns) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    return residuum <
           std::accumulate(ns.begin(), ns.end(), 1, std::multiplies<>()) *
               fudge_factor * std::numeric_limits<T>::epsilon();
}
}  // namespace

TEST_CASE("Poisson Solver Residuum Suite") {
    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    constexpr std::size_t nx = 21, ny = 32, nz = 25;
    constexpr float lenx = 2 * std::numbers::pi_v<float>;
    constexpr float leny = lenx * 1.1f, lenz = lenx / 1.1f;
    constexpr float dx = lenx / nx, dy = leny / ny, dz = lenz / nz;
    constexpr float inv_dx_squared = 1.0f / (dx * dx);
    constexpr float inv_dy_squared = 1.0f / (dy * dy);
    constexpr float inv_dz_squared = 1.0f / (dz * dz);
    std::function<float()> residuum_calculation_strategy;

    SUBCASE("1D") {
        std::array<float, nx> phi{};
        std::array<float, nx> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE("Boundary: Periodic | Periodic") {
            constexpr std::array<ps::BCPair, 1> kBcs = {
                {{ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic}}};

            std::unique_ptr<ps::Solver<1, kBcs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx},
                        std::array{inv_dx_squared, inv_dx_squared});
                };

                solver = std::make_unique<ps::Solver<1, kBcs>>(
                    std::array{lenx}, std::array{nx},
                    ps::FiniteDifference2Approximation<float>{});
            }

            // TODO: Test for FD4

            solver->Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            float residuum = residuum_calculation_strategy();
            CHECK_MESSAGE(ResiduumBelowThreshold(residuum, std::array{nx}),
                          "Residuum is ", residuum);
        }

        // TODO: Test for other BCs
    }

    SUBCASE("2D") {
        std::array<float, nx * ny> phi{};
        std::array<float, nx * ny> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic") {
            constexpr std::array<ps::BCPair, 2> kBcs = {
                {{ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic},
                 {ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic}}};

            std::unique_ptr<ps::Solver<2, kBcs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx, ny},
                        std::array{inv_dx_squared, inv_dx_squared,
                                   inv_dy_squared, inv_dy_squared});
                };

                solver = std::make_unique<ps::Solver<2, kBcs>>(
                    std::array{lenx, leny}, std::array{nx, ny},
                    ps::FiniteDifference2Approximation<float>{});
            }

            // TODO: Test for FD4

            solver->Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            float residuum = residuum_calculation_strategy();
            CHECK_MESSAGE(ResiduumBelowThreshold(residuum, std::array{nx, ny}),
                          "Residuum is ", residuum);
        }

        // TODO: Tests for other BCs
    }

    SUBCASE("3D") {
        std::array<float, nx * ny * nz> phi{};
        std::array<float, nx * ny * nz> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic | Periodic | Periodic") {
            constexpr std::array<ps::BCPair, 3> kBcs = {
                {{ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic},
                 {ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic},
                 {ps::BoundaryCondition::kPeriodic,
                  ps::BoundaryCondition::kPeriodic}}};

            std::unique_ptr<ps::Solver<3, kBcs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx, ny, nz},
                        std::array{inv_dx_squared, inv_dx_squared,
                                   inv_dy_squared, inv_dy_squared,
                                   inv_dz_squared, inv_dz_squared});
                };

                solver = std::make_unique<ps::Solver<3, kBcs>>(
                    std::array{lenx, leny, lenz}, std::array{nx, ny, nz},
                    ps::FiniteDifference2Approximation<float>{});
            }

            solver->Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            float residuum = residuum_calculation_strategy();
            CHECK_MESSAGE(
                ResiduumBelowThreshold(residuum, std::array{nx, ny, nz}),
                "Residuum is ", residuum);
        }

        // TODO: Test for other BCs
    }
}
