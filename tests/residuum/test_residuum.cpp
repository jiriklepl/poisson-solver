#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "initialization.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/solver.hpp"
#include "vendor/doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <numeric>
#include <random>

namespace ps = poisson_solver;
using BC = ps::BoundaryCondition;
using namespace tests;

namespace {
constexpr auto P = BC::kPeriodic;
constexpr auto D = BC::kDirichlet;
constexpr auto N = BC::kNeumann;
constexpr auto DS = BC::kDirichletStag;
constexpr auto NS = BC::kNeumannStag;

template <std::floating_point T>
struct ResiduumResult {
    bool success;
    T max_residuum;
};

// Coeffs order: [west, east, south, north, bottom, top]
template <std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
double GetMaxResiduumFullPeriodic(std::span<double> phi, std::span<double> rhs,
                                  std::array<std::size_t, Dim> ns,
                                  std::array<double, 2 * Dim> coeffs) {
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

    double max_residuum = 0.0;
    const double coeff_center =
        std::accumulate(coeffs.begin(), coeffs.end(), 0.0);

    for (std::size_t i = 0; i < size; ++i) {
        double neighbor_sum = 0.0;
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
        const double local_residuum =
            std::abs(coeff_center * phi[i] - neighbor_sum);
        max_residuum = std::max(max_residuum, local_residuum);
    }

    return max_residuum;
}

template <std::floating_point T, std::size_t Dim>
ResiduumResult<T> ResiduumBelowThreshold(T residuum,
                                         std::array<std::size_t, Dim> ns) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    T max_residuum = std::accumulate(ns.begin(), ns.end(), std::size_t{1},
                                     std::multiplies<>()) *
                     fudge_factor * std::numeric_limits<T>::epsilon();
    bool success = residuum < max_residuum;
    return {.success = success, .max_residuum = max_residuum};
}
}  // namespace

TEST_CASE("Poisson Solver Residuum Suite") {
    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    constexpr std::size_t nx = 21, ny = 32, nz = 25;
    constexpr double lenx = 2 * std::numbers::pi_v<double>;
    constexpr double leny = lenx * 1.1, lenz = lenx / 1.1;
    constexpr double dx = lenx / nx, dy = leny / ny, dz = lenz / nz;
    constexpr double inv_dx_squared = 1.0 / (dx * dx);
    constexpr double inv_dy_squared = 1.0 / (dy * dy);
    constexpr double inv_dz_squared = 1.0 / (dz * dz);
    std::function<double()> residuum_calculation_strategy;

    SUBCASE("1D") {
        std::array<double, nx> phi{};
        std::array<double, nx> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE("Boundary: Periodic | Periodic") {
            constexpr std::array<ps::BCPair, 1> kBCs = {{{P, P}}};

            std::unique_ptr<ps::Solver<1, kBCs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx},
                        std::array{inv_dx_squared, inv_dx_squared});
                };

                solver = std::make_unique<ps::Solver<1, kBCs>>(
                    std::array{lenx}, std::array{nx},
                    ps::FiniteDifference2Approximation<double>{});
            }

            // TODO: Test for FD4

            solver->Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            double residuum = residuum_calculation_strategy();
            ResiduumResult<double> result =
                ResiduumBelowThreshold(residuum, std::array{nx});
            CHECK_MESSAGE(result.success, "Residuum is ", residuum,
                          " and max allowed is ", result.max_residuum);
        }

        // TODO: Test for other BCs
    }

    SUBCASE("2D") {
        std::array<double, nx * ny> phi{};
        std::array<double, nx * ny> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic") {
            constexpr std::array<ps::BCPair, 2> kBCs = {{{P, P}, {P, P}}};

            std::unique_ptr<ps::Solver<2, kBCs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx, ny},
                        std::array{inv_dx_squared, inv_dx_squared,
                                   inv_dy_squared, inv_dy_squared});
                };

                solver = std::make_unique<ps::Solver<2, kBCs>>(
                    std::array{lenx, leny}, std::array{nx, ny},
                    ps::FiniteDifference2Approximation<double>{});
            }

            // TODO: Test for FD4

            solver->Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            double residuum = residuum_calculation_strategy();
            ResiduumResult<double> result =
                ResiduumBelowThreshold(residuum, std::array{nx, ny});
            CHECK_MESSAGE(result.success, "Residuum is ", residuum,
                          " and max allowed is ", result.max_residuum);
        }

        // TODO: Tests for other BCs
    }

    SUBCASE("3D") {
        std::array<double, nx * ny * nz> phi{};
        std::array<double, nx * ny * nz> rhs{};
        Initialize(phi, rhs, seed);

        SUBCASE(
            "Boundary: Periodic | Periodic | Periodic | Periodic | Periodic | "
            "Periodic") {
            constexpr std::array<ps::BCPair, 3> kBCs = {
                {{P, P}, {P, P}, {P, P}}};

            std::unique_ptr<ps::Solver<3, kBCs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduumFullPeriodic(
                        phi, rhs, std::array{nx, ny, nz},
                        std::array{inv_dx_squared, inv_dx_squared,
                                   inv_dy_squared, inv_dy_squared,
                                   inv_dz_squared, inv_dz_squared});
                };

                SUBCASE("C++") {
                    solver = std::make_unique<ps::Solver<3, kBCs>>(
                        std::array{lenx, leny, lenz}, std::array{nx, ny, nz},
                        ps::FiniteDifference2Approximation<double>{});

                    solver->Execute({phi.data(), phi.size()},
                                    {rhs.data(), rhs.size()});
                    double residuum = residuum_calculation_strategy();
                    ResiduumResult<double> result = ResiduumBelowThreshold(
                        residuum, std::array{nx, ny, nz});
                    CHECK_MESSAGE(result.success, "Residuum is ", residuum,
                                  " and max allowed is ", result.max_residuum);
                }

                SUBCASE("C API") {
                    // From "c_binding.h"
                    constexpr auto P_c = kPeriodic;
                    const int ns[] = {static_cast<int>(nx),
                                      static_cast<int>(ny),
                                      static_cast<int>(nz)};
                    const double lens[] = {lenx, leny, lenz};
                    const BoundaryCondition bcs[] = {P_c, P_c, P_c,
                                                     P_c, P_c, P_c};
                    const Approximation approx = kFD2;

                    PoissonSolver solver;
                    PoissonSolverNew(&solver, 3, ns, lens, bcs, approx);

                    PoissonSolverExecute(&solver, phi.data(), rhs.data());

                    double residuum = residuum_calculation_strategy();
                    ResiduumResult<double> result = ResiduumBelowThreshold(
                        residuum, std::array{nx, ny, nz});
                    CHECK_MESSAGE(result.success, "Residuum is ", residuum,
                                  " and max allowed is ", result.max_residuum);

                    PoissonSolverFinalize(&solver);
                }
            }
        }

        // TODO: Test for other BCs
    }
}
