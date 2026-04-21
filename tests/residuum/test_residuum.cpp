#define DOCTEST_CONFIG_IMPLEMENT
#include "initialization.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/fft_policies.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/solver.hpp"
#include "doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

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
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
T GetMaxResiduumFullPeriodic(std::span<T> phi,
                             std::span<T> rhs,
                             std::array<std::size_t, Dim> ns,
                             std::array<T, 2 * Dim> coeffs) {
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

    T max_residuum = 0.0;
    const T coeff_center = std::accumulate(coeffs.begin(), coeffs.end(), 0.0);

    for (std::size_t i = 0; i < size; ++i) {
        T neighbor_sum = 0.0;
        auto multidim_idx = flat_to_multidim_idx(i);
        auto neighbor_idxs = get_neighbor_idxs(multidim_idx);

        for (std::size_t d = 0; d < Dim; ++d) {
            const auto &[left_neigh_idx, right_neigh_idx] = neighbor_idxs[d];
            std::size_t flat_left_idx = multidim_to_flat_idx(left_neigh_idx);
            std::size_t flat_right_idx = multidim_to_flat_idx(right_neigh_idx);
            neighbor_sum += phi[flat_left_idx] * coeffs[2 * d];
            neighbor_sum += phi[flat_right_idx] * coeffs[2 * d + 1];
        }

        neighbor_sum -= rhs[i];
        const T local_residuum = std::abs(coeff_center * phi[i] - neighbor_sum);
        max_residuum = std::max(max_residuum, local_residuum);
    }

    return max_residuum;
}

template <std::floating_point T, std::size_t Dim>
ResiduumResult<T> ResiduumBelowThreshold(T residuum,
                                         std::array<std::size_t, Dim> ns) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    T max_residuum =
        std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>()) *
        fudge_factor * std::numeric_limits<T>::epsilon();
    bool success = residuum < max_residuum;
    return {.success = success, .max_residuum = max_residuum};
}

BoundaryCondition ToCBoundary(BC cxx_bc) {
    using CxxBC = ps::BoundaryCondition;

    constexpr auto cxxP = CxxBC::kPeriodic;
    constexpr auto cxxD = CxxBC::kDirichlet;
    constexpr auto cxxN = CxxBC::kNeumann;
    constexpr auto cxxDS = CxxBC::kDirichletStag;
    constexpr auto cxxNS = CxxBC::kNeumannStag;

    switch (cxx_bc) {
        case cxxP:
            return kPeriodic;
        case cxxD:
            return kDirichlet;
        case cxxN:
            return kNeumann;
        case cxxDS:
            return kDirichletStag;
        case cxxNS:
            return kNeumannStag;
        default:
            throw std::invalid_argument("Unknown boundary condition.");
    }
}

// Runs tests for both the C++ and C APIs
template <std::floating_point T, std::size_t Dim>
void RunApis(std::span<T> phi,
             std::span<T> rhs,
             std::array<std::size_t, Dim> ns,
             std::array<T, Dim> lens,
             std::array<ps::BCPair, Dim> bcs,
             const ps::IApproximation<T> &approx,
             Approximation c_approx,
             std::function<T()> calc_residuum) {
    SUBCASE("API: C++") {
        ps::Solver<T, Dim, ps::VkFFTPolicy> solver(ns, lens, bcs, approx);
        solver.Execute(phi, rhs);

        T residuum = calc_residuum();
        auto result = ResiduumBelowThreshold<T>(residuum, ns);
        CHECK_MESSAGE(result.success,
                      "Residuum is ",
                      residuum,
                      " and max allowed is ",
                      result.max_residuum);
    }

    SUBCASE("API: C Binding") {
        PoissonSolver solver;
        std::array<int, Dim> c_ns;
        std::array<T, Dim> c_lens;
        std::array<BoundaryCondition, 2 * Dim> c_bcs;

        for (std::size_t i = 0; i < Dim; ++i) {
            c_ns[i] = static_cast<int>(ns[i]);
            c_lens[i] = lens[i];
            c_bcs[2 * i] = ToCBoundary(bcs[i].first);
            c_bcs[2 * i + 1] = ToCBoundary(bcs[i].second);
        }

        if constexpr (std::is_same_v<T, double>) {
            PoissonSolverNew(&solver,
                             Dim,
                             c_ns.data(),
                             c_lens.data(),
                             c_bcs.data(),
                             c_approx);
            PoissonSolverExecute(&solver, phi.data(), rhs.data());
        } else {
            PoissonSolverNewFloat(&solver,
                                  Dim,
                                  c_ns.data(),
                                  c_lens.data(),
                                  c_bcs.data(),
                                  c_approx);
            PoissonSolverExecuteFloat(&solver, phi.data(), rhs.data());
        }

        T residuum = calc_residuum();
        auto result = ResiduumBelowThreshold<T>(residuum, ns);
        CHECK_MESSAGE(result.success,
                      "Residuum is ",
                      residuum,
                      " and max allowed is ",
                      result.max_residuum);

        if constexpr (std::is_same_v<T, double>) {
            PoissonSolverFinalize(&solver);
        } else {
            PoissonSolverFinalizeFloat(&solver);
        }
    }
}

}  // namespace

int main(int argc, char *argv[]) {
    doctest::Context context(argc, argv);
    int res = context.run();

    return res;
}

TEST_CASE_TEMPLATE("Poisson Solver Residuum Suite", T, float, double) {
    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    constexpr std::size_t nx = 21, ny = 32, nz = 25;
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1, lenz = lenx / 1.1;

    constexpr T dx = lenx / nx, dy = leny / ny, dz = lenz / nz;
    constexpr T inv_dx_squared = 1.0 / (dx * dx);
    constexpr T inv_dy_squared = 1.0 / (dy * dy);
    constexpr T inv_dz_squared = 1.0 / (dz * dz);

    SUBCASE("1D") {
        std::array<T, nx> phi{};
        std::array<T, nx> rhs{};
        Initialize<T>(phi, rhs, seed);
        std::array<std::size_t, 1> ns{nx};

        SUBCASE("Boundary: Periodic | Periodic") {
            std::array<ps::BCPair, 1> bcs = {{{P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return GetMaxResiduumFullPeriodic<T, 1>(
                        phi, rhs, ns, {inv_dx_squared, inv_dx_squared});
                };
                RunApis<T, 1>(phi,
                              rhs,
                              ns,
                              {lenx},
                              bcs,
                              ps::FiniteDifference2Approximation<T>{},
                              kFD2,
                              calc);
            }

            // TODO: Test for FD4
        }

        // TODO: Test for other BCs
    }

    SUBCASE("2D") {
        std::array<T, nx * ny> phi{};
        std::array<T, nx * ny> rhs{};
        Initialize<T>(phi, rhs, seed);
        std::array<std::size_t, 2> ns{nx, ny};

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic") {
            std::array<ps::BCPair, 2> bcs = {{{P, P}, {P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return GetMaxResiduumFullPeriodic<T, 2>(phi,
                                                            rhs,
                                                            ns,
                                                            {inv_dx_squared,
                                                             inv_dx_squared,
                                                             inv_dy_squared,
                                                             inv_dy_squared});
                };
                RunApis<T, 2>(phi,
                              rhs,
                              ns,
                              {lenx, leny},
                              bcs,
                              ps::FiniteDifference2Approximation<T>{},
                              kFD2,
                              calc);
            }
            // TODO: Test for FD4
        }

        // TODO: Test for other BCs
    }

    SUBCASE("3D") {
        std::array<T, nx * ny * nz> phi{};
        std::array<T, nx * ny * nz> rhs{};
        Initialize<T>(phi, rhs, seed);
        std::array<std::size_t, 3> ns{nx, ny, nz};

        SUBCASE(
            "Boundary: Periodic | Periodic | Periodic | Periodic | Periodic | "
            "Periodic") {
            std::array<ps::BCPair, 3> bcs = {{{P, P}, {P, P}, {P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return GetMaxResiduumFullPeriodic<T, 3>(phi,
                                                            rhs,
                                                            ns,
                                                            {inv_dx_squared,
                                                             inv_dx_squared,
                                                             inv_dy_squared,
                                                             inv_dy_squared,
                                                             inv_dz_squared,
                                                             inv_dz_squared});
                };
                RunApis<T, 3>(phi,
                              rhs,
                              ns,
                              {lenx, leny, lenz},
                              bcs,
                              ps::FiniteDifference2Approximation<T>{},
                              kFD2,
                              calc);
            }

            // TODO: Test for FD4
        }

        // TODO: Test for other BCs
    }
}
