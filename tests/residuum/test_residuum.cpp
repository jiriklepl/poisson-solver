#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "initialization.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/fft_policies.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

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

template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class MaxResiduumCalculator {
public:
    MaxResiduumCalculator(std::array<std::size_t, Dim> ns)
        : ns_(ns) {}

    T operator()(std::span<T> phi,
                 std::span<T> rhs,
                 std::array<ps::BCPair, Dim> bcs,
                 std::array<T, 2 * Dim> coeffs) {
        if (rhs.size() != phi.size()) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }

        const std::size_t size = rhs.size();
        T max_residuum = 0.0;

        for (std::size_t i = 0; i < size; ++i) {
            T neighbor_sum = 0.0;
            T coeff_center = 0.0;
            auto midx = FlatToMultidimIdx(i);

            for (std::size_t d = 0; d < Dim; ++d) {
                const T coeff_left = coeffs[2 * d];
                const T coeff_right = coeffs[2 * d + 1];

                if (midx[d] > 0) {
                    auto left_idx = midx;
                    left_idx[d] -= 1;
                    neighbor_sum +=
                        phi[MultidimToFlatIdx(left_idx)] * coeff_left;
                    coeff_center += coeff_left;
                } else {
                    ApplyBorderBC(phi,
                                  midx,
                                  d,
                                  Side::kLeft,
                                  bcs[d].first,
                                  coeff_left,
                                  neighbor_sum,
                                  coeff_center);
                }

                if (midx[d] < ns_[d] - 1) {
                    auto right_idx = midx;
                    right_idx[d] += 1;
                    neighbor_sum +=
                        phi[MultidimToFlatIdx(right_idx)] * coeff_right;
                    coeff_center += coeff_right;
                } else {
                    ApplyBorderBC(phi,
                                  midx,
                                  d,
                                  Side::kRight,
                                  bcs[d].second,
                                  coeff_right,
                                  neighbor_sum,
                                  coeff_center);
                }
            }

            neighbor_sum -= rhs[i];
            const T local_residuum =
                std::abs(coeff_center * phi[i] - neighbor_sum);
            max_residuum = std::max(max_residuum, local_residuum);
        }
        return max_residuum;
    }

private:
    enum class Side { kLeft, kRight };

    using MultidimIdx = std::array<std::size_t, Dim>;

    std::array<std::size_t, Dim> ns_;

    void ApplyBorderBC(std::span<T> phi,
                       MultidimIdx midx,
                       std::size_t dim,
                       Side side,
                       BC bc,
                       T coeff,
                       T &neighbor_sum,
                       T &coeff_center) {
        switch (bc) {
            case P: {
                auto wrap_idx = midx;
                wrap_idx[dim] = (side == Side::kLeft) ? (ns_[dim] - 1) : 0;
                neighbor_sum += phi[MultidimToFlatIdx(wrap_idx)] * coeff;
                coeff_center += coeff;
                break;
            }
            case D: {
                coeff_center += coeff;
                break;
            }
            case N: {
                auto inner_idx = midx;
                inner_idx[dim] = (side == Side::kLeft) ? 1 : (ns_[dim] - 2);
                neighbor_sum += phi[MultidimToFlatIdx(inner_idx)] * coeff;
                coeff_center += coeff;
                break;
            }
            case DS: {
                neighbor_sum -= phi[MultidimToFlatIdx(midx)] * coeff;
                coeff_center += coeff;
                break;
            }
            case NS: {
                // nop
                break;
            }
            default: {
                throw std::invalid_argument("Unknown boundary condition.");
            }
        }
    }

    std::size_t MultidimToFlatIdx(MultidimIdx idx) const {
        return ps::FlatIndex<Dim>(idx, ns_);
    }

    MultidimIdx FlatToMultidimIdx(std::size_t idx) const {
        MultidimIdx midx{};
        for (std::size_t i = Dim; i > 0; --i) {
            midx[i - 1] = idx % ns_[i - 1];
            idx /= ns_[i - 1];
        }
        return midx;
    };
};

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

    SUBCASE("1D") {
        std::array<T, nx> phi{};
        std::array<T, nx> rhs{};
        Initialize<T>(phi, rhs, seed);
        std::array<std::size_t, 1> ns{nx};
        MaxResiduumCalculator<T, 1> mrc(ns);

        SUBCASE("Boundary: Periodic | Periodic") {
            constexpr T dx = lenx / nx;
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            std::array<ps::BCPair, 1> bcs = {{{P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi, rhs, bcs, {inv_dx_squared, inv_dx_squared});
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

        SUBCASE("Boundary: Dirichlet | Dirichlet") {
            constexpr T dx = lenx / (nx + 1);
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            std::array<ps::BCPair, 1> bcs = {{{D, D}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi, rhs, bcs, {inv_dx_squared, inv_dx_squared});
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
        MaxResiduumCalculator<T, 2> mrc(ns);

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic") {
            constexpr T dx = lenx / nx, dy = leny / ny;
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            constexpr T inv_dy_squared = 1.0 / (dy * dy);
            std::array<ps::BCPair, 2> bcs = {{{P, P}, {P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi,
                               rhs,
                               bcs,
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

        SUBCASE("Boundary: Dirichlet | Dirichlet | Dirichlet | Dirichlet") {
            constexpr T dx = lenx / (nx + 1), dy = leny / (ny + 1);
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            constexpr T inv_dy_squared = 1.0 / (dy * dy);
            std::array<ps::BCPair, 2> bcs = {{{D, D}, {D, D}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi,
                               rhs,
                               bcs,
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
        MaxResiduumCalculator<T, 3> mrc(ns);

        SUBCASE(
            "Boundary: Periodic | Periodic | Periodic | Periodic | Periodic | "
            "Periodic") {
            constexpr T dx = lenx / nx, dy = leny / ny, dz = lenz / nz;
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            constexpr T inv_dy_squared = 1.0 / (dy * dy);
            constexpr T inv_dz_squared = 1.0 / (dz * dz);
            std::array<ps::BCPair, 3> bcs = {{{P, P}, {P, P}, {P, P}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi,
                               rhs,
                               bcs,
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

        SUBCASE(
            "Boundary: Dirichlet | Dirichlet | Dirichlet | Dirichlet | "
            "Dirichlet | Dirichlet") {
            constexpr T dx = lenx / (nx + 1), dy = leny / (ny + 1),
                        dz = lenz / (nz + 1);
            constexpr T inv_dx_squared = 1.0 / (dx * dx);
            constexpr T inv_dy_squared = 1.0 / (dy * dy);
            constexpr T inv_dz_squared = 1.0 / (dz * dz);
            std::array<ps::BCPair, 3> bcs = {{{D, D}, {D, D}, {D, D}}};

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                auto calc = [&]() {
                    return mrc(phi,
                               rhs,
                               bcs,
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
