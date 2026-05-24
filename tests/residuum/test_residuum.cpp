#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "initialization.hpp"
#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/solver.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ps = poisson_solver;
namespace psd = ps::detail;
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
struct TestConfig {
    std::string name{};
    ps::BCs<Dim> bcs{};
    std::array<std::size_t, Dim> dx_divisors{};
    std::function<void(std::span<T>)> rhs_adjust = nullptr;
};

template <std::size_t Dim>
psd::MultidimIdx<Dim> FlatToMultidimIdx(std::size_t idx,
                                        ps::GridSizes<Dim> ns) {
    psd::MultidimIdx<Dim> midx{};
    for (std::size_t i = 0; i < Dim; ++i) {
        midx[i] = idx % ns[i];
        idx /= ns[i];
    }
    return midx;
};

template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class MaxResiduumCalculator {
public:
    MaxResiduumCalculator(ps::GridSizes<Dim> ns)
        : ns_(ns) {}

    T operator()(std::span<T> phi,
                 std::span<const T> rhs,
                 ps::BCs<Dim> bcs,
                 std::array<T, 2 * Dim> coeffs) const {
        assert(rhs.size() == phi.size());

        const std::size_t size = rhs.size();
        T max_residuum = 0.0;

        for (std::size_t i = 0; i < size; ++i) {
            T neighbor_sum = 0.0;
            T coeff_center = 0.0;
            auto midx = FlatToMultidimIdx<Dim>(i, ns_);

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

    ps::GridSizes<Dim> ns_;

    void ApplyBorderBC(std::span<T> phi,
                       psd::MultidimIdx<Dim> midx,
                       std::size_t dim,
                       Side side,
                       BC bc,
                       T coeff,
                       T &neighbor_sum,
                       T &coeff_center) const {
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
                assert(false && "Unknown boundary condition.");
            }
        }
    }

    std::size_t MultidimToFlatIdx(psd::MultidimIdx<Dim> midx) const {
        return psd::FlatIndex<Dim>(midx, ns_);
    }
};

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
            assert(false && "Unknown boundary condition.");
    }
}

template <std::floating_point T>
std::function<void(std::span<T>)> MakePPDRhsCorrector() {
    using namespace std::ranges;
    return [](std::span<T> rhs) {
        for_each(rhs, [](T &val) {
            val += static_cast<T>(1.0);
        });
    };
}

template <std::floating_point T, std::size_t Dim>
std::function<void(std::span<T>)> MakeNeumannRhsCorrector(
    ps::GridSizes<Dim> ns) {
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
std::function<void(std::span<T>)> MakeNeumannStagRhsCorrector() {
    return [](std::span<T> rhs) {
        T sum = std::accumulate(rhs.begin(), rhs.end(), T{0.0});
        std::ranges::for_each(rhs, [sum](T &elem) {
            elem -= sum;
        });
    };
}

template <std::floating_point T, std::size_t Dim>
ResiduumResult<T> ResiduumBelowThreshold(T residuum, ps::GridSizes<Dim> ns) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    T max_residuum =
        std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>()) *
        fudge_factor * std::numeric_limits<T>::epsilon();
    bool success = residuum < max_residuum;
    return {.success = success, .max_residuum = max_residuum};
}

// Runs tests for both the C++ and C APIs
template <std::floating_point T, std::size_t Dim>
void RunApis(std::span<T> phi,
             std::span<const T> rhs,
             ps::GridSizes<Dim> ns,
             std::array<T, Dim> lens,
             ps::BCs<Dim> bcs,
             const ps::IApproximation<T> &approx,
             Approximation c_approx,
             const std::function<T()> &calc_residuum,
             ps::GridSizes<Dim> ngs_phi = {},
             ps::GridSizes<Dim> ngs_rhs = {}) {
    SUBCASE("API: C++") {
        ps::Solver<T, Dim> solver(ns, lens, bcs, approx);
        solver.Execute(phi, rhs, ngs_phi, ngs_rhs);

        T residuum = calc_residuum();

        // If residuum is exactly equal to 0.0, something might be off.
        CHECK_MESSAGE(residuum != T{0.0},
                      "Residuum is ",
                      residuum,
                      " which is exactly 0.0 and thats weird.");

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
        std::array<int, Dim> c_ngs_phi;
        std::array<int, Dim> c_ngs_rhs;

        for (std::size_t i = 0; i < Dim; ++i) {
            c_ns[i] = static_cast<int>(ns[i]);
            c_lens[i] = lens[i];
            c_bcs[2 * i] = ToCBoundary(bcs[i].first);
            c_bcs[2 * i + 1] = ToCBoundary(bcs[i].second);
            c_ngs_phi[i] = static_cast<int>(ngs_phi[i]);
            c_ngs_rhs[i] = static_cast<int>(ngs_rhs[i]);
        }

        if constexpr (std::is_same_v<T, double>) {
            PoissonSolverNew(&solver,
                             Dim,
                             c_ns.data(),
                             c_lens.data(),
                             c_bcs.data(),
                             c_approx);
            PoissonSolverExecute(&solver,
                                 phi.data(),
                                 rhs.data(),
                                 c_ngs_phi.data(),
                                 c_ngs_rhs.data());
        } else {
            PoissonSolverNewFloat(&solver,
                                  Dim,
                                  c_ns.data(),
                                  c_lens.data(),
                                  c_bcs.data(),
                                  c_approx);
            PoissonSolverExecuteFloat(&solver,
                                      phi.data(),
                                      rhs.data(),
                                      c_ngs_phi.data(),
                                      c_ngs_rhs.data());
        }

        T residuum = calc_residuum();

        // If residuum is exactly equal to 0.0, something might be off.
        CHECK_MESSAGE(residuum != T{0.0},
                      "Residuum is ",
                      residuum,
                      " which is exactly 0.0 and thats weird.");

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

template <std::floating_point T, std::size_t Dim>
void RunTestConfigs(std::span<const TestConfig<T, Dim>> configs,
                    std::span<T> phi,
                    std::span<const T> rhs,
                    ps::GridSizes<Dim> ns,
                    std::array<T, Dim> lens,
                    MaxResiduumCalculator<T, Dim> mrc) {
    for (const auto &tc : configs) {
        SUBCASE("Boundary: " + tc.name) {
            std::vector<T> local_rhs(rhs.begin(), rhs.end());

            if (tc.rhs_adjust) {
                tc.rhs_adjust(local_rhs);
            }

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                std::array<T, 2 * Dim> coeffs{};
                for (std::size_t d = 0; d < Dim; ++d) {
                    T dx = lens[d] / tc.dx_divisors[d];
                    T inv_dx2 = static_cast<T>(1.0) / (dx * dx);

                    coeffs[2 * d] = inv_dx2;
                    coeffs[2 * d + 1] = inv_dx2;
                }

                auto calc = [&]() {
                    return mrc(phi, local_rhs, tc.bcs, coeffs);
                };

                RunApis<T, Dim>(phi,
                                local_rhs,
                                ns,
                                lens,
                                tc.bcs,
                                ps::FiniteDifference2Approximation<T>{},
                                kFD2,
                                calc);
            }

            // TODO: Test for FD4
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
        constexpr ps::GridSizes<1> ns1d{nx};
        constexpr std::array<T, 1> lens1d = {lenx};
        std::array<T, nx> phi1d{};
        std::array<T, nx> rhs1d{};
        MaxResiduumCalculator<T, 1> mrc1d(ns1d);

        std::vector<TestConfig<T, 1>> configs = {
            {"Periodic | Periodic", {{{P, P}}}, {{nx}}, nullptr},
            {"Dirichlet | Dirichlet", {{{D, D}}}, {{(nx + 1)}}, nullptr},
            {"Neumann | Neumann",
             {{{N, N}}},
             {{(nx - 1)}},
             MakeNeumannRhsCorrector<T, 1>(ns1d)},
            {"DirichletStag | DirichletStag", {{{DS, DS}}}, {{nx}}, nullptr},
            {"NeumannStag | NeumannStag",
             {{{NS, NS}}},
             {{nx}},
             MakeNeumannStagRhsCorrector<T>()},
            {"NeumannStag | DirichletStag", {{{NS, DS}}}, {{nx}}, nullptr},
            {"DirichletStag | NeumannStag", {{{DS, NS}}}, {{nx}}, nullptr},
            {"Neumann | Dirichlet", {{{D, N}}}, {{nx}}, nullptr},
            {"Dirichlet | Neumann", {{{N, D}}}, {{nx}}, nullptr}};

        Initialize<T>(phi1d, rhs1d, seed);
        RunTestConfigs<T, 1>(configs, phi1d, rhs1d, ns1d, lens1d, mrc1d);
    }

    SUBCASE("2D") {
        constexpr ps::GridSizes<2> ns2d{nx, ny};
        constexpr std::array<T, 2> lens2d = {lenx, leny};
        std::array<T, nx * ny> phi2d{};
        std::array<T, nx * ny> rhs2d{};
        MaxResiduumCalculator<T, 2> mrc2d(ns2d);

        std::vector<TestConfig<T, 2>> configs = {
            {"Periodic | Periodic | Periodic | Periodic",
             {{{P, P}, {P, P}}},
             {{nx, ny}},
             nullptr},
            {"Dirichlet | Dirichlet | Dirichlet | Dirichlet",
             {{{D, D}, {D, D}}},
             {{(nx + 1), (ny + 1)}},
             nullptr},
            {"Neumann | Neumann | Neumann | Neumann",
             {{{N, N}, {N, N}}},
             {{(nx - 1), (ny - 1)}},
             MakeNeumannRhsCorrector<T, 2>(ns2d)},
            {"DirichletStag | DirichletStag | DirichletStag | DirichletStag",
             {{{DS, DS}, {DS, DS}}},
             {{nx, ny}},
             nullptr},
            {"NeumannStag | NeumannStag | NeumannStag | NeumannStag",
             {{{NS, NS}, {NS, NS}}},
             {{nx, ny}},
             MakeNeumannStagRhsCorrector<T>()}};

        Initialize<T>(phi2d, rhs2d, seed);
        RunTestConfigs<T, 2>(configs, phi2d, rhs2d, ns2d, lens2d, mrc2d);
    }

    SUBCASE("3D") {
        constexpr ps::GridSizes<3> ns3d{nx, ny, nz};
        constexpr std::array<T, 3> lens3d = {lenx, leny, lenz};
        std::array<T, nx * ny * nz> phi3d{};
        std::array<T, nx * ny * nz> rhs3d{};
        MaxResiduumCalculator<T, 3> mrc3d(ns3d);

        std::vector<TestConfig<T, 3>> configs = {
            {"Periodic | Periodic | Periodic | Periodic | Periodic | Periodic",
             {{{P, P}, {P, P}, {P, P}}},
             {{nx, ny, nz}},
             nullptr},
            {"Dirichlet | Dirichlet | Dirichlet | Dirichlet | Dirichlet | "
             "Dirichlet",
             {{{D, D}, {D, D}, {D, D}}},
             {{(nx + 1), (ny + 1), (nz + 1)}},
             nullptr},
            {"Neumann | Neumann | Neumann | Neumann | Neumann | Neumann",
             {{{N, N}, {N, N}, {N, N}}},
             {{(nx - 1), (ny - 1), (nz - 1)}},
             MakeNeumannRhsCorrector<T, 3>(ns3d)},
            {"DirichletStag | DirichletStag | DirichletStag | DirichletStag | "
             "DirichletStag | "
             "DirichletStag",
             {{{DS, DS}, {DS, DS}, {DS, DS}}},
             {{nx, ny, nz}},
             nullptr},
            {"NeumannStag | NeumannStag | NeumannStag | NeumannStag | "
             "NeumannStag | "
             "NeumannStag",
             {{{NS, NS}, {NS, NS}, {NS, NS}}},
             {{nx, ny, nz}},
             MakeNeumannStagRhsCorrector<T>()},
            {"Periodic | Periodic | Periodic | Periodic | NeumannStag | "
             "NeumannStag",
             {{{P, P}, {P, P}, {NS, NS}}},
             {{nx, ny, nz}},
             nullptr},
            {"Periodic | Periodic | Periodic | Periodic | DirichletStag | "
             "DirichletStag",
             {{{P, P}, {P, P}, {DS, DS}}},
             {{nx, ny, nz}},
             MakePPDRhsCorrector<T>()},
            {"Periodic | Periodic | Periodic | Periodic | DirichletStag | "
             "NeumannStag",
             {{{P, P}, {P, P}, {DS, NS}}},
             {{nx, ny, nz}},
             MakePPDRhsCorrector<T>()},
            {"Periodic | Periodic | Periodic | Periodic | NeumannStag | "
             "DirichletStag",
             {{{P, P}, {P, P}, {NS, DS}}},
             {{nx, ny, nz}},
             MakePPDRhsCorrector<T>()},
            {"Periodic | Periodic | Periodic | Periodic | Dirichlet | "
             "Neumann",
             {{{P, P}, {P, P}, {D, N}}},
             {{nx, ny, nz}},
             MakePPDRhsCorrector<T>()},
            {"Periodic | Periodic | Periodic | Periodic | Neumann | "
             "Dirichlet",
             {{{P, P}, {P, P}, {N, D}}},
             {{nx, ny, nz}},
             MakePPDRhsCorrector<T>()},
            {"NeumannStag | NeumannStag | Periodic | Periodic | Periodic | "
             "Periodic",
             {{{NS, NS}, {P, P}, {P, P}}},
             {{nx, ny, nz}},
             MakeNeumannStagRhsCorrector<T>()},
            {"Periodic | Periodic | NeumannStag | NeumannStag | Periodic | "
             "Periodic",
             {{{P, P}, {NS, NS}, {P, P}}},
             {{nx, ny, nz}},
             MakeNeumannStagRhsCorrector<T>()},
            {"Periodic | Periodic | NeumannStag | NeumannStag | NeumannStag | "
             "NeumannStag",
             {{{P, P}, {NS, NS}, {NS, NS}}},
             {{nx, ny, nz}},
             MakeNeumannStagRhsCorrector<T>()},
            {"DirichletStag | DirichletStag | DirichletStag | DirichletStag | "
             "Periodic | Periodic",
             {{{DS, DS}, {DS, DS}, {P, P}}},
             {{nx, ny, nz}},
             nullptr},
            {"DirichletStag | DirichletStag | DirichletStag | DirichletStag | "
             "NeumannStag | NeumannStag",
             {{{DS, DS}, {DS, DS}, {NS, NS}}},
             {{nx, ny, nz}},
             nullptr},
            {"DirichletStag | DirichletStag | DirichletStag | DirichletStag | "
             "NeumannStag | DirichletStag",
             {{{DS, DS}, {DS, DS}, {NS, DS}}},
             {{nx, ny, nz}},
             nullptr},
            {"Dirichlet | Dirichlet | Dirichlet | Dirichlet | Neumann | "
             "Dirichlet",
             {{{D, D}, {D, D}, {N, D}}},
             {{nx + 1, ny + 1, nz}},
             nullptr}};

        Initialize<T>(phi3d, rhs3d, seed);
        RunTestConfigs<T, 3>(configs, phi3d, rhs3d, ns3d, lens3d, mrc3d);
    }
}
