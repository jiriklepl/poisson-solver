#include <stdexcept>
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/c_binding.h"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/solver.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

namespace {
namespace ps = ::poisson_solver;
namespace psd = ps::detail;
namespace pst = ps::tests;

using ps::BC;
using ps::BCs;
using ps::GridSizes;
using ps::IApproximation;
using pst::TestConfig;

constexpr BC P = BC::Periodic;
constexpr BC D = BC::Dirichlet;
constexpr BC N = BC::Neumann;
constexpr BC DS = BC::DirichletStag;
constexpr BC NS = BC::NeumannStag;

template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class MaxResiduumCalculator {
public:
    explicit MaxResiduumCalculator(GridSizes<Dim> ns)
        : ns_(ns) {}

    auto operator()(std::span<T> phi,
                    std::span<const T> rhs,
                    BCs<Dim> bcs,
                    std::array<T, 2 * Dim> coeffs) const -> T {
        using pst::FlatToMultidimIdx;

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
                    neighbor_sum += phi[FlatIdx(left_idx)] * coeff_left;
                    coeff_center += coeff_left;
                } else {
                    ApplyBorderBC(phi,
                                  midx,
                                  d,
                                  Side::Left,
                                  bcs[d].first,
                                  coeff_left,
                                  neighbor_sum,
                                  coeff_center);
                }

                if (midx[d] < ns_[d] - 1) {
                    auto right_idx = midx;
                    right_idx[d] += 1;
                    neighbor_sum += phi[FlatIdx(right_idx)] * coeff_right;
                    coeff_center += coeff_right;
                } else {
                    ApplyBorderBC(phi,
                                  midx,
                                  d,
                                  Side::Right,
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
    using MultidimIdx = psd::MultidimIdx<Dim>;

    enum class Side { Left, Right };

    GridSizes<Dim> ns_;

    void ApplyBorderBC(std::span<T> phi,
                       MultidimIdx midx,
                       std::size_t dim,
                       Side side,
                       BC bc,
                       T coeff,
                       T &neighbor_sum,
                       T &coeff_center) const {
        switch (bc) {
            case P: {
                auto wrap_idx = midx;
                wrap_idx[dim] = (side == Side::Left) ? (ns_[dim] - 1) : 0;
                neighbor_sum += phi[FlatIdx(wrap_idx)] * coeff;
                coeff_center += coeff;
                break;
            }
            case D: {
                coeff_center += coeff;
                break;
            }
            case N: {
                auto inner_idx = midx;
                inner_idx[dim] = (side == Side::Left) ? 1 : (ns_[dim] - 2);
                neighbor_sum += phi[FlatIdx(inner_idx)] * coeff;
                coeff_center += coeff;
                break;
            }
            case DS: {
                neighbor_sum -= phi[FlatIdx(midx)] * coeff;
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

    auto FlatIdx(MultidimIdx midx) const -> std::size_t {
        using psd::FlatIndex;
        return FlatIndex<Dim>(midx, ns_);
    }
};

auto ToCBoundary(BC cxx_bc) -> BoundaryCondition {
    switch (cxx_bc) {
        case P:
            return Periodic;
        case D:
            return Dirichlet;
        case N:
            return Neumann;
        case DS:
            return DirichletStag;
        case NS:
            return NeumannStag;
        default:
            assert(false && "Unknown boundary condition.");

            // To prevent UB by not returning from a non-void function.
            throw std::runtime_error("We should never get here!");
    }
}

// Runs tests for both the C++ and C APIs
template <std::floating_point T, std::size_t Dim>
void RunApis(std::span<T> phi,
             std::span<const T> rhs,
             GridSizes<Dim> ns,
             std::array<T, Dim> lens,
             BCs<Dim> bcs,
             const IApproximation<T> &approx,
             Approximation c_approx,
             const std::function<T()> &calc_residuum,
             GridSizes<Dim> ngs_phi = {},
             GridSizes<Dim> ngs_rhs = {}) {
    using ps::Solver;
    using pst::ResiduumBelowThreshold;

    SUBCASE("API: C++") {
        Solver<T, Dim> solver(ns, lens, bcs, approx);
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
                    GridSizes<Dim> ns,
                    std::array<T, Dim> lens,
                    MaxResiduumCalculator<T, Dim> mrc) {
    for (const auto &tc : configs) {
        SUBCASE("Boundary: " + tc.Name()) {
            std::vector<T> local_rhs(rhs.begin(), rhs.end());

            tc.rhs_adjust(local_rhs);

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                using ps::FiniteDifference2Approximation;

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
                                FiniteDifference2Approximation<T>{},
                                FD2,
                                calc);
            }

            // TODO: Test for FD4
        }
    }
}

TEST_CASE_TEMPLATE("Poisson Solver Residuum Suite", T, float, double) {
    using pst::MakeNeumannRhsCorrector;
    using pst::MakeNeumannStagRhsCorrector;
    using pst::MakeNopCorrector;
    using pst::MakePPDRhsCorrector;

    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    // Values taken from `PoisFFT` tests
    constexpr std::size_t nx = 21;
    constexpr std::size_t ny = 32;
    constexpr std::size_t nz = 25;
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1;
    constexpr T lenz = lenx / 1.1;

    SUBCASE("1D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<1> ns1d{nx};
        constexpr std::array<T, 1> lens1d = {lenx};
        std::array<T, nx> phi1d{};
        std::array<T, nx> rhs1d{};
        MaxResiduumCalculator<T, 1> mrc1d(ns1d);

        std::vector<TestConfig<T, 1>> configs = {
            {{{{P, P}}}, {nx}},
            {{{{D, D}}}, {(nx + 1)}},
            {{{{N, N}}}, {(nx - 1)}, MakeNeumannRhsCorrector<T, 1>(ns1d)},
            {{{{DS, DS}}}, {nx}},
            {{{{NS, NS}}}, {nx}, MakeNeumannStagRhsCorrector<T>()},
            {{{{NS, DS}}}, {nx}},
            {{{{DS, NS}}}, {nx}},
            {{{{D, N}}}, {nx}},
            {{{{N, D}}}, {nx}}};

        Initialize<T>(phi1d, rhs1d, seed);
        RunTestConfigs<T, 1>(configs, phi1d, rhs1d, ns1d, lens1d, mrc1d);
    }

    SUBCASE("2D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<2> ns2d{nx, ny};
        constexpr std::array<T, 2> lens2d = {lenx, leny};
        std::array<T, nx * ny> phi2d{};
        std::array<T, nx * ny> rhs2d{};
        MaxResiduumCalculator<T, 2> mrc2d(ns2d);

        std::vector<TestConfig<T, 2>> configs = {
            {{{{P, P}, {P, P}}}, {nx, ny}},
            {{{{D, D}, {D, D}}}, {(nx + 1), (ny + 1)}},
            {{{{N, N}, {N, N}}},
             {(nx - 1), (ny - 1)},
             MakeNeumannRhsCorrector<T, 2>(ns2d)},
            {{{{DS, DS}, {DS, DS}}}, {nx, ny}},
            {{{{NS, NS}, {NS, NS}}},
             {nx, ny},
             MakeNeumannStagRhsCorrector<T>()}};

        Initialize<T>(phi2d, rhs2d, seed);
        RunTestConfigs<T, 2>(configs, phi2d, rhs2d, ns2d, lens2d, mrc2d);
    }

    SUBCASE("3D") {
        using pst::Initialize;
        using pst::TestConfig;

        constexpr GridSizes<3> ns3d{nx, ny, nz};
        constexpr std::array<T, 3> lens3d = {lenx, leny, lenz};
        std::array<T, nx * ny * nz> phi3d{};
        std::array<T, nx * ny * nz> rhs3d{};
        MaxResiduumCalculator<T, 3> mrc3d(ns3d);

        std::vector<TestConfig<T, 3>> configs = {
            {{{{P, P}, {P, P}, {P, P}}}, {nx, ny, nz}},
            {{{{D, D}, {D, D}, {D, D}}}, {(nx + 1), (ny + 1), (nz + 1)}},
            {{{{N, N}, {N, N}, {N, N}}},
             {(nx - 1), (ny - 1), (nz - 1)},
             MakeNeumannRhsCorrector<T, 3>(ns3d)},
            {{{{DS, DS}, {DS, DS}, {DS, DS}}}, {nx, ny, nz}},
            {{{{NS, NS}, {NS, NS}, {NS, NS}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {NS, NS}}}, {{nx, ny, nz}}},
            {{{{P, P}, {P, P}, {DS, DS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {DS, NS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {NS, DS}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {D, N}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{P, P}, {P, P}, {N, D}}},
             {nx, ny, nz},
             MakePPDRhsCorrector<T>()},
            {{{{NS, NS}, {P, P}, {P, P}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {NS, NS}, {P, P}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{P, P}, {NS, NS}, {NS, NS}}},
             {nx, ny, nz},
             MakeNeumannStagRhsCorrector<T>()},
            {{{{DS, DS}, {DS, DS}, {P, P}}}, {nx, ny, nz}},
            {{{{DS, DS}, {DS, DS}, {NS, NS}}}, {nx, ny, nz}},
            {{{{DS, DS}, {DS, DS}, {NS, DS}}}, {nx, ny, nz}},
            {{{{D, D}, {D, D}, {N, D}}}, {nx + 1, ny + 1, nz}},
            {{{{NS, DS}, {NS, NS}, {NS, NS}}}, {nx, ny, nz}},
            {{{{N, D}, {N, N}, {N, N}}}, {nx, ny - 1, nz - 1}}};

        Initialize<T>(phi3d, rhs3d, seed);
        RunTestConfigs<T, 3>(configs, phi3d, rhs3d, ns3d, lens3d, mrc3d);
    }
}
}  // namespace

auto main(int argc, char *argv[]) -> int {
    doctest::Context context(argc, argv);
    int res = context.run();

    return res;
}
