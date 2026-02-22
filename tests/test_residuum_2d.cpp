#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/solver.hpp"
#include "poisson-solver/workspace.hpp"

#include <cmath>
#include <functional>
#include <numbers>
#include <random>

namespace {
void Initialize2D(poisson_solver::Workspace2D<float> &phi,
                  poisson_solver::Workspace2D<float> &rhs, std::size_t nx,
                  std::size_t ny, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float rhs_sum = 0.0f;
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            phi(i, j) = dist(rng);
            float rhs_val = dist(rng);
            rhs(i, j) = rhs_val;
            rhs_sum += rhs_val;
        }
    }

    float avg = rhs_sum / (nx * ny);
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            rhs(i, j) -= avg;
        }
    }
}

float GetMaxResiduum2DFullPeriodic(poisson_solver::Workspace2D<float> &phi,
                                   poisson_solver::Workspace2D<float> &rhs,
                                   std::size_t nx, std::size_t ny,
                                   float coeff_west, float coeff_east,
                                   float coeff_south, float coeff_north) {
    float max_residuum = 0.0f;
    const float coeff_center =
        coeff_west + coeff_east + coeff_south + coeff_north;

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            float neighbor_sum = 0.0f;

            if (i > 0)
                neighbor_sum += phi(i - 1, j) * coeff_west;
            else
                neighbor_sum += phi(nx - 1, j) * coeff_west;

            if (i < nx - 1)
                neighbor_sum += phi(i + 1, j) * coeff_east;
            else
                neighbor_sum += phi(0, j) * coeff_east;

            if (j > 0)
                neighbor_sum += phi(i, j - 1) * coeff_south;
            else
                neighbor_sum += phi(i, ny - 1) * coeff_south;

            if (j < ny - 1)
                neighbor_sum += phi(i, j + 1) * coeff_north;
            else
                neighbor_sum += phi(i, 0) * coeff_north;

            neighbor_sum -= rhs(i, j);
            const float local_residuum =
                std::abs(coeff_center * phi(i, j) - neighbor_sum);
            max_residuum = std::max(max_residuum, local_residuum);
        }
    }

    return max_residuum;
}

template <std::floating_point T>
bool ResiduumBelowThreshold(T residuum, std::size_t nx, std::size_t ny) {
    constexpr T fudge_factor = static_cast<T>(100.0);
    return residuum <
           nx * ny * fudge_factor * std::numeric_limits<T>::epsilon();
}
}  // namespace

TEST_CASE("Poisson Solver Residuum Suite") {
    std::random_device dev;
    unsigned seed = dev();
    INFO("Initialized with seed ", seed);

    SUBCASE("2D") {
        constexpr std::size_t nx = 21;
        constexpr std::size_t ny = 32;
        constexpr float lenx = 2 * std::numbers::pi_v<float>;
        constexpr float leny = lenx * 1.1f;
        constexpr float dx = lenx / nx;
        constexpr float dy = leny / ny;
        constexpr float inv_dx_squared = 1.0f / (dx * dx);
        constexpr float inv_dy_squared = 1.0f / (dy * dy);

        poisson_solver::Workspace2D<float> phi{nx, ny};
        poisson_solver::Workspace2D<float> rhs{nx, ny};

        Initialize2D(phi, rhs, nx, ny, seed);

        // Variables to be configured by inner subcases
        std::function<float()> residuum_calculation_strategy;

        SUBCASE("Boundary: Periodic | Periodic | Periodic | Periodic") {
            constexpr poisson_solver::BoundaryConditions<2> kBcs = {
                {poisson_solver::BoundaryCondition::kPeriodic,
                 poisson_solver::BoundaryCondition::kPeriodic},
                {poisson_solver::BoundaryCondition::kPeriodic,
                 poisson_solver::BoundaryCondition::kPeriodic}};

            std::unique_ptr<poisson_solver::Solver2D<kBcs>> solver;

            // TODO: Test for Spectral approx

            SUBCASE("Approximation: FD2") {
                residuum_calculation_strategy = [&]() {
                    return GetMaxResiduum2DFullPeriodic(
                        phi, rhs, nx, ny, inv_dx_squared, inv_dx_squared,
                        inv_dy_squared, inv_dy_squared);
                };

                solver = std::make_unique<poisson_solver::Solver2D<kBcs>>(
                    lenx, leny, nx, ny,
                    poisson_solver::FiniteDifference2Approximation<float>{});
            }

            // TODO: Test for FD4

            solver->Execute(phi, rhs);
            float residuum = residuum_calculation_strategy();
            CHECK_MESSAGE(ResiduumBelowThreshold(residuum, nx, ny),
                          "Residuum is ", residuum);
        }
    }

    // TODO: Tests for other BCs
    // TODO: Tests of other dimensions
}
