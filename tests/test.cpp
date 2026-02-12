#include "poisson-solver/approximations.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/solver.hpp"
#include "poisson-solver/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>

void InitializePhiAndRhs(poisson_solver::Workspace2D<float> &phi,
                         poisson_solver::Workspace2D<float> &rhs,
                         std::size_t n_x, std::size_t n_y) {
    std::random_device device;
    std::mt19937 rng(device());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float rhs_sum = 0.0f;
    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            phi(i, j) = dist(rng);

            float rhs_value = dist(rng);
            rhs(i, j) = rhs_value;

            rhs_sum += rhs_value;
        }
    }

    float avg = rhs_sum / (n_x * n_y);

    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            rhs(i, j) -= avg;
        }
    }
}

float GetMaxResiduum2DFullPeriodic(poisson_solver::Workspace2D<float> &phi,
                                   poisson_solver::Workspace2D<float> &rhs,
                                   std::size_t n_x, std::size_t n_y,
                                   float coeff_west, float coeff_east,
                                   float coeff_south, float coeff_north) {
    float max_residuum = 0.0f;
    const float coeff_center =
        coeff_west + coeff_east + coeff_south + coeff_north;

    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            float neighbor_sum = 0.0f;

            if (i > 0)
                neighbor_sum += phi(i - 1, j) * coeff_west;
            else
                neighbor_sum += phi(n_x - 1, j) * coeff_west;

            if (i < n_x - 1)
                neighbor_sum += phi(i + 1, j) * coeff_east;
            else
                neighbor_sum += phi(0, j) * coeff_east;

            if (j > 0)
                neighbor_sum += phi(i, j - 1) * coeff_south;
            else
                neighbor_sum += phi(i, n_y - 1) * coeff_south;

            if (j < n_y - 1)
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

int main(int argc, char *argv[]) {
    std::cout << "Creating a solver\n";

    float len_x = 2 * std::numbers::pi_v<float>;
    float len_y = len_x * 1.1f;

    std::size_t n_x = 21;
    std::size_t n_y = 32;

    float dx = len_x / n_x;
    float dy = len_y / n_y;

    poisson_solver::Workspace2D<float> phi{n_x, n_y};
    poisson_solver::Workspace2D<float> rhs{n_x, n_y};

    InitializePhiAndRhs(phi, rhs, n_x, n_y);

    poisson_solver::Solver2D solver(
        len_x, len_y, n_x, n_y,
        {{poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic},
         {poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic}},
        poisson_solver::FiniteDifference2Approximation<float>{});

    try {
        solver.Execute(phi, rhs);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const float inv_dx_squared = 1.0f / (dx * dx);
    const float inv_dy_squared = 1.0f / (dy * dy);

    float residuum = GetMaxResiduum2DFullPeriodic(
        phi, rhs, n_x, n_y, inv_dx_squared, inv_dx_squared, inv_dy_squared,
        inv_dy_squared);

    if (poisson_solver::ResiduumBelowThreshold(residuum, n_x, n_y)) {
        std::cout << "Residuum FD2 OK: " << residuum << "\n";
    } else {
        std::cout << "Residuum FD2 FAIL: " << residuum << "\n";
        return 1;
    }

    return 0;
}
