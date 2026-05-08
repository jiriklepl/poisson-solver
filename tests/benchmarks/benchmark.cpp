#include "initialization.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/solver.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace ps = poisson_solver;
using namespace tests;

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr
            << "Usage: warmup_iters measurement_iters grid_side_size_begin "
               "grid_side_size_end grid_side_size_step out_file\n";
        return 1;
    }
    const std::size_t warmup_iters = std::stol(argv[1]);
    const std::size_t measurement_iters = std::stol(argv[2]);
    std::size_t grid_size = std::stol(argv[3]);
    const std::size_t grid_size_end = std::stol(argv[4]);
    const std::size_t grid_size_step = std::stol(argv[5]);
    const std::string out_filename = argv[6];

    std::ofstream out_file(out_filename);
    if (!out_file.is_open()) {
        std::cerr << "Cannot open file \"" << out_filename << "\"" << "\n";
        return 1;
    }

    std::random_device dev;
    unsigned seed = dev();
    constexpr double lenx = 2 * std::numbers::pi_v<double>;
    constexpr double leny = lenx * 1.1, lenz = lenx / 1.1;

    constexpr ps::BCs<3> kBCs = {
        {{ps::BoundaryCondition::kPeriodic, ps::BoundaryCondition::kPeriodic},
         {ps::BoundaryCondition::kPeriodic, ps::BoundaryCondition::kPeriodic},
         {ps::BoundaryCondition::kPeriodic, ps::BoundaryCondition::kPeriodic}}};

    std::vector<std::chrono::steady_clock::duration> times(measurement_iters);
    out_file << "grid_size,index,time_ns" << "\n";

    // Measurements
    for (; grid_size <= grid_size_end; grid_size += grid_size_step) {
        std::vector<double> phi(grid_size * grid_size * grid_size);
        std::vector<double> rhs(grid_size * grid_size * grid_size);
        Initialize(phi, rhs, seed);

        ps::Solver<3, kBCs> solver{
            std::array{grid_size, grid_size, grid_size},
            std::array{lenx, leny, lenz},
            ps::FiniteDifference2Approximation<double>{}};

        // Warmup
        for (std::size_t i = 0; i < warmup_iters; ++i) {
            solver.Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            seed = dev();
            Initialize(phi, rhs, seed);
        }

        for (std::size_t i = 0; i < measurement_iters; ++i) {
            auto start = std::chrono::steady_clock::now();
            solver.Execute({phi.data(), phi.size()}, {rhs.data(), rhs.size()});
            auto end = std::chrono::steady_clock::now();
            times[i] = end - start;
            seed = dev();
            Initialize(phi, rhs, seed);
        }

        for (std::size_t i = 0; i < measurement_iters; ++i) {
            out_file << grid_size << "," << i << "," << times[i].count()
                     << "\n";
        }
    }

    out_file.close();

    return 0;
}
