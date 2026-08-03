#include "test_helpers.hpp"
#include "poisson-solver/aliases.hpp"
#include "poisson-solver/solver.hpp"

#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace poisson_solver;

inline auto ParseBC(const std::string &str) -> BC {
    if (str == "P") return BC::Periodic;
    if (str == "D") return BC::Dirichlet;
    if (str == "N") return BC::Neumann;
    if (str == "DS") return BC::DirichletStag;
    if (str == "NS") return BC::NeumannStag;
    throw std::runtime_error("Unknown BC: " + str);
}

template <std::floating_point T, std::size_t Dim>
void RunBenchmarks(std::size_t dim,
                   bool use_gpu,
                   std::size_t measurement_iters,
                   std::size_t grid_size,
                   std::size_t grid_size_end,
                   std::size_t grid_size_step,
                   std::ofstream &out_file,
                   const std::vector<BC> &parsed_bcs,
                   unsigned seed) {
    constexpr T lenx = 2 * std::numbers::pi_v<T>;
    constexpr T leny = lenx * 1.1;
    constexpr T lenz = lenx / 1.1;
    std::array<T, Dim> lens;
    if constexpr (Dim >= 1)
        lens[0] = lenx;
    if constexpr (Dim >= 2)
        lens[1] = leny;
    if constexpr (Dim >= 3)
        lens[2] = lenz;

    BCs<Dim> bcs;
    for (std::size_t i = 0; i < Dim; ++i) {
        bcs[i].first = parsed_bcs[2 * i];
        bcs[i].second = parsed_bcs[2 * i + 1];
    }

    tests::TestConfig<T, Dim> cfg(bcs, {});
    std::string bc_name = cfg.Name();

    T *d_phi = nullptr;
    T *d_rhs = nullptr;

    if (use_gpu) {
        auto max_grid_size = grid_size_end * grid_size_end * grid_size_end;
        cudaMalloc(&d_phi, max_grid_size * sizeof(T));
        cudaMalloc(&d_rhs, max_grid_size * sizeof(T));
    }

    for (std::size_t size = grid_size; size <= grid_size_end;
         size += grid_size_step) {
        std::cout << "Measuring grid size " << size << "\n";
        GridSizes<Dim> ns;
        std::size_t total_elements = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            ns[i] = size;
            total_elements *= size;
        }

        std::vector<T> h_phi(total_elements);
        std::vector<T> h_rhs(total_elements);

        Solver<T, Dim> solver(ns, lens, bcs);

        for (std::size_t iter = 0; iter < measurement_iters; ++iter) {
            std::cout << "    Iteration " << iter << "\n";
            tests::Initialize<T>(h_phi, h_rhs, seed + iter);

            ExecutionTimes times{};
            if (use_gpu) {
                cudaMemcpy(d_rhs,
                           h_rhs.data(),
                           total_elements * sizeof(T),
                           cudaMemcpyHostToDevice);

                times = solver.ExecuteGPU(
                    std::span<T>{d_phi, total_elements},
                    std::span<const T>{d_rhs, total_elements},
                    GridSizes<Dim>{},
                    GridSizes<Dim>{});
            } else {
                times = solver.Execute(h_phi, h_rhs, GridSizes<Dim>{}, GridSizes<Dim>{});
            }

            out_file << Dim << "," << (sizeof(T) == 4 ? "float" : "double")
                     << "," << (use_gpu ? "GPU" : "CPU") << ","
                     << "\"" << bc_name << "\"," << size << ","
                     << times.copy_time_ms << "," << times.reshape_time_ms
                     << "," << times.fft_time_ms << ","
                     << times.kernel_time_ms << "\n";
        }
    }
    if (use_gpu) {
        cudaFree(d_phi);
        cudaFree(d_rhs);
    }
}

template <std::floating_point T, typename... Args>
void RunBenchmarks(std::size_t dim, Args &&...args) {
    switch (dim) {
        case 1:
            return RunBenchmarks<T, 1>(dim, std::forward<Args>(args)...);
        case 2:
            return RunBenchmarks<T, 2>(dim, std::forward<Args>(args)...);
        case 3:
            return RunBenchmarks<T, 3>(dim, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Dim out of bounds");
    }
}

auto main(int argc, char *argv[]) -> int {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " float_precision solver_dim use_gpu measurement_iters "
                     "grid_side_size_begin grid_side_size_end "
                     "grid_side_size_step out_file BC_1 BC_2 ...\n";
        return 1;
    }
    std::string float_precision = argv[1];
    std::size_t dim = std::atoll(argv[2]);
    std::string use_gpu_str = argv[3];
    bool use_gpu =
        (use_gpu_str == "true" || use_gpu_str == "1" || use_gpu_str == "yes");
    std::size_t measurement_iters = std::atoll(argv[4]);
    std::size_t grid_size = std::atoll(argv[5]);
    std::size_t grid_size_end = std::atoll(argv[6]);
    std::size_t grid_size_step = std::atoll(argv[7]);
    std::string out_filename = argv[8];

    if (static_cast<std::size_t>(argc) != 9 + 2 * dim) {
        std::cerr << "Error: Expected " << 2 * dim
                  << " BC arguments for dimension " << dim << "\n";
        return 1;
    }

    std::vector<BC> parsed_bcs;
    for (int i = 9; i < argc; ++i) {
        parsed_bcs.push_back(ParseBC(argv[i]));
    }

    std::ofstream out_file{out_filename, std::ios::app};
    if (!out_file.is_open()) {
        std::cerr << "Cannot open file \"" << out_filename << "\"" << "\n";
        return 1;
    }

    std::random_device dev;
    unsigned seed = dev();

    if (float_precision == "float") {
        RunBenchmarks<float>(dim,
                             use_gpu,
                             measurement_iters,
                             grid_size,
                             grid_size_end,
                             grid_size_step,
                             out_file,
                             parsed_bcs,
                             seed);
    } else if (float_precision == "double") {
        RunBenchmarks<double>(dim,
                              use_gpu,
                              measurement_iters,
                              grid_size,
                              grid_size_end,
                              grid_size_step,
                              out_file,
                              parsed_bcs,
                              seed);
    } else {
        throw std::runtime_error("Precision is not 'float' or 'double'");
    }

    return 0;
}
