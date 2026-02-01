#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/kernel.cuh"
#include "poisson-solver/workspace.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cufft.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

namespace poisson_solver {
// TODO: Incomplete, doesn't work for all possible scenarios
class Solver2D {
private:
    // Grid and geometry
    const float len_x;
    const float len_y;

    const std::size_t n_x;
    const std::size_t n_y;
    const std::size_t cnt;

    // Physics and math
    const std::array<BoundaryCondition, 4> bcs;

    std::vector<float> denom_x;
    std::vector<float> denom_y;

    const Approximation approx;

    float norm_factor;

    // FFT
    cufftHandle forward_plan;
    cufftHandle backward_plan;

    Workspace2D<cufftComplex> complex_workspace;
    cufftComplex *cu_complex_workspace;

public:
    Solver2D(float len_x, float len_y, std::size_t n_x, std::size_t n_y,
             std::array<BoundaryCondition, 4> bcs,
             Approximation approx = Approximation::kSpectral)
        : len_x(len_x),
          len_y(len_y),
          n_x(n_x),
          n_y(n_y),
          cnt(n_x * n_y),
          bcs(bcs),
          denom_x(n_x),
          denom_y(n_y),
          approx(approx),
          norm_factor(cnt),
          complex_workspace(n_x, n_y) {
        if (std::ranges::any_of(bcs, [](BoundaryCondition bc) {
                return bc != BoundaryCondition::kNeumann;
            })) {
            std::cerr << "Error: Only Neumann BCs are allowed currently!\n";
            return;
        }

        if (approx != Approximation::kSpectral) {
            std::cerr
                << "Error: Only spectral approximation is allowed currently!\n";
            return;
        }

        CalculateEigenvalues(len_x, n_x, denom_x);
        CalculateEigenvalues(len_y, n_y, denom_y);
    }

    void Execute();
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
