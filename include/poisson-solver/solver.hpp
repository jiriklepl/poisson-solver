#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"

#include <array>
#include <vector>

class Solver2D {
    // Grid and geometry
    const float len_x;
    const float len_y;

    const unsigned n_x;
    const unsigned n_y;
    const unsigned cnt;

    // Physics and math
    const std::array<BoundaryCondition, 4> bcs;

    std::vector<float> denom_x;
    std::vector<float> denom_y;

    const Approximation approx;

    float norm_factor;

public:
    Solver2D(float len_x, float len_y, unsigned n_x, unsigned n_y,
             std::array<BoundaryCondition, 4> bcs,
             Approximation approx = Approximation::kSpectral)
        : len_x(len_x),
          len_y(len_y),
          n_x(n_x),
          n_y(n_y),
          cnt(n_x * n_y),
          bcs(bcs),
          approx(approx) {
        denom_x = std::vector<float>();
        denom_y = std::vector<float>();
        norm_factor = 1;
    }
};

#endif  // SOLVER_HPP
