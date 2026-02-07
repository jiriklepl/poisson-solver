#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/kernel.cuh"
#include "poisson-solver/workspace.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace poisson_solver {
// TODO: Works only for all bcs == Periodic
class Solver2D {
private:
    // Grid and geometry
    const float len_x;
    const float len_y;

    const std::size_t n_x;
    const std::size_t n_y;
    const std::size_t cnt;

    // Physics and math
    const BoundaryConditions<2> bcs;

    std::vector<float> denom_x;
    std::vector<float> denom_y;

    cufftReal *cu_denom_x;
    cufftReal *cu_denom_y;

    float norm_factor;

    // FFT
    cufftHandle forward_plan;
    cufftHandle backward_plan;

    Workspace2D<cufftComplex> complex_workspace;
    cufftComplex *cu_complex_workspace;

public:
    Solver2D(float len_x, float len_y, std::size_t n_x, std::size_t n_y,
             BoundaryConditions<2> bcs,
             const IApproximation<float> &approximation =
                 SpectralApproximation<float>{});

    void Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs);
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
