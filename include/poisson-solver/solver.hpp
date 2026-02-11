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
public:
    Solver2D(
        float lenx, float leny, std::size_t nx, std::size_t ny,
        BoundaryConditions<2> bcs,
        const IApproximation<float> &approx = SpectralApproximation<float>{});

    void Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs);

private:
    const std::size_t nx_;
    const std::size_t ny_;
    const std::size_t cnt_;

    // Physics and math
    const BoundaryConditions<2> bcs_;

    std::vector<float> denomx_;
    std::vector<float> denomy_;

    cufftReal *cu_denomx_;
    cufftReal *cu_denomy_;

    float norm_factor_;

    // FFT
    cufftHandle forward_plan_;
    cufftHandle backward_plan_;

    Workspace2D<cufftComplex> cmplx_workspace_;
    cufftComplex *cu_cmplx_workspace_;
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
