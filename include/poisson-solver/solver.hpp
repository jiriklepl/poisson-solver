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
        const IApproximation<float> &approx = SpectralApproximation<float>{},
        dim3 cuda_block_size_2d = {16, 16});

    // TODO: Define moves, for now they're deleted
    // (either create plan unique wrapper, similar to device ptrs, and then
    // default them, or manually define the moves)
    Solver2D(const Solver2D &) = delete;
    Solver2D &operator=(const Solver2D &) = delete;
    Solver2D(Solver2D &&) = delete;
    Solver2D operator=(Solver2D &&) = delete;
    virtual ~Solver2D();

    void Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs);

private:
    // Grid sizes
    const std::size_t nx_;
    const std::size_t ny_;
    const std::size_t cnt_;

    // Physics and math
    const BoundaryConditions<2> bcs_;
    std::vector<float> denomx_;
    std::vector<float> denomy_;
    CudaUniquePtr<cufftReal> cu_denomx_;
    CudaUniquePtr<cufftReal> cu_denomy_;
    float norm_factor_;

    // FFT
    // Cufft defined forward/backward during execution.
    // Initialization requires only the domain size, thus only one plan needed.
    cufftHandle plan_;
    CudaUniquePtr<cufftReal> cu_rhs_;
    CudaUniquePtr<cufftReal> cu_phi_;
    CudaUniquePtr<cufftComplex> cu_cmplx_workspace_;

    // Device settings
    const dim3 block_size_2d_;
    const dim3 grid_size_2d_;
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
