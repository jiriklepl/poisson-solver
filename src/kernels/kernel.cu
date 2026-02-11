#include "poisson-solver/kernel.cuh"

namespace poisson_solver {
__global__ void Solve2DFullPeriodic(cufftComplex *complex_workspace,
                                    const cufftReal *denom_x,
                                    const cufftReal *denom_y, std::size_t n_x,
                                    std::size_t n_y) {
    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= n_x || j >= n_y) return;

    const std::size_t idx = i * n_y + j;

    if (i == 0 && j == 0) {
        complex_workspace[idx] = make_cuComplex(0, 0);
        return;
    }

    cufftReal denom = (denom_x[i] + denom_y[j]);
    complex_workspace[idx].x /= denom;
    complex_workspace[idx].y /= denom;
}

void RunSolver2DFullPeriodic(cufftComplex *complex_workspace,
                             const cufftReal *denom_x, const cufftReal *denom_y,
                             std::size_t n_x, std::size_t n_y,
                             dim3 block_size_2d) {
    const std::size_t grid_size_x =
        (n_x + block_size_2d.x - 1) / block_size_2d.x;
    const std::size_t grid_size_y =
        (n_y + block_size_2d.y - 1) / block_size_2d.y;
    const dim3 grid_size{static_cast<unsigned>(grid_size_x),
                         static_cast<unsigned>(grid_size_y)};

    Solve2DFullPeriodic<<<grid_size, block_size_2d>>>(
        complex_workspace, denom_x, denom_y, n_x, n_y);
}
}  // namespace poisson_solver
