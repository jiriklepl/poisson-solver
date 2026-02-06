#include "poisson-solver/kernel.cuh"

namespace poisson_solver {
__global__ void Solve2DFullPeriodic(cufftComplex *complex_workspace,
                                    const cufftReal *kDenom_x,
                                    const cufftReal *kDenom_y, std::size_t n_x,
                                    std::size_t n_y) {
    std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n_x || y >= n_y) {
        return;
    }

    std::size_t idx = y * n_x + x;

    if (x == 0 && y == 0) {
        complex_workspace[idx] = make_cuComplex(0, 0);
        return;
    }

    cufftReal denom = (kDenom_x[idx] + kDenom_y[idx]);

    complex_workspace[idx].x = cuCrealf(complex_workspace[idx]) / denom;
    complex_workspace[idx].y = cuCimagf(complex_workspace[idx]) / denom;
}

void RunSolver2DFullPeriodic(cufftComplex *complex_workspace,
                             const cufftReal *kDenom_x,
                             const cufftReal *kDenom_y, std::size_t n_x,
                             std::size_t n_y, std::size_t block_size) {
    const std::size_t kTotalPoints = n_x * n_y;
    const std::size_t kGridSize = (kTotalPoints + block_size - 1) / block_size;

    Solve2DFullPeriodic<<<kGridSize, block_size>>>(complex_workspace, kDenom_x,
                                                   kDenom_y, n_x, n_y);
}
}  // namespace poisson_solver
