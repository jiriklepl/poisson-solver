#include "poisson-solver/kernel.cuh"

namespace poisson_solver {
__global__ void Solve2DFullPeriodic(cufftComplex *cmplx_workspace,
                                    const cufftReal *denomx,
                                    const cufftReal *denomy, std::size_t nx,
                                    std::size_t ny) {
    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    const std::size_t idx = i * ny + j;

    if (i == 0 && j == 0) {
        cmplx_workspace[idx] = make_cuComplex(0, 0);
        return;
    }

    const cufftReal denom = (denomx[i] + denomy[j]);
    cmplx_workspace[idx].x /= denom;
    cmplx_workspace[idx].y /= denom;
}

void RunSolver2DFullPeriodic(cufftComplex *cmplx_workspace,
                             const cufftReal *denomx, const cufftReal *denomy,
                             std::size_t nx, std::size_t ny,
                             dim3 block_size_2d) {
    const std::size_t grid_size_x =
        (nx + block_size_2d.x - 1) / block_size_2d.x;
    const std::size_t grid_size_y =
        (ny + block_size_2d.y - 1) / block_size_2d.y;

    const dim3 grid_size{static_cast<unsigned>(grid_size_x),
                         static_cast<unsigned>(grid_size_y)};

    Solve2DFullPeriodic<<<grid_size, block_size_2d>>>(cmplx_workspace, denomx,
                                                      denomy, nx, ny);
}
}  // namespace poisson_solver
