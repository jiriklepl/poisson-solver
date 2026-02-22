#include "poisson-solver/kernel.cuh"

namespace poisson_solver {
__global__ void InitCmplxWorkspace(cufftComplex *cmplx_workspace,
                                   const cufftReal *rhs, std::size_t nx,
                                   std::size_t ny) {
    const std::size_t stride_x = gridDim.x * blockDim.x;
    const std::size_t stride_y = gridDim.y * blockDim.y;

    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < nx; i += stride_x) {
        std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
        for (; j < ny; j += stride_y) {
            const std::size_t idx = i * ny + j;
            cmplx_workspace[idx] = make_cuComplex(rhs[idx], 0);
        }
    }
}

__global__ void DivideCmplxWorkspaceByDenoms(cufftComplex *cmplx_workspace,
                                             std::size_t nx, std::size_t ny,
                                             const cufftReal *denomx,
                                             const cufftReal *denomy) {
    const std::size_t stride_x = gridDim.x * blockDim.x;
    const std::size_t stride_y = gridDim.y * blockDim.y;

    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < nx; i += stride_x) {
        std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
        for (; j < ny; j += stride_y) {
            const std::size_t idx = i * ny + j;
            if (i == 0 && j == 0) {
                cmplx_workspace[idx] = make_cuComplex(0, 0);
                continue;
            }
            const cufftReal denom = (denomx[i] + denomy[j]);
            cmplx_workspace[idx].x /= denom;
            cmplx_workspace[idx].y /= denom;
        }
    }
}

__global__ void NormalizeCmplxWorkspaceToPhi(
    cufftReal *phi, const cufftComplex *cmplx_workspace, std::size_t nx,
    std::size_t ny,  float norm_factor) {
    const std::size_t stride_x = gridDim.x * blockDim.x;
    const std::size_t stride_y = gridDim.y * blockDim.y;

    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < nx; i += stride_x) {
        std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
        for (; j < ny; j += stride_y) {
            const std::size_t idx = i * ny + j;
            phi[idx] = cmplx_workspace[idx].x / norm_factor;
        }
    }
}

void RunInitCmplxWorkspace(cufftComplex *cmplx_workspace, const cufftReal *rhs,
                           std::size_t nx, std::size_t ny, dim3 grid_size_2d,
                           dim3 block_size_2d) {
    InitCmplxWorkspace<<<grid_size_2d, block_size_2d>>>(cmplx_workspace, rhs,
                                                        nx, ny);
}

void RunDivideCmplxWorkspaceByDenoms(cufftComplex *cmplx_workspace,
                                     std::size_t nx, std::size_t ny,
                                     const cufftReal *denomx,
                                     const cufftReal *denomy, dim3 grid_size_2d,
                                     dim3 block_size_2d) {
    DivideCmplxWorkspaceByDenoms<<<grid_size_2d, block_size_2d>>>(
        cmplx_workspace, nx, ny, denomx, denomy);
}

void RunNormalization(cufftReal *phi, const cufftComplex *cmplx_workspace,
                      std::size_t nx, std::size_t ny, float norm_factor,
                      dim3 grid_size_2d, dim3 block_size_2d) {
    NormalizeCmplxWorkspaceToPhi<<<grid_size_2d, block_size_2d>>>(
        phi, cmplx_workspace, nx, ny, norm_factor);
}
}  // namespace poisson_solver
