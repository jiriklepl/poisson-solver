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
template <BoundaryConditions<2> BCs>
class Solver2D {
public:
    Solver2D(
        float lenx, float leny, std::size_t nx, std::size_t ny,
        const IApproximation<float> &approx = SpectralApproximation<float>{},
        dim3 cuda_block_size_2d = {16, 16})
        : nx_(nx),
          ny_(ny),
          cnt_(nx_ * ny_),
          denomx_(nx_),
          denomy_(ny_),
          block_size_2d_{cuda_block_size_2d},
          grid_size_2d_{(static_cast<unsigned>(nx_) * block_size_2d_.x - 1) /
                            block_size_2d_.x,
                        (static_cast<unsigned>(ny_) * block_size_2d_.y - 1) /
                            block_size_2d_.y} {
        if constexpr (std::ranges::any_of(
                          BCs.Flatten(), [](BoundaryCondition bc) {
                              return bc != BoundaryCondition::kPeriodic;
                          })) {
            throw std::invalid_argument(
                "Error: Only periodic BCs are allowed currently!\n");
        }

        using XPolicy = AxisPolicy<BCs.x, float>;
        using YPolicy = AxisPolicy<BCs.y, float>;

        norm_factor_ = XPolicy::CalculateNormalizationFactor(nx_) *
                       YPolicy::CalculateNormalizationFactor(ny_);

        // TODO: We can calculate them directly on the GPU, no?
        CalculateEigenvalues<BCs.x>(lenx, nx_, approx, denomx_);
        CalculateEigenvalues<BCs.y>(leny, ny_, approx, denomy_);

        cu_rhs_ = make_cuda_unique<cufftReal>(cnt_);
        cu_phi_ = make_cuda_unique<cufftReal>(cnt_);
        cu_cmplx_workspace_ = make_cuda_unique<cufftComplex>(cnt_);
        cu_denomx_ = make_cuda_unique<cufftReal>(nx_);
        cu_denomy_ = make_cuda_unique<cufftReal>(ny_);

        // Eigenvals don't change between iterations.
        // Thus can be copied immediately.
        CUCH(cudaMemcpy(cu_denomx_.get(), denomx_.data(),
                        nx_ * sizeof(cufftReal), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_denomy_.get(), denomy_.data(),
                        ny_ * sizeof(cufftReal), cudaMemcpyHostToDevice));

        CUFFTCH(cufftPlan2d(&plan_, nx_, ny_, CUFFT_C2C));
    }

    // TODO: Define moves, for now they're deleted
    // (either create plan unique wrapper, similar to device ptrs, and then
    // default them, or manually define the moves)
    Solver2D(const Solver2D &) = delete;
    Solver2D &operator=(const Solver2D &) = delete;
    Solver2D(Solver2D &&) = delete;
    Solver2D operator=(Solver2D &&) = delete;
    virtual ~Solver2D() { CUFFTCH(cufftDestroy(plan_)); }

    void Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs) {
        if (phi.Size() != cnt_ || rhs.Size() != cnt_) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }
        // TODO: No need for this copy if we are guaranteed the rhs didn't
        // change
        CUCH(cudaMemcpy(cu_rhs_.get(), rhs.Ptr(), cnt_ * sizeof(cufftReal),
                        cudaMemcpyHostToDevice));
        RunInitCmplxWorkspace(cu_cmplx_workspace_.get(), cu_rhs_.get(), nx_,
                              ny_, grid_size_2d_, block_size_2d_);

        CUFFTCH(cufftExecC2C(plan_, cu_cmplx_workspace_.get(),
                             cu_cmplx_workspace_.get(), CUFFT_FORWARD));
        RunDivideCmplxWorkspaceByDenoms(cu_cmplx_workspace_.get(), nx_, ny_,
                                        cu_denomx_.get(), cu_denomy_.get(),
                                        grid_size_2d_, block_size_2d_);
        CUFFTCH(cufftExecC2C(plan_, cu_cmplx_workspace_.get(),
                             cu_cmplx_workspace_.get(), CUFFT_INVERSE));

        RunNormalization(cu_phi_.get(), cu_cmplx_workspace_.get(), nx_, ny_,
                         norm_factor_, grid_size_2d_, block_size_2d_);
        CUCH(cudaMemcpy(phi.Ptr(), cu_phi_.get(), cnt_ * sizeof(cufftReal),
                        cudaMemcpyDeviceToHost));
    }

private:
    // Grid sizes
    const std::size_t nx_;
    const std::size_t ny_;
    const std::size_t cnt_;

    // Physics and math
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
