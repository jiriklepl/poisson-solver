#include "poisson-solver/solver.hpp"

namespace poisson_solver {
Solver2D::Solver2D(float lenx, float leny, std::size_t nx, std::size_t ny,
                   BoundaryConditions<2> bcs,
                   const IApproximation<float> &approx, dim3 cuda_block_size_2d)
    : nx_(nx),
      ny_(ny),
      cnt_(nx_ * ny_),
      bcs_(bcs),
      denomx_(nx_),
      denomy_(ny_),
      block_size_2d_{cuda_block_size_2d},
      grid_size_2d_{(static_cast<unsigned>(nx_) * block_size_2d_.x - 1) /
                        block_size_2d_.x,
                    (static_cast<unsigned>(ny_) * block_size_2d_.y - 1) /
                        block_size_2d_.y} {
    if (std::ranges::any_of(bcs.Flatten(), [](BoundaryCondition bc) {
            return bc != BoundaryCondition::kPeriodic;
        })) {
        throw std::invalid_argument(
            "Error: Only periodic BCs are allowed currently!\n");
    }

    norm_factor_ = CalculateNormalizationFactor(nx_, bcs_.x) *
                   CalculateNormalizationFactor(ny_, bcs_.y);

    // TODO: We can calculate them directly on the GPU, no?
    CalculateEigenvalues(lenx, nx_, bcs_.x, approx, denomx_);
    CalculateEigenvalues(leny, ny_, bcs_.y, approx, denomy_);

    CUCH(cudaMalloc(&cu_rhs_, cnt_ * sizeof(cufftReal)));
    CUCH(cudaMalloc(&cu_phi_, cnt_ * sizeof(cufftReal)));
    CUCH(cudaMalloc(&cu_cmplx_workspace_, cnt_ * sizeof(cufftComplex)));
    CUCH(cudaMalloc(&cu_denomx_, nx_ * sizeof(cufftReal)));
    CUCH(cudaMalloc(&cu_denomy_, ny_ * sizeof(cufftReal)));

    // Eigenvals don't change between iterations.
    // Thus can be copied immediately.
    CUCH(cudaMemcpy(cu_denomx_, denomx_.data(), nx_ * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(cu_denomy_, denomy_.data(), ny_ * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));

    CUFFTCH(cufftPlan2d(&forward_plan_, nx_, ny_, CUFFT_C2C));
    CUFFTCH(cufftPlan2d(&backward_plan_, nx_, ny_, CUFFT_C2C));
}

void Solver2D::Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs) {
    if (phi.Size() != cnt_ || rhs.Size() != cnt_) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }
    // TODO: No need for this copy if we are guaranteed the rhs didn't change
    CUCH(cudaMemcpy(cu_rhs_, rhs.Ptr(), cnt_ * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));

    RunInitCmplxWorkspace(cu_cmplx_workspace_, cu_rhs_, nx_, ny_, grid_size_2d_,
                          block_size_2d_);

    CUFFTCH(cufftExecC2C(forward_plan_, cu_cmplx_workspace_,
                         cu_cmplx_workspace_, CUFFT_FORWARD));

    RunDivideCmplxWorkspaceByDenoms(cu_cmplx_workspace_, nx_, ny_, cu_denomx_,
                                    cu_denomy_, grid_size_2d_, block_size_2d_);

    CUFFTCH(cufftExecC2C(backward_plan_, cu_cmplx_workspace_,
                         cu_cmplx_workspace_, CUFFT_INVERSE));

    RunNormalization(cu_phi_, cu_cmplx_workspace_, nx_, ny_, norm_factor_,
                     grid_size_2d_, block_size_2d_);

    CUCH(cudaMemcpy(phi.Ptr(), cu_phi_, cnt_ * sizeof(cufftReal),
                    cudaMemcpyDeviceToHost));
}
}  // namespace poisson_solver
