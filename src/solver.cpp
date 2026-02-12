#include "poisson-solver/solver.hpp"

namespace poisson_solver {
Solver2D::Solver2D(float lenx, float leny, std::size_t nx, std::size_t ny,
                   BoundaryConditions<2> bcs,
                   const IApproximation<float> &approx)
    : nx_(nx),
      ny_(ny),
      cnt_(nx * ny),
      bcs_(bcs),
      denomx_(nx),
      denomy_(ny),
      cmplx_workspace_(nx, ny) {
    if (std::ranges::any_of(bcs.Flatten(), [](BoundaryCondition bc) {
            return bc != BoundaryCondition::kPeriodic;
        })) {
        throw std::invalid_argument(
            "Error: Only periodic BCs are allowed currently!\n");
    }

    norm_factor_ = CalculateNormalizationFactor(nx_, bcs_.x) *
                   CalculateNormalizationFactor(ny_, bcs_.y);

    CalculateEigenvalues(lenx, nx_, bcs_.x, approx, denomx_);
    CalculateEigenvalues(leny, ny_, bcs_.y, approx, denomy_);

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

    constexpr dim3 kBlockSize2D = {16, 16};

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < nx_; ++i) {
        for (std::size_t j = 0; j < ny_; ++j) {
            cmplx_workspace_(i, j) = make_cuComplex(rhs(i, j), 0);
        }
    }

    CUCH(cudaMemcpy(cu_cmplx_workspace_, cmplx_workspace_.Ptr(),
                    cnt_ * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    CUFFTCH(cufftExecC2C(forward_plan_, cu_cmplx_workspace_,
                         cu_cmplx_workspace_, CUFFT_FORWARD));

    RunSolver2DFullPeriodic(cu_cmplx_workspace_, cu_denomx_, cu_denomy_, nx_,
                            ny_, kBlockSize2D);

    // Check kernel launch and execution errors
    CUCH(cudaDeviceSynchronize());

    CUFFTCH(cufftExecC2C(backward_plan_, cu_cmplx_workspace_,
                         cu_cmplx_workspace_, CUFFT_INVERSE));

    CUCH(cudaMemcpy(cmplx_workspace_.Ptr(), cu_cmplx_workspace_,
                    cnt_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < nx_; ++i) {
        for (std::size_t j = 0; j < ny_; ++j) {
            phi(i, j) = cmplx_workspace_(i, j).x / norm_factor_;
        }
    }
}
}  // namespace poisson_solver
