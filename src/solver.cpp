#include "poisson-solver/solver.hpp"

namespace poisson_solver {
Solver2D::Solver2D(float len_x, float len_y, std::size_t n_x, std::size_t n_y,
                   BoundaryConditions<2> bcs,
                   const IApproximation<float> &approximation)
    : len_x(len_x),
      len_y(len_y),
      n_x(n_x),
      n_y(n_y),
      cnt(n_x * n_y),
      bcs(bcs),
      denom_x(n_x),
      denom_y(n_y),
      complex_workspace(n_x, n_y) {
    if (std::ranges::any_of(bcs.Flatten(), [](BoundaryCondition bc) {
            return bc != BoundaryCondition::kPeriodic;
        })) {
        throw std::invalid_argument(
            "Error: Only periodic BCs are allowed currently!\n");
    }

    norm_factor = CalculateNormalizationFactor(n_x, bcs.x) *
                  CalculateNormalizationFactor(n_y, bcs.y);

    CalculateEigenvalues(len_x, n_x, bcs.x, approximation, denom_x);
    CalculateEigenvalues(len_y, n_y, bcs.y, approximation, denom_y);

    CUCH(cudaMalloc(&cu_complex_workspace, cnt * sizeof(cufftComplex)));
    CUCH(cudaMalloc(&cu_denom_x, n_x * sizeof(cufftReal)));
    CUCH(cudaMalloc(&cu_denom_y, n_y * sizeof(cufftReal)));

    CUFFTCH(cufftPlan2d(&forward_plan, n_x, n_y, CUFFT_C2C));
    CUFFTCH(cufftPlan2d(&backward_plan, n_x, n_y, CUFFT_C2C));
}

void Solver2D::Execute(Workspace2D<float> &phi, const Workspace2D<float> &rhs) {
    if (phi.size() != cnt || rhs.size() != cnt) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }

    constexpr dim3 kBlockSize2D = {16, 16};

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            complex_workspace(i, j) = make_cuComplex(rhs(i, j), 0);
        }
    }

    CUCH(cudaMemcpy(cu_complex_workspace, complex_workspace.ptr(),
                    cnt * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    CUFFTCH(cufftExecC2C(forward_plan, cu_complex_workspace,
                         cu_complex_workspace, CUFFT_FORWARD));

    CUCH(cudaMemcpy(cu_denom_x, denom_x.data(), n_x * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(cu_denom_y, denom_y.data(), n_y * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));

    RunSolver2DFullPeriodic(cu_complex_workspace, cu_denom_x, cu_denom_y, n_x,
                            n_y, kBlockSize2D);

    // Check kernel launch and execution errors
    CUCH(cudaDeviceSynchronize());

    CUFFTCH(cufftExecC2C(backward_plan, cu_complex_workspace,
                         cu_complex_workspace, CUFFT_INVERSE));

    CUCH(cudaMemcpy(complex_workspace.ptr(), cu_complex_workspace,
                    cnt * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            phi(i, j) = cuCrealf(complex_workspace(i, j)) / norm_factor;
        }
    }
}
}  // namespace poisson_solver
