#include "poisson-solver/solver.hpp"

namespace poisson_solver {
Solver2D::Solver2D(float len_x, float len_y, std::size_t n_x, std::size_t n_y,
                   BoundaryConditions<2> bcs, Approximation approx)
    : len_x(len_x),
      len_y(len_y),
      n_x(n_x),
      n_y(n_y),
      cnt(n_x * n_y),
      bcs(bcs),
      denom_x(n_x),
      denom_y(n_y),
      approx(approx),
      complex_workspace(n_x, n_y) {
    if (std::ranges::any_of(bcs.Flatten(), [](BoundaryCondition bc) {
            return bc != BoundaryCondition::kPeriodic;
        })) {
        throw std::invalid_argument(
            "Error: Only periodic BCs are allowed currently!\n");
    }

    if (approx != Approximation::kSpectral) {
        throw std::invalid_argument(
            "Error: Only spectral approximation is allowed currently!\n");
    }

    norm_factor = CalculateNormalizationFactor(n_x, bcs.x) *
                  CalculateNormalizationFactor(n_y, bcs.y);

    CalculateEigenvalues(len_x, n_x, bcs.x, denom_x);
    CalculateEigenvalues(len_y, n_y, bcs.y, denom_y);

    CUCH(cudaMalloc(&cu_complex_workspace, cnt * sizeof(cufftComplex)));

    CUFFTCH(cufftPlan2d(&forward_plan, n_x, n_y, CUFFT_C2C));
    CUFFTCH(cufftPlan2d(&backward_plan, n_x, n_y, CUFFT_C2C));
}

void Solver2D::Execute(Workspace2D<float> &phi,
                       const Workspace2D<float> &kRHS) {
    if (phi.size() != cnt || kRHS.size() != cnt) {
        throw std::runtime_error("Incorrect size of phi or rhs.");
    }

    constexpr std::size_t kBlockSize = 256;

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            complex_workspace(i, j) = make_cuComplex(kRHS(i, j), 0);
        }
    }

    // Copy workspace to device
    CUCH(cudaMemcpy(cu_complex_workspace, complex_workspace.ptr(),
                    cnt * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // Forward FFT
    CUFFTCH(cufftExecC2C(forward_plan, cu_complex_workspace,
                         cu_complex_workspace, CUFFT_FORWARD));

    // Copy denominators to device
    CUCH(cudaMemcpy(cu_denom_x, denom_x.data(), n_x * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(cu_denom_y, denom_y.data(), n_y * sizeof(cufftReal),
                    cudaMemcpyHostToDevice));

    RunSolver2DFullPeriodic(cu_complex_workspace, denom_x.data(),
                            denom_y.data(), n_x, n_y, kBlockSize);

    // Backward FFT
    CUFFTCH(cufftExecC2C(backward_plan, cu_complex_workspace,
                         cu_complex_workspace, CUFFT_INVERSE));

    // Copy workspace back to host
    CUCH(cudaMemcpy(complex_workspace.ptr(), cu_complex_workspace,
                    cnt * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // TODO - Do this on GPU (this is very small - cuda GRAPHS? Is it even worth
    // it?)
    for (std::size_t i = 0; i < n_x; ++i) {
        for (std::size_t j = 0; j < n_y; ++j) {
            phi(i, j) = cuCrealf(complex_workspace(i, j)) / norm_factor;
        }
    }
}
}  // namespace poisson_solver
