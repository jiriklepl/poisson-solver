#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/axis_policies.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/runners.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <heffte.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cuComplex.h>

namespace poisson_solver {
// TODO: Works only for all bcs == Periodic
template <std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class Solver {
public:
    Solver(
        std::array<std::size_t, Dim> ns, std::array<double, Dim> lens,
        std::array<BCPair, Dim> bcs,
        const IApproximation<double>& approx = SpectralApproximation<double>{},
        dim3 cuda_block_size = DefaultBlockSize())
        : ns_(ns),
          bcs_(bcs),
          cnt_(std::reduce(ns_.begin(), ns_.end(), std::size_t{1},
                           std::multiplies<>{})),
          denoms_(MakeDenoms(ns_, std::make_index_sequence<Dim>{})),
          norm_factor_(CalculateNormFactor(ns_, bcs_)),
          block_size_(cuda_block_size),
          grid_size_(CalculateGridSize(ns, block_size_)) {
        if (!std::ranges::none_of(bcs_, [](BCPair bc) {
                return bc.first != BoundaryCondition::kPeriodic ||
                       bc.second != BoundaryCondition::kPeriodic;
            })) {
            throw std::invalid_argument(
                "Invalid boundary condition combination.");
        }

        // TODO: We can calculate them directly on the GPU, no?
        CalculateEigenvalues(denoms_[0], lens[0], ns_[0], bcs_[0], approx);
        if constexpr (Dim >= 2) {
            CalculateEigenvalues(denoms_[1], lens[1], ns_[1], bcs_[1], approx);
        }
        if constexpr (Dim >= 3) {
            CalculateEigenvalues(denoms_[2], lens[2], ns_[2], bcs_[2], approx);
        }

        cu_rhs_ = make_cuda_unique<double>(cnt_);
        cu_phi_ = make_cuda_unique<double>(cnt_);
        cu_cmplx_workspace_ = make_cuda_unique<cuDoubleComplex>(cnt_);
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denoms_[i] = make_cuda_unique<double>(ns_[i]);

            // Eigenvals don't change between iterations.
            // Thus can be copied immediately.
            CUCH(cudaMemcpy(cu_denoms_[i].get(), denoms_[i].data(),
                            ns_[i] * sizeof(double),
                            cudaMemcpyHostToDevice));
        }

        int nx = ns_[0];
        int ny = (Dim >= 2) ? ns_[1] : 1;
        int nz = (Dim >= 3) ? ns_[2] : 1;
        // `box3d` has column-major ordering!!!
        heffte::box3d<> grid_box({0, 0, 0}, {nz - 1, ny - 1, nx - 1});
        heffte::plan_options options =
            heffte::default_options<heffte::backend::cufft>();
        fft_plan_ = std::make_unique<heffte::fft3d<heffte::backend::cufft>>(
            grid_box, grid_box, MPI_COMM_SELF, options);
    }

    // TODO: Define moves, for now they're deleted
    // (either create plan unique wrapper, similar to device ptrs, and then
    // default them, or manually define the moves)
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver operator=(Solver&&) = delete;

    void Execute(std::span<double> phi, std::span<const double> rhs) {
        if (phi.size() != cnt_ || rhs.size() != cnt_) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }

        CUCH(cudaMemcpy(cu_rhs_.get(), rhs.data(),
                        cnt_ * sizeof(double),
                        cudaMemcpyHostToDevice));
        RunInitCmplxWorkspace(cu_cmplx_workspace_.get(), cu_rhs_.get(), ns_,
                              grid_size_, block_size_);
        CUCH(cudaGetLastError());
        fft_plan_->forward(cu_cmplx_workspace_.get(), cu_cmplx_workspace_.get(),
                           heffte::scale::none);

        // TODO: Do this only once in the ctor
        std::array<const double*, Dim> cu_denom_ptrs;
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denom_ptrs[i] = cu_denoms_[i].get();
        }
        RunDivideCmplxWorkspaceByDenoms(cu_cmplx_workspace_.get(), ns_,
                                        cu_denom_ptrs, grid_size_, block_size_);
        CUCH(cudaGetLastError());
        fft_plan_->backward(cu_cmplx_workspace_.get(),
                            cu_cmplx_workspace_.get(), heffte::scale::none);
        RunNormalization(cu_phi_.get(), cu_cmplx_workspace_.get(), ns_,
                         norm_factor_, grid_size_, block_size_);
        CUCH(cudaGetLastError());
        CUCH(cudaMemcpy(phi.data(), cu_phi_.get(),
                        cnt_ * sizeof(double),
                        cudaMemcpyDeviceToHost));
    }

    std::size_t GetTotalGridSize() { return cnt_; }

private:
    // Grid
    const std::array<std::size_t, Dim> ns_;
    const std::array<BCPair, Dim> bcs_;
    const std::size_t cnt_;

    // Physics and math
    std::array<std::vector<double>, Dim> denoms_;
    std::array<CudaUniquePtr<double>, Dim> cu_denoms_;
    const double norm_factor_;

    // FFT
    std::unique_ptr<heffte::fft3d<heffte::backend::cufft>> fft_plan_;
    CudaUniquePtr<double> cu_rhs_;
    CudaUniquePtr<double> cu_phi_;
    CudaUniquePtr<cuDoubleComplex> cu_cmplx_workspace_;

    // Device settings
    const dim3 block_size_;
    const dim3 grid_size_;

    template <std::size_t... Is>
    static std::array<std::vector<double>, Dim> MakeDenoms(
        const std::array<std::size_t, Dim>& ns, std::index_sequence<Is...>) {
        return {std::vector<double>(ns[Is])...};
    }

    static double CalculateNormFactor(std::array<std::size_t, Dim> ns,
                                      std::array<BCPair, Dim> bcs) {
        double factor = 1.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            auto policy = GetAxisPolicy<double>(bcs[i]);
            factor *= policy->CalculateNormalizationFactor(ns[i]);
        }
        return factor;
    }

    static constexpr dim3 DefaultBlockSize() {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else if constexpr (Dim == 3) {
            return {8, 8, 4};
        }
    }

    static dim3 CalculateGridSize(const std::array<std::size_t, Dim>& ns,
                                  dim3 block) {
        if constexpr (Dim == 1) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x};
        } else if constexpr (Dim == 2) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y};
        } else if constexpr (Dim == 3) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y,
                    (static_cast<unsigned>(ns[2]) + block.z - 1) / block.z};
        }
    }

    // TODO: Delete T parameter as soon as `Solver` is templated by a float type
    template <std::floating_point T>
    static void CalculateEigenvalues(std::vector<T>& eigenvals, T len,
                                     std::size_t n, BCPair bc,
                                     const IApproximation<T>& approx) {
        auto policy = GetAxisPolicy<T>(bc);
        eigenvals.resize(n);

        const T grid_dx = policy->GetGridDx(len, n);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;

        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(policy->GetApproximationInput(i, n, dkx)) *
                           inv_grid_dx_squared;
        }
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
