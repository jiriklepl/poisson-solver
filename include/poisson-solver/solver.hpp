#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/axis_strategies.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/fft_policies.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/runners.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
// TODO: Works only for all bcs == Periodic
template <std::floating_point T,
          std::size_t Dim,
          template <std::floating_point> typename FFTPolicy>
    requires(Dim <= 3 && Dim >= 1)
class Solver {
public:
    Solver(std::array<std::size_t, Dim> ns,
           std::array<T, Dim> lens,
           std::array<BCPair, Dim> bcs,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = DefaultBlockSize())
        : ns_(ns),
          bcs_(bcs),
          cnt_(std::reduce(
              ns_.begin(), ns_.end(), std::size_t{1}, std::multiplies<>{})),
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

        cu_rhs_ = make_cuda_unique<T>(cnt_);
        cu_phi_ = make_cuda_unique<T>(cnt_);
        cu_cmplx_workspace_ = make_cuda_unique<CuComplexTrait_t<T>>(cnt_);
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denoms_[i] = make_cuda_unique<T>(ns_[i]);

            // Eigenvals don't change between iterations.
            // Thus can be copied immediately.
            CUCH(cudaMemcpy(cu_denoms_[i].get(),
                            denoms_[i].data(),
                            ns_[i] * sizeof(T),
                            cudaMemcpyHostToDevice));
        }

        FFTPolicy<T>::Init(fft_plan_, ns_);
    }

    void Execute(std::span<T> phi, std::span<const T> rhs) {
        if (phi.size() != cnt_ || rhs.size() != cnt_) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }

        CUCH(cudaMemcpy(cu_rhs_.get(),
                        rhs.data(),
                        cnt_ * sizeof(T),
                        cudaMemcpyHostToDevice));
        RunInitCmplxWorkspace(cu_cmplx_workspace_.get(),
                              cu_rhs_.get(),
                              ns_,
                              grid_size_,
                              block_size_);
        CUCH(cudaGetLastError());
        FFTPolicy<T>::Forward(cu_cmplx_workspace_.get(), fft_plan_);

        // TODO: Do this only once in the ctor
        std::array<const T *, Dim> cu_denom_ptrs;
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denom_ptrs[i] = cu_denoms_[i].get();
        }
        RunDivideCmplxWorkspaceByDenoms(cu_cmplx_workspace_.get(),
                                        ns_,
                                        cu_denom_ptrs,
                                        grid_size_,
                                        block_size_);
        CUCH(cudaGetLastError());

        FFTPolicy<T>::Backward(cu_cmplx_workspace_.get(), fft_plan_);
        RunNormalization(cu_phi_.get(),
                         cu_cmplx_workspace_.get(),
                         ns_,
                         norm_factor_,
                         grid_size_,
                         block_size_);
        CUCH(cudaGetLastError());
        CUCH(cudaMemcpy(phi.data(),
                        cu_phi_.get(),
                        cnt_ * sizeof(T),
                        cudaMemcpyDeviceToHost));
    }

    std::size_t GetTotalGridSize() const {
        return cnt_;
    }

private:
    // Grid
    std::array<std::size_t, Dim> ns_;
    std::array<BCPair, Dim> bcs_;
    std::size_t cnt_;

    // Physics and math
    std::array<std::vector<T>, Dim> denoms_;
    std::array<CudaUniquePtr<T>, Dim> cu_denoms_;
    T norm_factor_;

    // FFT
    typename FFTPolicy<T>::Plan fft_plan_;
    CudaUniquePtr<T> cu_rhs_;
    CudaUniquePtr<T> cu_phi_;
    CudaUniquePtr<CuComplexTrait_t<T>> cu_cmplx_workspace_;

    // Device settings
    dim3 block_size_;
    dim3 grid_size_;

    template <std::size_t... Is>
    static std::array<std::vector<T>, Dim> MakeDenoms(
        const std::array<std::size_t, Dim> &ns, std::index_sequence<Is...>) {
        return {std::vector<T>(ns[Is])...};
    }

    static T CalculateNormFactor(std::array<std::size_t, Dim> ns,
                                 std::array<BCPair, Dim> bcs) {
        T factor = 1.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            auto strategy = GetAxisStrategy<T>(bcs[i]);
            factor *= strategy->CalculateNormalizationFactor(ns[i]);
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

    static dim3 CalculateGridSize(const std::array<std::size_t, Dim> &ns,
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

    static void CalculateEigenvalues(std::vector<T> &eigenvals,
                                     T len,
                                     std::size_t n,
                                     BCPair bc,
                                     const IApproximation<T> &approx) {
        auto strategy = GetAxisStrategy<T>(bc);
        eigenvals.resize(n);

        const T grid_dx = strategy->GetGridDx(len, n);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;

        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(strategy->GetApproximationInput(i, n, dkx)) *
                           inv_grid_dx_squared;
        }
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
