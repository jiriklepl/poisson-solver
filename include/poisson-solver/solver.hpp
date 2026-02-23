#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/helpers.hpp"
#include "poisson-solver/runners.hpp"
#include "poisson-solver/workspace.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <cufft.h>
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
template <std::size_t Dim, std::array<BCPair, Dim> BCs>
    requires((Dim <= 3 && Dim >= 1) &&
             std::ranges::none_of(BCs,
                                  [](BoundaryCondition bc) {
                                      return bc != BoundaryCondition::kPeriodic;
                                  }))
class Solver {
public:
    Solver(std::array<float, Dim> lens, std::array<std::size_t, Dim> ns,
           const IApproximation<float> &approx = SpectralApproximation<float>{},
           dim3 cuda_block_size = DefaultBlockSize())
        : ns_(ns),
          cnt_(std::reduce(ns_.begin(), ns_.end(), std::size_t{1},
                           std::multiplies<>{})),
          denoms_(MakeDenoms(ns_, std::make_index_sequence<Dim>{})),
          norm_factor_(
              CalculateNormFactor(ns_, std::make_index_sequence<Dim>{})),
          block_size_(cuda_block_size),
          grid_size_(CalculateGridSize(ns, block_size_)) {
        // TODO: We can calculate them directly on the GPU, no?
        CalculateEigenvalues<BCs[0]>(lens[0], ns_[0], approx, denoms_[0]);
        if constexpr (Dim >= 2) {
            CalculateEigenvalues<BCs[1]>(lens[1], ns_[1], approx, denoms_[1]);
        }
        if constexpr (Dim >= 3) {
            CalculateEigenvalues<BCs[2]>(lens[2], ns_[2], approx, denoms_[2]);
        }

        cu_rhs_ = make_cuda_unique<cufftReal>(cnt_);
        cu_phi_ = make_cuda_unique<cufftReal>(cnt_);
        cu_cmplx_workspace_ = make_cuda_unique<cufftComplex>(cnt_);
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denoms_[i] = make_cuda_unique<cufftReal>(ns_[i]);

            // Eigenvals don't change between iterations.
            // Thus can be copied immediately.
            CUCH(cudaMemcpy(cu_denoms_[i].get(), denoms_[i].data(),
                            ns_[i] * sizeof(cufftReal),
                            cudaMemcpyHostToDevice));
        }

        if constexpr (Dim == 1) {
            CUFFTCH(cufftPlan1d(&plan_, ns_[0], CUFFT_C2C, 1));
        } else if (Dim == 2) {
            CUFFTCH(cufftPlan2d(&plan_, ns_[0], ns_[1], CUFFT_C2C));
        } else if (Dim == 3) {
            CUFFTCH(cufftPlan3d(&plan_, ns_[0], ns_[1], ns_[2], CUFFT_C2C));
        }
    }

    // TODO: Define moves, for now they're deleted
    // (either create plan unique wrapper, similar to device ptrs, and then
    // default them, or manually define the moves)
    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;
    Solver(Solver &&) = delete;
    Solver operator=(Solver &&) = delete;
    virtual ~Solver() { CUFFTCH(cufftDestroy(plan_)); }

    void Execute(std::span<float> phi, std::span<float> rhs) {
        // TODO: Don't check this in release builds
        if (phi.size() != cnt_ || rhs.size() != cnt_) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }

        // TODO: No need for this copy if we are guaranteed the rhs didn't
        // change
        CUCH(cudaMemcpy(cu_rhs_.get(), rhs.data(), cnt_ * sizeof(cufftReal),
                        cudaMemcpyHostToDevice));
        RunInitCmplxWorkspace(cu_cmplx_workspace_.get(), cu_rhs_.get(), ns_,
                              grid_size_, block_size_);

        CUFFTCH(cufftExecC2C(plan_, cu_cmplx_workspace_.get(),
                             cu_cmplx_workspace_.get(), CUFFT_FORWARD));
        std::array<const cufftReal *, Dim> cu_denom_ptrs;
        for (std::size_t i = 0; i < Dim; ++i) {
            cu_denom_ptrs[i] = cu_denoms_[i].get();
        }
        RunDivideCmplxWorkspaceByDenoms(cu_cmplx_workspace_.get(), ns_,
                                        cu_denom_ptrs, grid_size_, block_size_);
        CUFFTCH(cufftExecC2C(plan_, cu_cmplx_workspace_.get(),
                             cu_cmplx_workspace_.get(), CUFFT_INVERSE));

        RunNormalization(cu_phi_.get(), cu_cmplx_workspace_.get(), ns_,
                         norm_factor_, grid_size_, block_size_);
        CUCH(cudaMemcpy(phi.data(), cu_phi_.get(), cnt_ * sizeof(cufftReal),
                        cudaMemcpyDeviceToHost));
    }

private:
    // Grid sizes
    const std::array<std::size_t, Dim> ns_;
    const std::size_t cnt_;

    // Physics and math
    std::array<std::vector<float>, Dim> denoms_;
    std::array<CudaUniquePtr<cufftReal>, Dim> cu_denoms_;
    const float norm_factor_;

    // FFT
    // Cufft defined forward/backward during execution.
    // Initialization requires only the domain size, thus only one plan needed.
    cufftHandle plan_;
    CudaUniquePtr<cufftReal> cu_rhs_;
    CudaUniquePtr<cufftReal> cu_phi_;
    CudaUniquePtr<cufftComplex> cu_cmplx_workspace_;

    // Device settings
    const dim3 block_size_;
    const dim3 grid_size_;

    template <std::size_t... Is>
    static std::array<std::vector<float>, Dim> MakeDenoms(
        const std::array<std::size_t, Dim> &ns, std::index_sequence<Is...>) {
        return {std::vector<float>(ns[Is])...};
    }

    template <std::size_t... Is>
    static float CalculateNormFactor(const std::array<std::size_t, Dim> &ns,
                                     std::index_sequence<Is...>) {
        return (
            AxisPolicy<BCs[Is], float>::CalculateNormalizationFactor(ns[Is]) *
            ...);
    }

    static constexpr dim3 DefaultBlockSize() {
        if constexpr (Dim == 1) {
            return {256};
        } else if (Dim == 2) {
            return {16, 16};
        } else if (Dim == 3) {
            return {8, 8, 4};
        }
    }

    static dim3 CalculateGridSize(const std::array<std::size_t, Dim> &ns,
                                  dim3 block) {
        if constexpr (Dim == 1) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x};
        } else if (Dim == 2) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y};
        } else if (Dim == 3) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y,
                    (static_cast<unsigned>(ns[2]) + block.z - 1) / block.z};
        }
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
