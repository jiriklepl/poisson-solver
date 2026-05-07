#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/axis_strategies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/runners/common_real.hpp"
#include "poisson-solver/detail/runners/full_dirichlet.hpp"
#include "poisson-solver/detail/runners/full_neumann.hpp"
#include "poisson-solver/detail/runners/full_periodic.hpp"
#include "poisson-solver/fft_policies.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poisson_solver {
// TODO: Doesn't work for all 3D cases
template <std::floating_point T,
          std::size_t Dim,
          template <std::floating_point, std::size_t> typename FFTPolicy>
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
          cnt_(ComputeTotalGridSize(ns_)),
          norm_factor_(ComputeNormFactor(ns_, bcs_)),
          block_size_(cuda_block_size),
          grid_size_(ComputeGridSize(ns_, block_size_)) {
        AllocDeviceBuffers();

        // Denoms are constant for the solver lifetime.
        // They can be copied just once, here.
        // It is important to have buffers already allocated before copying.
        const auto denoms = GetEigenvalues(ns_, lens, bcs_, approx);
        CopyDenomsToDevice(denoms);
        cu_denom_observers_ = CreateDenomObservers();

        solve_ = GetSolveMethod(bcs);

        FFTPolicy<T, Dim>::Init(fft_plan_, ns_, bcs);
    }

    void Execute(std::span<T> phi, std::span<const T> rhs) {
        if (phi.size() != cnt_ || rhs.size() != cnt_) {
            throw std::runtime_error("Incorrect size of phi or rhs.");
        }

        (this->*solve_)(phi, rhs);
        CUCH(cudaMemcpy(phi.data(),
                        cu_phi_.get(),
                        cnt_ * sizeof(T),
                        cudaMemcpyDeviceToHost));
    }

    std::size_t GetTotalGridSize() const {
        return cnt_;
    }

private:
    using SolveFunction = void (Solver::*)(std::span<T>, std::span<const T>);
    using HostOneAxisDenoms = std::vector<T>;
    using HostDenoms = std::array<HostOneAxisDenoms, Dim>;

    // === Grid ===
    std::array<std::size_t, Dim> ns_;
    std::array<BCPair, Dim> bcs_;
    std::size_t cnt_;

    // === Physics and math ===
    std::array<detail::CudaUniquePtr<T>, Dim> cu_denoms_;

    // These will be passed into the kernels.
    std::array<const T *, Dim> cu_denom_observers_;

    T norm_factor_;
    SolveFunction solve_;

    // === FFT ===
    typename FFTPolicy<T, Dim>::Plan fft_plan_;

    // RHS needed only in case we initialize complex workspace from it.
    std::optional<detail::CudaUniquePtr<T>> cu_rhs_;
    detail::CudaUniquePtr<T> cu_phi_;

    // Only one of those is used
    std::optional<detail::CudaUniquePtr<detail::CuComplexTrait<T>>>
        cu_cmplx_workspace_;
    std::optional<detail::CudaUniquePtr<T>> cu_real_workspace_;

    // === Device settings ===
    dim3 block_size_;
    dim3 grid_size_;

    static constexpr dim3 DefaultBlockSize() {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else if constexpr (Dim == 3) {
            return {8, 8, 4};
        }
    }

    static std::size_t ComputeTotalGridSize(std::array<std::size_t, Dim> ns) {
        return std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>{});
    }

    static T ComputeNormFactor(std::array<std::size_t, Dim> ns,
                               std::array<BCPair, Dim> bcs) {
        using detail::GetAxisStrategy;

        T factor = 1.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            auto strategy = GetAxisStrategy<T>(bcs[i]);
            factor *= strategy->ComputeNormalizationFactor(ns[i]);
        }
        return factor;
    }

    static dim3 ComputeGridSize(std::array<std::size_t, Dim> ns, dim3 block) {
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

    void AllocDeviceWorkspace() {
        using detail::CuComplexTrait;
        using detail::MakeCudaUnique;

        // TODO: Only allocate one of those depending on the BCs (or solve
        // method? could be cleaner). For now both are allocated as idk when
        // specifically each is needed.
        cu_cmplx_workspace_ = MakeCudaUnique<CuComplexTrait<T>>(cnt_);
        cu_real_workspace_ = MakeCudaUnique<T>(cnt_);
    }

    void AllocDeviceRhs() {
        using detail::MakeCudaUnique;

        // TODO: rhs is needed only in case that complex workspace is needed.
        cu_rhs_ = MakeCudaUnique<T>(cnt_);
    }

    void AllocDevicePhi() {
        using detail::MakeCudaUnique;
        cu_phi_ = MakeCudaUnique<T>(cnt_);
    }

    void AllocDeviceDenoms() {
        using detail::MakeCudaUnique;
        for (std::size_t d = 0; d < Dim; ++d) {
            cu_denoms_[d] = MakeCudaUnique<T>(ns_[d]);
        }
    }

    void AllocDeviceBuffers() {
        AllocDeviceWorkspace();
        AllocDeviceRhs();
        AllocDevicePhi();
        AllocDeviceDenoms();
    }

    static HostOneAxisDenoms GetOneAxisEigenvalues(
        T len, std::size_t n, BCPair bcp, const IApproximation<T> &approx) {
        using detail::GetAxisStrategy;

        HostOneAxisDenoms eigenvals(n);
        auto strategy = GetAxisStrategy<T>(bcp);

        const T grid_dx = strategy->GetGridDx(len, n);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;
        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(strategy->GetApproximationInput(i, n, dkx)) *
                           inv_grid_dx_squared;
        }

        return eigenvals;
    }

    static HostDenoms GetEigenvalues(std::array<std::size_t, Dim> ns,
                                     std::array<T, Dim> lens,
                                     std::array<BCPair, Dim> bcs,
                                     const IApproximation<T> &approx) {
        // TODO: This could be done on GPU
        HostDenoms denoms{};
        for (std::size_t d = 0; d < Dim; ++d) {
            denoms[d] = GetOneAxisEigenvalues(ns[d], lens[d], bcs[d], approx);
        }
        return denoms;
    }

    void CopyDenomsToDevice(const HostDenoms &denoms) {
        for (std::size_t d = 0; d < Dim; ++d) {
            CUCH(cudaMemcpy(cu_denoms_[d].get(),
                            denoms[d].data(),
                            ns_[d] * sizeof(T),
                            cudaMemcpyHostToDevice));
        }
    }

    std::array<const T *, Dim> CreateDenomObservers() {
        std::array<const T *, Dim> observers{};
        for (std::size_t d = 0; d < Dim; ++d) {
            observers[d] = cu_denoms_[d].get();
        }
        return observers;
    }

    void SolveFullPeriodic(std::span<T> phi, std::span<const T> rhs) {
        using namespace detail::runners::full_periodic;

        CUCH(cudaMemcpy(cu_rhs_.value().get(),
                        rhs.data(),
                        cnt_ * sizeof(T),
                        cudaMemcpyHostToDevice));

        Init(cu_cmplx_workspace_.value().get(),
             cu_rhs_.value().get(),
             ns_,
             grid_size_,
             block_size_);

        FFTPolicy<T, Dim>::Forward(cu_cmplx_workspace_.value().get(),
                                   fft_plan_);

        DivideByDenoms(cu_cmplx_workspace_.value().get(),
                       ns_,
                       cu_denom_observers_,
                       grid_size_,
                       block_size_);

        FFTPolicy<T, Dim>::Backward(cu_cmplx_workspace_.value().get(),
                                    fft_plan_);
        Normalize(cu_phi_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  grid_size_,
                  block_size_);
    }

    void SolveFullDirichlet(std::span<T> phi, std::span<const T> rhs) {
        using namespace detail::runners::full_dirichlet;
        using detail::runners::Normalize;

        CUCH(cudaMemcpy(cu_real_workspace_.value().get(),
                        rhs.data(),
                        cnt_ * sizeof(T),
                        cudaMemcpyHostToDevice));

        FFTPolicy<T, Dim>::Forward(cu_real_workspace_.value().get(), fft_plan_);

        DivideByDenoms(cu_real_workspace_.value().get(),
                       ns_,
                       cu_denom_observers_,
                       grid_size_,
                       block_size_);

        FFTPolicy<T, Dim>::Backward(cu_real_workspace_.value().get(),
                                    fft_plan_);
        Normalize(cu_phi_.get(),
                  cu_real_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  grid_size_,
                  block_size_);
    }

    void SolveFullNeumann(std::span<T> phi, std::span<const T> rhs) {
        using namespace detail::runners::full_neumann;
        using detail::runners::Normalize;

        CUCH(cudaMemcpy(cu_real_workspace_.value().get(),
                        rhs.data(),
                        cnt_ * sizeof(T),
                        cudaMemcpyHostToDevice));

        FFTPolicy<T, Dim>::Forward(cu_real_workspace_.value().get(), fft_plan_);

        DivideByDenoms(cu_real_workspace_.value().get(),
                       ns_,
                       cu_denom_observers_,
                       grid_size_,
                       block_size_);

        FFTPolicy<T, Dim>::Backward(cu_real_workspace_.value().get(),
                                    fft_plan_);
        Normalize(cu_phi_.get(),
                  cu_real_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  grid_size_,
                  block_size_);
    }

    void SolvePPNs(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void SolvePPD(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void SolveNsPP(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void SolvePNsNs(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void SolvePNsP(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void SolveDDP(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void Solve2Real1Real(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    void Solve1Real2Real(std::span<T> phi, std::span<const T> rhs) {
        // TODO
        throw std::runtime_error("Not implemented.");
    }

    static SolveFunction GetSolveMethod1D(std::array<BCPair, 1> bcs) {
        using namespace std::ranges;

        if (all_of(bcs, IsPeriodic)) {
            return &Solver::SolveFullPeriodic;
        } else if (all_of(bcs, IsDirichlet) || all_of(bcs, IsDirichletStag)) {
            return &Solver::SolveFullDirichlet;
        } else if (all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag)) {
            return &Solver::SolveFullNeumann;
        } else if (none_of(bcs, HasPeriodic) &&
                   any_of(bcs, HasDirichletOrStag)) {
            return &Solver::SolveFullDirichlet;
        } else {
            throw std::invalid_argument(
                "Invalid boundary conditions combination.");
        }
    }

    static SolveFunction GetSolveMethod2D(std::array<BCPair, 2> bcs) {
        using namespace std::ranges;

        if (all_of(bcs, IsPeriodic)) {
            return &Solver::SolveFullPeriodic;
        } else if (all_of(bcs, IsDirichlet) || all_of(bcs, IsDirichletStag)) {
            return &Solver::SolveFullDirichlet;
        } else if (all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag)) {
            return &Solver::SolveFullNeumann;
        } else {
            throw std::invalid_argument(
                "Invalid boundary conditions combination.");
        }
    }

    static SolveFunction GetSolveMethod3D(std::array<BCPair, 3> bcs) {
        using namespace std::ranges;

        auto [bcx, bcy, bcz] = bcs;

        if (all_of(bcs, IsPeriodic)) {
            return &Solver::SolveFullPeriodic;
        } else if (all_of(bcs, IsDirichlet) || all_of(bcs, IsDirichletStag)) {
            return &Solver::SolveFullDirichlet;
        } else if (all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag)) {
            return &Solver::SolveFullNeumann;
        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            return &Solver::SolvePPNs;
        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && !HasPeriodic(bcz) &&
                   HasDirichletOrStag(bcz)) {
            return &Solver::SolvePPD;
        } else if (IsNeumannOrStag(bcx) && IsPeriodic(bcy) && IsPeriodic(bcz)) {
            return &Solver::SolveNsPP;
        } else if (IsPeriodic(bcx) &&
                   ((IsNeumann(bcy) && IsNeumann(bcz)) ||
                    (IsNeumannStag(bcy) && IsNeumannStag(bcz)))) {
            return &Solver::SolvePNsNs;
        } else if (IsPeriodic(bcx) && IsNeumannOrStag(bcy) && IsPeriodic(bcz)) {
            return &Solver::SolvePNsP;
        } else if (((IsDirichlet(bcx) && IsDirichlet(bcy)) ||
                    (IsDirichletStag(bcx) && IsDirichletStag(bcy))) &&
                   IsPeriodic(bcz)) {
            return &Solver::SolveDDP;
        } else if (none_of(bcs, IsPeriodic) && (bcx == bcy) && (bcx != bcz) &&
                   any_of(bcs, HasDirichletOrStag)) {
            return &Solver::Solve2Real1Real;
        } else if (none_of(bcs, IsPeriodic) && (bcy == bcz) && (bcx != bcy) &&
                   any_of(bcs, HasDirichletOrStag)) {
            return &Solver::Solve1Real2Real;
        } else {
            throw std::invalid_argument(
                "Invalid boundary conditions combination.");
        }
    }

    static SolveFunction GetSolveMethod(std::array<BCPair, Dim> bcs) {
        if constexpr (Dim == 1) {
            return GetSolveMethod1D(bcs);
        } else if constexpr (Dim == 2) {
            return GetSolveMethod2D(bcs);
        } else if constexpr (Dim == 3) {
            return GetSolveMethod3D(bcs);
        }
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
