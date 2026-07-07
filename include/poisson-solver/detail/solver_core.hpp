#ifndef SOLVER_CORE_HPP
#define SOLVER_CORE_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <numbers>
#include <numeric>
#include <optional>
#include <vector>

namespace poisson_solver::detail {
/**
 * @brief Class encapsulating core solver data.
 * Every solver is composed (at least) of this core.
 *
 * @tparam T Floating point type (only `float` and `double` supported).
 * @tparam Dim Number of grid dimensions.
 * @tparam FFTDim Number of grid dimensions transformed via FFT.
 */
template <std::floating_point T, std::size_t Dim, std::size_t FFTDim = Dim>
    requires(FFTDim <= Dim)
struct SolverCore {
public:
    using HostOneAxisDenoms = std::vector<T>;
    using HostDenoms = std::array<HostOneAxisDenoms, FFTDim>;
    using CudaUniquePtrT = detail::CudaUniquePtr<T>;
    using CudaUniquePtrCmplxT =
        detail::CudaUniquePtr<detail::CuComplexTrait<T>>;
    using DenomConstPtrs = detail::DenomConstPtrs<T, FFTDim>;

    // FFT needs to know the dimensionality of the whole grid.
    // Transformation of only some dimensions are handled using
    // the 1D and 2D methods of FFTPolicy.
    using CoreFFTPolicy = FFTPolicy<T, Dim>;

    // === Plans ===
    typename CoreFFTPolicy::Plan fft_plan;

    // === Grid ===
    GridSizes<Dim> ns;
    std::size_t cnt;

    // === Physics and Math ===
    std::array<CudaUniquePtrT, FFTDim> cu_denoms;

    // These will be passed into the kernels.
    DenomConstPtrs cu_denom_observers;

    T norm_factor;

    // === Workspaces ===

    // Only some solvers need complex workspace.
    std::optional<CudaUniquePtrCmplxT> cu_cmplx_workspace;

    // This is always used - either straight for computation
    // or to initialize complex workspace from it.
    CudaUniquePtrT cu_real_workspace;

    // === Device Settings ===
    dim3 cu_block_size;
    dim3 cu_grid_size;

    SolverCore(GridSizes<Dim> ns,
               std::array<T, FFTDim> lens,
               const IApproximation<T> &approx,
               dim3 cuda_block_size,
               const ISolverPlan<T, FFTDim> &sp)
        : fft_plan(CoreFFTPolicy::Init(ns, sp.fft_layout)),
          ns(ns),
          cnt(ComputeTotalGridSize(ns)),
          norm_factor(ComputeNormFactor(ns, sp)),
          cu_block_size(cuda_block_size),
          cu_grid_size(ComputeCUDAGridSize(ns, cuda_block_size)) {
        AllocDeviceBuffers();

        // Denoms are constant for the solver lifetime.
        // They can be copied just once, here.
        const auto denoms = GetEigenvalues(ns, lens, approx, sp);
        CopyDenomsToDevice(denoms);
        cu_denom_observers = CreateDenomObservers();
    }

private:
    static auto ComputeTotalGridSize(GridSizes<Dim> ns) -> std::size_t {
        return std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>{});
    }

    auto ComputeNormFactor(GridSizes<Dim> ns,
                           const ISolverPlan<T, FFTDim> &sp) const -> T {
        T factor = 1.0;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            const auto &strategy = sp.axis_strategies[d];
            factor *= strategy->ComputeNormalizationFactor(ns[d]);
        }
        return factor;
    }

    /**
     * @note The dimensionality of the result is determined by `FFTDim`.
     * The computations in dimensions not transformed via FFT
     * (e.g. Z in `NonuniformZSolver`) contain dependencies in that dimension.
     * This forces the GPU threads to parallelize only on the remaining dims
     * (e.g. X and Y in `NonuniformZSolver`).
     */
    static auto ComputeCUDAGridSize(GridSizes<Dim> ns, dim3 block) -> dim3 {
        if constexpr (FFTDim == 1) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x};
        } else if constexpr (FFTDim == 2) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y};
        } else if constexpr (FFTDim == 3) {
            return {(static_cast<unsigned>(ns[0]) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ns[1]) + block.y - 1) / block.y,
                    (static_cast<unsigned>(ns[2]) + block.z - 1) / block.z};
        }
    }

    void AllocDeviceWorkspaces() {
        using detail::CuComplexTrait;
        using detail::MakeCudaUnique;

        // TODO: Only allocate one of those depending on the BCs (or solve
        // method? could be cleaner). For now both are allocated as idk when
        // specifically each is needed.
        cu_cmplx_workspace = MakeCudaUnique<CuComplexTrait<T>>(cnt);
        cu_real_workspace = MakeCudaUnique<T>(cnt);
    }

    void AllocDeviceDenoms() {
        using detail::MakeCudaUnique;
        for (std::size_t d = 0; d < FFTDim; ++d) {
            cu_denoms[d] = MakeCudaUnique<T>(ns[d]);
        }
    }

    void AllocDeviceBuffers() {
        AllocDeviceWorkspaces();
        AllocDeviceDenoms();
    }

    auto GetOneAxisEigenvalues(std::size_t n,
                               T len,
                               std::size_t axis,
                               const IApproximation<T> &approx,
                               const ISolverPlan<T, FFTDim> &sp) const
        -> HostOneAxisDenoms {
        HostOneAxisDenoms eigenvals(n);
        const auto &strategy = sp.axis_strategies[axis];

        const T grid_dx = strategy->GetGridDx(n, len);
        const T dkx = std::numbers::pi_v<T> * grid_dx / len;
        const T inv_grid_dx_squared = static_cast<T>(1.0) / (grid_dx * grid_dx);

        for (std::size_t i = 0; i < n; ++i) {
            eigenvals[i] = approx(strategy->GetApproximationInput(i, n, dkx)) *
                           inv_grid_dx_squared;
        }

        return eigenvals;
    }

    auto GetEigenvalues(GridSizes<Dim> ns,
                        std::array<T, FFTDim> lens,
                        const IApproximation<T> &approx,
                        const ISolverPlan<T, FFTDim> &sp) const -> HostDenoms {
        // TODO: This could be done on GPU
        HostDenoms denoms{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            denoms[d] = GetOneAxisEigenvalues(ns[d], lens[d], d, approx, sp);
        }
        return denoms;
    }

    /**
     * @note Device `denom` buffers have to be already allocated.
     */
    void CopyDenomsToDevice(const HostDenoms &denoms) {
        for (std::size_t d = 0; d < FFTDim; ++d) {
            CUCH(cudaMemcpy(cu_denoms[d].get(),
                            denoms[d].data(),
                            ns[d] * sizeof(T),
                            cudaMemcpyHostToDevice));
        }
    }

    /**
     * @note Device `denom` buffers have to be already allocated.
     */
    auto CreateDenomObservers() -> DenomConstPtrs {
        DenomConstPtrs observers{};
        for (std::size_t d = 0; d < FFTDim; ++d) {
            observers[d] = cu_denoms[d].get();
        }
        return observers;
    }
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_CORE_HPP
