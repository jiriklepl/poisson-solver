#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/runners/common_cmplx.hpp"
#include "poisson-solver/detail/runners/common_real.hpp"
#include "poisson-solver/detail/runners/full_neumann.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace poisson_solver {
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
/**
 * @brief Abstract solver class for both uniform and nonuniform solvers.
 * Its childer are solvers of the equation \f$ \delta \phi = b \f$.
 *
 * @tparam T Floating point type (only `float` and `double` supported).
 * @tparam Dim Number of grid dimensions.
 */
class ISolver {
public:
    virtual ~ISolver() = default;
    virtual void Execute(std::span<T> phi,
                         std::span<const T> rhs,
                         GridSizes<Dim> ngs_phi,
                         GridSizes<Dim> ngs_rhs) = 0;
    virtual auto GetTotalGridSize() const -> std::size_t = 0;
};

template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
/**
 * @brief Solver for a uniform grid.
 */
class Solver : public ISolver<T, Dim> {
public:
    /**
     * @brief Constructs a poisson solver.
     *
     * @param ns Sizes of the grid in each dimension (without ghost cells).
     * @param lens Real lengths of the grid in each dimension.
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     *
     * @note The sovler assumes that the first index is the fastest changing,
     * second is middle and third is slowest.
     */
    Solver(GridSizes<Dim> ns,
           std::array<T, Dim> lens,
           BCs<Dim> bcs,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = DefaultBlockSize())
        : solver_plan_(bcs),
          ns_(ns),
          bcs_(bcs),
          cnt_(ComputeTotalGridSize(ns_)),
          norm_factor_(ComputeNormFactor(ns_)),
          cu_block_size_(cuda_block_size),
          cu_grid_size_(ComputeGridSize(ns_, cu_block_size_)) {
        AllocDeviceBuffers();

        // Denoms are constant for the solver lifetime.
        // They can be copied just once, here.
        // It is important to have buffers already allocated before copying.
        const auto denoms = GetEigenvalues(ns_, lens, approx);
        CopyDenomsToDevice(denoms);
        cu_denom_observers_ = CreateDenomObservers();

        fft_plan_ = FFTPolicy::Init(ns_, solver_plan_.fft_layout);
    }

    /**
     * @brief Solves the equation \f$ \delta \phi = b \f$.
     *
     * @param[out] phi
     * @param[in] rhs
     * @param[in] ngs_phi Number of ghost cells (on one side)
     * \p phi has in each dimension.
     * @param[in] ngs_rhs Number of ghost cells (on one side)
     * \p rhs has in each dimension.
     */
    void Execute(std::span<T> phi,
                 std::span<const T> rhs,
                 GridSizes<Dim> ngs_phi,
                 GridSizes<Dim> ngs_rhs) override {
        // `const_cast` necessary due to
        // CUDA C-like API interaction.
        // Nothing is actually modified.
        detail::MemcpyActiveGrid(const_cast<T *>(rhs.data()),
                                 cu_real_workspace_.get(),
                                 ns_,
                                 ngs_rhs,
                                 cudaMemcpyHostToDevice);

        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(phi.data(),
                                 cu_real_workspace_.get(),
                                 ns_,
                                 ngs_phi,
                                 cudaMemcpyDeviceToHost);
    }

    auto GetTotalGridSize() const -> std::size_t override {
        return cnt_;
    }

private:
    friend struct detail::SolverPlan<T, Dim>;

    using SolveMethod = void (Solver::*)(std::span<T>, std::span<const T>);
    using HostOneAxisDenoms = std::vector<T>;
    using HostDenoms = std::array<HostOneAxisDenoms, Dim>;
    using CudaUniquePtrT = detail::CudaUniquePtr<T>;
    using CudaUniquePtrCmplxT =
        detail::CudaUniquePtr<detail::CuComplexTrait<T>>;
    using DenomConstPtrs = detail::DenomConstPtrs<T, Dim>;
    using FFTPolicy = detail::FFTPolicy<T, Dim>;

    detail::SolverPlan<T, Dim> solver_plan_;

    // === Grid ===
    GridSizes<Dim> ns_;
    BCs<Dim> bcs_;
    std::size_t cnt_;

    // === Physics and math ===
    std::array<CudaUniquePtrT, Dim> cu_denoms_;

    // These will be passed into the kernels.
    DenomConstPtrs cu_denom_observers_;

    T norm_factor_;

    // === FFT ===
    typename FFTPolicy::Plan fft_plan_;

    // === Workspaces ===

    // Only some solvers need complex workspace.
    std::optional<CudaUniquePtrCmplxT> cu_cmplx_workspace_;

    // This is always used - either straight for computation
    // or to initialize complex workspace from it.
    CudaUniquePtrT cu_real_workspace_;

    // === Device settings ===
    dim3 cu_block_size_;
    dim3 cu_grid_size_;

    static constexpr auto DefaultBlockSize() -> dim3 {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else if constexpr (Dim == 3) {
            return {8, 8, 4};
        }
    }

    static auto ComputeTotalGridSize(GridSizes<Dim> ns) -> std::size_t {
        return std::accumulate(
            ns.begin(), ns.end(), std::size_t{1}, std::multiplies<>{});
    }

    /**
     * @note `solver_plan_` has to be initialized
     * before this function is called.
     */
    auto ComputeNormFactor(GridSizes<Dim> ns) const -> T {
        T factor = 1.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            const auto &strategy = solver_plan_.axis_strategies[d];
            factor *= strategy->ComputeNormalizationFactor(ns[d]);
        }
        return factor;
    }

    static auto ComputeGridSize(GridSizes<Dim> ns, dim3 block) -> dim3 {
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

    void AllocDeviceDenoms() {
        using detail::MakeCudaUnique;
        for (std::size_t d = 0; d < Dim; ++d) {
            cu_denoms_[d] = MakeCudaUnique<T>(ns_[d]);
        }
    }

    void AllocDeviceBuffers() {
        AllocDeviceWorkspace();
        AllocDeviceDenoms();
    }

    /**
     * @note `solver_plan_` has to be initialized
     * before this function is called.
     */
    auto GetOneAxisEigenvalues(std::size_t n,
                               T len,
                               std::size_t axis,
                               const IApproximation<T> &approx) const
        -> HostOneAxisDenoms {
        HostOneAxisDenoms eigenvals(n);
        const auto &strategy = solver_plan_.axis_strategies[axis];

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
                        std::array<T, Dim> lens,
                        const IApproximation<T> &approx) const -> HostDenoms {
        // TODO: This could be done on GPU
        HostDenoms denoms{};
        for (std::size_t d = 0; d < Dim; ++d) {
            denoms[d] = GetOneAxisEigenvalues(ns[d], lens[d], d, approx);
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

    auto CreateDenomObservers() -> DenomConstPtrs {
        DenomConstPtrs observers{};
        for (std::size_t d = 0; d < Dim; ++d) {
            observers[d] = cu_denoms_[d].get();
        }
        return observers;
    }

    void SolveFullPeriodic() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward(cu_cmplx_workspace_.value().get(), fft_plan_);

        DivideByDenomsAndZero(cu_cmplx_workspace_.value().get(),
                              ns_,
                              cu_denom_observers_,
                              cu_grid_size_,
                              cu_block_size_);

        FFTPolicy::Backward(cu_cmplx_workspace_.value().get(), fft_plan_);
        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
    }

    void SolveFullDirichlet() {
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward(cu_real_workspace_.get(), fft_plan_);

        DivideByDenoms(cu_real_workspace_.get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);

        FFTPolicy::Backward(cu_real_workspace_.get(), fft_plan_);
        Normalize(cu_real_workspace_.get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
    }

    void SolveFullNeumann() {
        using detail::runners::common_real::Normalize;
        using detail::runners::full_neumann::DivideByDenoms;

        FFTPolicy::Forward(cu_real_workspace_.get(), fft_plan_);

        DivideByDenoms(cu_real_workspace_.get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);

        FFTPolicy::Backward(cu_real_workspace_.get(), fft_plan_);
        Normalize(cu_real_workspace_.get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
    }

    void SolvePPNs() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward2D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenomsAndZero(cu_cmplx_workspace_.value().get(),
                              ns_,
                              cu_denom_observers_,
                              cu_grid_size_,
                              cu_block_size_);
        FFTPolicy::Backward2D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);
    }

    void SolvePPD() {
        using detail::runners::common_cmplx::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward2D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenoms(cu_cmplx_workspace_.value().get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);
        FFTPolicy::Backward2D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);
    }

    void SolveNsPP() {
        // TODO: Same body as PPNs.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward2D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenomsAndZero(cu_cmplx_workspace_.value().get(),
                              ns_,
                              cu_denom_observers_,
                              cu_grid_size_,
                              cu_block_size_);
        FFTPolicy::Backward2D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);
    }

    void SolvePNsNs() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward2D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward1D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenomsAndZero(cu_cmplx_workspace_.value().get(),
                              ns_,
                              cu_denom_observers_,
                              cu_grid_size_,
                              cu_block_size_);
        FFTPolicy::Backward1D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward2D(cu_real_workspace_.get(), fft_plan_);
    }

    void SolvePNsP() {
        // TODO: Same body as PPNs.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward2D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenomsAndZero(cu_cmplx_workspace_.value().get(),
                              ns_,
                              cu_denom_observers_,
                              cu_grid_size_,
                              cu_block_size_);
        FFTPolicy::Backward2D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);
    }

    void SolveDDP() {
        using detail::runners::common_cmplx::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward2D(cu_real_workspace_.get(), fft_plan_);
        Init(cu_cmplx_workspace_.value().get(),
             cu_real_workspace_.get(),
             ns_,
             cu_grid_size_,
             cu_block_size_);

        FFTPolicy::Forward1D(cu_cmplx_workspace_.value().get(), fft_plan_);
        DivideByDenoms(cu_cmplx_workspace_.value().get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);
        FFTPolicy::Backward1D(cu_cmplx_workspace_.value().get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  cu_cmplx_workspace_.value().get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward2D(cu_real_workspace_.get(), fft_plan_);
    }

    void Solve2Real1Real() {
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward2D(cu_real_workspace_.get(), fft_plan_);

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        DivideByDenoms(cu_real_workspace_.get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward2D(cu_real_workspace_.get(), fft_plan_);
    }

    void Solve1Real2Real() {
        // TODO: Same body as 2Real1Real.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward2D(cu_real_workspace_.get(), fft_plan_);

        FFTPolicy::Forward1D(cu_real_workspace_.get(), fft_plan_);
        DivideByDenoms(cu_real_workspace_.get(),
                       ns_,
                       cu_denom_observers_,
                       cu_grid_size_,
                       cu_block_size_);
        FFTPolicy::Backward1D(cu_real_workspace_.get(), fft_plan_);

        Normalize(cu_real_workspace_.get(),
                  ns_,
                  norm_factor_,
                  cu_grid_size_,
                  cu_block_size_);
        FFTPolicy::Backward2D(cu_real_workspace_.get(), fft_plan_);
    }
};

template <std::floating_point T, std::size_t Dim>
    requires(Dim == 3)
/**
 * @brief Solver for a non-uniform grid.
 *
 * @warning Only works for 3D grids.
 */
class NonuniformZSolver : public ISolver<T, Dim> {
    // TODO: Start implementing and see what you
    // need to put from Solver to ISolver.
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
