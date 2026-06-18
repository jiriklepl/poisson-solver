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
#include "poisson-solver/detail/solver_core.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <span>

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

protected:
    static constexpr auto DefaultBlockSize() -> dim3 {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else if constexpr (Dim == 3) {
            return {8, 8, 4};
        }
    }
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
     * @note The solver assumes that the first index is the fastest changing,
     * second is middle and third is slowest.
     */
    Solver(GridSizes<Dim> ns,
           std::array<T, Dim> lens,
           BCs<Dim> bcs,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = ISolver<T, Dim>::DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns, lens, approx, cuda_block_size, solver_plan_) {}

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
                                 core_.cu_real_workspace.get(),
                                 core_.ns,
                                 ngs_rhs,
                                 cudaMemcpyHostToDevice);

        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(phi.data(),
                                 core_.cu_real_workspace.get(),
                                 core_.ns,
                                 ngs_phi,
                                 cudaMemcpyDeviceToHost);
    }

    auto GetTotalGridSize() const -> std::size_t override {
        return core_.cnt;
    }

private:
    friend struct detail::SolverPlan<T, Dim>;
    using FFTPolicy = detail::FFTPolicy<T, Dim>;

    detail::SolverPlan<T, Dim> solver_plan_;
    detail::SolverCore<T, Dim> core_;

    void SolveFullPeriodic() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward(core_.cu_cmplx_workspace.value().get(),
                           core_.fft_plan);

        DivideByDenomsAndZero(core_.cu_cmplx_workspace.value().get(),
                              core_.ns,
                              core_.cu_denom_observers,
                              core_.cu_grid_size,
                              core_.cu_block_size);

        FFTPolicy::Backward(core_.cu_cmplx_workspace.value().get(),
                            core_.fft_plan);
        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
    }

    void SolveFullDirichlet() {
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward(core_.cu_real_workspace.get(), core_.fft_plan);

        DivideByDenoms(core_.cu_real_workspace.get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);

        FFTPolicy::Backward(core_.cu_real_workspace.get(), core_.fft_plan);
        Normalize(core_.cu_real_workspace.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
    }

    void SolveFullNeumann() {
        using detail::runners::common_real::Normalize;
        using detail::runners::full_neumann::DivideByDenoms;

        FFTPolicy::Forward(core_.cu_real_workspace.get(), core_.fft_plan);

        DivideByDenoms(core_.cu_real_workspace.get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);

        FFTPolicy::Backward(core_.cu_real_workspace.get(), core_.fft_plan);
        Normalize(core_.cu_real_workspace.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
    }

    void SolvePPNs() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward2D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenomsAndZero(core_.cu_cmplx_workspace.value().get(),
                              core_.ns,
                              core_.cu_denom_observers,
                              core_.cu_grid_size,
                              core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void SolvePPD() {
        using detail::runners::common_cmplx::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward2D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace.value().get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void SolveNsPP() {
        // TODO: Same body as PPNs.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward2D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenomsAndZero(core_.cu_cmplx_workspace.value().get(),
                              core_.ns,
                              core_.cu_denom_observers,
                              core_.cu_grid_size,
                              core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void SolvePNsNs() {
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward2D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward1D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenomsAndZero(core_.cu_cmplx_workspace.value().get(),
                              core_.ns,
                              core_.cu_denom_observers,
                              core_.cu_grid_size,
                              core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void SolvePNsP() {
        // TODO: Same body as PPNs.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_cmplx::DivideByDenomsAndZero;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward2D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenomsAndZero(core_.cu_cmplx_workspace.value().get(),
                              core_.ns,
                              core_.cu_denom_observers,
                              core_.cu_grid_size,
                              core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void SolveDDP() {
        using detail::runners::common_cmplx::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        FFTPolicy::Forward2D(core_.cu_real_workspace.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward1D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace.value().get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_cmplx_workspace.value().get(),
                              core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.cu_cmplx_workspace.value().get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void Solve2Real1Real() {
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward2D(core_.cu_real_workspace.get(), core_.fft_plan);

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_real_workspace.get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace.get(), core_.fft_plan);
    }

    void Solve1Real2Real() {
        // TODO: Same body as 2Real1Real.
        // Only is change in `fft_plan_` construction
        // but this doesn't concern this function.
        using detail::runners::common_real::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward2D(core_.cu_real_workspace.get(), core_.fft_plan);

        FFTPolicy::Forward1D(core_.cu_real_workspace.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_real_workspace.get(),
                       core_.ns,
                       core_.cu_denom_observers,
                       core_.cu_grid_size,
                       core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace.get(), core_.fft_plan);
    }
};

template <std::floating_point T>
/**
 * @brief Solver for a non-uniform grid.
 *
 * @warning Only works for 3D grids.
 */
class NonuniformZSolver : public ISolver<T, 3> {
public:
    NonuniformZSolver(
        GridSizes<3> ns,
        std::array<T, 2> lens,
        BCs<3> bcs,
        const IApproximation<T> &approx = SpectralApproximation<T>{},
        dim3 cuda_block_size = ISolver<T, 3>::DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns, lens, approx, cuda_block_size, solver_plan_) {}

private:
    detail::NonuniformZSolverPlan<T> solver_plan_;
    detail::SolverCore<T, 3, 2> core_;
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
