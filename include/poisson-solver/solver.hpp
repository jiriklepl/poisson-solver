#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "poisson-solver/aliases.hpp"
#include "poisson-solver/approximations.hpp"
#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/fft_policies.hpp"
#include "poisson-solver/detail/helpers.hpp"
#include "poisson-solver/detail/runners/common.hpp"
#include "poisson-solver/detail/runners/common_cmplx.hpp"
#include "poisson-solver/detail/runners/common_real.hpp"
#include "poisson-solver/detail/runners/nonuniform_z.hpp"
#include "poisson-solver/detail/solver_core.hpp"
#include "poisson-solver/detail/solver_plan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <span>
#include <stdexcept>

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
        using detail::runners::DivideByDenomsAndZero;
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
        using detail::runners::DivideByDenoms;
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
        using detail::runners::DivideByDenomsAndZero;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward(core_.cu_real_workspace.get(), core_.fft_plan);

        DivideByDenomsAndZero(core_.cu_real_workspace.get(),
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
        using detail::runners::DivideByDenomsAndZero;
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
        using detail::runners::DivideByDenoms;
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
        using detail::runners::DivideByDenomsAndZero;
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
        using detail::runners::DivideByDenomsAndZero;
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
        using detail::runners::DivideByDenomsAndZero;
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
        using detail::runners::DivideByDenoms;
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
        using detail::runners::DivideByDenoms;
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
        using detail::runners::DivideByDenoms;
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
    /**
     * @brief Constructs a poisson solver with nonuniform Z axis.
     *
     * @param ns Sizes of the grid in each dimension (without ghost cells).
     * @param lens Real lengths of the grid in each dimension.
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * @param z Coordinates representing the cell centers along the Z-axis.
     * Must contain at least `ns[2]` elements.
     * @param z_u Coordinates representing the cell faces along the Z-axis.
     * Must contain at least `ns[2] + 1` elements.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     * The `z` component will be unused (always set to 1).
     */
    NonuniformZSolver(
        GridSizes<3> ns,
        std::array<T, 2> lens,
        std::span<const T> z,
        std::span<const T> z_u,
        BCs<3> bcs,
        const IApproximation<T> &approx = SpectralApproximation<T>{},
        dim3 cuda_block_size = DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns, lens, approx, cuda_block_size, solver_plan_) {
        cuda_block_size.z = 1;

        auto nz = ns[2];
        AllocateDeviceBuffers(nz);
        auto mat = BuildTridiagMatrix(z, z_u, nz);
        CopyTridiagMatrixToDevice(mat, nz);
    }

    void Execute(std::span<T> phi,
                 std::span<const T> rhs,
                 GridSizes<3> ngs_phi,
                 GridSizes<3> ngs_rhs) override {
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
    friend struct detail::NonuniformZSolverPlan<T>;

    // Plan has to be initialized before core
    detail::NonuniformZSolverPlan<T> solver_plan_;
    detail::SolverCore<T, 3, 2> core_;

    using FFTPolicy = decltype(core_)::CoreFFTPolicy;
    using CudaUniquePtrT = decltype(core_)::CudaUniquePtrT;

    using TridiagMatrixDiagonal = std::vector<T>;
    using TridiagMatrix = std::array<TridiagMatrixDiagonal, 3>;

    CudaUniquePtrT cu_mat_a_;
    CudaUniquePtrT cu_mat_b_;
    CudaUniquePtrT cu_mat_c_;

    CudaUniquePtrT cu_cp_workspace_;

    /**
     * @note Shadows the `DefaultBlockSize` method in `ISolver` because
     * `NonuniformZSolver` doesn't parallelize in Z dimension.
     *
     * @return Default block size for `NonuniformZSolver`.
     */
    static constexpr auto DefaultBlockSize() -> dim3 {
        return {16, 16, 1};
    }

    void AllocateDeviceBuffers(std::size_t nz) {
        using detail::MakeCudaUnique;

        cu_mat_a_ = MakeCudaUnique<T>(nz);
        cu_mat_b_ = MakeCudaUnique<T>(nz);
        cu_mat_c_ = MakeCudaUnique<T>(nz);
        cu_cp_workspace_ = MakeCudaUnique<T>(core_.cnt);
    }

    static auto BuildMatrixDiagonalA(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t nz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_a(nz, 0);
        for (std::size_t i = 1; i < nz; ++i) {
            mat_a[i] = 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_a;
    }

    static auto BuildMatrixDiagonalB(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t nz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_b(nz, 0);
        if (nz == 1) {
            mat_b[0] = 0;
        } else {
            mat_b[0] = -1.0 / (z[1] - z[0]) / (z_u[1] - z_u[0]);
        }

        for (std::size_t i = 1; i < nz - 1; ++i) {
            mat_b[i] = -1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
            mat_b[i] -= 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }

        if (nz > 1) {
            std::size_t i = nz - 1;
            mat_b[i] = -1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_b;
    }

    static auto BuildMatrixDiagonalC(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t nz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_c(nz, 0);
        for (std::size_t i = 0; i < nz - 1; ++i) {
            mat_c[i] = 1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_c;
    }

    static auto BuildTridiagMatrix(std::span<const T> z,
                                   std::span<const T> z_u,
                                   std::size_t nz) -> TridiagMatrix {
        if (z.size() < nz) {
            throw std::invalid_argument("The size of `z` is too small.");
        }
        if (z_u.size() < nz + 1) {
            throw std::invalid_argument("The size of `z_u` is too small.");
        }

        return {BuildMatrixDiagonalA(z, z_u, nz),
                BuildMatrixDiagonalB(z, z_u, nz),
                BuildMatrixDiagonalC(z, z_u, nz)};
    }

    /**
     * @brief Copies all three diagonals to the device.
     *
     * @param[in] mat The three diagonals (all allocated to have size \p nz).
     * @param[in] nz Main diagonal size.
     */
    void CopyTridiagMatrixToDevice(TridiagMatrix mat, std::size_t nz) {
        CUCH(cudaMemcpy(cu_mat_a_.get(),
                        mat[0].data(),
                        nz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_b_.get(),
                        mat[1].data(),
                        nz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_c_.get(),
                        mat[2].data(),
                        nz * sizeof(T),
                        cudaMemcpyHostToDevice));
    }

    void SolvePPNs() {
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        Init(core_.cu_cmplx_workspace.value().get(),
             core_.cu_real_workspace.get(),
             core_.ns,
             core_.cu_grid_size,
             core_.cu_block_size);

        FFTPolicy::Forward2D(core_.cu_cmplx_workspace.value().get(),
                             core_.fft_plan);

        TridiagZ(core_.cu_cmplx_workspace.value().get(),
                 cu_cp_workspace_.get(),
                 core_.ns,
                 core_.cu_denom_observers,
                 cu_mat_a_.get(),
                 cu_mat_b_.get(),
                 cu_mat_c_.get(),
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
    }

    void SolveFullNeumann() {
        using detail::runners::common_real::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        FFTPolicy::Forward2D(core_.cu_real_workspace.get(), core_.fft_plan);

        TridiagZ(core_.cu_real_workspace.get(),
                 cu_cp_workspace_.get(),
                 core_.ns,
                 core_.cu_denom_observers,
                 cu_mat_a_.get(),
                 cu_mat_b_.get(),
                 cu_mat_c_.get(),
                 core_.cu_grid_size,
                 core_.cu_block_size);

        FFTPolicy::Backward2D(core_.cu_real_workspace.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.cu_grid_size,
                  core_.cu_block_size);
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
