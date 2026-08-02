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

#ifdef USE_MPI
#include <mpi.h>
#endif  // USE_MPI

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
    virtual void ExecuteGPU(std::span<T> d_phi,
                            std::span<const T> d_rhs,
                            GridSizes<Dim> ngs_phi,
                            GridSizes<Dim> ngs_rhs) = 0;
    virtual auto GetTotalGridSize() const -> std::size_t = 0;

protected:
    static constexpr auto DefaultBlockSize() -> dim3 {
        if constexpr (Dim == 1) {
            return {256};
        } else if constexpr (Dim == 2) {
            return {16, 16};
        } else {
            // Dim == 3
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
     * @brief Constructs a local poisson solver.
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

#ifdef USE_MPI
    /**
     * @brief Constructs a distributed poisson solver.
     *
     * @param ns Local sizes of the grid in each dimension (without ghost
     * cells).
     * @param lens Real lengths of the grid in each dimension (global).
     * @param bcs Boundary conditions for each axis in the order {{west, east},
     * {south, north}, {bottom, top}}.
     * For more information @see boundary_conditions.hpp.
     * @param gns Global sizes of the grid in each dimension (without ghost
     * cells).
     * @param offs Offset of local data.
     * @param comm The MPI communicator.
     * @param approx Desired approximation function. @see approximations.hpp
     * @param cuda_block_size Optional CUDA block size used in every kernel.
     */
    Solver(GridSizes<Dim> ns,
           std::array<T, Dim> lens,
           BCs<Dim> bcs,
           GridSizes<Dim> gns,
           GridOffsets<Dim> offs,
           MPI_Comm comm,
           const IApproximation<T> &approx = SpectralApproximation<T>{},
           dim3 cuda_block_size = ISolver<T, Dim>::DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns,
                lens,
                gns,
                offs,
                comm,
                approx,
                cuda_block_size,
                solver_plan_) {}
#endif  // USE_MPI

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
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_rhs,
                                 cudaMemcpyHostToDevice,
                                 true);

        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(phi.data(),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_phi,
                                 cudaMemcpyDeviceToHost,
                                 false);
    }

    void ExecuteGPU(std::span<T> d_phi,
                    std::span<const T> d_rhs,
                    GridSizes<Dim> ngs_phi,
                    GridSizes<Dim> ngs_rhs) override {
        detail::MemcpyActiveGrid(const_cast<T *>(d_rhs.data()),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_rhs,
                                 cudaMemcpyDeviceToDevice,
                                 true);

        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(d_phi.data(),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_phi,
                                 cudaMemcpyDeviceToDevice,
                                 false);
    }

    auto GetTotalGridSize() const -> std::size_t override {
        return core_.cnt;
    }

private:
    friend struct detail::SolverPlan<T, Dim>;
    using FFTPolicy = detail::FFTPolicy<T, Dim>;

    // Plan has to be initialized before core
    detail::SolverPlan<T, Dim> solver_plan_;
    detail::SolverCore<T, Dim> core_;

    void SolveFullPeriodic() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace1.get(),
             core_.ns,
             core_.ComputeCUDAGridSize(core_.ns),
             core_.cu_block_size);

        FFTPolicy::Forward(core_.cu_cmplx_workspace1.get(), core_.fft_plan);

        DivideByDenoms(core_.cu_cmplx_workspace1.get(),
                       core_.ns,
                       core_.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(core_.ns),
                       core_.cu_block_size,
                       true);

        FFTPolicy::Backward(core_.cu_cmplx_workspace1.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace1.get(),
                  core_.cu_cmplx_workspace1.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(core_.ns),
                  core_.cu_block_size);
    }

    void SolveFullDirichlet() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward(core_.cu_real_workspace1.get(), core_.fft_plan);

        DivideByDenoms(core_.cu_real_workspace1.get(),
                       core_.ns,
                       core_.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(core_.ns),
                       core_.cu_block_size);

        FFTPolicy::Backward(core_.cu_real_workspace1.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace1.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(core_.ns),
                  core_.cu_block_size);
    }

    void SolveFullNeumann() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        FFTPolicy::Forward(core_.cu_real_workspace1.get(), core_.fft_plan);

        DivideByDenoms(core_.cu_real_workspace1.get(),
                       core_.ns,
                       core_.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(core_.ns),
                       core_.cu_block_size,
                       true);

        FFTPolicy::Backward(core_.cu_real_workspace1.get(), core_.fft_plan);

        Normalize(core_.cu_real_workspace1.get(),
                  core_.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(core_.ns),
                  core_.cu_block_size);
    }

    void SolvePPNs() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward1D(core_.cu_real_workspace2.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace2.get(),
             new_layout.ns,
             core_.ComputeCUDAGridSize(new_layout.ns),
             core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace2.get(), core_.cu_cmplx_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace2.get(),
                       new_layout.ns,
                       new_layout.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(new_layout.ns),
                       core_.cu_block_size,
                       true);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace1.get(), core_.cu_cmplx_workspace2.get());
        Normalize(core_.cu_real_workspace2.get(),
                  core_.cu_cmplx_workspace1.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace2.get(), core_.fft_plan);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void SolvePPD() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward1D(core_.cu_real_workspace2.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace2.get(),
             new_layout.ns,
             core_.ComputeCUDAGridSize(new_layout.ns),
             core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace2.get(), core_.cu_cmplx_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace2.get(),
                       new_layout.ns,
                       new_layout.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(new_layout.ns),
                       core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace1.get(), core_.cu_cmplx_workspace2.get());
        Normalize(core_.cu_real_workspace2.get(),
                  core_.cu_cmplx_workspace1.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace2.get(), core_.fft_plan);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void SolveNsPP() {
        // Same body as PPNs.
        // Only is change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        SolvePPNs();
    }

    void SolvePNsNs() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_real_workspace2.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace2.get(),
             new_layout.ns,
             core_.ComputeCUDAGridSize(new_layout.ns),
             core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace2.get(), core_.cu_cmplx_workspace1.get());
        FFTPolicy::Forward1D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace2.get(),
                       new_layout.ns,
                       new_layout.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(new_layout.ns),
                       core_.cu_block_size,
                       true);
        FFTPolicy::Backward1D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace1.get(), core_.cu_cmplx_workspace2.get());
        Normalize(core_.cu_real_workspace2.get(),
                  core_.cu_cmplx_workspace1.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace2.get(), core_.fft_plan);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void SolvePNsP() {
        // Same body as PPNs.
        // Only is change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        SolvePPNs();
    }

    void SolveDDP() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_real_workspace2.get(), core_.fft_plan);
        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace2.get(),
             new_layout.ns,
             core_.ComputeCUDAGridSize(new_layout.ns),
             core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace2.get(), core_.cu_cmplx_workspace1.get());
        FFTPolicy::Forward1D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_cmplx_workspace2.get(),
                       new_layout.ns,
                       new_layout.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(new_layout.ns),
                       core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_cmplx_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace1.get(), core_.cu_cmplx_workspace2.get());
        Normalize(core_.cu_real_workspace2.get(),
                  core_.cu_cmplx_workspace1.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace2.get(), core_.fft_plan);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void Solve2Real1Real() {
        using detail::runners::DivideByDenoms;
        using detail::runners::common_real::Normalize;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_real_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        FFTPolicy::Forward1D(core_.cu_real_workspace1.get(), core_.fft_plan);
        DivideByDenoms(core_.cu_real_workspace1.get(),
                       new_layout.ns,
                       new_layout.offs,
                       core_.cu_denom_observers,
                       core_.ComputeCUDAGridSize(new_layout.ns),
                       core_.cu_block_size);
        FFTPolicy::Backward1D(core_.cu_real_workspace1.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        Normalize(core_.cu_real_workspace2.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);
        FFTPolicy::Backward2D(core_.cu_real_workspace2.get(), core_.fft_plan);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void Solve1Real2Real() {
        // Same body as 2Real1Real.
        // Only is change is in `fft_plan` construction
        // and `reshaper` configuration
        // but this doesn't concern this function.
        Solve2Real1Real();
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
        Init(z, z_u, ns);
    }

#ifdef USE_MPI
    NonuniformZSolver(
        GridSizes<3> ns,
        std::array<T, 2> lens,
        std::span<const T> z,
        std::span<const T> z_u,
        BCs<3> bcs,
        GridSizes<3> gns,
        GridOffsets<3> offs,
        MPI_Comm comm,
        const IApproximation<T> &approx = SpectralApproximation<T>{},
        dim3 cuda_block_size = DefaultBlockSize())
        : solver_plan_(bcs),
          core_(ns,
                lens,
                gns,
                offs,
                comm,
                approx,
                cuda_block_size,
                solver_plan_) {
        Init(z, z_u, gns);
    }
#endif  // USE_MPI

    void Execute(std::span<T> phi,
                 std::span<const T> rhs,
                 GridSizes<3> ngs_phi,
                 GridSizes<3> ngs_rhs) override {
        // `const_cast` necessary due to
        // CUDA C-like API interaction.
        // Nothing is actually modified.
        detail::MemcpyActiveGrid(const_cast<T *>(rhs.data()),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_rhs,
                                 cudaMemcpyHostToDevice,
                                 true);
        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(phi.data(),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_phi,
                                 cudaMemcpyDeviceToHost,
                                 false);
    }

    void ExecuteGPU(std::span<T> d_phi,
                    std::span<const T> d_rhs,
                    GridSizes<3> ngs_phi,
                    GridSizes<3> ngs_rhs) override {
        detail::MemcpyActiveGrid(const_cast<T *>(d_rhs.data()),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_rhs,
                                 cudaMemcpyDeviceToDevice,
                                 true);
        (this->*solver_plan_.solve_method)();

        detail::MemcpyActiveGrid(d_phi.data(),
                                 core_.cu_real_workspace1.get(),
                                 core_.ns,
                                 ngs_phi,
                                 cudaMemcpyDeviceToDevice,
                                 false);
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

    void Init(std::span<const T> z, std::span<const T> z_u, GridSizes<3> gns) {
        core_.cu_block_size.z = 1;

        auto gnz = gns[2];
        AllocateDeviceBuffers(gnz);
        auto mat = BuildTridiagMatrix(z, z_u, gnz);
        CopyTridiagMatrixToDevice(mat, gnz);
    }

    /**
     * @note Shadows the `DefaultBlockSize` method in `ISolver` because
     * `NonuniformZSolver` doesn't parallelize in Z dimension.
     *
     * @return Default block size for `NonuniformZSolver`.
     */
    static constexpr auto DefaultBlockSize() -> dim3 {
        return {16, 16, 1};
    }

    void AllocateDeviceBuffers(std::size_t gnz) {
        using detail::MakeCudaUnique;

        cu_mat_a_ = MakeCudaUnique<T>(gnz);
        cu_mat_b_ = MakeCudaUnique<T>(gnz);
        cu_mat_c_ = MakeCudaUnique<T>(gnz);
        cu_cp_workspace_ =
            MakeCudaUnique<T>(core_.reshaper->MaxLocalElements());
    }

    static auto BuildMatrixDiagonalA(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_a(gnz, 0);
        for (std::size_t i = 1; i < gnz; ++i) {
            mat_a[i] = 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_a;
    }

    static auto BuildMatrixDiagonalB(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_b(gnz, 0);
        if (gnz == 1) {
            mat_b[0] = 0;
        } else {
            mat_b[0] = -1.0 / (z[1] - z[0]) / (z_u[1] - z_u[0]);
        }

        for (std::size_t i = 1; i < gnz - 1; ++i) {
            mat_b[i] = -1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
            mat_b[i] -= 1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }

        if (gnz > 1) {
            std::size_t i = gnz - 1;
            mat_b[i] = -1.0 / (z[i] - z[i - 1]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_b;
    }

    static auto BuildMatrixDiagonalC(std::span<const T> z,
                                     std::span<const T> z_u,
                                     std::size_t gnz) -> TridiagMatrixDiagonal {
        TridiagMatrixDiagonal mat_c(gnz, 0);
        for (std::size_t i = 0; i < gnz - 1; ++i) {
            mat_c[i] = 1.0 / (z[i + 1] - z[i]) / (z_u[i + 1] - z_u[i]);
        }
        return mat_c;
    }

    static auto BuildTridiagMatrix(std::span<const T> z,
                                   std::span<const T> z_u,
                                   std::size_t gnz) -> TridiagMatrix {
        if (z.size() < gnz) {
            throw std::invalid_argument("The size of `z` is too small.");
        }
        if (z_u.size() < gnz + 1) {
            throw std::invalid_argument("The size of `z_u` is too small.");
        }

        return {BuildMatrixDiagonalA(z, z_u, gnz),
                BuildMatrixDiagonalB(z, z_u, gnz),
                BuildMatrixDiagonalC(z, z_u, gnz)};
    }

    /**
     * @brief Copies all three diagonals to the device.
     *
     * @param[in] mat The three diagonals (all allocated to have size \p nz).
     * @param[in] nz Main diagonal size.
     */
    void CopyTridiagMatrixToDevice(TridiagMatrix mat, std::size_t gnz) {
        CUCH(cudaMemcpy(cu_mat_a_.get(),
                        mat[0].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_b_.get(),
                        mat[1].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(cu_mat_c_.get(),
                        mat[2].data(),
                        gnz * sizeof(T),
                        cudaMemcpyHostToDevice));
    }

    void SolvePPNs() {
        using detail::runners::common_cmplx::Init;
        using detail::runners::common_cmplx::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        Init(core_.cu_cmplx_workspace1.get(),
             core_.cu_real_workspace2.get(),
             new_layout.ns,
             core_.ComputeCUDAGridSize(new_layout.ns),
             core_.cu_block_size);
        FFTPolicy::Forward2D(core_.cu_cmplx_workspace1.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace2.get(), core_.cu_cmplx_workspace1.get());
        TridiagZ(core_.cu_cmplx_workspace2.get(),
                 cu_cp_workspace_.get(),
                 new_layout.ns,
                 {new_layout.offs[0], new_layout.offs[1]},
                 core_.cu_denom_observers,
                 cu_mat_a_.get(),
                 cu_mat_b_.get(),
                 cu_mat_c_.get(),
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeCmplx(
            core_.cu_cmplx_workspace1.get(), core_.cu_cmplx_workspace2.get());
        FFTPolicy::Backward2D(core_.cu_cmplx_workspace1.get(), core_.fft_plan);
        Normalize(core_.cu_real_workspace2.get(),
                  core_.cu_cmplx_workspace1.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }

    void SolveFullNeumann() {
        using detail::runners::common_real::Normalize;
        using detail::runners::nonuniform_z::TridiagZ;

        auto new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Forward2D(core_.cu_real_workspace2.get(), core_.fft_plan);

        new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace1.get(), core_.cu_real_workspace2.get());
        TridiagZ(core_.cu_real_workspace1.get(),
                 cu_cp_workspace_.get(),
                 new_layout.ns,
                 {new_layout.offs[0], new_layout.offs[1]},
                 core_.cu_denom_observers,
                 cu_mat_a_.get(),
                 cu_mat_b_.get(),
                 cu_mat_c_.get(),
                 core_.ComputeCUDAGridSize(new_layout.ns),
                 core_.cu_block_size);

        new_layout = core_.reshaper->ReshapeReal(
            core_.cu_real_workspace2.get(), core_.cu_real_workspace1.get());
        FFTPolicy::Backward2D(core_.cu_real_workspace2.get(), core_.fft_plan);
        Normalize(core_.cu_real_workspace2.get(),
                  new_layout.ns,
                  core_.norm_factor,
                  core_.ComputeCUDAGridSize(new_layout.ns),
                  core_.cu_block_size);

        core_.reshaper->ReshapeReal(core_.cu_real_workspace1.get(),
                                    core_.cu_real_workspace2.get());
    }
};
}  // namespace poisson_solver

#endif  // SOLVER_HPP
